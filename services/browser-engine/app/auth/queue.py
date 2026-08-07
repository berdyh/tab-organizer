"""Parallel authentication request queue."""

import asyncio
import hashlib
import ipaddress
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

from cryptography.fernet import Fernet

# OS keyring coordinates for the durable credential-store key. DORMANT in the
# shipped image: `keyring` is not listed in services/browser-engine/requirements.txt
# (verified: it appears in no requirements*.txt under services/), so the `import
# keyring` in `_load_key_from_keyring` below always raises ImportError in every
# built container and this path is skipped. Resolution falls through to
# CREDENTIAL_ENCRYPTION_KEY (see .env.example), which ships blank -- so the
# credential store is fail-closed and the auth-scraping feature is INOPERATIVE
# until an operator sets that variable. See ../MODULE.md for the full picture.
_KEYRING_SERVICE = "tab-organizer-credential-store"
_KEYRING_USERNAME = "fernet-key"


class CredentialStoreError(RuntimeError):
    """Raised when credential storage cannot be secured (fail-closed).

    Carries a structured {code, cause, fix} so callers can surface an
    actionable error instead of silently encrypting with a throwaway key.
    """

    def __init__(self, code: str, cause: str, fix: str):
        self.code = code
        self.cause = cause
        self.fix = fix
        super().__init__(f"{code}: {cause} Fix: {fix}")

    def to_dict(self) -> dict:
        return {"code": self.code, "cause": self.cause, "fix": self.fix}


@dataclass
class AuthRequest:
    """A pending authentication request."""

    id: str
    domain: str
    url: str
    auth_type: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"  # pending, provided, failed, expired
    session_id: Optional[str] = None
    form_fields: Optional[list[str]] = None
    oauth_provider: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def canonical_credential_host(value: Optional[str]) -> str:
    """Reduce a domain string or URL to the store's canonical host key.

    Mirrors `AuthQueue._extract_domain`'s normalization (lowercase, leading
    `www.` removed) minus the port. That mirroring is the whole point: the
    store ALREADY treats `example.com` and `www.example.com` as one key in both
    directions — a credential submitted while scraping `www` is stored under
    the apex and handed back for the apex, and one submitted for the apex is
    handed back for `www`. Scope coverage has to agree with the lookup that
    selected the credential in the first place, or the redirect rule
    contradicts it.

    Accepts a bare domain, a `host:port`, a bracketed IPv6 literal, or a full
    URL, because `domain` on the submission API is a free-text string.
    """
    raw = (value or "").strip()
    if "//" in raw:
        raw = urlparse(raw).netloc or ""
    raw = raw.rsplit("@", 1)[-1]
    if raw.startswith("["):
        raw = raw[1:].partition("]")[0]
    else:
        raw = raw.partition(":")[0]
    raw = raw.rstrip(".").lower()
    if raw.startswith("www.") and len(raw) > 4:
        raw = raw[4:]
    return raw


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class CredentialScope:
    """The set of hosts a stored credential may be sent to.

    Recorded AT SUBMISSION TIME, never inferred from a host seen later. That
    distinction is the design: "same registrable domain" is the rule everyone
    reaches for and it cannot be computed correctly without a public-suffix
    list (this repo carries none and must not grow one for this) — without a
    PSL, `co.uk` and `s3.amazonaws.com` look like registrable domains and every
    tenant of them becomes one scope. So the parent domain is whatever the
    submitter named, and widening past exact host is an EXPLICIT per-credential
    opt-in rather than something derived from string shape.

    `include_subdomains` defaults to False, and the submission API sends only a
    domain string, so every credential stored through it gets the exact-host
    scope. Exact host here means the store's own host key, i.e. the apex/`www`
    pair the store already treats as one credential — see
    `canonical_credential_host`. That is what unbreaks the ordinary
    `example.com -> www.example.com` canonical redirect without handing the
    cookie to any host the store would not already have served it to.
    """

    domain: str
    include_subdomains: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", canonical_credential_host(self.domain))
        if not self.domain:
            raise ValueError("A credential scope requires a domain.")
        # Coarse guard, deliberately NOT a public-suffix check: subdomain
        # opt-in on a bare TLD (`com`) or an IP literal is never a legitimate
        # submission and is catastrophic if it happens. A two-label public
        # suffix such as `co.uk` still passes, which is why the flag stays an
        # explicit operator statement about a domain they named — nothing here
        # ever derives a parent domain from a host.
        if self.include_subdomains and (
            _is_ip_literal(self.domain) or "." not in self.domain
        ):
            raise ValueError(
                "include_subdomains requires a multi-label parent domain, "
                f"not {self.domain!r}."
            )

    def covers(self, host: Optional[str]) -> bool:
        """Whether `host` is inside this credential's recorded scope.

        The subdomain test compares on a LABEL BOUNDARY (`.` + domain), not a
        bare string suffix: `"evil-example.com".endswith("example.com")` is
        True and would hand an attacker-registered lookalike the user's
        credential, which is exactly the mistake this predicate exists to
        avoid. `evil-example.com` ends with `-example.com`, never
        `.example.com`, so it is refused at both branches.
        """
        candidate = canonical_credential_host(host)
        if not candidate:
            return False
        if candidate == self.domain:
            return True
        if not self.include_subdomains:
            return False
        return candidate.endswith(f".{self.domain}")


def _default_scope(domain: str) -> Optional[CredentialScope]:
    """Exact-host scope for a submission that named no scope.

    Returns None — meaning "no scope recorded" — when the domain cannot be
    canonicalized at all. Consumers read None as the strictest possible rule
    (exact host, compared against the origin the request started at), so an
    unparseable domain degrades toward less credential reach, never more.
    """
    try:
        return CredentialScope(domain=domain)
    except ValueError:
        return None


@dataclass
class StoredCredentials:
    """Encrypted credentials storage."""

    domain: str
    auth_type: str
    encrypted_data: bytes
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    # None means no scope was recorded (a credential stored before scopes
    # existed, or one written by a caller that predates them). Consumers must
    # treat that as exact-host, i.e. the pre-scope behavior.
    scope: Optional[CredentialScope] = None


class CredentialStore:
    """Secure credential storage with encryption."""

    def __init__(self, encryption_key: Optional[str] = None):
        self._credentials: dict[str, StoredCredentials] = {}
        # Fail closed: never invent a throwaway key. Resolution is deferred to
        # a stored error so the service still starts; store() is what refuses.
        self._fernet: Optional[Fernet] = None
        self._init_error: Optional[CredentialStoreError] = None
        try:
            self._fernet = self._resolve_fernet(encryption_key)
        except CredentialStoreError as error:
            self._init_error = error

    def _load_key_from_keyring(self) -> Optional[str]:
        """Return a durable Fernet key from the OS keyring, if reachable.

        Returns None when no keyring backend is available in-process, so the
        caller falls through to CREDENTIAL_ENCRYPTION_KEY / fail-closed. In the
        shipped image this ALWAYS returns None: `keyring` is not an installed
        dependency (not in requirements.txt), so the import below always raises
        ImportError. The branch is kept for a future environment that installs
        it explicitly, not one this repo builds today.
        """
        try:
            import keyring
        except ImportError:
            return None
        try:
            existing = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
        except Exception:
            return None
        if existing:
            return existing
        # No key yet — mint a durable one and persist it to the keyring. This is
        # not "inventing a key": it survives restarts and lives in secure OS
        # storage, unlike a per-process throwaway.
        new_key = Fernet.generate_key().decode()
        try:
            keyring.set_password(_KEYRING_SERVICE, _KEYRING_USERNAME, new_key)
        except Exception:
            return None
        return new_key

    def _resolve_fernet(self, encryption_key: Optional[str]) -> Fernet:
        """Resolve the encryption key: explicit arg, OS keyring, then env var.

        The OS-keyring step is a no-op in the shipped image (see
        `_load_key_from_keyring`), so in practice this resolves to
        CREDENTIAL_ENCRYPTION_KEY or nothing. Raises CredentialStoreError
        (fail-closed) when none is available.
        """
        key = encryption_key or self._load_key_from_keyring()
        if not key:
            key = os.getenv("CREDENTIAL_ENCRYPTION_KEY")
        if not key:
            raise CredentialStoreError(
                code="credential_store_unconfigured",
                cause=(
                    "No OS keyring backend is available in-process and "
                    "CREDENTIAL_ENCRYPTION_KEY is not set."
                ),
                fix=(
                    "Set CREDENTIAL_ENCRYPTION_KEY to a Fernet.generate_key() "
                    "value, or run where an OS keyring (Secret Service) is "
                    "reachable. The store refuses credentials until then."
                ),
            )
        try:
            return Fernet(key.encode() if isinstance(key, str) else key)
        except Exception as exc:
            raise CredentialStoreError(
                code="credential_key_invalid",
                cause=f"The configured credential key is not a valid Fernet key: {exc}.",
                fix=(
                    "CREDENTIAL_ENCRYPTION_KEY must be a url-safe base64-encoded "
                    "32-byte key produced by Fernet.generate_key()."
                ),
            )

    @property
    def is_ready(self) -> bool:
        """Whether the store can encrypt (a key was resolved)."""
        return self._fernet is not None

    def store(
        self,
        domain: str,
        auth_type: str,
        credentials: dict,
        expires_at: Optional[datetime] = None,
        scope: Optional[CredentialScope] = None,
    ) -> None:
        """Store encrypted credentials for a domain.

        Raises CredentialStoreError (fail-closed) if no encryption key could be
        resolved; callers must surface this rather than accept credentials.

        `scope` records which hosts the credential may travel to. Omitting it
        records the exact-host default rather than leaving the credential
        scopeless, so the scraper never has to guess for anything this method
        wrote.
        """
        import json

        if self._fernet is None:
            assert self._init_error is not None
            raise self._init_error

        data = json.dumps(credentials).encode()
        encrypted = self._fernet.encrypt(data)

        self._credentials[domain] = StoredCredentials(
            domain=domain,
            auth_type=auth_type,
            encrypted_data=encrypted,
            expires_at=expires_at,
            scope=scope or _default_scope(domain),
        )

    def retrieve(self, domain: str) -> Optional[dict]:
        """Retrieve and decrypt credentials for a domain."""
        import json

        stored = self._credentials.get(domain)
        if not stored:
            return None

        # Check expiration
        if stored.expires_at and datetime.utcnow() > stored.expires_at:
            del self._credentials[domain]
            return None

        # No key configured means nothing can be decrypted. This was already the
        # outcome -- `None.decrypt` raised AttributeError straight into the
        # `except Exception` below -- so the explicit check changes nothing but
        # makes the fail-closed path deliberate instead of incidental.
        if self._fernet is None:
            return None

        try:
            decrypted = self._fernet.decrypt(stored.encrypted_data)
            return json.loads(decrypted.decode())
        except Exception:
            return None

    def get_scope(self, domain: str) -> Optional[CredentialScope]:
        """Return the scope recorded for a domain's credentials, if any.

        None means either "no such credential" or "stored without a scope";
        both must be read as the strictest rule, so the two cases deliberately
        do not need to be told apart.
        """
        stored = self._credentials.get(domain)
        if not stored:
            return None
        if stored.expires_at and datetime.utcnow() > stored.expires_at:
            del self._credentials[domain]
            return None
        return stored.scope

    def has_credentials(self, domain: str) -> bool:
        """Check if credentials exist for a domain."""
        stored = self._credentials.get(domain)
        if not stored:
            return False
        if stored.expires_at and datetime.utcnow() > stored.expires_at:
            del self._credentials[domain]
            return False
        return True

    def remove(self, domain: str) -> bool:
        """Remove credentials for a domain."""
        if domain in self._credentials:
            del self._credentials[domain]
            return True
        return False

    def clear(self) -> int:
        """Clear all stored credentials."""
        count = len(self._credentials)
        self._credentials.clear()
        return count


class AuthQueue:
    """
    Parallel authentication request queue.

    Allows scraping to continue for public sites while waiting
    for credentials for authenticated sites.
    """

    def __init__(self, encryption_key: Optional[str] = None):
        self._pending: dict[str, AuthRequest] = {}  # domain → request
        self._credential_store = CredentialStore(encryption_key)
        self._lock = asyncio.Lock()

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain

    def _generate_id(self, domain: str) -> str:
        """Generate unique ID for auth request."""
        timestamp = datetime.utcnow().isoformat()
        return hashlib.sha256(f"{domain}:{timestamp}".encode()).hexdigest()[:16]

    async def request_auth(
        self,
        url: str,
        auth_type: str,
        session_id: Optional[str] = None,
        form_fields: Optional[list[str]] = None,
        oauth_provider: Optional[str] = None,
    ) -> AuthRequest:
        """
        Queue an authentication request.

        If credentials already exist for the domain, returns immediately.
        Otherwise, queues the request for user input.
        """
        domain = self._extract_domain(url)

        async with self._lock:
            # Check if we already have credentials
            if self._credential_store.has_credentials(domain):
                return AuthRequest(
                    id="existing",
                    domain=domain,
                    url=url,
                    auth_type=auth_type,
                    status="provided",
                )

            # Check if request already pending
            if domain in self._pending:
                return self._pending[domain]

            # Create new request
            request = AuthRequest(
                id=self._generate_id(domain),
                domain=domain,
                url=url,
                auth_type=auth_type,
                session_id=session_id,
                form_fields=form_fields,
                oauth_provider=oauth_provider,
            )

            self._pending[domain] = request
            return request

    async def provide_credentials(
        self,
        domain: str,
        credentials: dict,
        expires_at: Optional[datetime] = None,
        include_subdomains: bool = False,
    ) -> bool:
        """
        Provide credentials for a pending auth request.

        Triggers any registered callbacks for the domain.

        `include_subdomains` is the explicit, per-credential opt-in that lets
        the credential reach `*.domain`; it defaults to False, which is what
        the HTTP submission surface (`POST /auth/credentials`, which carries
        only a domain string) therefore records for every credential. Passing
        True for a domain that cannot carry subdomains raises ValueError rather
        than silently storing a scope that means something else.
        """
        scope = (
            CredentialScope(domain=domain, include_subdomains=True)
            if include_subdomains
            else None
        )
        async with self._lock:
            request = self._pending.get(domain)
            if not request:
                return False

            # Store credentials
            self._credential_store.store(
                domain=domain,
                auth_type=request.auth_type,
                credentials=credentials,
                expires_at=expires_at,
                scope=scope,
            )

            # Update request status
            request.status = "provided"

            # Remove from pending
            del self._pending[domain]

        return True

    def get_credentials(self, url: str) -> Optional[dict]:
        """Get stored credentials for a URL's domain."""
        domain = self._extract_domain(url)
        return self._credential_store.retrieve(domain)

    def get_credential_scope(self, url: str) -> Optional[CredentialScope]:
        """Scope recorded for the credentials a URL would be scraped with.

        The scraper asks for this alongside `get_credentials` so the redirect
        rule can test "is this hop still inside the scope the user submitted
        for" instead of "is it the same host I started on".
        """
        domain = self._extract_domain(url)
        return self._credential_store.get_scope(domain)

    def has_credentials(self, url: str) -> bool:
        """Check if credentials exist for a URL's domain."""
        domain = self._extract_domain(url)
        return self._credential_store.has_credentials(domain)

    def get_pending_count(self) -> int:
        """Get count of pending auth requests."""
        return len(self._pending)

    def get_pending_for_session(self, session_id: str) -> list[AuthRequest]:
        """Get pending auth requests for a specific session."""
        return [r for r in self._pending.values() if r.session_id == session_id]

    async def cancel_request(self, domain: str) -> bool:
        """Cancel a pending auth request."""
        async with self._lock:
            if domain in self._pending:
                self._pending[domain].status = "cancelled"
                del self._pending[domain]
                return True
            return False

    async def expire_old_requests(self, max_age_seconds: int = 3600) -> int:
        """Expire old pending requests."""
        now = datetime.utcnow()
        expired = []

        async with self._lock:
            for domain, request in self._pending.items():
                age = (now - request.created_at).total_seconds()
                if age > max_age_seconds:
                    expired.append(domain)

            for domain in expired:
                self._pending[domain].status = "expired"
                del self._pending[domain]

        return len(expired)

    def to_dict(self) -> dict:
        """Convert queue state to dictionary."""
        return {
            "pending_count": len(self._pending),
            "pending": [
                {
                    "id": r.id,
                    "domain": r.domain,
                    "url": r.url,
                    "auth_type": r.auth_type,
                    "created_at": r.created_at.isoformat(),
                    "form_fields": r.form_fields,
                    "oauth_provider": r.oauth_provider,
                }
                for r in self._pending.values()
            ],
        }
