"""Parallel authentication request queue."""

import asyncio
import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Awaitable
from urllib.parse import urlparse

from cryptography.fernet import Fernet


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


@dataclass
class StoredCredentials:
    """Encrypted credentials storage."""
    domain: str
    auth_type: str
    encrypted_data: bytes
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


class CredentialStore:
    """Secure credential storage with encryption."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        key = encryption_key or os.getenv("CREDENTIAL_ENCRYPTION_KEY")
        if key:
            # Ensure key is valid Fernet key (32 url-safe base64-encoded bytes)
            self._fernet = Fernet(key.encode() if isinstance(key, str) else key)
        else:
            # Generate a session-scoped key
            self._fernet = Fernet(Fernet.generate_key())
        
        self._credentials: dict[str, StoredCredentials] = {}
    
    def store(
        self,
        domain: str,
        auth_type: str,
        credentials: dict,
        expires_at: Optional[datetime] = None,
    ) -> None:
        """Store encrypted credentials for a domain."""
        import json
        
        data = json.dumps(credentials).encode()
        encrypted = self._fernet.encrypt(data)
        
        self._credentials[domain] = StoredCredentials(
            domain=domain,
            auth_type=auth_type,
            encrypted_data=encrypted,
            expires_at=expires_at,
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
        
        try:
            decrypted = self._fernet.decrypt(stored.encrypted_data)
            return json.loads(decrypted.decode())
        except Exception:
            return None
    
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
    
    def __init__(self):
        self._pending: dict[str, AuthRequest] = {}  # domain → request
        self._credential_store = CredentialStore()
        self._callbacks: dict[str, list[Callable[[dict], Awaitable[None]]]] = {}
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
    ) -> bool:
        """
        Provide credentials for a pending auth request.
        
        Triggers any registered callbacks for the domain.
        """
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
            )
            
            # Update request status
            request.status = "provided"
            
            # Remove from pending
            del self._pending[domain]
        
        # Trigger callbacks
        if domain in self._callbacks:
            for callback in self._callbacks[domain]:
                try:
                    await callback(credentials)
                except Exception:
                    pass
            del self._callbacks[domain]
        
        return True
    
    def get_credentials(self, url: str) -> Optional[dict]:
        """Get stored credentials for a URL's domain."""
        domain = self._extract_domain(url)
        return self._credential_store.retrieve(domain)
    
    def has_credentials(self, url: str) -> bool:
        """Check if credentials exist for a URL's domain."""
        domain = self._extract_domain(url)
        return self._credential_store.has_credentials(domain)
    
    def get_pending(self) -> list[AuthRequest]:
        """Get all pending auth requests."""
        return list(self._pending.values())
    
    def get_pending_count(self) -> int:
        """Get count of pending auth requests."""
        return len(self._pending)
    
    def get_pending_for_session(self, session_id: str) -> list[AuthRequest]:
        """Get pending auth requests for a specific session."""
        return [
            r for r in self._pending.values()
            if r.session_id == session_id
        ]
    
    async def register_callback(
        self,
        domain: str,
        callback: Callable[[dict], Awaitable[None]],
    ) -> None:
        """Register a callback for when credentials are provided."""
        if domain not in self._callbacks:
            self._callbacks[domain] = []
        self._callbacks[domain].append(callback)
    
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
