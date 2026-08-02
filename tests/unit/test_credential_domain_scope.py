"""Stored credentials carry the domain scope they were submitted for.

Before this, the store held a bare ``{name: value}`` dict with no domain
metadata, so the redirect rule had nothing to compare against except the exact
host the request started on — and an ordinary
``https://example.com/report -> https://www.example.com/report`` canonical
redirect dropped the user's cookies and captured the logged-out page.

The fix records a `CredentialScope` at submission time. These tests cover both
halves of it: the drops that should stop happening, and the retentions that
must remain impossible. The sharpest of the latter is the lookalike domain —
``evil-example.com`` must never match ``example.com``, which a bare
``endswith`` suffix test would allow.

The wire-level tests drive the real `httpx.AsyncClient` through a mock
transport so they observe the headers httpx actually emits, not the scraper's
own bookkeeping.
"""

import socket
import sys
import types

import httpx
import pytest
from cryptography.fernet import Fernet

from services import url_safety


def _install_playwright_stub() -> None:
    """Let unit tests import browser-engine code without browser binaries."""
    playwright_module = types.ModuleType("playwright")
    async_api = types.ModuleType("playwright.async_api")
    async_api.Browser = object
    async_api.Page = object
    async_api.TimeoutError = TimeoutError
    async_api.async_playwright = lambda: None
    playwright_module.async_api = async_api
    sys.modules["playwright"] = playwright_module
    sys.modules["playwright.async_api"] = async_api


try:
    import playwright.async_api  # noqa: F401
except ModuleNotFoundError:
    _install_playwright_stub()

from services.browser_engine.app.auth.queue import (  # noqa: E402
    AuthQueue,
    CredentialScope,
    CredentialStore,
    StoredCredentials,
    canonical_credential_host,
)
from services.browser_engine.app.scraper.engine import ScraperEngine  # noqa: E402

COOKIE_CREDENTIALS = {"type": "cookie", "cookies": {"session": "COOKIEVAL"}}


class RedirectChain:
    """Serve a fixed sequence of Location headers, recording every hop."""

    def __init__(self, *locations):
        self._locations = locations
        self.hops = []

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.hops.append(
            {
                "host": request.headers.get("host"),
                "cookie": request.headers.get("cookie"),
            }
        )
        index = len(self.hops) - 1
        location = self._locations[index] if index < len(self._locations) else None
        if location:
            return httpx.Response(302, headers={"location": location})
        return httpx.Response(200, html="<html><title>Page</title>body</html>")


@pytest.fixture
def public_dns(monkeypatch):
    """Resolve every scrape host to one public IP so hops stay in-process."""
    monkeypatch.delenv("SCRAPE_ALLOW_PRIVATE_NETWORKS", raising=False)
    monkeypatch.setattr(
        url_safety.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )


def _pin_client_transport(monkeypatch, chain: RedirectChain) -> None:
    """Make the clients built inside the scraper speak to the mock transport."""
    real_client = httpx.AsyncClient
    transport = chain.transport

    def factory(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


def _no_keyring(monkeypatch):
    monkeypatch.setattr(CredentialStore, "_load_key_from_keyring", lambda self: None)


def _explicit_key_store(monkeypatch) -> CredentialStore:
    """The store is inoperative unless a key is supplied; supply one."""
    _no_keyring(monkeypatch)
    monkeypatch.delenv("CREDENTIAL_ENCRYPTION_KEY", raising=False)
    store = CredentialStore(encryption_key=Fernet.generate_key().decode())
    assert store.is_ready is True
    return store


# --- The scope predicate --------------------------------------------------


class TestCredentialScopeCoverage:
    def test_exact_host_is_covered(self):
        assert CredentialScope("example.com").covers("example.com") is True

    def test_apex_and_www_are_one_scope_by_default(self):
        """Not a widening: the store already keys both onto one credential.

        `AuthQueue._extract_domain` strips a leading `www.` when storing AND
        when retrieving, so a credential submitted at `www.example.com` is
        already served for `example.com` and vice versa. The scope agreeing
        with that adds no host the store would not already hand it to.
        """
        assert CredentialScope("example.com").covers("www.example.com") is True
        assert CredentialScope("www.example.com").covers("example.com") is True
        assert CredentialScope("www.example.com").domain == "example.com"

    def test_sibling_subdomain_is_not_covered_by_default(self):
        scope = CredentialScope("example.com")
        assert scope.covers("api.example.com") is False
        assert scope.covers("uploads.example.com") is False
        assert scope.covers("www.www.example.com") is False

    def test_lookalike_domain_is_never_covered(self):
        """The mistake a bare `endswith` suffix test would make.

        `"evil-example.com".endswith("example.com")` is True, so a suffix rule
        hands an attacker-registered lookalike the user's session cookie. The
        label-boundary test refuses it under both flag settings.
        """
        for include_subdomains in (False, True):
            scope = CredentialScope("example.com", include_subdomains)
            assert scope.covers("evil-example.com") is False
            assert scope.covers("evilexample.com") is False
            assert scope.covers("a.evil-example.com") is False
            assert scope.covers("example.com.evil.tld") is False
            assert scope.covers("notexample.com") is False

    def test_subdomain_opt_in_covers_labels_below_the_recorded_parent(self):
        scope = CredentialScope("example.com", include_subdomains=True)
        assert scope.covers("api.example.com") is True
        assert scope.covers("a.b.example.com") is True
        assert scope.covers("example.com") is True
        assert scope.covers("other.tld") is False

    def test_subdomain_opt_in_refuses_a_parent_that_cannot_have_subdomains(self):
        """Coarse guard only — deliberately not a public-suffix check."""
        for bad in ("com", "localhost", "127.0.0.1", "::1"):
            with pytest.raises(ValueError):
                CredentialScope(bad, include_subdomains=True)
        # The same domains are fine as exact-host scopes.
        assert CredentialScope("localhost").covers("localhost") is True

    def test_scope_normalizes_ports_userinfo_case_and_trailing_dot(self):
        assert CredentialScope("EXAMPLE.com:8443").domain == "example.com"
        assert CredentialScope("https://user@Example.COM./x").domain == "example.com"
        assert CredentialScope("example.com").covers("EXAMPLE.COM.") is True

    def test_empty_or_unparseable_domain_is_refused(self):
        for bad in ("", "   ", "https://"):
            with pytest.raises(ValueError):
                CredentialScope(bad)

    def test_canonical_host_helper_matches_the_store_key_rule(self):
        assert canonical_credential_host("www.example.com") == "example.com"
        assert canonical_credential_host("[::1]:9222") == "::1"
        assert canonical_credential_host(None) == ""


# --- What the store records -----------------------------------------------


class TestStoredScopeMetadata:
    def test_store_records_an_exact_host_scope_by_default(self, monkeypatch):
        store = _explicit_key_store(monkeypatch)

        store.store("example.com", "cookie", {"session": "x"})

        scope = store.get_scope("example.com")
        assert scope == CredentialScope("example.com", include_subdomains=False)

    def test_store_keeps_an_explicit_subdomain_scope(self, monkeypatch):
        store = _explicit_key_store(monkeypatch)

        store.store(
            "example.com",
            "cookie",
            {"session": "x"},
            scope=CredentialScope("example.com", include_subdomains=True),
        )

        assert store.get_scope("example.com").include_subdomains is True

    def test_a_credential_stored_without_a_scope_reports_none(self, monkeypatch):
        """Migration case: nothing recorded a scope, so nothing is claimed."""
        store = _explicit_key_store(monkeypatch)
        store._credentials["legacy.example"] = StoredCredentials(
            domain="legacy.example",
            auth_type="cookie",
            encrypted_data=b"",
        )

        assert store.get_scope("legacy.example") is None

    def test_store_is_still_fail_closed_without_a_key(self, monkeypatch):
        _no_keyring(monkeypatch)
        monkeypatch.delenv("CREDENTIAL_ENCRYPTION_KEY", raising=False)

        store = CredentialStore()

        assert store.is_ready is False
        assert store.get_scope("example.com") is None

    @pytest.mark.asyncio
    async def test_queue_default_submission_records_the_exact_host_scope(
        self, monkeypatch
    ):
        """The HTTP surface sends only a domain string; this is what it gets."""
        _no_keyring(monkeypatch)
        monkeypatch.delenv("CREDENTIAL_ENCRYPTION_KEY", raising=False)
        queue = AuthQueue(encryption_key=Fernet.generate_key().decode())
        await queue.request_auth("https://example.com/login", "cookie")

        assert await queue.provide_credentials("example.com", {"session": "x"}) is True

        scope = queue.get_credential_scope("https://example.com/report")
        assert scope == CredentialScope("example.com")
        assert queue.get_credential_scope("https://www.example.com/report") == scope
        assert queue.get_credential_scope("https://api.example.com/x") is None

    @pytest.mark.asyncio
    async def test_queue_subdomain_opt_in_is_explicit(self, monkeypatch):
        _no_keyring(monkeypatch)
        monkeypatch.delenv("CREDENTIAL_ENCRYPTION_KEY", raising=False)
        queue = AuthQueue(encryption_key=Fernet.generate_key().decode())
        await queue.request_auth("https://example.com/login", "cookie")

        await queue.provide_credentials(
            "example.com", {"session": "x"}, include_subdomains=True
        )

        assert queue.get_credential_scope("https://example.com/x").include_subdomains


# --- What rides the wire --------------------------------------------------


@pytest.mark.asyncio
async def test_apex_to_www_redirect_keeps_credentials_in_scope(monkeypatch, public_dns):
    """The flow the finding named: an ordinary canonical redirect."""
    chain = RedirectChain("https://www.example.com/report")
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_cookie_auth(
        "https://example.com/report",
        COOKIE_CREDENTIALS,
        CredentialScope("example.com"),
    )

    assert [hop["host"] for hop in chain.hops] == ["example.com", "www.example.com"]
    assert [hop["cookie"] for hop in chain.hops] == [
        "session=COOKIEVAL",
        "session=COOKIEVAL",
    ]
    assert result.auth_used is True
    assert "credential_scope_drop" not in result.metadata


@pytest.mark.asyncio
async def test_host_outside_the_scope_is_still_dropped(monkeypatch, public_dns):
    chain = RedirectChain("https://api.example.com/report")
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_cookie_auth(
        "https://example.com/report",
        COOKIE_CREDENTIALS,
        CredentialScope("example.com"),
    )

    assert [hop["cookie"] for hop in chain.hops] == ["session=COOKIEVAL", None]
    assert result.auth_used is False
    drop = result.metadata["credential_scope_drop"]
    assert drop["dropped_at_host"] == "api.example.com"
    assert drop["credential_scope_domain"] == "example.com"
    assert drop["credential_scope_subdomains"] is False


@pytest.mark.asyncio
async def test_lookalike_host_is_dropped_even_with_subdomains_allowed(
    monkeypatch, public_dns
):
    """`evil-example.com` is a different registration, not a subdomain."""
    chain = RedirectChain("https://evil-example.com/collect")
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_cookie_auth(
        "https://example.com/report",
        COOKIE_CREDENTIALS,
        CredentialScope("example.com", include_subdomains=True),
    )

    assert [hop["host"] for hop in chain.hops] == [
        "example.com",
        "evil-example.com",
    ]
    assert [hop["cookie"] for hop in chain.hops] == ["session=COOKIEVAL", None]
    assert result.auth_used is False


@pytest.mark.asyncio
async def test_subdomain_opt_in_keeps_credentials_on_a_real_subdomain(
    monkeypatch, public_dns
):
    chain = RedirectChain("https://api.example.com/report")
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_cookie_auth(
        "https://example.com/report",
        COOKIE_CREDENTIALS,
        CredentialScope("example.com", include_subdomains=True),
    )

    assert [hop["cookie"] for hop in chain.hops] == [
        "session=COOKIEVAL",
        "session=COOKIEVAL",
    ]
    assert result.auth_used is True


@pytest.mark.asyncio
async def test_scopeless_credentials_keep_the_strict_exact_host_rule(
    monkeypatch, public_dns
):
    """Documented default for anything stored before scopes existed.

    No scope recorded means no claim about `www`, so the apex -> www hop drops
    exactly as it did before this change — the migration behavior is "nothing
    gets wider without a recorded scope".
    """
    chain = RedirectChain("https://www.example.com/report")
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_cookie_auth(
        "https://example.com/report",
        COOKIE_CREDENTIALS,
        None,
    )

    assert [hop["cookie"] for hop in chain.hops] == ["session=COOKIEVAL", None]
    assert result.auth_used is False
    assert result.metadata["credential_scope_drop"]["credential_scope_domain"] is None


@pytest.mark.asyncio
async def test_scope_is_authoritative_even_on_the_first_hop(monkeypatch, public_dns):
    """A scope narrows as well as widens: it is not merely an extra allowance."""
    chain = RedirectChain()
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_cookie_auth(
        "https://other.tld/report",
        COOKIE_CREDENTIALS,
        CredentialScope("example.com"),
    )

    assert [hop["cookie"] for hop in chain.hops] == [None]
    assert result.auth_used is False


@pytest.mark.asyncio
async def test_drop_stays_sticky_when_the_chain_returns_into_scope(
    monkeypatch, public_dns
):
    """Re-arming would let the middle hop choose the authenticated request."""
    chain = RedirectChain(
        "https://attacker.tld/bounce",
        "https://www.example.com/report",
    )
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_cookie_auth(
        "https://example.com/report",
        COOKIE_CREDENTIALS,
        CredentialScope("example.com", include_subdomains=True),
    )

    assert [hop["cookie"] for hop in chain.hops] == ["session=COOKIEVAL", None, None]
    assert result.auth_used is False


@pytest.mark.asyncio
async def test_scheme_downgrade_inside_the_scope_is_still_a_crossing(
    monkeypatch, public_dns
):
    chain = RedirectChain("http://www.example.com/report")
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_cookie_auth(
        "https://example.com/report",
        COOKIE_CREDENTIALS,
        CredentialScope("example.com", include_subdomains=True),
    )

    assert [hop["cookie"] for hop in chain.hops] == ["session=COOKIEVAL", None]
    assert result.auth_used is False


@pytest.mark.asyncio
async def test_port_change_inside_the_scope_is_still_a_crossing(
    monkeypatch, public_dns
):
    chain = RedirectChain("https://www.example.com:8443/report")
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_cookie_auth(
        "https://example.com/report",
        COOKIE_CREDENTIALS,
        CredentialScope("example.com", include_subdomains=True),
    )

    assert [hop["cookie"] for hop in chain.hops] == ["session=COOKIEVAL", None]
    assert result.auth_used is False


@pytest.mark.asyncio
async def test_scrape_url_uses_the_scope_recorded_by_the_auth_queue(
    monkeypatch, public_dns
):
    """End to end: submission through the queue unbreaks apex -> www."""
    _no_keyring(monkeypatch)
    monkeypatch.delenv("CREDENTIAL_ENCRYPTION_KEY", raising=False)
    queue = AuthQueue(encryption_key=Fernet.generate_key().decode())
    await queue.request_auth("https://example.com/report", "cookie")
    await queue.provide_credentials("example.com", COOKIE_CREDENTIALS)

    chain = RedirectChain("https://www.example.com/report")
    _pin_client_transport(monkeypatch, chain)
    engine = ScraperEngine(respect_robots=False)
    engine.set_auth_queue(queue)

    result = await engine.scrape_url("https://example.com/report")

    assert [hop["cookie"] for hop in chain.hops] == [
        "session=COOKIEVAL",
        "session=COOKIEVAL",
    ]
    assert result.auth_used is True


@pytest.mark.asyncio
async def test_scrape_url_degrades_to_strict_for_a_queue_without_scopes(
    monkeypatch, public_dns
):
    """An injected queue that predates scopes must not raise, and must not widen."""

    class LegacyQueue:
        def has_credentials(self, url):
            return True

        def get_credentials(self, url):
            return COOKIE_CREDENTIALS

    chain = RedirectChain("https://www.example.com/report")
    _pin_client_transport(monkeypatch, chain)
    engine = ScraperEngine(respect_robots=False)
    engine.set_auth_queue(LegacyQueue())

    result = await engine.scrape_url("https://example.com/report")

    assert [hop["cookie"] for hop in chain.hops] == ["session=COOKIEVAL", None]
    assert result.auth_used is False
