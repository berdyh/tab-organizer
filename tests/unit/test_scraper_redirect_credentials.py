"""Stored credentials must not follow a redirect off their own origin.

The scraper follows redirects manually (`_safe_httpx_request`) while reusing a
single httpx client. httpx applies client-level auth and a bare-domain cookie
jar to every request unconditionally, so attaching either to the CLIENT hands
the user's intranet credentials to whatever host a redirect names. These tests
drive the real `httpx.AsyncClient` through a mock transport so they observe the
headers httpx actually puts on the wire, not the scraper's own bookkeeping.
"""

import base64
import contextlib
import logging
import socket
import sys
import types

import httpx
import pytest

from services import observability, url_safety


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

from services.browser_engine.app.scraper import engine  # noqa: E402
from services.browser_engine.app.scraper.engine import ScraperEngine  # noqa: E402

BASIC_HEADER = "Basic " + base64.b64encode(b"user:s3cret").decode()
BASIC_CREDENTIALS = {"type": "basic", "username": "user", "password": "s3cret"}
COOKIE_CREDENTIALS = {"type": "cookie", "cookies": {"session": "COOKIEVAL"}}


class RedirectChain:
    """Serve a fixed sequence of Location headers, recording every hop.

    `set_cookies` maps a zero-based hop index to a `Set-Cookie` value, which is
    how a server (or an attacker-controlled hop) plants state in the shared
    client's cookie jar.
    """

    def __init__(self, *locations, set_cookies=None):
        self._locations = locations
        self._set_cookies = set_cookies or {}
        self.hops = []

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.hops.append(
            {
                "host": request.headers.get("host"),
                "authorization": request.headers.get("authorization"),
                "cookie": request.headers.get("cookie"),
                "cookie_headers": request.headers.get_list("cookie"),
                # Case-insensitive lookup, and the RAW header list, so a test
                # can assert on a header nobody thought to name a key for.
                "headers": request.headers,
                "raw": list(request.headers.raw),
            }
        )
        index = len(self.hops) - 1
        headers = {}
        set_cookie = self._set_cookies.get(index)
        if set_cookie:
            headers["set-cookie"] = set_cookie
        location = self._locations[index] if index < len(self._locations) else None
        if location:
            headers["location"] = location
            return httpx.Response(302, headers=headers)
        return httpx.Response(
            200,
            headers=headers,
            html="<html><title>Page</title>body</html>",
        )


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


@pytest.mark.asyncio
async def test_basic_auth_is_dropped_on_cross_host_redirect(monkeypatch, public_dns):
    chain = RedirectChain("https://attacker.tld/collect")
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_basic_auth(
        "https://intranet.example/report",
        BASIC_CREDENTIALS,
    )

    assert result.status == "success"
    assert [hop["host"] for hop in chain.hops] == [
        "intranet.example",
        "attacker.tld",
    ]
    assert chain.hops[0]["authorization"] == BASIC_HEADER
    assert chain.hops[1]["authorization"] is None


@pytest.mark.asyncio
async def test_basic_auth_survives_same_host_redirect(monkeypatch, public_dns):
    chain = RedirectChain("https://intranet.example/report/final")
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_basic_auth(
        "https://intranet.example/report",
        BASIC_CREDENTIALS,
    )

    assert result.status == "success"
    assert [hop["authorization"] for hop in chain.hops] == [
        BASIC_HEADER,
        BASIC_HEADER,
    ]


@pytest.mark.asyncio
async def test_session_cookie_is_dropped_on_cross_host_redirect(
    monkeypatch, public_dns
):
    chain = RedirectChain("https://attacker.tld/collect")
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_cookie_auth(
        "https://intranet.example/report",
        COOKIE_CREDENTIALS,
    )

    assert result.status == "success"
    assert chain.hops[0]["cookie"] == "session=COOKIEVAL"
    assert chain.hops[1]["cookie"] is None


@pytest.mark.asyncio
async def test_session_cookie_survives_same_host_redirect(monkeypatch, public_dns):
    chain = RedirectChain("https://intranet.example/report/final")
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_cookie_auth(
        "https://intranet.example/report",
        COOKIE_CREDENTIALS,
    )

    assert result.status == "success"
    assert [hop["cookie"] for hop in chain.hops] == [
        "session=COOKIEVAL",
        "session=COOKIEVAL",
    ]


@pytest.mark.asyncio
async def test_credentials_are_not_rearmed_when_the_chain_returns_home(
    monkeypatch, public_dns
):
    """a.example -> attacker.tld -> a.example must stay unauthenticated.

    Re-arming would let whoever controls the middle hop choose which
    authenticated request on the credential origin gets made and captured.
    """
    chain = RedirectChain(
        "https://attacker.tld/bounce",
        "https://intranet.example/admin/secrets",
    )
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_basic_auth(
        "https://intranet.example/report",
        BASIC_CREDENTIALS,
    )

    assert result.status == "success"
    assert [hop["host"] for hop in chain.hops] == [
        "intranet.example",
        "attacker.tld",
        "intranet.example",
    ]
    assert [hop["authorization"] for hop in chain.hops] == [BASIC_HEADER, None, None]


@pytest.mark.asyncio
async def test_scheme_downgrade_on_same_host_drops_credentials(monkeypatch, public_dns):
    chain = RedirectChain("http://intranet.example/report")
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_cookie_auth(
        "https://intranet.example/report",
        COOKIE_CREDENTIALS,
    )

    assert result.status == "success"
    assert [hop["cookie"] for hop in chain.hops] == ["session=COOKIEVAL", None]


@pytest.mark.asyncio
async def test_scheme_upgrade_on_same_host_keeps_credentials(monkeypatch, public_dns):
    chain = RedirectChain("https://intranet.example/report")
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_cookie_auth(
        "http://intranet.example/report",
        COOKIE_CREDENTIALS,
    )

    assert result.status == "success"
    assert [hop["cookie"] for hop in chain.hops] == [
        "session=COOKIEVAL",
        "session=COOKIEVAL",
    ]


@pytest.mark.asyncio
async def test_port_change_on_same_host_drops_credentials(monkeypatch, public_dns):
    chain = RedirectChain("https://intranet.example:8443/report")
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_cookie_auth(
        "https://intranet.example/report",
        COOKIE_CREDENTIALS,
    )

    assert result.status == "success"
    assert [hop["host"] for hop in chain.hops] == [
        "intranet.example",
        "intranet.example:8443",
    ]
    assert [hop["cookie"] for hop in chain.hops] == ["session=COOKIEVAL", None]


@pytest.mark.asyncio
async def test_forwarded_page_credentials_are_dropped_cross_host(
    monkeypatch, public_dns
):
    """The browser route handler forwards page headers verbatim.

    `_safe_browser_route_handler` replays a Playwright request's own headers,
    which after a form login include the logged-in `Cookie` (and may include
    `Authorization`). Those are credentials too and follow the same rule.
    """
    chain = RedirectChain("https://attacker.tld/collect")

    async with httpx.AsyncClient(transport=chain.transport) as client:
        await engine._safe_httpx_request(
            client,
            "GET",
            "https://intranet.example/report",
            headers={
                "User-Agent": "TabOrganizer",
                "Authorization": BASIC_HEADER,
                "Cookie": "session=COOKIEVAL",
            },
        )

    assert chain.hops[0]["authorization"] == BASIC_HEADER
    assert chain.hops[0]["cookie"] == "session=COOKIEVAL"
    assert chain.hops[1]["authorization"] is None
    assert chain.hops[1]["cookie"] is None


# --- The client's own cookie jar (R1) -------------------------------------
#
# `_safe_httpx_request` reuses ONE `httpx.AsyncClient` for up to 10 hops, and
# every response's `Set-Cookie` lands in `client.cookies`. Two facts make that
# jar a cross-origin credential channel that the per-request credential scoping
# above does not touch:
#
# 1. `http.cookiejar.CookieJar.add_cookie_header` injects jar cookies only when
#    the request carries no `Cookie` header — and the scoping rule strips that
#    header exactly on the credential-dead hop, which is the hop where the jar
#    is then free to inject.
# 2. `resolve_scrape_targets` rewrites each hop to the resolved IP literal (the
#    DNS-rebinding defence), so the jar's domain key is the ADDRESS, not the
#    hostname. Any two hosts sharing an address — a CDN edge, a load balancer,
#    a reverse proxy, shared hosting — share one jar entry.
#
# The precondition is only "two hostnames resolve to one IP", which the
# `public_dns` fixture models exactly.


@pytest.mark.asyncio
async def test_server_set_cookie_does_not_ride_to_a_cross_host_redirect(
    monkeypatch, public_dns
):
    """A rotated session cookie must not reach the redirect target.

    The credential origin answers with `Set-Cookie: session=ROTATED` and a 302
    off-origin. Hop 2 has its `Cookie` header stripped by the scoping rule,
    which is precisely what lets the jar inject the rotated value instead.
    """
    chain = RedirectChain(
        "https://attacker.tld/collect",
        set_cookies={0: "session=ROTATED; Path=/"},
    )
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_cookie_auth(
        "https://intranet.example/report",
        COOKIE_CREDENTIALS,
    )

    assert result.status == "success"
    assert [hop["host"] for hop in chain.hops] == [
        "intranet.example",
        "attacker.tld",
    ]
    assert chain.hops[0]["cookie"] == "session=COOKIEVAL"
    assert chain.hops[1]["cookie"] is None


@pytest.mark.asyncio
async def test_server_set_cookie_does_not_ride_off_the_basic_auth_origin(
    monkeypatch, public_dns
):
    """The Basic-auth path shares the same jar even though it sends no cookies."""
    chain = RedirectChain(
        "https://attacker.tld/collect",
        set_cookies={0: "sid=SERVERSIDE; Path=/"},
    )
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_basic_auth(
        "https://intranet.example/report",
        BASIC_CREDENTIALS,
    )

    assert result.status == "success"
    assert chain.hops[1]["authorization"] is None
    assert chain.hops[1]["cookie"] is None


@pytest.mark.asyncio
async def test_ambient_browser_cookie_is_not_resupplied_from_the_jar(
    monkeypatch, public_dns
):
    """The default-install variant: `_safe_browser_route_handler` forwards headers.

    The handler replays a Playwright request's own headers, so after a login the
    ambient browser-session `Cookie` is on hop 1. If the origin rotates it, the
    jar re-supplies the rotated value on the cross-host hop even though the
    forwarded header was correctly stripped.
    """
    chain = RedirectChain(
        "https://attacker.tld/collect",
        set_cookies={0: "session=ROTATED; Path=/"},
    )

    async with httpx.AsyncClient(transport=chain.transport) as client:
        await engine._safe_httpx_request(
            client,
            "GET",
            "https://intranet.example/report",
            headers={
                "User-Agent": "TabOrganizer",
                "Cookie": "session=AMBIENT",
            },
        )

    assert chain.hops[0]["cookie"] == "session=AMBIENT"
    assert chain.hops[1]["cookie"] is None


@pytest.mark.asyncio
async def test_attacker_set_cookie_is_not_delivered_back_to_the_credential_origin(
    monkeypatch, public_dns
):
    """intranet -> attacker -> intranet must not carry the attacker's cookie home.

    Handing the attacker's `Set-Cookie` to the credential origin on hop 3 is the
    authenticated-request-forgery primitive the sticky no-re-arm rule exists to
    prevent: the middle hop chooses both the request and part of its state.
    """
    chain = RedirectChain(
        "https://attacker.tld/bounce",
        "https://intranet.example/admin/secrets",
        set_cookies={1: "session=ATTACKERCHOSEN; Path=/"},
    )
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_cookie_auth(
        "https://intranet.example/report",
        COOKIE_CREDENTIALS,
    )

    assert result.status == "success"
    assert [hop["host"] for hop in chain.hops] == [
        "intranet.example",
        "attacker.tld",
        "intranet.example",
    ]
    assert chain.hops[2]["cookie"] is None


# --- Header-case collision (R3) -------------------------------------------


@pytest.mark.asyncio
async def test_stored_and_ambient_cookies_emit_one_valid_cookie_header(
    monkeypatch, public_dns
):
    """RFC 6265 allows at most one `Cookie` header, joined with `; `.

    Setting the key `"Cookie"` next to a caller-supplied lowercase `"cookie"`
    leaves two entries in the header dict, which httpx renders as two header
    lines (read back comma-joined). Servers parse that as one garbage cookie
    name, so the authenticated request silently degrades.
    """
    chain = RedirectChain()

    async with httpx.AsyncClient(transport=chain.transport) as client:
        await engine._safe_httpx_request(
            client,
            "GET",
            "https://intranet.example/report",
            headers={"User-Agent": "TabOrganizer", "cookie": "ambient=AMB"},
            cookies={"session": "COOKIEVAL"},
        )

    assert chain.hops[0]["cookie_headers"] == ["ambient=AMB; session=COOKIEVAL"]
    assert "," not in chain.hops[0]["cookie"]


# --- A drop must never be silent (R2) -------------------------------------
#
# `https://example.com/report -> https://www.example.com/report` is an ordinary
# canonical redirect. The strict host comparison is kept on purpose — the
# credential store holds a bare `{name: value}` dict with no domain metadata,
# so nothing in it says the cookie was meant for `www` — but the consequence is
# that the scrape returns the logged-OUT page. Reporting that as
# `status='success', auth_used=True` files public content as an authenticated
# capture and feeds a false positive to the "authenticated capture => local
# embeddings" gate.


@pytest.mark.asyncio
async def test_apex_to_www_drop_is_recorded_in_result_metadata(monkeypatch, public_dns):
    chain = RedirectChain("https://www.example.com/report")
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_cookie_auth(
        "https://example.com/report",
        COOKIE_CREDENTIALS,
    )

    assert result.status == "success"
    assert [hop["cookie"] for hop in chain.hops] == ["session=COOKIEVAL", None]

    drop = result.metadata["credential_scope_drop"]
    assert drop["reason"] == "redirect_left_credential_origin"
    assert drop["auth_type"] == "cookie"
    assert drop["credential_origin_host"] == "example.com"
    assert drop["dropped_at_host"] == "www.example.com"


@pytest.mark.asyncio
async def test_dropped_credentials_are_not_reported_as_an_authenticated_capture(
    monkeypatch, public_dns
):
    """The captured body came from an unauthenticated request; say so."""
    chain = RedirectChain("https://www.example.com/report")
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_basic_auth(
        "https://example.com/report",
        BASIC_CREDENTIALS,
    )

    assert result.auth_used is False


@pytest.mark.asyncio
async def test_surviving_credentials_still_report_an_authenticated_capture(
    monkeypatch, public_dns
):
    """The non-vacuity half: an unbroken chain keeps auth_used and adds no drop."""
    chain = RedirectChain("https://intranet.example/report/final")
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_cookie_auth(
        "https://intranet.example/report",
        COOKIE_CREDENTIALS,
    )

    assert result.auth_used is True
    assert "credential_scope_drop" not in result.metadata


class _RecordingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def _captured_events():
    """Capture `log_event` output off the logger it actually targets.

    `configure_logging` sets `propagate = False` on the per-service logger, so
    pytest's root-level `caplog` sees nothing as soon as any service has
    configured logging in the same process — which is why this cannot use
    `caplog` and stay green in the full unit run.
    """
    logger = logging.getLogger(f"taborganizer.{observability._service_name}")
    handler = _RecordingHandler()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


@pytest.mark.asyncio
async def test_credential_drop_is_logged_structurally(monkeypatch, public_dns):
    """An operator has to be able to see this without reading the metadata blob."""
    chain = RedirectChain("https://www.example.com/report")
    _pin_client_transport(monkeypatch, chain)

    with _captured_events() as records:
        await ScraperEngine(respect_robots=False)._scrape_cookie_auth(
            "https://example.com/report",
            COOKIE_CREDENTIALS,
        )

    events = [
        record
        for record in records
        if getattr(record, "event", None) == "scrape.credentials_dropped_on_redirect"
    ]
    assert len(events) == 1
    assert events[0].levelno == logging.WARNING
    fields = events[0].fields
    assert fields["credential_origin_host"] == "example.com"
    assert fields["dropped_at_host"] == "www.example.com"
    # Hosts only: a redirect target's path/query can carry tokens.
    assert "COOKIEVAL" not in str(fields)
    assert "/report" not in str(fields)


# --- Non-standard credential headers (R4/N1) ------------------------------
#
# `_safe_browser_route_handler` forwards a Playwright request's headers
# verbatim. Removing the two NAMED credential headers on a cross-origin hop
# closed two instances and left the class open: a page authenticated by
# `X-Api-Key`, or carrying an `X-Csrf-Token`, handed both to the redirect
# target, and `Referer` handed over the full secret URL of the credentialed
# origin. Unlike the client-jar leak this needs no shared IP — any cross-host
# redirect is enough — so the rule is an ALLOWLIST of headers an anonymous
# client sends to any host, and the default is deny.

PAGE_HEADERS = {
    "User-Agent": "TabOrganizer/1.0",
    "Accept": "text/html",
    "Accept-Encoding": "gzip",
    "Accept-Language": "en-GB",
    "X-Csrf-Token": "CSRF",
    "X-Api-Key": "APIKEY",
    "Authorization": BASIC_HEADER,
    "Cookie": "session=COOKIEVAL",
    "Referer": "https://intranet.example/secret-page",
}
TRANSPORT_HEADERS = {
    "user-agent": "TabOrganizer/1.0",
    "accept": "text/html",
    "accept-encoding": "gzip",
    "accept-language": "en-GB",
}


def _transport_view(hop) -> dict:
    return {name: hop["headers"].get(name) for name in TRANSPORT_HEADERS}


@pytest.mark.asyncio
async def test_non_standard_credential_headers_are_dropped_cross_host(
    monkeypatch, public_dns
):
    """`x-api-key`/`x-csrf-token`/`referer` must not reach the redirect target."""
    chain = RedirectChain("https://attacker.tld/collect")

    async with httpx.AsyncClient(transport=chain.transport) as client:
        await engine._safe_httpx_request(
            client,
            "GET",
            "https://intranet.example/secret-page",
            headers=engine._headers_for_safe_browser_fetch(PAGE_HEADERS),
        )

    assert [hop["host"] for hop in chain.hops] == ["intranet.example", "attacker.tld"]
    assert chain.hops[0]["headers"].get("x-csrf-token") == "CSRF"
    assert chain.hops[0]["headers"].get("x-api-key") == "APIKEY"

    crossed = chain.hops[1]
    assert crossed["headers"].get("x-csrf-token") is None
    assert crossed["headers"].get("x-api-key") is None
    assert crossed["headers"].get("referer") is None
    assert crossed["authorization"] is None
    assert crossed["cookie"] is None
    # Nothing token-shaped survives anywhere in the raw header list, including
    # any header this test did not think to name.
    raw = str(crossed["raw"])
    for secret in ("CSRF", "APIKEY", "COOKIEVAL", "secret-page", BASIC_HEADER):
        assert secret not in raw
    # ...while the transport allowlist is still intact. Asserting the exact
    # values matters: httpx substitutes its own `accept`/`accept-encoding`/
    # `user-agent` defaults whenever the header is missing, so a regression that
    # dropped them would otherwise look like a pass.
    assert _transport_view(crossed) == TRANSPORT_HEADERS


@pytest.mark.asyncio
async def test_page_headers_survive_a_same_host_redirect(monkeypatch, public_dns):
    """The non-vacuity half: a live hop still carries the page's own headers.

    These are going back to the host they came from. Stripping them here would
    break ordinary scraping — CSRF-guarded pages and API-key-authenticated
    endpoints would start returning 403 on any internal redirect.
    """
    chain = RedirectChain("https://intranet.example/secret-page/final")

    async with httpx.AsyncClient(transport=chain.transport) as client:
        await engine._safe_httpx_request(
            client,
            "GET",
            "https://intranet.example/secret-page",
            headers=engine._headers_for_safe_browser_fetch(PAGE_HEADERS),
        )

    live = chain.hops[1]
    assert live["headers"].get("x-csrf-token") == "CSRF"
    assert live["headers"].get("x-api-key") == "APIKEY"
    assert live["headers"].get("referer") == "https://intranet.example/secret-page"
    assert live["authorization"] == BASIC_HEADER
    assert live["cookie"] == "session=COOKIEVAL"
    assert _transport_view(live) == TRANSPORT_HEADERS


@pytest.mark.asyncio
async def test_cross_origin_referer_is_reduced_to_its_origin_on_a_live_hop(
    monkeypatch, public_dns
):
    """`Referer` names a different host than the one it is sent to.

    A page on the credentialed origin pulling a subresource from a third party
    is hop 1 of its OWN request, so it is a live hop and the cross-origin rule
    above never fires — yet the full secret URL, query string included, goes to
    the third party. Browsers send origin-only cross-origin; so do we.
    """
    chain = RedirectChain()

    async with httpx.AsyncClient(transport=chain.transport) as client:
        await engine._safe_httpx_request(
            client,
            "GET",
            "https://cdn.example/img.png",
            headers={
                "User-Agent": "TabOrganizer/1.0",
                "Referer": "https://intranet.example/secret-page?token=TOKENVAL",
            },
        )

    assert chain.hops[0]["headers"].get("referer") == "https://intranet.example/"
    raw = str(chain.hops[0]["raw"])
    assert "secret-page" not in raw
    assert "TOKENVAL" not in raw


@pytest.mark.asyncio
async def test_referer_is_dropped_entirely_on_a_scheme_downgrade(
    monkeypatch, public_dns
):
    """https -> http would put the internal origin on the wire in cleartext."""
    chain = RedirectChain()

    async with httpx.AsyncClient(transport=chain.transport) as client:
        await engine._safe_httpx_request(
            client,
            "GET",
            "http://cdn.example/img.png",
            headers={
                "User-Agent": "TabOrganizer/1.0",
                "Referer": "https://intranet.example/secret-page",
            },
        )

    assert chain.hops[0]["headers"].get("referer") is None


# --- Header injection through rendered values (R5/N2) ---------------------
#
# Cookies are rendered into a `Cookie` header by hand (`_cookie_header`),
# because a jar keyed on the request URL cannot work once the URL has been
# rewritten to an IP literal. httpx does not validate header values, so a
# stored value of "v\r\nX-Injected: yes" was emitted as a literal second header
# line — the raw tuple observed was (b'Cookie', b'session=v\r\nX-Injected: yes').
# The whole request is refused rather than the cookie being dropped: dropping
# would return the logged-OUT page as an authenticated success, which is the
# silent degradation this module already had to fix once.


@pytest.mark.asyncio
async def test_crlf_in_a_stored_cookie_emits_no_request_at_all(monkeypatch, public_dns):
    chain = RedirectChain()
    _pin_client_transport(monkeypatch, chain)

    result = await ScraperEngine(respect_robots=False)._scrape_cookie_auth(
        "https://intranet.example/report",
        {"type": "cookie", "cookies": {"session": "v\r\nX-Injected: yes"}},
    )

    assert result.status == "failed"
    assert chain.hops == []
    # The error travels into the ledger and the logs; it names the cookie, never
    # its value.
    assert "session" in result.error
    assert "X-Injected" not in result.error
    assert "\r" not in result.error


def test_cookie_rendering_rejects_control_characters_in_names_and_values():
    with pytest.raises(ValueError):
        engine._cookie_header({"session": "v\r\nX-Injected: yes"})
    with pytest.raises(ValueError):
        engine._cookie_header({"session": "v\x00truncated"})
    with pytest.raises(ValueError):
        engine._cookie_header({"a\r\nX-Injected: yes": "v"})
    assert engine._cookie_header({"session": "COOKIEVAL"}) == "session=COOKIEVAL"


@pytest.mark.asyncio
async def test_crlf_in_a_forwarded_cookie_header_is_refused(monkeypatch, public_dns):
    """The route handler forwards page headers verbatim, stored cookies or not.

    Caught at the wire choke point (`_request_kwargs_for_target`) rather than in
    the cookie renderer, so every path that builds headers is covered by one
    check instead of each being audited separately.
    """
    chain = RedirectChain()

    async with httpx.AsyncClient(transport=chain.transport) as client:
        with pytest.raises(ValueError):
            await engine._safe_httpx_request(
                client,
                "GET",
                "https://intranet.example/report",
                headers={
                    "User-Agent": "TabOrganizer",
                    "Cookie": "a=b\r\nX-Injected: yes",
                },
            )

    assert chain.hops == []
