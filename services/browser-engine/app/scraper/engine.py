"""Non-blocking web scraper with parallel auth support."""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Awaitable, Callable, Optional
from urllib.parse import urljoin, urlparse

import httpx
from playwright.async_api import Browser
from playwright.async_api import TimeoutError as PlaywrightTimeout
from playwright.async_api import async_playwright

from services.observability import log_event
from services.url_safety import resolve_scrape_targets, validate_scrape_url

from ..auth.queue import CredentialScope

REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
REQUEST_HEADERS_TO_DROP = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
RESPONSE_HEADERS_TO_DROP = {
    "connection",
    "content-encoding",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
# Everything a hop that has LEFT the credential origin may keep. An ALLOWLIST,
# not a denylist: `{"authorization", "cookie"}` closed two instances and left
# the class open — `x-api-key`, `x-csrf-token`, `x-auth-*` and every vendor
# token header rode a cross-host redirect untouched, and unlike the client
# cookie jar this needs no shared IP, only a redirect. Each entry must be
# something an anonymous client sends to any host on the internet:
#   accept, accept-encoding — content/transfer negotiation for THIS hop; they
#     describe what we can parse, not who we are.
#   accept-language — every browser sends it cross-origin; dropping it makes the
#     redirect target serve a different language than the origin did.
#   user-agent — sent to every host; on the httpx path it is our own static
#     `ScraperEngine.USER_AGENT`, and hosts routinely 403 a UA-less client.
# `referer` is deliberately NOT here: see `_referer_scoped_headers`.
CROSS_ORIGIN_HEADER_ALLOWLIST = frozenset(
    {"accept", "accept-encoding", "accept-language", "user-agent"}
)
# CR/LF end a header line and NUL truncates it, so a value carrying one is
# header injection, not content — RFC 6265 excludes CTLs from cookie-value.
HEADER_UNSAFE_CHARACTERS = re.compile(r"[\r\n\x00]")


@dataclass
class CredentialHopTrace:
    """Whether stored credentials survived a redirect chain.

    `_origin_keeps_credentials` admits exactly the hosts inside the scope the
    credential was SUBMITTED for (`CredentialScope`), and falls back to exact
    host when no scope was recorded. A drop is therefore rarer than it was, but
    it is still not rare: a hop outside the scope returns the logged-OUT page,
    and without this trace that is indexed as an authenticated capture.
    """

    credential_origin: Optional[str] = None
    hops: int = 0
    dropped_at: Optional[str] = None
    scope: Optional[CredentialScope] = None

    @property
    def credentials_dropped(self) -> bool:
        return self.dropped_at is not None

    def record_hop(self, url: str, credentials_live: bool) -> None:
        self.hops += 1
        if not credentials_live and self.dropped_at is None:
            self.dropped_at = url


def _credential_drop_fields(trace: CredentialHopTrace) -> dict:
    """Operator-facing description of a drop — hosts only, never full URLs.

    A redirect target's path and query can carry tokens; the host is enough to
    answer "why did my authenticated scrape return public content?".
    """
    return {
        "credential_origin_host": _host_of(trace.credential_origin),
        "dropped_at_host": _host_of(trace.dropped_at),
        "hops": trace.hops,
        # Flat scalars, not a nested object: these are splatted into a
        # JSON-lines log event as well as into result metadata. A null domain
        # means no scope was recorded and the strict exact-host rule applied.
        "credential_scope_domain": trace.scope.domain if trace.scope else None,
        "credential_scope_subdomains": bool(
            trace.scope and trace.scope.include_subdomains
        ),
    }


def _host_of(url: Optional[str]) -> Optional[str]:
    return (urlparse(url).hostname or None) if url else None


async def _safe_httpx_get(
    client: httpx.AsyncClient,
    url: str,
    **kwargs,
) -> httpx.Response:
    return await _safe_httpx_request(client, "GET", url, **kwargs)


async def _safe_httpx_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    auth: Optional[httpx.Auth] = None,
    cookies: Optional[dict] = None,
    credential_scope: Optional[CredentialScope] = None,
    trace: Optional[CredentialHopTrace] = None,
    **kwargs,
) -> httpx.Response:
    """Follow redirects while connecting only to vetted network targets.

    `auth`/`cookies` must be handed to this function, never to the client:
    httpx applies client-level auth and a bare-domain cookie jar to every
    request unconditionally, so a redirect naming an attacker host would be
    served the user's stored credentials. Credentials ride only while a hop
    stays inside `credential_scope` — the scope recorded when the user
    submitted them — and are never re-armed after the first crossing (see
    `_origin_keeps_credentials`). Omitting the scope means exact host, which is
    what every caller with no recorded scope gets. Caller-supplied
    headers — the browser route handler forwards page headers verbatim — are
    reduced by the same rule to `CROSS_ORIGIN_HEADER_ALLOWLIST` once the chain
    has left that origin, so a bearer token in a non-standard header does not
    survive where `Authorization` would not.

    Pass a `CredentialHopTrace` to learn whether the credentials survived; a
    silent drop makes a logged-out page look like an authenticated capture.
    """
    current_url = validate_scrape_url(url)
    credential_origin = current_url
    credentials_live = True
    current_method = method.upper()
    current_kwargs = dict(kwargs)
    if trace is not None:
        trace.credential_origin = credential_origin
        trace.scope = credential_scope

    for _ in range(10):
        # The client's jar is not a second credential store; it is an ambient
        # one. `resolve_scrape_targets` rewrites every hop to a vetted IP
        # literal, so a jar entry is keyed on the ADDRESS and cannot tell
        # intranet.example from attacker.tld when they share a CDN edge, load
        # balancer, reverse proxy, or host. Worse, `add_cookie_header` injects
        # only when the request has no `Cookie` header — exactly the hop the
        # rule below strips. Cleared on EVERY hop, not just on a crossing: a
        # jar that cannot scope has nothing to contribute on any hop, keeping
        # one rule instead of two interacting ones, and making the invariant
        # "credentials are per-request, never per-client" literally true. Cost
        # is a mid-chain server-set cookie not carried forward on a same-origin
        # redirect; no scraper flow depends on that (stored cookies are
        # re-attached per live hop, and the browser path builds a fresh client
        # per intercepted request while Playwright keeps its own jar).
        client.cookies.clear()
        credentials_live = credentials_live and _origin_keeps_credentials(
            credential_origin, current_url, credential_scope
        )
        if trace is not None:
            trace.record_hop(current_url, credentials_live)
        targets = resolve_scrape_targets(current_url)
        response = await _request_first_safe_target(
            client,
            current_method,
            targets,
            **_credential_scoped_kwargs(
                current_kwargs, auth, cookies, credentials_live, current_url
            ),
        )
        location = response.headers.get("location")
        if response.status_code not in REDIRECT_STATUS_CODES or not location:
            return response
        if response.status_code == 303 or (
            response.status_code in {301, 302} and current_method not in {"GET", "HEAD"}
        ):
            current_method = "GET"
            current_kwargs = {
                key: value for key, value in current_kwargs.items() if key != "content"
            }
        current_url = validate_scrape_url(urljoin(current_url, location))
    raise httpx.TooManyRedirects("Exceeded safe redirect limit")


def _origin_keeps_credentials(
    origin_url: str,
    target_url: str,
    scope: Optional[CredentialScope] = None,
) -> bool:
    """Whether stored credentials may ride to `target_url`.

    The host question is "is this host inside the scope the credential was
    submitted for" (`CredentialScope.covers`), NOT "is it the host I started
    on". With no scope recorded the answer falls back to exact host — the
    pre-scope rule — so anything the store wrote before scopes existed, and the
    caller-header path (`_safe_browser_route_handler` forwards a page's ambient
    headers, which carry no recorded scope at all), keeps the strict behavior.

    Everything else is still compared against the ORIGIN THE REQUEST STARTED
    AT, and the caller drops credentials permanently on the first mismatch
    rather than re-arming on a later hop that returns: with a "same as the
    previous hop" rule a chain a.example -> attacker.tld -> a.example would
    hand the attacker the choice of which authenticated a.example request gets
    made and captured. Browsers likewise never restore `Authorization` after a
    cross-origin hop. The scope widens WHICH HOSTS qualify; it does not make a
    drop recoverable.

    A scheme downgrade is a crossing: https -> http on the same host puts the
    Basic credential and the session cookie on the wire in cleartext, which is
    the disclosure being prevented. An http -> https upgrade of an otherwise
    identical origin keeps them, since it strictly increases confidentiality.
    """
    origin = urlparse(origin_url)
    target = urlparse(target_url)
    if not _host_in_credential_scope(origin, target, scope):
        return False

    origin_scheme = origin.scheme.lower()
    target_scheme = target.scheme.lower()
    if origin_scheme == target_scheme:
        return _effective_port(origin) == _effective_port(target)
    return (
        (origin_scheme, target_scheme) == ("http", "https")
        and origin.port is None
        and target.port is None
    )


def _host_in_credential_scope(origin, target, scope: Optional[CredentialScope]) -> bool:
    """Host half of the live/dead-hop decision.

    A missing scope is the strictest answer, never the most permissive one, so
    every path that never learned about scopes keeps exact-host behavior. When
    a scope IS present it is authoritative in both directions: a hop the scope
    does not cover is dead even if it is the very origin the request started
    at, so a credential can never be spent outside the scope it was submitted
    for.
    """
    target_host = (target.hostname or "").lower()
    if not target_host:
        return False
    if scope is None:
        return (origin.hostname or "").lower() == target_host
    return scope.covers(target_host)


def _effective_port(parsed) -> int:
    return parsed.port or (443 if parsed.scheme.lower() == "https" else 80)


def _credential_scoped_kwargs(kwargs, auth, cookies, credentials_live, target_url):
    """Attach or strip every credential-bearing input for a single hop.

    On a hop that has left the credential origin the caller's headers are
    reduced to `CROSS_ORIGIN_HEADER_ALLOWLIST` rather than having known-bad
    names removed. Enumerating the bad ones only ever closes the instances
    named: the route handler forwards a page's headers verbatim, and a page
    authenticated by `X-Api-Key` or carrying an `X-Csrf-Token` handed both to
    the redirect target while `Authorization` and `Cookie` were correctly
    absent. An allowlist makes the default deny, so the next header a site
    invents is covered without an edit here.

    Live hops are untouched apart from `Referer`: those headers are going back
    to the host they came from, and stripping them would break ordinary
    scraping.
    """
    headers = dict(kwargs.get("headers") or {})
    if not credentials_live:
        return {
            **kwargs,
            "headers": {
                key: value
                for key, value in headers.items()
                if key.lower() in CROSS_ORIGIN_HEADER_ALLOWLIST
            },
        }

    headers = _referer_scoped_headers(headers, target_url)
    hop_kwargs = dict(kwargs)
    if auth is not None:
        hop_kwargs["auth"] = auth
    if cookies:
        headers = _headers_with_cookies(headers, cookies)
    hop_kwargs["headers"] = headers
    return hop_kwargs


def _referer_scoped_headers(headers: dict, target_url: str) -> dict:
    """Reduce a cross-origin `Referer` to its origin before it leaves.

    `Referer` is the one caller header that names a DIFFERENT host than the one
    it is being sent to, so the allowlist above is not enough on its own. A page
    on `https://intranet.example/secret-page` pulling an image from
    `cdn.example` sends the full secret URL — path, query, and any token in it —
    on hop 1, which is a LIVE hop for that fetch, so no redirect is involved and
    the cross-origin rule never fires. Browsers default to
    `strict-origin-when-cross-origin` for exactly this reason and we match it:
    origin only when the target is a different origin, nothing at all on an
    https -> http downgrade (the origin would travel in cleartext). Same-origin
    requests keep the full value — that host already knows its own paths, and
    hotlink checks read the origin either way.
    """
    if not any(key.lower() == "referer" for key in headers):
        return headers

    scoped = {key: value for key, value in headers.items() if key.lower() != "referer"}
    value = next(
        headers[key] for key in reversed(list(headers)) if key.lower() == "referer"
    )
    referer = urlparse(value)
    target = urlparse(target_url)
    referer_origin = _origin_of(referer)
    if referer_origin is None:
        return scoped
    if referer_origin == _origin_of(target):
        scoped["Referer"] = value
    elif not (referer.scheme.lower() == "https" and target.scheme.lower() == "http"):
        scoped["Referer"] = referer_origin
    return scoped


def _origin_of(parsed) -> Optional[str]:
    host = (parsed.hostname or "").lower()
    if not host:
        return None
    scheme = parsed.scheme.lower()
    port = _effective_port(parsed)
    netloc = host if port == (443 if scheme == "https" else 80) else f"{host}:{port}"
    return f"{scheme}://{netloc}/"


def _reject_control_characters(label: str, value: str) -> str:
    """Fail the request rather than silently dropping the offending value.

    Dropping a poisoned cookie would produce a logged-OUT capture still reported
    as an authenticated success — the silent-degradation failure this module
    already had to fix once — whereas CR/LF/NUL in a cookie or header value is
    never legitimate content, so refusing costs no real scrape. The message
    names the field and never the value: the value is the credential, and it
    travels on into `ScrapeResult.error` and the logs.
    """
    if HEADER_UNSAFE_CHARACTERS.search(value):
        raise ValueError(f"{label} contains control characters (CR, LF, or NUL)")
    return value


def _headers_with_cookies(headers: dict, cookies) -> dict:
    """Fold stored cookies into ONE `Cookie` header, case-insensitively.

    Header names are case-insensitive, so writing the key `"Cookie"` next to a
    caller-supplied `"cookie"` leaves two dict entries and httpx emits two
    `Cookie` header lines. RFC 6265 allows at most one, and a server that reads
    only the first silently serves logged-out content. The caller's value is
    merged rather than replaced: on a live hop both it and the stored cookie
    belong to the same origin, and they are dropped together on a crossing.
    """
    merged = {key: value for key, value in headers.items() if key.lower() != "cookie"}
    existing = "; ".join(
        value for key, value in headers.items() if key.lower() == "cookie" and value
    )
    rendered = _cookie_header(cookies)
    merged["Cookie"] = f"{existing}; {rendered}" if existing else rendered
    return merged


def _cookie_header(cookies) -> str:
    """Render cookies as one header instead of seeding the client's jar.

    A jar keyed on the request URL cannot work here: `resolve_scrape_targets`
    rewrites the URL to a vetted IP literal, so a domain-scoped cookie would
    never match, and a bare-domain cookie matches every host.

    Rendering by hand means rendering CR/LF by hand too: a stored value of
    `"v\\r\\nX-Injected: yes"` became a literal second header line, since httpx
    does not validate header values. Checked per cookie rather than only at the
    wire choke point so the error can say WHICH cookie is poisoned.
    """
    rendered = []
    for name, value in httpx.Cookies(cookies).items():
        _reject_control_characters("A stored cookie name", name)
        _reject_control_characters(f"Stored cookie {name!r}", value)
        rendered.append(f"{name}={value}")
    return "; ".join(rendered)


async def _request_first_safe_target(
    client: httpx.AsyncClient,
    method: str,
    targets,
    **kwargs,
) -> httpx.Response:
    last_error = None
    for target in targets:
        try:
            return await client.request(
                method,
                target.request_url,
                follow_redirects=False,
                **_request_kwargs_for_target(target, kwargs),
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as error:
            last_error = error
            continue
    if last_error:
        raise last_error
    raise ValueError("Scrape URL host could not be resolved safely")


def _request_kwargs_for_target(target, kwargs):
    headers = dict(kwargs.get("headers") or {})
    if target.host_header:
        headers = {**headers, "Host": target.host_header}
    _reject_header_injection(headers)

    extensions = dict(kwargs.get("extensions") or {})
    if target.sni_hostname:
        extensions = {**extensions, "sni_hostname": target.sni_hostname}

    request_kwargs = {**kwargs, "headers": headers}
    if extensions:
        request_kwargs = {**request_kwargs, "extensions": extensions}
    return request_kwargs


def _reject_header_injection(headers: dict) -> None:
    """Last stop before the wire: no header name or value may carry CR/LF/NUL.

    Placed at the one choke point every outbound hop passes through, so a value
    smuggled in through ANY of the paths that build headers — stored cookies,
    page headers forwarded verbatim by the route handler, the resolved `Host` —
    is caught without each path having to be audited separately. The auth path
    needs no equivalent: `httpx.BasicAuth` base64-encodes the credential, so a
    CR/LF in a username or password cannot reach the header line.
    """
    for key, value in headers.items():
        _reject_control_characters("A header name", str(key))
        _reject_control_characters(f"Header {str(key).lower()!r} value", str(value))


def _safe_browser_route_handler(timeout: int):
    async def handler(route, request) -> None:
        try:
            headers = _headers_for_safe_browser_fetch(request.headers)
            content = None
            if request.method.upper() not in {"GET", "HEAD"}:
                content = request.post_data_buffer
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await _safe_httpx_request(
                    client,
                    request.method,
                    request.url,
                    headers=headers,
                    content=content,
                )
            await route.fulfill(
                status=response.status_code,
                headers=_headers_for_browser_fulfill(response.headers),
                body=b"" if request.method.upper() == "HEAD" else response.content,
            )
        except Exception:
            await route.abort()

    return handler


def _headers_for_safe_browser_fetch(headers):
    return {
        key: value
        for key, value in dict(headers).items()
        if key.lower() not in REQUEST_HEADERS_TO_DROP
    }


def _headers_for_browser_fulfill(headers):
    return {
        key: value
        for key, value in dict(headers).items()
        if key.lower() not in RESPONSE_HEADERS_TO_DROP
    }


@dataclass
class ScrapeResult:
    """Result of scraping a URL."""

    url: str
    status: str  # success, failed, auth_required, timeout, blocked
    title: Optional[str] = None
    content: Optional[str] = None
    html: Optional[str] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    scraped_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)
    # True only when the result came from the credential-store path
    # (_scrape_with_auth basic/cookie/form). Ambient auth (cookies already in a
    # browser context, session reuse, auth-wall pages queued as auth_required) is
    # invisible today, so auth_used=false there is a known under-report — see the
    # scraper MODULE card. The unknown-auth-type fallback to plain httpx stays
    # false. Propagates capture -> ledger -> /index metadata (decision 37 hook).
    auth_used: bool = False


class ContentExtractor:
    """Extract readable content from HTML."""

    # Tags to remove entirely
    REMOVE_TAGS = [
        "script",
        "style",
        "noscript",
        "iframe",
        "svg",
        "canvas",
        "video",
        "audio",
        "nav",
        "footer",
        "header",
        "aside",
    ]

    # Tags that indicate main content
    CONTENT_TAGS = ["article", "main", "section", "div"]

    def extract_title(self, html: str) -> str:
        """Extract page title from HTML."""
        # Try <title> tag
        match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Try <h1> tag
        match = re.search(r"<h1[^>]*>([^<]+)</h1>", html, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return ""

    def extract_text(self, html: str) -> str:
        """Extract readable text from HTML."""
        # Remove unwanted tags
        text = html
        for tag in self.REMOVE_TAGS:
            text = re.sub(
                rf"<{tag}[^>]*>.*?</{tag}>", "", text, flags=re.IGNORECASE | re.DOTALL
            )

        # Remove all remaining HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Decode HTML entities
        text = self._decode_entities(text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _decode_entities(self, text: str) -> str:
        """Decode common HTML entities."""
        entities = {
            "&nbsp;": " ",
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "&quot;": '"',
            "&#39;": "'",
            "&apos;": "'",
        }
        for entity, char in entities.items():
            text = text.replace(entity, char)
        return text

    def extract_metadata(self, html: str) -> dict:
        """Extract metadata from HTML."""
        metadata = {}

        # Extract meta description
        match = re.search(
            r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']',
            html,
            re.IGNORECASE,
        )
        if match:
            metadata["description"] = match.group(1)

        # Extract meta keywords
        match = re.search(
            r'<meta[^>]*name=["\']keywords["\'][^>]*content=["\']([^"\']+)["\']',
            html,
            re.IGNORECASE,
        )
        if match:
            metadata["keywords"] = match.group(1).split(",")

        # Extract Open Graph data
        og_patterns = [
            (
                "og_title",
                r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\']([^"\']+)["\']',
            ),
            (
                "og_description",
                r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']+)["\']',
            ),
            (
                "og_image",
                r'<meta[^>]*property=["\']og:image["\'][^>]*content=["\']([^"\']+)["\']',
            ),
        ]

        for key, pattern in og_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                metadata[key] = match.group(1)

        return metadata


class RobotsChecker:
    """Check robots.txt compliance."""

    def __init__(self):
        self._cache: dict[str, dict] = {}  # domain → rules

    async def is_allowed(self, url: str, user_agent: str = "*") -> bool:
        """Check if URL is allowed by robots.txt."""
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"

        if domain not in self._cache:
            await self._fetch_robots(domain)

        rules = self._cache.get(domain, {})
        path = parsed.path or "/"

        # Check disallow rules
        disallow = rules.get("disallow", [])
        for pattern in disallow:
            if path.startswith(pattern):
                return False

        return True

    async def _fetch_robots(self, domain: str) -> None:
        """Fetch and parse robots.txt."""
        try:
            async with httpx.AsyncClient() as client:
                response = await _safe_httpx_get(
                    client,
                    f"{domain}/robots.txt",
                    timeout=10.0,
                )
                if response.status_code == 200:
                    self._cache[domain] = self._parse_robots(response.text)
                else:
                    self._cache[domain] = {}
        except Exception:
            self._cache[domain] = {}

    def _parse_robots(self, content: str) -> dict:
        """Parse robots.txt content."""
        rules = {"disallow": [], "allow": []}
        current_agent = None

        for line in content.split("\n"):
            line = line.strip().lower()

            if line.startswith("user-agent:"):
                current_agent = line.split(":", 1)[1].strip()
            elif current_agent in ("*", "tab-organizer"):
                if line.startswith("disallow:"):
                    path = line.split(":", 1)[1].strip()
                    if path:
                        rules["disallow"].append(path)
                elif line.startswith("allow:"):
                    path = line.split(":", 1)[1].strip()
                    if path:
                        rules["allow"].append(path)

        return rules


class ScraperEngine:
    """Non-blocking web scraper with parallel auth support."""

    USER_AGENT = (
        "Mozilla/5.0 (compatible; TabOrganizer/1.0; +https://github.com/tab-organizer)"
    )

    def __init__(
        self,
        max_concurrent: int = 10,
        timeout: int = 30,
        respect_robots: bool = True,
    ):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.respect_robots = respect_robots

        self._browser: Optional[Browser] = None
        self._playwright = None
        self._browser_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._extractor = ContentExtractor()
        self._robots = RobotsChecker()
        self._auth_queue = None
        self._auth_detector = None

    def set_auth_queue(self, queue) -> None:
        """Set the auth queue for handling authentication."""
        self._auth_queue = queue

    def set_auth_detector(self, detector) -> None:
        """Set the auth detector."""
        self._auth_detector = detector

    async def _get_browser(self) -> Browser:
        """Get or create browser instance."""
        async with self._browser_lock:
            if self._browser is None:
                self._playwright = await async_playwright().start()
                try:
                    self._browser = await self._playwright.chromium.launch(
                        headless=True,
                        args=["--no-sandbox", "--disable-dev-shm-usage"],
                    )
                except Exception:
                    await self._playwright.stop()
                    self._playwright = None
                    raise
        return self._browser

    async def close(self) -> None:
        """Close browser instance."""
        async with self._browser_lock:
            browser = self._browser
            playwright = self._playwright
            self._browser = None
            self._playwright = None

            try:
                if browser:
                    await browser.close()
            finally:
                if playwright:
                    await playwright.stop()

    async def scrape_url(
        self,
        url: str,
        session_id: Optional[str] = None,
        use_browser: bool = False,
    ) -> ScrapeResult:
        """
        Scrape a single URL.

        Args:
            url: URL to scrape
            session_id: Session ID for auth queue
            use_browser: Use Playwright instead of httpx
        """
        try:
            validate_scrape_url(url)
        except ValueError as error:
            return ScrapeResult(url=url, status="failed", error=str(error))

        async with self._semaphore:
            # Check robots.txt
            if self.respect_robots:
                if not await self._robots.is_allowed(url):
                    return ScrapeResult(
                        url=url,
                        status="blocked",
                        error="Blocked by robots.txt",
                    )

            # Check for existing credentials
            if self._auth_queue and self._auth_queue.has_credentials(url):
                credentials = self._auth_queue.get_credentials(url)
                return await self._scrape_with_auth(
                    url,
                    credentials,
                    session_id,
                    scope=self._credential_scope_for(url),
                )

            # Try scraping
            if use_browser:
                return await self._scrape_with_browser(url, session_id)
            else:
                return await self._scrape_with_httpx(url, session_id)

    async def _scrape_with_httpx(
        self,
        url: str,
        session_id: Optional[str] = None,
    ) -> ScrapeResult:
        """Scrape URL using httpx."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await _safe_httpx_get(
                    client,
                    url,
                    headers={"User-Agent": self.USER_AGENT},
                )

                html = response.text

                # Check for auth requirement
                if self._auth_detector:
                    auth_result = self._auth_detector.detect_from_response(
                        response.status_code,
                        dict(response.headers),
                        url,
                    )

                    # A 401/403 can be a bot-challenge whose only tell is in the
                    # body (Cloudflare "Just a moment...", PerimeterX captcha);
                    # re-check with the full response so such pages are reported
                    # blocked instead of entering the credential queue (WI0-B6).
                    if auth_result.requires_auth and response.status_code in (
                        401,
                        403,
                    ):
                        auth_result = self._auth_detector.detect(
                            url=url,
                            status_code=response.status_code,
                            headers=dict(response.headers),
                            html=html,
                        )

                    if auth_result.requires_auth:
                        if self._auth_queue:
                            await self._auth_queue.request_auth(
                                url=url,
                                auth_type=auth_result.auth_type,
                                session_id=session_id,
                                form_fields=auth_result.form_fields,
                                oauth_provider=auth_result.oauth_provider,
                            )
                        return ScrapeResult(
                            url=url,
                            status="auth_required",
                            status_code=response.status_code,
                            html=html,
                            metadata={"auth_type": auth_result.auth_type},
                        )

                # Check HTML for auth indicators
                if self._auth_detector:
                    auth_result = self._auth_detector.detect_from_html(html, url)
                    if auth_result.requires_auth and auth_result.confidence > 0.7:
                        if self._auth_queue:
                            await self._auth_queue.request_auth(
                                url=url,
                                auth_type=auth_result.auth_type,
                                session_id=session_id,
                                form_fields=auth_result.form_fields,
                            )
                        return ScrapeResult(
                            url=url,
                            status="auth_required",
                            status_code=response.status_code,
                            html=html,
                            metadata={"auth_type": auth_result.auth_type},
                        )

                # Extract content
                title = self._extractor.extract_title(html)
                content = self._extractor.extract_text(html)
                metadata = self._extractor.extract_metadata(html)

                return ScrapeResult(
                    url=url,
                    status="success",
                    title=title,
                    content=content,
                    html=html,
                    status_code=response.status_code,
                    metadata=metadata,
                )

        except httpx.TimeoutException:
            return ScrapeResult(
                url=url,
                status="timeout",
                error="Request timed out",
            )
        except Exception as e:
            return ScrapeResult(
                url=url,
                status="failed",
                error=str(e),
            )

    async def _scrape_with_browser(
        self,
        url: str,
        session_id: Optional[str] = None,
    ) -> ScrapeResult:
        """Scrape URL using Playwright browser."""
        try:
            browser = await self._get_browser()
            page = await browser.new_page()

            try:
                await page.set_extra_http_headers({"User-Agent": self.USER_AGENT})
                await page.route("**/*", _safe_browser_route_handler(self.timeout))

                response = await page.goto(url, timeout=self.timeout * 1000)

                if response is None:
                    return ScrapeResult(
                        url=url,
                        status="failed",
                        error="No response received",
                    )

                status_code = response.status

                # Wait for content to load
                await page.wait_for_load_state("domcontentloaded")

                html = await page.content()

                # Check for auth
                if self._auth_detector:
                    auth_result = self._auth_detector.detect(
                        url=url,
                        status_code=status_code,
                        headers=dict(response.headers),
                        html=html,
                    )

                    if auth_result.requires_auth:
                        if self._auth_queue:
                            await self._auth_queue.request_auth(
                                url=url,
                                auth_type=auth_result.auth_type,
                                session_id=session_id,
                                form_fields=auth_result.form_fields,
                                oauth_provider=auth_result.oauth_provider,
                            )
                        return ScrapeResult(
                            url=url,
                            status="auth_required",
                            status_code=status_code,
                            html=html,
                            metadata={"auth_type": auth_result.auth_type},
                        )

                title = await page.title()
                content = self._extractor.extract_text(html)
                metadata = self._extractor.extract_metadata(html)

                return ScrapeResult(
                    url=url,
                    status="success",
                    title=title,
                    content=content,
                    html=html,
                    status_code=status_code,
                    metadata=metadata,
                )

            finally:
                await page.close()

        except PlaywrightTimeout:
            return ScrapeResult(
                url=url,
                status="timeout",
                error="Browser timeout",
            )
        except Exception as e:
            return ScrapeResult(
                url=url,
                status="failed",
                error=str(e),
            )

    def _credential_scope_for(self, url: str) -> Optional[CredentialScope]:
        """Scope recorded when these credentials were submitted, or None.

        None means "no scope recorded", which the redirect rule reads as exact
        host. Resolved through `getattr` so an injected auth queue that predates
        scopes degrades to the strict rule instead of raising.
        """
        getter = getattr(self._auth_queue, "get_credential_scope", None)
        if getter is None:
            return None
        return getter(url)

    async def _scrape_with_auth(
        self,
        url: str,
        credentials: dict,
        session_id: Optional[str] = None,
        scope: Optional[CredentialScope] = None,
    ) -> ScrapeResult:
        """Scrape URL with authentication."""
        auth_type = credentials.get("type", "basic")

        if auth_type == "basic":
            return await self._scrape_basic_auth(url, credentials, scope)
        elif auth_type == "cookie":
            return await self._scrape_cookie_auth(url, credentials, scope)
        elif auth_type == "form":
            return await self._scrape_form_auth(url, credentials, session_id)
        else:
            return await self._scrape_with_httpx(url, session_id)

    def _authenticated_result(
        self,
        *,
        url: str,
        title: str,
        content: str,
        html: str,
        status_code: Optional[int],
        metadata: dict,
        trace: CredentialHopTrace,
        auth_type: str,
    ) -> ScrapeResult:
        """Build a credential-store result, telling the truth about the drop.

        When the chain left the credential origin, the body that was actually
        captured came from an UNAUTHENTICATED request, so `auth_used` stays
        false: it feeds the "authenticated capture => local embeddings" gate,
        and a false positive there mislabels public content as sensitive while
        hiding that the scrape silently returned the logged-out page. The drop
        is recorded in `metadata` and logged so an operator can see why.
        """
        if not trace.credentials_dropped:
            return ScrapeResult(
                url=url,
                status="success",
                title=title,
                content=content,
                html=html,
                status_code=status_code,
                metadata=metadata,
                auth_used=True,
            )

        fields = _credential_drop_fields(trace)
        log_event(
            "scrape.credentials_dropped_on_redirect",
            level=logging.WARNING,
            auth_type=auth_type,
            **fields,
        )
        return ScrapeResult(
            url=url,
            status="success",
            title=title,
            content=content,
            html=html,
            status_code=status_code,
            metadata={
                **metadata,
                "credential_scope_drop": {
                    "reason": "redirect_left_credential_origin",
                    "auth_type": auth_type,
                    **fields,
                },
            },
            auth_used=False,
        )

    async def _scrape_basic_auth(
        self,
        url: str,
        credentials: dict,
        scope: Optional[CredentialScope] = None,
    ) -> ScrapeResult:
        """Scrape with HTTP Basic Auth."""
        try:
            auth = httpx.BasicAuth(
                credentials.get("username", ""),
                credentials.get("password", ""),
            )

            # auth goes to the request, not the client: a client-level auth is
            # replayed onto whatever host a redirect names.
            trace = CredentialHopTrace()
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await _safe_httpx_get(
                    client,
                    url,
                    headers={"User-Agent": self.USER_AGENT},
                    auth=auth,
                    credential_scope=scope,
                    trace=trace,
                )

                html = response.text
                title = self._extractor.extract_title(html)
                content = self._extractor.extract_text(html)
                metadata = self._extractor.extract_metadata(html)

                return self._authenticated_result(
                    url=url,
                    title=title,
                    content=content,
                    html=html,
                    status_code=response.status_code,
                    metadata=metadata,
                    trace=trace,
                    auth_type="basic",
                )

        except Exception as e:
            return ScrapeResult(
                url=url,
                status="failed",
                error=str(e),
            )

    async def _scrape_cookie_auth(
        self,
        url: str,
        credentials: dict,
        scope: Optional[CredentialScope] = None,
    ) -> ScrapeResult:
        """Scrape with cookie authentication."""
        try:
            cookies = credentials.get("cookies", {})

            # cookies go to the request, not the client jar: a bare-domain jar
            # is replayed onto whatever host a redirect names.
            trace = CredentialHopTrace()
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await _safe_httpx_get(
                    client,
                    url,
                    headers={"User-Agent": self.USER_AGENT},
                    cookies=cookies,
                    credential_scope=scope,
                    trace=trace,
                )

                html = response.text
                title = self._extractor.extract_title(html)
                content = self._extractor.extract_text(html)
                metadata = self._extractor.extract_metadata(html)

                return self._authenticated_result(
                    url=url,
                    title=title,
                    content=content,
                    html=html,
                    status_code=response.status_code,
                    metadata=metadata,
                    trace=trace,
                    auth_type="cookie",
                )

        except Exception as e:
            return ScrapeResult(
                url=url,
                status="failed",
                error=str(e),
            )

    async def _scrape_form_auth(
        self,
        url: str,
        credentials: dict,
        session_id: Optional[str] = None,
    ) -> ScrapeResult:
        """Scrape with form-based authentication using browser."""
        try:
            browser = await self._get_browser()
            page = await browser.new_page()

            try:
                login_url = credentials.get("login_url", url)
                await page.route("**/*", _safe_browser_route_handler(self.timeout))
                await page.goto(login_url, timeout=self.timeout * 1000)

                # Fill form fields
                username_field = credentials.get("username_field", "username")
                password_field = credentials.get("password_field", "password")

                await page.fill(
                    f'input[name="{username_field}"], input[type="email"], input[type="text"]',
                    credentials.get("username", ""),
                )
                await page.fill(
                    f'input[name="{password_field}"], input[type="password"]',
                    credentials.get("password", ""),
                )

                # Submit form
                await page.click('button[type="submit"], input[type="submit"]')
                await page.wait_for_load_state("networkidle")

                # Navigate to target URL
                if page.url != url:
                    await page.goto(url, timeout=self.timeout * 1000)

                html = await page.content()
                title = await page.title()
                content = self._extractor.extract_text(html)
                metadata = self._extractor.extract_metadata(html)

                return ScrapeResult(
                    url=url,
                    status="success",
                    title=title,
                    content=content,
                    html=html,
                    metadata=metadata,
                    auth_used=True,
                )

            finally:
                await page.close()

        except Exception as e:
            return ScrapeResult(
                url=url,
                status="failed",
                error=str(e),
            )

    async def scrape_batch(
        self,
        urls: list[str],
        session_id: Optional[str] = None,
        callback: Optional[Callable[[ScrapeResult], Awaitable[None]]] = None,
        use_browser: bool = False,
    ) -> list[ScrapeResult]:
        """
        Scrape multiple URLs in parallel.

        Non-blocking: continues scraping public sites while
        waiting for auth on protected sites.
        """
        tasks = []

        for url in urls:
            task = asyncio.create_task(
                self._scrape_with_callback(url, session_id, callback, use_browser)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        final_results = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                final_results.append(
                    ScrapeResult(
                        url=url,
                        status="failed",
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)

        return final_results

    async def _scrape_with_callback(
        self,
        url: str,
        session_id: Optional[str],
        callback: Optional[Callable[[ScrapeResult], Awaitable[None]]],
        use_browser: bool = False,
    ) -> ScrapeResult:
        """Scrape URL and call callback with result."""
        result = await self.scrape_url(
            url,
            session_id=session_id,
            use_browser=use_browser,
        )

        if callback:
            try:
                await callback(result)
            except Exception as error:
                result.metadata = {**result.metadata, "callback_error": str(error)}

        return result
