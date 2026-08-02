"""Attach-mode Chromium DevTools Protocol tab import."""

import asyncio
import ipaddress
import json
import logging
import re
import socket
from dataclasses import dataclass, field
from typing import Any, NoReturn, Optional, Union
from urllib.parse import urlparse, urlunparse

import httpx

from services.observability import log_event
from services.url_safety import validate_scrape_url

LOCAL_CDP_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}
DEFAULT_CDP_URL = "http://host.docker.internal:9222"

IPAddress = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]

# CDP attach is a local-only invariant, so the resolved-address gate is a
# deliberate ALLOWLIST of genuinely host-local networks. Everything else is
# refused by default.
#
# An allowlist is used because both obvious "is it local?" predicates in the
# stdlib are unsound here:
#   * `is_private` is far too broad. It is True for the 6to4 range 2002::/16
#     (so `2002:808:808::` -- which encodes the *global* IPv4 8.8.8.8 -- reads
#     as "private"), for the benchmark range 198.18.0.0/15, for 192.0.0.0/24,
#     and for the documentation ranges 2001:db8::/32 and 2001:2::/48. All of
#     those are routable off-host through a container's default gateway.
#   * `not is_global` is no better: every range listed above is also
#     `is_global == False`.
# A denylist of "dangerous" subclasses only ever closes the instances someone
# thought to enumerate; enumerating what is *allowed* fails closed against the
# ranges nobody thought about.
#
# IPv4 and IPv6 link-local (169.254.0.0/16 -- the cloud metadata address
# 169.254.169.254 -- and fe80::/10) are deliberately absent. fe80::/10 lies
# outside fc00::/7, so IPv6 link-local is excluded structurally rather than by
# a carve-out; `test_local_cdp_networks_exclude_link_local` pins that.
#
# ADDING A NETWORK HERE IS A SECURITY DECISION: it widens what a poisoned or
# misconfigured resolver can point the local-only CDP client at.
LOCAL_CDP_NETWORKS: tuple[Union[ipaddress.IPv4Network, ipaddress.IPv6Network], ...] = (
    ipaddress.ip_network("127.0.0.0/8"),  # IPv4 loopback
    ipaddress.ip_network("10.0.0.0/8"),  # RFC1918
    ipaddress.ip_network("172.16.0.0/12"),  # RFC1918; docker0 bridge lives here
    ipaddress.ip_network("192.168.0.0/16"),  # RFC1918
    ipaddress.ip_network("::1/128"),  # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),  # IPv6 unique-local (excludes fe80::/10)
)

# Chrome's remote-debugging HTTP server rejects any Host header that is not an
# IP address or the literal "localhost" ("Host header is specified and is not
# an IP address or localhost"). These forms are accepted as-is; every other
# allowed input host (notably `host.docker.internal`) must be resolved to its
# IP before the URL is handed to Playwright. See WI0 B2.
CHROME_HOST_HEADER_SAFE = {"localhost", "127.0.0.1", "::1"}

# The websocket endpoint the debug server advertises is attacker-reachable data
# (see `resolve_cdp_ws_endpoint`), so reading it is bounded like any untrusted
# response: one short timeout, no redirects, no proxy env, and a body cap.
CDP_WS_SCHEMES = {"ws", "wss"}
CDP_VERSION_TIMEOUT_SECONDS = 10.0
CDP_VERSION_MAX_BYTES = 64 * 1024


class CDPConnectionError(RuntimeError):
    """A CDP attach failure carrying an actionable {code, cause, fix} message."""

    def __init__(self, code: str, cause: str, fix: str):
        self.code = code
        self.cause = cause
        self.fix = fix
        super().__init__(f"{code}: {cause} | fix: {fix}")


def _format_netloc(hostname: str, port: Optional[int]) -> str:
    """Build a netloc, re-bracketing IPv6 literals so urlparse can read it back.

    `urlparse` strips the brackets off a bracketed IPv6 host when exposing
    `.hostname` (e.g. `[::1]` -> `::1`), so rebuilding a netloc from that
    unbracketed form must re-add them or the result is ambiguous ("::1:9222"
    parses as a different, broken host/port split).
    """
    host = f"[{hostname}]" if ":" in hostname else hostname
    return f"{host}:{port}" if port else host


def validate_cdp_url(url: str) -> str:
    """Return a normalized local CDP URL or raise ValueError."""
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Chrome debugging endpoint must use http or https")
    if not parsed.hostname:
        raise ValueError("Chrome debugging endpoint must include a host")
    if parsed.username or parsed.password or parsed.path not in {"", "/"}:
        raise ValueError(
            "Chrome debugging endpoint must not include credentials or a path"
        )
    if parsed.params or parsed.query or parsed.fragment:
        raise ValueError("Chrome debugging endpoint must not include query parameters")

    hostname = parsed.hostname.lower()
    if hostname not in LOCAL_CDP_HOSTS:
        raise ValueError(
            "Chrome debugging endpoint must be a local Chrome debugging endpoint"
        )

    netloc = _format_netloc(hostname, parsed.port)

    return urlunparse((parsed.scheme, netloc, "", "", "", ""))


def _parse_resolved_address(ip_text: str, hostname: str) -> IPAddress:
    """Parse one resolver answer into an address, or raise an actionable error."""
    try:
        return ipaddress.ip_address(ip_text)
    except ValueError as error:
        raise CDPConnectionError(
            code="cdp_resolved_address_invalid",
            cause=(
                f"resolver returned a non-IP value {ip_text!r} for Chrome debug "
                f"host {hostname!r}: {error}"
            ),
            fix="Check the container's resolver/hosts entry for the debug host.",
        ) from error


def _is_local_cdp_address(address: IPAddress) -> bool:
    """Return whether an address falls inside the local-attach allowlist."""
    return any(address in network for network in LOCAL_CDP_NETWORKS)


def _preferred_connect_address(candidates: list[IPAddress]) -> IPAddress:
    """Pick which allowlisted address to actually dial.

    IPv4 wins over IPv6, and resolver order breaks ties inside a family. This
    is deliberate rather than "whatever `getaddrinfo` returned first": under
    AF_UNSPEC the OS applies RFC 6724 sorting, which prefers IPv6, so a
    dual-stack `host.docker.internal` would silently move the attach target
    from the A record to the AAAA record. The documented bridge in
    `_cdp_connect_failure` ("socat ... bind=172.17.0.1") is IPv4-only, so the
    IPv6 answer is the one that will not accept the connection.
    """
    for address in candidates:
        if address.version == 4:
            return address
    return candidates[0]


def resolve_cdp_connect_url(validated_url: str) -> str:
    """Return the URL to actually connect to, with the host resolved to an IP.

    `validate_cdp_url` keeps an allowlist for *input* (it accepts the friendly
    `host.docker.internal` name a user configures), but Chrome's debug port
    rejects that name in the Host header. Resolve any non-IP, non-localhost
    host to its IP before connecting; leave localhost/IP forms untouched (WI0
    B2).

    Every answer the resolver returns is checked against `LOCAL_CDP_NETWORKS`.
    Unsafe answers are *filtered out* and the connection is made to the
    preferred survivor; the resolution is refused only when no answer is
    local. Rejecting the whole lookup because one answer was unsafe would
    break a real and common configuration -- `host.docker.internal` answering
    A=172.17.0.1 alongside AAAA=fe80::1 -- without buying any security: the
    URL returned here carries a pinned IP literal, so the unsafe answer is
    never dialed and cannot be swapped in later (that is what closes DNS
    rebinding, not the reject-if-any behavior). Filtered answers are logged so
    a poisoning attempt is still visible.
    """
    parsed = urlparse(validated_url)
    hostname = (parsed.hostname or "").lower()
    if hostname in CHROME_HOST_HEADER_SAFE:
        return validated_url

    try:
        # AF_UNSPEC + SOCK_STREAM covers both IPv4 and IPv6 answers; a
        # v4-only lookup (the previous socket.gethostbyname) would silently
        # ignore any AAAA record the same hostname resolves to.
        addr_infos = socket.getaddrinfo(
            hostname, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
        )
    except OSError as error:
        raise CDPConnectionError(
            code="cdp_host_unresolvable",
            cause=f"could not resolve Chrome debug host {hostname!r}: {error}",
            fix=(
                "Ensure the container can resolve the debug host (Docker's "
                "host-gateway maps host.docker.internal); on Linux add "
                "'--add-host=host.docker.internal:host-gateway'."
            ),
        ) from error

    resolved_ips: list[str] = []
    seen: set[str] = set()
    for info in addr_infos:
        ip_text = info[4][0]
        if ip_text not in seen:
            seen.add(ip_text)
            resolved_ips.append(ip_text)

    if not resolved_ips:
        raise CDPConnectionError(
            code="cdp_host_unresolvable",
            cause=f"resolver returned no addresses for Chrome debug host {hostname!r}",
            fix=(
                "Ensure the container can resolve the debug host (Docker's "
                "host-gateway maps host.docker.internal); on Linux add "
                "'--add-host=host.docker.internal:host-gateway'."
            ),
        )

    allowed: list[IPAddress] = []
    rejected: list[str] = []
    for ip_text in resolved_ips:
        address = _parse_resolved_address(ip_text, hostname)
        if _is_local_cdp_address(address):
            allowed.append(address)
        else:
            rejected.append(ip_text)

    if not allowed:
        raise CDPConnectionError(
            code="cdp_resolved_address_not_local",
            cause=(
                f"Chrome debug host {hostname!r} resolved to "
                f"{', '.join(rejected)}, which is not in the local-attach "
                "allowlist (loopback, RFC1918, or IPv6 unique-local); refusing "
                "to attach a local-only CDP client to a non-local target"
            ),
            fix=(
                "CDP attach is local-only by design. Confirm the container's "
                "resolver/hosts file maps this debug host to the Docker "
                "bridge gateway or loopback (e.g. 'docker network inspect "
                "bridge'), not to a public or otherwise remote address."
            ),
        )

    if rejected:
        log_event(
            "cdp.resolved_address_filtered",
            level=logging.WARNING,
            host=hostname,
            rejected=rejected,
            allowed=[str(address) for address in allowed],
        )

    netloc = _format_netloc(str(_preferred_connect_address(allowed)), parsed.port)
    return urlunparse((parsed.scheme, netloc, "", "", "", ""))


def _cdp_connect_failure(cdp_url: str, error: Exception) -> CDPConnectionError:
    """Wrap a raw connect failure in an actionable {code, cause, fix} error."""
    return CDPConnectionError(
        code="cdp_connect_failed",
        cause=f"could not attach to Chrome debug endpoint {cdp_url}: {error}",
        fix=(
            "Chrome's debug port defaults to 127.0.0.1, which is unreachable "
            "from the container. Bridge it to the Docker bridge interface only "
            "(never the whole LAN): run a host-side "
            "'socat TCP-LISTEN:9222,fork,bind=172.17.0.1 TCP:127.0.0.1:9222' "
            "(172.17.0.1 is the default docker0 bridge IP; confirm yours with "
            "'docker network inspect bridge'). Headed Chrome ignores "
            "--remote-debugging-address and stays on loopback, so the socat "
            "bridge is the reliable path for a headed browser. Only as a last "
            "resort bind the port to all interfaces "
            "('--remote-debugging-address=0.0.0.0' or socat 'bind=0.0.0.0') -- "
            "that exposes the UNAUTHENTICATED debug port to the LAN and must be "
            "firewalled. Host-side tab capture is the durable fix "
            "(plan decision 24)."
        ),
    )


def _ws_endpoint_refused(connect_url: str, advertised: Any, reason: str) -> NoReturn:
    """Log and raise for a debug endpoint that advertised an unusable socket."""
    log_event(
        "cdp.ws_endpoint_rejected",
        level=logging.WARNING,
        connect_url=connect_url,
        advertised=str(advertised)[:200],
        reason=reason,
    )
    raise CDPConnectionError(
        code="cdp_ws_endpoint_not_local",
        cause=(
            f"Chrome debug endpoint {connect_url} advertised websocket endpoint "
            f"{str(advertised)[:200]!r}, which {reason}; refusing to attach a "
            "local-only CDP client to an endpoint it did not validate"
        ),
        fix=(
            "The websocket endpoint a debug server advertises must be the same "
            "host and port that was attached to. Confirm CDP_URL points at your "
            "own Chrome debug port and that nothing (a proxy, a redirect, or "
            "another process bound to that address) is answering for it."
        ),
    )


def _same_cdp_host(advertised_host: str, pinned_host: str) -> bool:
    """Return whether an advertised host is the exact host that was pinned.

    Compared as addresses when both parse as one, so alternate spellings of the
    same IP (`::1` vs `0:0:0:0:0:0:0:1`) match, and as exact lowercased strings
    otherwise. Deliberately strict: `127.0.0.1` does not match `localhost`,
    because name-to-address equivalence is exactly the resolver-controlled step
    the pinning exists to remove.
    """
    advertised = advertised_host.lower()
    pinned = pinned_host.lower()
    if advertised == pinned:
        return True
    try:
        return ipaddress.ip_address(advertised) == ipaddress.ip_address(pinned)
    except ValueError:
        return False


def validate_ws_debugger_url(advertised: Any, connect_url: str) -> str:
    """Return the advertised websocket endpoint or raise `CDPConnectionError`.

    Applies the same local-only rules the connect URL passed: ws/wss scheme, no
    credentials, the exact pinned host, the exact pinned port, and -- checked
    independently of the host comparison so a hole in one is not a hole in both
    -- an address inside `LOCAL_CDP_NETWORKS` (or one of the literal local host
    forms Chrome's Host-header check permits).
    """
    if not isinstance(advertised, str) or not advertised.strip():
        _ws_endpoint_refused(
            connect_url, advertised, "is missing or is not a websocket URL string"
        )

    candidate = advertised.strip()
    # Refused before parsing: a control character or space can mean one thing to
    # `urlparse` here and another to the WHATWG parser and websocket client on
    # the Node side, and CRLF in a path is request-smuggling material.
    if any(ord(char) < 0x21 or ord(char) == 0x7F for char in candidate):
        _ws_endpoint_refused(
            connect_url, advertised, "contains control characters or whitespace"
        )

    parsed = urlparse(candidate)
    if parsed.scheme not in CDP_WS_SCHEMES:
        _ws_endpoint_refused(connect_url, advertised, "does not use the ws/wss scheme")
    if parsed.username or parsed.password:
        _ws_endpoint_refused(connect_url, advertised, "carries embedded credentials")
    if not parsed.hostname:
        _ws_endpoint_refused(connect_url, advertised, "has no host")

    pinned = urlparse(connect_url)
    hostname = (parsed.hostname or "").lower()
    if not _same_cdp_host(hostname, (pinned.hostname or "").lower()):
        _ws_endpoint_refused(
            connect_url,
            advertised,
            f"points at host {hostname!r} rather than the attached endpoint",
        )
    if parsed.port != pinned.port:
        _ws_endpoint_refused(
            connect_url,
            advertised,
            f"points at port {parsed.port!r} rather than the attached port",
        )

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is None:
        if hostname not in CHROME_HOST_HEADER_SAFE:
            _ws_endpoint_refused(
                connect_url, advertised, "is not in the local-attach allowlist"
            )
    elif not _is_local_cdp_address(address):
        _ws_endpoint_refused(
            connect_url, advertised, "is not in the local-attach allowlist"
        )

    # Emitted in canonical form so the string Playwright parses cannot spell the
    # validated host any other way.
    emit_host = str(address) if address is not None else hostname
    return urlunparse(
        (
            parsed.scheme,
            _format_netloc(emit_host, parsed.port),
            parsed.path,
            parsed.params,
            parsed.query,
            "",
        )
    )


async def _fetch_cdp_version(connect_url: str) -> dict[str, Any]:
    """Read `/json/version` from the pinned endpoint as untrusted data."""
    url = f"{connect_url.rstrip('/')}/json/version"
    try:
        async with httpx.AsyncClient(
            timeout=CDP_VERSION_TIMEOUT_SECONDS,
            # A 3xx must not move this request off the pinned address, and
            # proxy env vars must not reroute a local-only call.
            follow_redirects=False,
            trust_env=False,
        ) as client:
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    _ws_endpoint_refused(
                        connect_url,
                        f"HTTP {response.status_code}",
                        "did not answer /json/version with 200",
                    )
                body = b""
                async for chunk in response.aiter_bytes():
                    body += chunk
                    if len(body) > CDP_VERSION_MAX_BYTES:
                        _ws_endpoint_refused(
                            connect_url,
                            f"{len(body)} bytes",
                            "returned an oversized /json/version body",
                        )
    except httpx.HTTPError as error:
        raise _cdp_connect_failure(connect_url, error) from error

    try:
        payload = json.loads(body)
    except ValueError as error:
        _ws_endpoint_refused(connect_url, body[:200], f"is not JSON: {error}")
    if not isinstance(payload, dict):
        _ws_endpoint_refused(connect_url, payload, "is not a /json/version object")
    return payload


async def resolve_cdp_ws_endpoint(connect_url: str) -> str:
    """Return the validated websocket endpoint to hand Playwright.

    `connect_over_cdp` given an http(s) URL fetches `/json/version` from it and
    then dials whatever `webSocketDebuggerUrl` the response body contains --
    following cross-host redirects and honouring proxy env vars on the way
    (playwright-core `server/chromium/chromium.js:urlToWSEndpoint` +
    `utils/network.js:httpRequest`, unchanged from 1.41 through 1.57). Pinning
    the connect address therefore pins only the *first* hop: anything that can
    answer on that address chooses the socket Playwright ends up on.

    So the fetch happens here instead, under this module's rules, and Playwright
    is handed a ws:// URL -- which `urlToWSEndpoint` returns untouched, skipping
    its fetch, its redirect chain and its proxy lookup entirely.

    `webSocketDebuggerUrl` is the only field of that response consumed by
    anything: Playwright ignores the rest (`Browser`, `Protocol-Version`,
    `User-Agent`, `V8-Version`, `WebKit-Version`), and neither this module nor
    Playwright's attach path reads `/json/list` -- target URLs arrive over the
    validated CDP socket and still pass `_is_importable_page`.
    """
    payload = await _fetch_cdp_version(connect_url)
    return validate_ws_debugger_url(payload.get("webSocketDebuggerUrl"), connect_url)


def _is_importable_page(url: str) -> bool:
    """Return whether a browser target should be imported as a content tab."""
    if not url:
        return False
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    try:
        validate_scrape_url(url)
    except ValueError:
        return False
    return True


def _extract_domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.lower()


def _readable_text_from_html(html: str) -> Optional[str]:
    """Extract readable text from HTML with optional trafilatura support."""
    try:
        import trafilatura  # type: ignore
    except Exception:
        return None

    try:
        text = trafilatura.extract(html)
    except Exception:
        return None
    return text.strip() if text else None


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


@dataclass(frozen=True)
class HarvestedTab:
    """A tab imported from an attached browser."""

    url: str
    title: str
    content: str
    html: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_document(self) -> dict[str, Any]:
        """Return the AI/backend document shape for this tab."""
        return {
            "id": self.url,
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "metadata": {
                **self.metadata,
                "source": "cdp",
                "domain": _extract_domain(self.url),
            },
        }


@dataclass(frozen=True)
class TabHarvestError:
    """A non-fatal tab import error."""

    url: str
    error: str


@dataclass(frozen=True)
class TabHarvestResult:
    """Result of importing tabs from an attached browser."""

    tabs: list[HarvestedTab]
    errors: list[TabHarvestError] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.tabs)

    @property
    def failed(self) -> int:
        return len(self.errors)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result."""
        return {
            "total": self.total,
            "imported": len(self.tabs),
            "failed": self.failed,
            "tabs": [tab.to_document() for tab in self.tabs],
            "errors": [error.__dict__ for error in self.errors],
        }


class CDPTabHarvester:
    """Attach to an existing Chromium instance and harvest open tabs."""

    def __init__(
        self,
        cdp_url: str = DEFAULT_CDP_URL,
        playwright_factory=None,
        max_concurrent: int = 25,
    ):
        self.cdp_url = validate_cdp_url(cdp_url)
        self.playwright_factory = playwright_factory
        self.max_concurrent = max(1, max_concurrent)

    async def _start_playwright(self):
        factory = self.playwright_factory
        if factory is None:
            from playwright.async_api import async_playwright

            factory = async_playwright()
        return await factory.start()

    async def _connect(self):
        playwright = await self._start_playwright()
        try:
            connect_url = resolve_cdp_connect_url(self.cdp_url)
            ws_endpoint = await resolve_cdp_ws_endpoint(connect_url)
        except CDPConnectionError:
            await playwright.stop()
            raise
        try:
            browser = await playwright.chromium.connect_over_cdp(ws_endpoint)
            return playwright, browser
        except Exception as error:
            await playwright.stop()
            raise _cdp_connect_failure(ws_endpoint, error) from error

    async def harvest(self, max_tabs: Optional[int] = None) -> TabHarvestResult:
        """Import visible HTTP(S) pages from the attached browser."""
        playwright, browser = await self._connect()
        try:
            pages = []
            for context in getattr(browser, "contexts", []):
                pages.extend(getattr(context, "pages", []))

            importable_pages = [
                page for page in pages if _is_importable_page(getattr(page, "url", ""))
            ]
            if max_tabs is not None:
                importable_pages = importable_pages[: max(0, max_tabs)]

            semaphore = asyncio.Semaphore(self.max_concurrent)

            async def harvest_page(page):
                async with semaphore:
                    return await self._harvest_page(page)

            results = await asyncio.gather(
                *(harvest_page(page) for page in importable_pages),
                return_exceptions=True,
            )

            tabs: list[HarvestedTab] = []
            errors: list[TabHarvestError] = []
            for page, result in zip(importable_pages, results):
                if isinstance(result, HarvestedTab):
                    tabs.append(result)
                elif isinstance(result, Exception):
                    errors.append(
                        TabHarvestError(
                            url=getattr(page, "url", ""),
                            error=str(result),
                        )
                    )
            return TabHarvestResult(tabs=tabs, errors=errors)
        finally:
            await playwright.stop()

    async def _harvest_page(self, page) -> HarvestedTab:
        url = getattr(page, "url", "")
        title = await page.title()
        html = await page.content()
        content = _readable_text_from_html(html)
        if not content:
            content = await page.evaluate(
                "() => document.body ? document.body.innerText : ''"
            )
        content = _collapse_whitespace(content)
        return HarvestedTab(
            url=url,
            title=title or url,
            content=content,
            html=html,
            metadata={"title": title or url},
        )

    async def open_urls(self, urls: list[str]) -> list[dict[str, str]]:
        """Open URLs in the attached browser without creating a new profile."""
        safe_urls = []
        for url in urls:
            validate_scrape_url(url)
            safe_urls.append(url)

        playwright, browser = await self._connect()
        try:
            contexts = getattr(browser, "contexts", [])
            if contexts:
                context = contexts[0]
            else:
                context = await browser.new_context()

            opened = []
            for url in safe_urls:
                page = await context.new_page()
                await page.goto(url)
                opened.append({"url": url, "status": "opened"})
            return opened
        finally:
            await playwright.stop()
