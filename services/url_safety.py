"""URL validation policy for outbound scraping."""

import ipaddress
import os
import socket
from dataclasses import dataclass
from urllib.parse import ParseResult, urlparse, urlunparse

SAFE_SCRAPE_SCHEMES = {"http", "https"}
LOCAL_HOSTNAMES = {
    "localhost",
    "host.docker.internal",
    "gateway.docker.internal",
    "docker.internal",
}


@dataclass(frozen=True)
class ResolvedScrapeTarget:
    """Network target for a scrape request after DNS safety checks."""

    request_url: str
    host_header: str | None = None
    sni_hostname: str | None = None


def private_scrape_urls_allowed() -> bool:
    """Return whether private-network scrape targets are explicitly allowed."""
    return os.getenv("SCRAPE_ALLOW_PRIVATE_NETWORKS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def normalize_scrape_url(url: str) -> str:
    """Validate and normalize a URL before it can enter scrape workflows."""
    parsed = _parse_url(url)
    host = parsed.hostname.lower() if parsed.hostname else ""
    if not private_scrape_urls_allowed():
        _reject_private_host(host)

    netloc = _normalized_netloc(host, parsed)
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme.lower(), netloc, path, "", parsed.query, ""))


def validate_scrape_url(url: str) -> str:
    """Return the original URL if it satisfies the outbound scrape policy."""
    normalize_scrape_url(url)
    return url


def scrape_url_host_is_ip_literal(url: str) -> bool:
    """Return whether the scrape URL host is already an IP literal."""
    parsed = _parse_url(url)
    try:
        ipaddress.ip_address(parsed.hostname or "")
    except ValueError:
        return False
    return True


def resolve_scrape_targets(url: str) -> list[ResolvedScrapeTarget]:
    """Resolve a scrape URL to vetted connect targets.

    When private-network scraping is disabled, callers should connect to these
    targets directly instead of letting the HTTP client perform a second DNS
    lookup that could be changed by DNS rebinding.
    """
    parsed = _parse_url(url)
    host = parsed.hostname.lower() if parsed.hostname else ""

    if private_scrape_urls_allowed():
        return [ResolvedScrapeTarget(_request_url(parsed, host))]

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        _reject_internal_host_name(host)
        addresses = _resolve_public_addresses(host, parsed)
    else:
        _reject_private_address(address)
        return [ResolvedScrapeTarget(_request_url(parsed, host))]

    return [
        ResolvedScrapeTarget(
            request_url=_request_url(parsed, raw_address),
            host_header=_normalized_netloc(host, parsed),
            sni_hostname=host if parsed.scheme.lower() == "https" else None,
        )
        for raw_address in addresses
    ]


def _parse_url(url: str) -> ParseResult:
    candidate = url.strip()
    if not candidate:
        raise ValueError("URL is required")
    if "://" not in candidate:
        candidate = f"https://{candidate}"

    parsed = urlparse(candidate)
    if parsed.scheme.lower() not in SAFE_SCRAPE_SCHEMES:
        raise ValueError("Scrape URL must use http or https")
    if not parsed.hostname:
        raise ValueError("Scrape URL must include a host")
    if parsed.username or parsed.password:
        raise ValueError("Scrape URL must not include credentials")
    return parsed


def _reject_private_host(host: str) -> None:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        _reject_internal_host_name(host)
        _reject_resolved_private_addresses(host)
        return

    _reject_private_address(address)


def _reject_internal_host_name(host: str) -> None:
    if host in LOCAL_HOSTNAMES or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("Scrape URL must not target local or internal hosts")
    if "." not in host:
        raise ValueError("Scrape URL must not target single-label internal hosts")


def _reject_resolved_private_addresses(host: str) -> None:
    try:
        addrinfos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return

    for raw_address in {addrinfo[4][0] for addrinfo in addrinfos}:
        try:
            address = ipaddress.ip_address(raw_address)
        except ValueError:
            continue
        _reject_private_address(address)


def _resolve_public_addresses(host: str, parsed: ParseResult) -> list[str]:
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    try:
        addrinfos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise ValueError("Scrape URL host could not be resolved safely") from error

    addresses = []
    for raw_address in {addrinfo[4][0] for addrinfo in addrinfos}:
        try:
            address = ipaddress.ip_address(raw_address)
        except ValueError:
            continue
        _reject_private_address(address)
        addresses.append(raw_address)

    if not addresses:
        raise ValueError("Scrape URL host could not be resolved safely")
    return addresses


def _reject_private_address(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> None:
    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        raise ValueError("Scrape URL must not target private network addresses")


def _normalized_netloc(host: str, parsed: ParseResult) -> str:
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{host}:{parsed.port}" if parsed.port else host


def _request_url(parsed: ParseResult, host: str) -> str:
    path = parsed.path or "/"
    return urlunparse(
        (
            parsed.scheme.lower(),
            _normalized_netloc(host, parsed),
            path,
            "",
            parsed.query,
            "",
        )
    )
