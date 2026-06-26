"""URL validation policy for outbound scraping."""

import ipaddress
import os
from urllib.parse import ParseResult, urlparse, urlunparse


SAFE_SCRAPE_SCHEMES = {"http", "https"}
LOCAL_HOSTNAMES = {
    "localhost",
    "host.docker.internal",
    "gateway.docker.internal",
    "docker.internal",
}


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
    if host in LOCAL_HOSTNAMES or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("Scrape URL must not target local or internal hosts")
    if "." not in host:
        raise ValueError("Scrape URL must not target single-label internal hosts")

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return

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
