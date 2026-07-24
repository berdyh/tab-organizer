"""Attach-mode Chromium DevTools Protocol tab import."""

import asyncio
import re
import socket
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

from services.url_safety import validate_scrape_url

LOCAL_CDP_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}
DEFAULT_CDP_URL = "http://host.docker.internal:9222"

# Chrome's remote-debugging HTTP server rejects any Host header that is not an
# IP address or the literal "localhost" ("Host header is specified and is not
# an IP address or localhost"). These forms are accepted as-is; every other
# allowed input host (notably `host.docker.internal`) must be resolved to its
# IP before the URL is handed to Playwright. See WI0 B2.
CHROME_HOST_HEADER_SAFE = {"localhost", "127.0.0.1", "::1"}


class CDPConnectionError(RuntimeError):
    """A CDP attach failure carrying an actionable {code, cause, fix} message."""

    def __init__(self, code: str, cause: str, fix: str):
        self.code = code
        self.cause = cause
        self.fix = fix
        super().__init__(f"{code}: {cause} | fix: {fix}")


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

    netloc = hostname
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"

    return urlunparse((parsed.scheme, netloc, "", "", "", ""))


def resolve_cdp_connect_url(validated_url: str) -> str:
    """Return the URL to actually connect to, with the host resolved to an IP.

    `validate_cdp_url` keeps an allowlist for *input* (it accepts the friendly
    `host.docker.internal` name a user configures), but Chrome's debug port
    rejects that name in the Host header. Resolve any non-IP, non-localhost
    host to its IP before connecting; leave localhost/IP forms untouched (WI0
    B2).
    """
    parsed = urlparse(validated_url)
    hostname = (parsed.hostname or "").lower()
    if hostname in CHROME_HOST_HEADER_SAFE:
        return validated_url

    try:
        ip = socket.gethostbyname(hostname)
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

    netloc = f"{ip}:{parsed.port}" if parsed.port else ip
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
        except CDPConnectionError:
            await playwright.stop()
            raise
        try:
            browser = await playwright.chromium.connect_over_cdp(connect_url)
            return playwright, browser
        except Exception as error:
            await playwright.stop()
            raise _cdp_connect_failure(connect_url, error) from error

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
