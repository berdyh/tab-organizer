"""Attach-mode Chromium DevTools Protocol tab import."""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

from services.url_safety import validate_scrape_url

LOCAL_CDP_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}
DEFAULT_CDP_URL = "http://host.docker.internal:9222"


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
            browser = await playwright.chromium.connect_over_cdp(self.cdp_url)
            return playwright, browser
        except Exception:
            await playwright.stop()
            raise

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
