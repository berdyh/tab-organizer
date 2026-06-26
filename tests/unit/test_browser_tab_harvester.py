"""Contracts for importing live Chromium tabs through CDP."""

import pytest

from services.browser_engine.app.tabs.cdp import CDPTabHarvester, validate_cdp_url


class FakePage:
    """Minimal Playwright page double for CDP import tests."""

    def __init__(self, url: str, title: str, text: str):
        self.url = url
        self._title = title
        self._text = text
        self.opened_urls: list[str] = []

    async def title(self):
        return self._title

    async def content(self):
        return f"<html><head><title>{self._title}</title></head><body>{self._text}</body></html>"

    async def evaluate(self, script):
        assert "innerText" in script
        return self._text

    async def goto(self, url):
        self.opened_urls.append(url)


class FakeContext:
    def __init__(self, pages):
        self.pages = pages
        self.created_pages: list[FakePage] = []

    async def new_page(self):
        page = FakePage("about:blank", "", "")
        self.created_pages.append(page)
        return page


class FakeBrowser:
    def __init__(self, pages):
        self.contexts = [FakeContext(pages)]
        self.closed = False

    async def close(self):
        self.closed = True


class FakeChromium:
    def __init__(self, browser):
        self.browser = browser
        self.connected_urls: list[str] = []

    async def connect_over_cdp(self, url):
        self.connected_urls.append(url)
        return self.browser


class FakePlaywright:
    def __init__(self, browser):
        self.chromium = FakeChromium(browser)
        self.stopped = False

    async def stop(self):
        self.stopped = True


class FakePlaywrightFactory:
    def __init__(self, playwright):
        self.playwright = playwright

    async def start(self):
        return self.playwright


def test_validate_cdp_url_allows_only_local_control_plane():
    assert validate_cdp_url("http://localhost:9222") == "http://localhost:9222"
    assert validate_cdp_url("http://127.0.0.1:9222/") == "http://127.0.0.1:9222"
    assert (
        validate_cdp_url("http://host.docker.internal:9222")
        == "http://host.docker.internal:9222"
    )

    with pytest.raises(ValueError, match="local Chrome debugging endpoint"):
        validate_cdp_url("http://example.com:9222")

    with pytest.raises(ValueError, match="http"):
        validate_cdp_url("file:///tmp/socket")


@pytest.mark.asyncio
async def test_harvester_imports_visible_tabs_without_closing_user_browser():
    browser = FakeBrowser(
        [
            FakePage(
                "https://example.com/research",
                "Research",
                "A useful browser tab about research.",
            ),
            FakePage("chrome://settings", "Settings", "Browser settings"),
            FakePage("about:blank", "", ""),
        ]
    )
    playwright = FakePlaywright(browser)

    harvester = CDPTabHarvester(
        cdp_url="http://localhost:9222",
        playwright_factory=FakePlaywrightFactory(playwright),
        max_concurrent=1,
    )

    result = await harvester.harvest()

    assert playwright.chromium.connected_urls == ["http://localhost:9222"]
    assert playwright.stopped is True
    assert browser.closed is False
    assert result.total == 1
    assert result.failed == 0
    assert result.tabs[0].url == "https://example.com/research"
    assert result.tabs[0].title == "Research"
    assert result.tabs[0].content == "A useful browser tab about research."


@pytest.mark.asyncio
async def test_harvester_opens_urls_in_attached_browser_without_new_profile():
    existing_page = FakePage("https://example.com/current", "Current", "Current")
    browser = FakeBrowser([existing_page])
    playwright = FakePlaywright(browser)

    harvester = CDPTabHarvester(
        cdp_url="http://localhost:9222",
        playwright_factory=FakePlaywrightFactory(playwright),
    )

    opened = await harvester.open_urls(["https://example.com/target"])

    assert opened == [{"url": "https://example.com/target", "status": "opened"}]
    created_page = browser.contexts[0].created_pages[0]
    assert created_page.opened_urls == ["https://example.com/target"]
    assert browser.closed is False
