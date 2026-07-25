"""Contracts for importing live Chromium tabs through CDP."""

import ipaddress
import socket
import sys
import types
from urllib.parse import urlparse

import pytest

from services.browser_engine.app.tabs import cdp as cdp_module
from services.browser_engine.app.tabs.cdp import (
    CDPConnectionError,
    CDPTabHarvester,
    resolve_cdp_connect_url,
    validate_cdp_url,
)


def _install_playwright_stub() -> None:
    """Let route tests import browser-engine code without browser binaries."""
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

from services.browser_engine.app import main as browser_main


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


def _stub_debug_version(monkeypatch, ws_url: str = None):
    """Answer `/json/version` the way Chrome does: echo the attached host:port.

    The harvester now reads that document itself and hands Playwright a
    validated ws:// endpoint, so every attach test needs a debug endpoint to
    answer. See `tests/unit/test_cdp_second_hop.py` for the validation rules.
    """

    async def _fake_fetch(connect_url: str) -> dict:
        parsed = urlparse(connect_url)
        advertised = ws_url or f"ws://{parsed.netloc}/devtools/browser/stub"
        return {"webSocketDebuggerUrl": advertised}

    monkeypatch.setattr(cdp_module, "_fetch_cdp_version", _fake_fetch)


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


def test_validate_cdp_url_rebrackets_ipv6_loopback():
    # urlparse strips brackets when exposing .hostname; rebuilding the netloc
    # from that unbracketed form must re-add them, or the result round-trips
    # into a broken host:port split ("::1:9222" instead of "[::1]:9222").
    validated = validate_cdp_url("http://[::1]:9222")
    assert validated == "http://[::1]:9222"

    reparsed = urlparse(validated)
    assert reparsed.hostname == "::1"
    assert reparsed.port == 9222


def test_resolve_cdp_connect_url_leaves_bracketed_ipv6_loopback_untouched(
    monkeypatch,
):
    def _fail(host, port, family=None, type=None):
        raise AssertionError("IPv6 loopback must not be DNS-resolved")

    monkeypatch.setattr(cdp_module.socket, "getaddrinfo", _fail)
    connect_url = resolve_cdp_connect_url(validate_cdp_url("http://[::1]:9222"))
    assert connect_url == "http://[::1]:9222"
    assert urlparse(connect_url).port == 9222


def _fake_getaddrinfo(*ips):
    """Build a socket.getaddrinfo stand-in returning the given IP literals."""

    def _resolve(host, port, family=None, type=None):
        infos = []
        for ip in ips:
            fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
            sockaddr = (ip, 0, 0, 0) if fam == socket.AF_INET6 else (ip, 0)
            infos.append((fam, socket.SOCK_STREAM, 6, "", sockaddr))
        return infos

    return _resolve


def test_resolve_cdp_connect_url_rewrites_host_docker_internal_to_ip(monkeypatch):
    # WI0 B2: the validator accepts host.docker.internal as input, but Chrome's
    # debug port rejects that Host header. Connect via the resolved IP instead.
    monkeypatch.setattr(
        cdp_module.socket, "getaddrinfo", _fake_getaddrinfo("172.17.0.1")
    )
    assert (
        resolve_cdp_connect_url("http://host.docker.internal:9222")
        == "http://172.17.0.1:9222"
    )


def test_resolve_cdp_connect_url_leaves_localhost_untouched(monkeypatch):
    # localhost/IP forms are valid Chrome Host headers and must not be resolved.
    def _fail(host, port, family=None, type=None):
        raise AssertionError("localhost must not be DNS-resolved")

    monkeypatch.setattr(cdp_module.socket, "getaddrinfo", _fail)
    assert resolve_cdp_connect_url("http://localhost:9222") == "http://localhost:9222"
    assert resolve_cdp_connect_url("http://127.0.0.1:9222") == "http://127.0.0.1:9222"


def test_resolve_cdp_connect_url_reports_unresolvable_host(monkeypatch):
    def _raise(host, port, family=None, type=None):
        raise OSError("Name or service not known")

    monkeypatch.setattr(cdp_module.socket, "getaddrinfo", _raise)
    with pytest.raises(CDPConnectionError) as excinfo:
        resolve_cdp_connect_url("http://host.docker.internal:9222")
    error = excinfo.value
    assert error.code == "cdp_host_unresolvable"
    assert "host.docker.internal" in error.cause
    assert error.fix


@pytest.mark.parametrize(
    "dangerous_ip",
    [
        # Every address below satisfies `is_private == True` in Python's
        # ipaddress module, so an `is_private or is_loopback` allow predicate
        # accepts all of them. They are also all `is_global == False`, so
        # `not is_global` accepts them too. Only an allowlist of genuinely
        # local networks refuses them.
        "198.18.0.1",  # benchmark range 198.18.0.0/15: routable off-host
        "192.0.0.170",  # 192.0.0.0/24 (NAT64/DS-Lite assignments): routable
        "2002:808:808::",  # 6to4 for the GLOBAL IPv4 8.8.8.8
        "2001:db8::1",  # IPv6 documentation range
        "2001:2::1",  # IPv6 benchmark range
        "169.254.169.254",  # IPv4 link-local: cloud metadata service
        "fe80::1",  # IPv6 link-local
        "0.0.0.0",  # unspecified
        "::",  # unspecified (v6)
        "240.0.0.1",  # reserved (Class E)
        "8.8.8.8",  # plain public address (regression guard)
        "224.0.0.1",  # multicast (regression guard; is_private is False)
    ],
)
def test_resolve_cdp_connect_url_rejects_addresses_outside_local_allowlist(
    monkeypatch, dangerous_ip
):
    monkeypatch.setattr(
        cdp_module.socket, "getaddrinfo", _fake_getaddrinfo(dangerous_ip)
    )
    with pytest.raises(CDPConnectionError) as excinfo:
        resolve_cdp_connect_url("http://host.docker.internal:9222")
    error = excinfo.value
    assert error.code == "cdp_resolved_address_not_local"
    assert dangerous_ip in error.cause
    assert error.fix


@pytest.mark.parametrize(
    "local_ip",
    [
        "172.17.0.1",  # default docker0 bridge -- MUST keep working
        "172.31.255.254",  # top of RFC1918 172.16.0.0/12
        "10.1.2.3",
        "192.168.1.10",
        "127.0.0.53",  # resolver-provided loopback
        "fd00::1",  # IPv6 unique-local
    ],
)
def test_resolve_cdp_connect_url_accepts_allowlisted_local_addresses(
    monkeypatch, local_ip
):
    monkeypatch.setattr(cdp_module.socket, "getaddrinfo", _fake_getaddrinfo(local_ip))
    expected_host = f"[{local_ip}]" if ":" in local_ip else local_ip
    assert (
        resolve_cdp_connect_url("http://host.docker.internal:9222")
        == f"http://{expected_host}:9222"
    )


def test_local_cdp_networks_exclude_link_local():
    # fe80::/10 sits outside fc00::/7, so IPv6 link-local is excluded by the
    # shape of the allowlist rather than by a carve-out. Pin that, because a
    # future "widen the IPv6 range" edit would silently re-admit it.
    for address in ("fe80::1", "fe80::dead:beef", "169.254.169.254"):
        parsed_address = ipaddress.ip_address(address)
        assert not any(
            parsed_address in network for network in cdp_module.LOCAL_CDP_NETWORKS
        )


def test_resolve_cdp_connect_url_filters_unsafe_answers_and_keeps_safe_one(monkeypatch):
    # A dual-stack host.docker.internal answering A=172.17.0.1 plus
    # AAAA=fe80::1 is a real, working configuration. Filter the unsafe answer
    # instead of refusing the host outright: the returned URL pins the safe IP
    # literal, so the rejected answer is never dialed.
    monkeypatch.setattr(
        cdp_module.socket,
        "getaddrinfo",
        _fake_getaddrinfo("fe80::1", "172.17.0.1", "8.8.8.8"),
    )
    assert (
        resolve_cdp_connect_url("http://host.docker.internal:9222")
        == "http://172.17.0.1:9222"
    )


def test_resolve_cdp_connect_url_prefers_ipv4_among_safe_answers(monkeypatch):
    # AF_UNSPEC + RFC 6724 ordering puts the IPv6 answer first. The documented
    # bridge ('socat ... bind=172.17.0.1') is IPv4-only, so selection must not
    # silently follow the resolver's preference.
    monkeypatch.setattr(
        cdp_module.socket,
        "getaddrinfo",
        _fake_getaddrinfo("fd00::1", "172.17.0.1"),
    )
    assert (
        resolve_cdp_connect_url("http://host.docker.internal:9222")
        == "http://172.17.0.1:9222"
    )


def test_resolve_cdp_connect_url_uses_ipv6_when_it_is_the_only_safe_answer(monkeypatch):
    monkeypatch.setattr(
        cdp_module.socket,
        "getaddrinfo",
        _fake_getaddrinfo("fd00::1", "8.8.8.8"),
    )
    connect_url = resolve_cdp_connect_url("http://host.docker.internal:9222")
    assert connect_url == "http://[fd00::1]:9222"
    assert urlparse(connect_url).port == 9222


@pytest.mark.asyncio
async def test_harvester_connects_via_resolved_ip(monkeypatch):
    monkeypatch.setattr(
        cdp_module.socket, "getaddrinfo", _fake_getaddrinfo("172.17.0.1")
    )
    _stub_debug_version(monkeypatch)
    browser = FakeBrowser([FakePage("https://example.com/a", "A", "content a")])
    playwright = FakePlaywright(browser)

    harvester = CDPTabHarvester(
        cdp_url="http://host.docker.internal:9222",
        playwright_factory=FakePlaywrightFactory(playwright),
        max_concurrent=1,
    )

    await harvester.harvest()

    # Validator keeps the friendly form; the wire connection uses the IP.
    assert harvester.cdp_url == "http://host.docker.internal:9222"
    assert playwright.chromium.connected_urls == [
        "ws://172.17.0.1:9222/devtools/browser/stub"
    ]


@pytest.mark.asyncio
async def test_harvester_attaches_over_bracketed_ipv6_loopback(monkeypatch):
    # End-to-end: an IPv6 loopback endpoint must survive validation, connect
    # resolution, and the actual attach with its brackets intact. An
    # unbracketed "::1:9222" would reparse into a different host.
    real_getaddrinfo = socket.getaddrinfo

    def _guarded(host, port, *args, **kwargs):
        # Page URLs still go through scrape URL safety, which resolves them;
        # only the CDP host must never reach the resolver.
        assert host != "::1", "IPv6 loopback must not be DNS-resolved"
        return real_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(cdp_module.socket, "getaddrinfo", _guarded)
    _stub_debug_version(monkeypatch)
    browser = FakeBrowser([FakePage("https://example.com/a", "A", "content a")])
    playwright = FakePlaywright(browser)

    harvester = CDPTabHarvester(
        cdp_url="http://[::1]:9222",
        playwright_factory=FakePlaywrightFactory(playwright),
        max_concurrent=1,
    )

    result = await harvester.harvest()

    assert harvester.cdp_url == "http://[::1]:9222"
    assert playwright.chromium.connected_urls == [
        "ws://[::1]:9222/devtools/browser/stub"
    ]
    assert urlparse(playwright.chromium.connected_urls[0]).hostname == "::1"
    assert urlparse(playwright.chromium.connected_urls[0]).port == 9222
    assert result.total == 1
    assert browser.closed is False


@pytest.mark.asyncio
async def test_harvester_connect_failure_raises_actionable_error(monkeypatch):
    monkeypatch.setattr(
        cdp_module.socket, "getaddrinfo", _fake_getaddrinfo("172.17.0.1")
    )
    _stub_debug_version(monkeypatch)

    class FailingChromium:
        def __init__(self):
            self.connected_urls: list[str] = []

        async def connect_over_cdp(self, url):
            self.connected_urls.append(url)
            raise ConnectionRefusedError("connection refused")

    class FailingPlaywright:
        def __init__(self):
            self.chromium = FailingChromium()
            self.stopped = False

        async def stop(self):
            self.stopped = True

    playwright = FailingPlaywright()
    harvester = CDPTabHarvester(
        cdp_url="http://host.docker.internal:9222",
        playwright_factory=FakePlaywrightFactory(playwright),
        max_concurrent=1,
    )

    with pytest.raises(CDPConnectionError) as excinfo:
        await harvester.harvest()

    error = excinfo.value
    assert error.code == "cdp_connect_failed"
    assert "172.17.0.1:9222" in error.cause
    assert "socat" in error.fix
    assert playwright.stopped is True


@pytest.mark.asyncio
async def test_harvester_imports_visible_tabs_without_closing_user_browser(monkeypatch):
    _stub_debug_version(monkeypatch)
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

    assert playwright.chromium.connected_urls == [
        "ws://localhost:9222/devtools/browser/stub"
    ]
    assert playwright.stopped is True
    assert browser.closed is False
    assert result.total == 1
    assert result.failed == 0
    assert result.tabs[0].url == "https://example.com/research"
    assert result.tabs[0].title == "Research"
    assert result.tabs[0].content == "A useful browser tab about research."


@pytest.mark.asyncio
async def test_harvester_opens_urls_in_attached_browser_without_new_profile(
    monkeypatch,
):
    _stub_debug_version(monkeypatch)
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


@pytest.mark.asyncio
async def test_browser_import_tabs_endpoint_requires_auth_and_returns_documents(
    monkeypatch,
):
    class FakeHarvester:
        def __init__(self, cdp_url, max_concurrent):
            self.cdp_url = cdp_url
            self.max_concurrent = max_concurrent

        async def harvest(self, max_tabs=None):
            assert max_tabs == 2000
            return type(
                "Result",
                (),
                {
                    "to_dict": lambda self: {
                        "total": 1,
                        "imported": 1,
                        "failed": 0,
                        "tabs": [
                            {
                                "id": "https://example.com/page",
                                "url": "https://example.com/page",
                                "title": "Example",
                                "content": "Example content",
                                "metadata": {"source": "cdp"},
                            }
                        ],
                        "errors": [],
                    }
                },
            )()

    monkeypatch.setenv("BROWSER_ENGINE_API_TOKEN", "browser-token")
    monkeypatch.setattr(browser_main, "CDPTabHarvester", FakeHarvester)

    with pytest.raises(browser_main.HTTPException) as missing:
        await browser_main.import_tabs_from_browser(
            browser_main.TabImportRequest(cdp_url="http://localhost:9222"),
            _auth=browser_main._require_browser_engine_auth(None),
        )
    assert missing.value.status_code == 401

    result = await browser_main.import_tabs_from_browser(
        browser_main.TabImportRequest(cdp_url="http://localhost:9222"),
        _auth=browser_main._require_browser_engine_auth("Bearer browser-token"),
    )

    assert result["status"] == "completed"
    assert result["imported"] == 1
    assert result["tabs"][0]["url"] == "https://example.com/page"
