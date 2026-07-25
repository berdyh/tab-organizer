"""The CDP attach's second hop must stay on the address that was pinned.

`resolve_cdp_connect_url` pins the debug endpoint to a validated local IP, but
handing that http(s) URL to `connect_over_cdp` only pins the *first* hop:
playwright-core's `urlToWSEndpoint` fetches `/json/version` and then dials
whatever `webSocketDebuggerUrl` the body contains -- following cross-host
redirects and honouring proxy env vars on the way. These tests cover the fetch
being done here instead, against the same allowlist, with a real stand-in debug
server on loopback rather than a mocked HTTP client.
"""

import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from services.browser_engine.app.tabs import cdp as cdp_module
from services.browser_engine.app.tabs.cdp import (
    CDPConnectionError,
    CDPTabHarvester,
    validate_ws_debugger_url,
)

PINNED = "http://172.17.0.1:9222"


class FakeChromium:
    def __init__(self):
        self.connected_urls: list[str] = []

    async def connect_over_cdp(self, url):
        self.connected_urls.append(url)
        return FakeBrowser()


class FakeBrowser:
    def __init__(self):
        self.contexts: list = []
        self.closed = False


class FakePlaywright:
    def __init__(self):
        self.chromium = FakeChromium()
        self.stopped = False

    async def stop(self):
        self.stopped = True


class FakePlaywrightFactory:
    def __init__(self, playwright):
        self.playwright = playwright

    async def start(self):
        return self.playwright


def _send(handler, body: bytes, status: int = 200, headers=()):
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    for name, value in headers:
        handler.send_header(name, value)
    handler.end_headers()
    handler.wfile.write(body)


@contextlib.contextmanager
def debug_endpoint(respond):
    """Serve a stand-in Chrome debug endpoint on loopback.

    `respond(handler, netloc)` writes the `/json/version` answer; `netloc` is
    the address the harvester actually attached to, which is what a real Chrome
    echoes into `webSocketDebuggerUrl` from the request's Host header.
    """
    state: dict = {"paths": []}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
            state["paths"].append(self.path)
            respond(self, state["netloc"])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state["netloc"] = f"127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{state['netloc']}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _harvester(cdp_url: str) -> tuple[CDPTabHarvester, FakePlaywright]:
    playwright = FakePlaywright()
    harvester = CDPTabHarvester(
        cdp_url=cdp_url,
        playwright_factory=FakePlaywrightFactory(playwright),
        max_concurrent=1,
    )
    return harvester, playwright


@pytest.mark.asyncio
async def test_endpoint_advertising_its_own_address_is_attached():
    def respond(handler, netloc):
        body = json.dumps(
            {
                "Browser": "Chrome/120.0",
                "webSocketDebuggerUrl": f"ws://{netloc}/devtools/browser/abc-123",
            }
        ).encode()
        _send(handler, body)

    with debug_endpoint(respond) as (url, state):
        harvester, playwright = _harvester(url)
        result = await harvester.harvest()

    assert result.total == 0
    assert state["paths"] == ["/json/version"]
    assert playwright.chromium.connected_urls == [
        f"ws://{state['netloc']}/devtools/browser/abc-123"
    ]
    assert playwright.stopped is True


@pytest.mark.asyncio
async def test_endpoint_advertising_offhost_socket_is_refused():
    def respond(handler, netloc):
        body = json.dumps(
            {"webSocketDebuggerUrl": "ws://evil.example:9222/devtools/browser/abc"}
        ).encode()
        _send(handler, body)

    with debug_endpoint(respond) as (url, _state):
        harvester, playwright = _harvester(url)
        with pytest.raises(CDPConnectionError) as excinfo:
            await harvester.harvest()

    error = excinfo.value
    assert error.code == "cdp_ws_endpoint_not_local"
    assert "evil.example" in error.cause
    assert playwright.chromium.connected_urls == []
    assert playwright.stopped is True


@pytest.mark.asyncio
async def test_open_urls_refuses_offhost_socket_too():
    # `open_urls` shares `_connect`, so the same gate must cover the write path.
    def respond(handler, netloc):
        body = json.dumps(
            {"webSocketDebuggerUrl": "ws://198.51.100.7:9222/devtools/browser/abc"}
        ).encode()
        _send(handler, body)

    with debug_endpoint(respond) as (url, _state):
        harvester, playwright = _harvester(url)
        with pytest.raises(CDPConnectionError) as excinfo:
            await harvester.open_urls(["https://example.com/target"])

    assert excinfo.value.code == "cdp_ws_endpoint_not_local"
    assert playwright.chromium.connected_urls == []


@pytest.mark.asyncio
async def test_version_fetch_does_not_follow_a_redirect_off_the_pinned_address():
    def respond(handler, netloc):
        _send(
            handler,
            b"{}",
            status=302,
            headers=(("Location", "http://evil.example/json/version"),),
        )

    with debug_endpoint(respond) as (url, state):
        harvester, playwright = _harvester(url)
        with pytest.raises(CDPConnectionError) as excinfo:
            await harvester.harvest()

    assert excinfo.value.code == "cdp_ws_endpoint_not_local"
    assert "did not answer /json/version with 200" in excinfo.value.cause
    assert state["paths"] == ["/json/version"]
    assert playwright.chromium.connected_urls == []


@pytest.mark.asyncio
async def test_version_fetch_ignores_proxy_environment(monkeypatch):
    # A proxy env var must not reroute a local-only call; port 1 is where the
    # request would land if the client trusted the environment.
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:1")

    def respond(handler, netloc):
        body = json.dumps(
            {"webSocketDebuggerUrl": f"ws://{netloc}/devtools/browser/abc"}
        ).encode()
        _send(handler, body)

    with debug_endpoint(respond) as (url, state):
        harvester, playwright = _harvester(url)
        await harvester.harvest()

    assert playwright.chromium.connected_urls == [
        f"ws://{state['netloc']}/devtools/browser/abc"
    ]


@pytest.mark.asyncio
async def test_oversized_version_body_is_refused():
    def respond(handler, netloc):
        body = json.dumps(
            {
                "webSocketDebuggerUrl": f"ws://{netloc}/devtools/browser/abc",
                # Fixed size, not derived from the constant: raising the cap
                # must fail this test rather than move it.
                "padding": "x" * (128 * 1024),
            }
        ).encode()
        _send(handler, body)

    with debug_endpoint(respond) as (url, _state):
        harvester, playwright = _harvester(url)
        with pytest.raises(CDPConnectionError) as excinfo:
            await harvester.harvest()

    assert excinfo.value.code == "cdp_ws_endpoint_not_local"
    assert "oversized" in excinfo.value.cause
    assert playwright.chromium.connected_urls == []


@pytest.mark.asyncio
async def test_non_json_version_body_is_refused():
    def respond(handler, netloc):
        _send(handler, b"<html>not a devtools server</html>")

    with debug_endpoint(respond) as (url, _state):
        harvester, playwright = _harvester(url)
        with pytest.raises(CDPConnectionError) as excinfo:
            await harvester.harvest()

    assert excinfo.value.code == "cdp_ws_endpoint_not_local"
    assert playwright.chromium.connected_urls == []


@pytest.mark.asyncio
async def test_unreachable_debug_port_keeps_the_actionable_connect_error():
    with debug_endpoint(lambda handler, netloc: None) as (url, _state):
        pass  # server is shut down before the harvester runs

    harvester, playwright = _harvester(url)
    with pytest.raises(CDPConnectionError) as excinfo:
        await harvester.harvest()

    error = excinfo.value
    assert error.code == "cdp_connect_failed"
    assert "socat" in error.fix
    assert playwright.stopped is True


@pytest.mark.parametrize(
    "advertised",
    [
        "ws://8.8.8.8:9222/devtools/browser/abc",
        "ws://[::ffff:8.8.8.8]:9222/devtools/browser/abc",
        "ws://[2002:808:808::]:9222/devtools/browser/abc",
        "ws://[64:ff9b::808:808]:9222/devtools/browser/abc",
        "ws://169.254.169.254:9222/devtools/browser/abc",
        "ws://evil.example:9222/devtools/browser/abc",
        "ws://127.0.0.1:9222/devtools/browser/abc",
        "ws://172.17.0.2:9222/devtools/browser/abc",
        "ws://172.17.0.1:9223/devtools/browser/abc",
        "ws://172.17.0.1/devtools/browser/abc",
        "ws://user:pass@172.17.0.1:9222/devtools/browser/abc",
        "ws://172.17.0.1:9222@evil.example/devtools/browser/abc",
        "http://172.17.0.1:9222/devtools/browser/abc",
        "https://172.17.0.1:9222/devtools/browser/abc",
        "file:///etc/passwd",
        "ws://172.17.0.1:9222/devtools/browser/abc\r\nHost: evil.example",
        "ws:///devtools/browser/abc",
        "",
        "   ",
        None,
        1234,
        {"webSocketDebuggerUrl": "ws://172.17.0.1:9222/x"},
    ],
)
def test_advertised_endpoint_must_be_the_pinned_socket(advertised):
    with pytest.raises(CDPConnectionError) as excinfo:
        validate_ws_debugger_url(advertised, PINNED)
    assert excinfo.value.code == "cdp_ws_endpoint_not_local"


@pytest.mark.parametrize(
    "advertised,expected",
    [
        (
            "ws://172.17.0.1:9222/devtools/browser/abc",
            "ws://172.17.0.1:9222/devtools/browser/abc",
        ),
        (
            "wss://172.17.0.1:9222/devtools/browser/abc",
            "wss://172.17.0.1:9222/devtools/browser/abc",
        ),
        (
            "  ws://172.17.0.1:9222/devtools/browser/abc  ",
            "ws://172.17.0.1:9222/devtools/browser/abc",
        ),
    ],
)
def test_advertised_endpoint_on_the_pinned_socket_is_accepted(advertised, expected):
    assert validate_ws_debugger_url(advertised, PINNED) == expected


def test_ipv6_pinned_socket_round_trips_with_brackets():
    assert (
        validate_ws_debugger_url(
            "ws://[::1]:9222/devtools/browser/abc", "http://[::1]:9222"
        )
        == "ws://[::1]:9222/devtools/browser/abc"
    )
    # Alternate spelling of the same address is still the same address.
    assert (
        validate_ws_debugger_url(
            "ws://[0:0:0:0:0:0:0:1]:9222/devtools/browser/abc", "http://[::1]:9222"
        )
        == "ws://[::1]:9222/devtools/browser/abc"
    )


def test_localhost_pinned_socket_requires_the_same_spelling():
    assert (
        validate_ws_debugger_url(
            "ws://localhost:9222/devtools/browser/abc", "http://localhost:9222"
        )
        == "ws://localhost:9222/devtools/browser/abc"
    )
    # Strict by design: name-to-address equivalence is the resolver-controlled
    # step the pinning exists to remove.
    with pytest.raises(CDPConnectionError):
        validate_ws_debugger_url(
            "ws://127.0.0.1:9222/devtools/browser/abc", "http://localhost:9222"
        )


def test_allowlist_is_checked_independently_of_the_host_match():
    # Belt and braces: even if a caller ever pinned a non-local address, the
    # advertised socket is still measured against LOCAL_CDP_NETWORKS.
    with pytest.raises(CDPConnectionError) as excinfo:
        validate_ws_debugger_url(
            "ws://8.8.8.8:9222/devtools/browser/abc", "http://8.8.8.8:9222"
        )
    assert excinfo.value.code == "cdp_ws_endpoint_not_local"
