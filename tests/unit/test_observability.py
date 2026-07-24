"""Structured logging and X-Request-ID propagation regressions."""

import io
import json
import logging
import sys
import types

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services import observability as obs


def _install_playwright_stub():
    """Allow importing browser-engine code without the browser binaries."""
    playwright_module = types.ModuleType("playwright")
    async_api = types.ModuleType("playwright.async_api")
    async_api.Browser = object
    async_api.Page = object
    async_api.TimeoutError = TimeoutError
    async_api.async_playwright = lambda: None
    playwright_module.async_api = async_api
    sys.modules.setdefault("playwright", playwright_module)
    sys.modules.setdefault("playwright.async_api", async_api)


try:
    import playwright.async_api  # noqa: F401
except ModuleNotFoundError:
    _install_playwright_stub()

from services.ai_engine.app import main as ai_main  # noqa: E402
from services.backend_core.app import main as backend_main  # noqa: E402
from services.backend_core.app.api import routes  # noqa: E402
from services.backend_core.app.sessions.manager import SessionManager  # noqa: E402
from services.browser_engine.app import main as browser_main  # noqa: E402
from services.browser_engine.app.scraper.engine import ScrapeResult  # noqa: E402


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _make_app():
    app = FastAPI()
    app.add_middleware(obs.RequestIDMiddleware, service="test")

    @app.get("/echo")
    def echo():
        return {"request_id": obs.get_request_id()}

    return app


def _attach_json_capture(service: str):
    """Capture emitted JSON lines from a service logger for assertions."""
    logger = logging.getLogger(f"taborganizer.{service}")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(obs._JsonFormatter())
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return stream, handler


def _read_events(stream: io.StringIO):
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Request-ID middleware
# --------------------------------------------------------------------------- #
def test_middleware_generates_request_id_and_binds_it_for_handler():
    client = TestClient(_make_app())
    resp = client.get("/echo")
    rid = resp.headers.get("X-Request-ID")
    assert rid
    assert resp.json()["request_id"] == rid


def test_middleware_accepts_and_echoes_inbound_request_id():
    client = TestClient(_make_app())
    resp = client.get("/echo", headers={"X-Request-ID": "trace-abc-123"})
    assert resp.headers["X-Request-ID"] == "trace-abc-123"
    assert resp.json()["request_id"] == "trace-abc-123"


def test_middleware_sanitizes_hostile_inbound_request_id():
    client = TestClient(_make_app())
    resp = client.get("/echo", headers={"X-Request-ID": "abc123def"})
    assert resp.headers["X-Request-ID"] == "abc123def"

    # Characters outside the allow-list are stripped (header/log injection safe).
    cleaned = obs._sanitize_request_id("bad id\r\ndrop; rm")
    assert cleaned == "badiddroprm"


@pytest.mark.parametrize(
    "app",
    [backend_main.app, ai_main.app, browser_main.app],
    ids=["backend-core", "ai-engine", "browser-engine"],
)
def test_each_service_echoes_request_id(app):
    client = TestClient(app)
    resp = client.get("/", headers={"X-Request-ID": "svc-trace-1"})
    assert resp.headers.get("X-Request-ID") == "svc-trace-1"


def test_each_service_installs_request_id_middleware():
    for app in (backend_main.app, ai_main.app, browser_main.app):
        classes = [m.cls for m in app.user_middleware]
        assert obs.RequestIDMiddleware in classes


# --------------------------------------------------------------------------- #
# Outbound propagation helper
# --------------------------------------------------------------------------- #
def test_request_id_headers_empty_when_unbound():
    token = obs.set_request_id(None)
    try:
        assert obs.request_id_headers() == {}
    finally:
        obs.reset_request_id(token)


def test_request_id_headers_present_when_bound():
    token = obs.set_request_id("bound-id")
    try:
        assert obs.request_id_headers() == {"X-Request-ID": "bound-id"}
    finally:
        obs.reset_request_id(token)


# --------------------------------------------------------------------------- #
# JSON formatter shape
# --------------------------------------------------------------------------- #
def test_json_formatter_emits_required_keys():
    obs.configure_logging("backend-core")
    stream, handler = _attach_json_capture("backend-core")
    logger = logging.getLogger("taborganizer.backend-core")
    try:
        token = obs.set_request_id("fmt-req")
        try:
            obs.log_event("unit.probe", answer=42)
        finally:
            obs.reset_request_id(token)
    finally:
        logger.removeHandler(handler)

    events = _read_events(stream)
    assert len(events) == 1
    record = events[0]
    assert set(record) >= {"ts", "level", "service", "request_id", "event"}
    assert record["service"] == "backend-core"
    assert record["event"] == "unit.probe"
    assert record["request_id"] == "fmt-req"
    assert record["answer"] == 42


# --------------------------------------------------------------------------- #
# Callback-failure structured record (WI0 B8: this was invisible before)
# --------------------------------------------------------------------------- #
def test_callback_failure_emits_structured_record(tmp_path, monkeypatch):
    obs.configure_logging("backend-core")
    manager = SessionManager(db_path=str(tmp_path / "backend.sqlite3"))
    session = manager.create_session("Logging Session")
    monkeypatch.setattr(routes, "session_manager", manager)

    stream, handler = _attach_json_capture("backend-core")
    logger = logging.getLogger("taborganizer.backend-core")
    try:
        response = routes.scrape_complete_callback(
            {
                "session_id": session.id,
                "url": "https://example.com/never-registered",
                "status": "success",
                "content": "body",
            }
        )
    finally:
        logger.removeHandler(handler)

    assert response == {"status": "error", "message": "URL not found in session"}

    events = _read_events(stream)
    failures = [e for e in events if e["event"] == "callback.scrape_complete_failed"]
    assert len(failures) == 1
    assert failures[0]["level"] == "warning"
    assert failures[0]["reason"] == "url_not_found_in_session"
    assert failures[0]["session_id"] == session.id


def test_callback_success_emits_content_length_not_body(tmp_path, monkeypatch):
    obs.configure_logging("backend-core")
    manager = SessionManager(db_path=str(tmp_path / "backend.sqlite3"))
    session = manager.create_session("Logging Session")
    manager.add_urls_to_session(session.id, ["https://example.com/page"])
    monkeypatch.setattr(routes, "session_manager", manager)

    stream, handler = _attach_json_capture("backend-core")
    logger = logging.getLogger("taborganizer.backend-core")
    try:
        routes.scrape_complete_callback(
            {
                "session_id": session.id,
                "url": "https://example.com/page",
                "status": "success",
                "content": "secret page body",
                "metadata": {"title": "Example"},
            }
        )
    finally:
        logger.removeHandler(handler)

    events = _read_events(stream)
    success = [e for e in events if e["event"] == "callback.scrape_complete"]
    assert len(success) == 1
    assert success[0]["content_length"] == len("secret page body")
    # The raw body must never appear in a log line.
    assert "secret page body" not in stream.getvalue()


# --------------------------------------------------------------------------- #
# Browser-engine background task propagates X-Request-ID downstream
# --------------------------------------------------------------------------- #
class _FakeScraper:
    def __init__(self, results):
        self.results = results
        self.closed = False

    async def scrape_batch(
        self, urls, session_id=None, callback=None, use_browser=False
    ):
        for result in self.results:
            if callback:
                await callback(result)
        return self.results

    async def close(self):
        self.closed = True


class _CapturingAsyncClient:
    def __init__(self, calls):
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return None

    async def post(self, url, **kwargs):
        self._calls.append((url, kwargs.get("headers", {})))
        return httpx.Response(
            200, request=httpx.Request("POST", url), json={"status": "ok"}
        )


@pytest.mark.asyncio
async def test_background_scrape_propagates_request_id_downstream(monkeypatch):
    session_id = "obs-propagate"
    result = ScrapeResult(
        url="https://example.com/",
        status="success",
        title="Example",
        content="content",
        status_code=200,
    )
    fake_scraper = _FakeScraper([result])
    monkeypatch.setattr(browser_main, "_new_scraper_engine", lambda: fake_scraper)

    calls = []
    monkeypatch.setattr(
        browser_main.httpx, "AsyncClient", lambda: _CapturingAsyncClient(calls)
    )
    monkeypatch.setenv("BACKEND_URL", "http://backend.test/")
    monkeypatch.setenv("AI_ENGINE_URL", "http://ai.test/")
    monkeypatch.setenv("BACKEND_CALLBACK_TOKEN", "callback-token")
    browser_main.scraping_tasks[session_id] = browser_main._new_scrape_task_info(1)

    try:
        await browser_main.scrape_urls_background(
            session_id, [result.url], False, "req-propagate-1"
        )
    finally:
        browser_main.scraping_tasks.pop(session_id, None)

    assert [url for url, _ in calls] == [
        "http://backend.test/api/v1/callback/scrape-complete",
        "http://ai.test/index",
    ]
    for _url, headers in calls:
        assert headers.get("X-Request-ID") == "req-propagate-1"
    # The context var is restored after the detached task completes.
    assert obs.get_request_id() is None
