"""Browser-engine background callback/index failure reporting tests."""

import asyncio
import sys
import types

import httpx
import pytest

from services import url_safety


def _install_framework_stubs() -> None:
    """Provide minimal FastAPI/Pydantic stubs for direct function tests."""
    fastapi_module = types.ModuleType("fastapi")
    middleware_module = types.ModuleType("fastapi.middleware")
    cors_module = types.ModuleType("fastapi.middleware.cors")
    pydantic_module = types.ModuleType("pydantic")

    class BackgroundTasks:
        def add_task(self, *args, **kwargs):
            return None

    class FastAPI:
        def __init__(self, *args, **kwargs):
            return None

        def add_middleware(self, *args, **kwargs):
            return None

        def get(self, *args, **kwargs):
            return lambda func: func

        def post(self, *args, **kwargs):
            return lambda func: func

        def delete(self, *args, **kwargs):
            return lambda func: func

        def on_event(self, *args, **kwargs):
            return lambda func: func

    class HTTPException(Exception):
        def __init__(self, status_code, detail=None):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class CORSMiddleware:
        pass

    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Header(default=None):
        return default

    def Depends(dependency):
        return dependency

    fastapi_module.BackgroundTasks = BackgroundTasks
    fastapi_module.Depends = Depends
    fastapi_module.FastAPI = FastAPI
    fastapi_module.Header = Header
    fastapi_module.HTTPException = HTTPException
    cors_module.CORSMiddleware = CORSMiddleware
    pydantic_module.BaseModel = BaseModel

    sys.modules["fastapi"] = fastapi_module
    sys.modules["fastapi.middleware"] = middleware_module
    sys.modules["fastapi.middleware.cors"] = cors_module
    sys.modules["pydantic"] = pydantic_module


def _install_playwright_stub() -> None:
    """Let unit tests import browser-engine code without browser binaries."""
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
    import fastapi  # noqa: F401
    import pydantic  # noqa: F401
except ModuleNotFoundError:
    _install_framework_stubs()

try:
    import playwright.async_api  # noqa: F401
except ModuleNotFoundError:
    _install_playwright_stub()

from services.browser_engine.app import main as browser_main
from services.browser_engine.app.scraper.engine import ScraperEngine, ScrapeResult


def test_service_token_headers_use_ai_engine_token(monkeypatch):
    monkeypatch.setenv("AI_ENGINE_API_TOKEN", "shared-token")

    assert browser_main._service_token_headers("AI_ENGINE_API_TOKEN") == {
        "Authorization": "Bearer shared-token"
    }


def test_service_token_headers_support_callback_token_fallback(monkeypatch):
    monkeypatch.setenv("AI_ENGINE_API_TOKEN", "ai-token")
    monkeypatch.setenv("BACKEND_CALLBACK_TOKEN", "callback-token")

    assert browser_main._service_token_headers(
        "BACKEND_CALLBACK_TOKEN", "AI_ENGINE_API_TOKEN"
    ) == {"Authorization": "Bearer callback-token"}

    monkeypatch.delenv("BACKEND_CALLBACK_TOKEN")
    assert browser_main._service_token_headers(
        "BACKEND_CALLBACK_TOKEN", "AI_ENGINE_API_TOKEN"
    ) == {"Authorization": "Bearer ai-token"}


def test_browser_engine_auth_requires_shared_token(monkeypatch):
    monkeypatch.delenv("BROWSER_ENGINE_API_TOKEN", raising=False)
    monkeypatch.delenv("AI_ENGINE_API_TOKEN", raising=False)
    monkeypatch.delenv("BACKEND_CALLBACK_TOKEN", raising=False)

    with pytest.raises(browser_main.HTTPException) as unconfigured:
        browser_main._require_browser_engine_auth(None)
    assert unconfigured.value.status_code == 401

    monkeypatch.setenv("AI_ENGINE_API_TOKEN", "shared-token")
    with pytest.raises(browser_main.HTTPException) as missing:
        browser_main._require_browser_engine_auth(None)
    assert missing.value.status_code == 401

    with pytest.raises(browser_main.HTTPException) as wrong:
        browser_main._require_browser_engine_auth("Bearer wrong")
    assert wrong.value.status_code == 401

    assert browser_main._require_browser_engine_auth("Bearer shared-token") is None


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost:8080/health",
        "http://127.0.0.1:8080/health",
        "http://10.0.0.2/private",
        "http://backend-core:8080/health",
        "http://host.docker.internal:8080/health",
    ],
)
def test_browser_engine_rejects_unsafe_scrape_urls(url):
    with pytest.raises(browser_main.HTTPException) as exc_info:
        browser_main._validate_scrape_urls([url])

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_safe_httpx_get_connects_to_vetted_ip_with_host_and_sni(monkeypatch):
    from services.browser_engine.app.scraper import engine

    monkeypatch.setattr(
        url_safety.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (
                url_safety.socket.AF_INET,
                url_safety.socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", 443),
            )
        ],
    )
    calls = []

    class CapturingClient:
        async def request(self, method, url, follow_redirects=False, **_kwargs):
            calls.append(
                {
                    "method": method,
                    "url": url,
                    "follow_redirects": follow_redirects,
                    "headers": _kwargs.get("headers"),
                    "extensions": _kwargs.get("extensions"),
                }
            )
            request = httpx.Request("GET", url)
            return httpx.Response(200, request=request)

    await engine._safe_httpx_get(
        CapturingClient(),
        "https://public-looking.test/page?q=1",
        headers={"User-Agent": "TabOrganizer"},
    )

    assert calls == [
        {
            "method": "GET",
            "url": "https://93.184.216.34/page?q=1",
            "follow_redirects": False,
            "headers": {
                "User-Agent": "TabOrganizer",
                "Host": "public-looking.test",
            },
            "extensions": {"sni_hostname": "public-looking.test"},
        }
    ]


@pytest.mark.asyncio
async def test_safe_httpx_get_rejects_unsafe_redirect_target(monkeypatch):
    from services.browser_engine.app.scraper import engine

    monkeypatch.setattr(
        url_safety.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (
                url_safety.socket.AF_INET,
                url_safety.socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", 443),
            )
        ],
    )
    calls = []

    class RedirectingClient:
        async def request(self, method, url, follow_redirects=False, **_kwargs):
            calls.append(
                {"method": method, "url": url, "follow_redirects": follow_redirects}
            )
            request = httpx.Request("GET", url)
            return httpx.Response(
                302,
                headers={"location": "http://127.0.0.1/private"},
                request=request,
            )

    with pytest.raises(ValueError, match="private network"):
        await engine._safe_httpx_get(
            RedirectingClient(),
            "https://public-looking.test/path",
        )

    assert calls == [
        {
            "method": "GET",
            "url": "https://93.184.216.34/path",
            "follow_redirects": False,
        }
    ]


@pytest.mark.asyncio
async def test_safe_browser_route_fetches_public_hostname_through_safe_httpx(
    monkeypatch,
):
    from services.browser_engine.app.scraper import engine

    monkeypatch.setattr(
        url_safety.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (
                url_safety.socket.AF_INET,
                url_safety.socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", 443),
            )
        ],
    )
    monkeypatch.delenv("SCRAPE_ALLOW_PRIVATE_NETWORKS", raising=False)
    calls = []

    class CapturingAsyncClient:
        def __init__(self, **_kwargs):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def request(self, method, url, follow_redirects=False, **_kwargs):
            calls.append(
                {
                    "method": method,
                    "url": url,
                    "follow_redirects": follow_redirects,
                    "headers": _kwargs.get("headers"),
                    "extensions": _kwargs.get("extensions"),
                }
            )
            request = httpx.Request(method, url)
            return httpx.Response(
                200,
                headers={
                    "content-type": "text/html",
                    "content-length": "999",
                },
                content=b"<html><title>Safe</title></html>",
                request=request,
            )

    class CapturingRoute:
        def __init__(self):
            self.fulfilled = None
            self.aborted = False

        async def fulfill(self, **kwargs):
            self.fulfilled = kwargs

        async def abort(self):
            self.aborted = True

    class BrowserRequest:
        method = "GET"
        url = "https://public-looking.test/page"
        headers = {"user-agent": "TabOrganizer", "host": "public-looking.test"}
        post_data_buffer = None

    monkeypatch.setattr(engine.httpx, "AsyncClient", CapturingAsyncClient)
    route = CapturingRoute()

    await engine._safe_browser_route_handler(timeout=5)(route, BrowserRequest())

    assert route.aborted is False
    assert route.fulfilled == {
        "status": 200,
        "headers": {"content-type": "text/html"},
        "body": b"<html><title>Safe</title></html>",
    }
    assert calls == [
        {
            "method": "GET",
            "url": "https://93.184.216.34/page",
            "follow_redirects": False,
            "headers": {
                "user-agent": "TabOrganizer",
                "Host": "public-looking.test",
            },
            "extensions": {"sni_hostname": "public-looking.test"},
        }
    ]


class FakeScraper:
    def __init__(self, results):
        self.results = results
        self.closed = False
        self.use_browser = None

    async def scrape_batch(
        self,
        urls,
        session_id=None,
        callback=None,
        use_browser=False,
    ):
        self.use_browser = use_browser
        assert urls == [result.url for result in self.results]
        for result in self.results:
            if callback:
                await callback(result)
        return self.results

    async def close(self):
        self.closed = True


def _use_background_scraper(monkeypatch, fake_scraper):
    monkeypatch.setattr(browser_main, "_new_scraper_engine", lambda: fake_scraper)


class FakeAsyncClient:
    def __init__(self, handler):
        self._handler = handler

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def post(self, url, **kwargs):
        return self._handler(url, kwargs)


def _response(url, status_code=200, json=None, text=None):
    response_kwargs = {}
    if json is not None:
        response_kwargs["json"] = json
    if text is not None:
        response_kwargs["text"] = text
    return httpx.Response(
        status_code,
        request=httpx.Request("POST", url),
        **response_kwargs,
    )


@pytest.mark.asyncio
async def test_backend_callback_http_failure_is_visible_without_stopping_batch(
    monkeypatch,
):
    session_id = "callback-http-failure"
    result = ScrapeResult(
        url="https://example.com/",
        status="success",
        title="Example Domain",
        content="Example Domain content",
        status_code=200,
    )
    fake_scraper = FakeScraper([result])
    calls = []

    def handler(url, kwargs):
        calls.append((url, kwargs))
        if url.endswith("/callback/scrape-complete"):
            return _response(url, status_code=503, text="backend unavailable")
        return _response(url, json={"status": "indexed"})

    _use_background_scraper(monkeypatch, fake_scraper)
    monkeypatch.setattr(
        browser_main.httpx,
        "AsyncClient",
        lambda: FakeAsyncClient(handler),
    )
    monkeypatch.setenv("BACKEND_URL", "http://backend.test/")
    monkeypatch.setenv("AI_ENGINE_URL", "http://ai.test/")
    monkeypatch.setenv("BACKEND_CALLBACK_TOKEN", "callback-token")
    browser_main.scraping_tasks[session_id] = browser_main._new_scrape_task_info(1)

    try:
        await browser_main.scrape_urls_background(session_id, [result.url], False)

        status = browser_main.scraping_tasks[session_id]
        assert status["completed"] == 1
        assert status["success"] == 1
        assert status["status"] == "completed_with_downstream_errors"
        assert status["backend_callback_failed"] == 1
        assert status["ai_index_failed"] == 0
        assert status["downstream_errors"][0]["source"] == "backend_callback"
        assert status["downstream_errors"][0]["url"] == result.url
        assert "backend unavailable" in status["downstream_errors"][0]["message"]
        assert [url for url, _kwargs in calls] == [
            "http://backend.test/api/v1/callback/scrape-complete",
            "http://ai.test/index",
        ]
        assert calls[0][1]["headers"] == {"Authorization": "Bearer callback-token"}
        assert fake_scraper.closed is True
    finally:
        browser_main.scraping_tasks.pop(session_id, None)


@pytest.mark.asyncio
async def test_backend_callback_http_detail_payload_is_visible(monkeypatch):
    session_id = "callback-http-detail"
    result = ScrapeResult(
        url="https://example.com/",
        status="success",
        title="Example Domain",
        content="Example Domain content",
        status_code=200,
    )
    fake_scraper = FakeScraper([result])

    def handler(url, kwargs):
        if url.endswith("/callback/scrape-complete"):
            return _response(
                url,
                status_code=401,
                json={"detail": "Invalid backend callback token"},
            )
        return _response(url, json={"status": "indexed"})

    _use_background_scraper(monkeypatch, fake_scraper)
    monkeypatch.setattr(
        browser_main.httpx,
        "AsyncClient",
        lambda: FakeAsyncClient(handler),
    )
    browser_main.scraping_tasks[session_id] = browser_main._new_scrape_task_info(1)

    try:
        await browser_main.scrape_urls_background(session_id, [result.url], False)

        status = browser_main.scraping_tasks[session_id]
        assert status["status"] == "completed_with_downstream_errors"
        assert status["backend_callback_failed"] == 1
        assert "Invalid backend callback token" in (
            status["downstream_errors"][0]["message"]
        )
    finally:
        browser_main.scraping_tasks.pop(session_id, None)


@pytest.mark.asyncio
async def test_backend_callback_error_payload_is_visible(monkeypatch):
    session_id = "callback-error-payload"
    result = ScrapeResult(
        url="https://example.org/",
        status="failed",
        error="blocked by remote site",
    )
    fake_scraper = FakeScraper([result])

    def handler(url, kwargs):
        return _response(url, json={"status": "error", "message": "Session not found"})

    _use_background_scraper(monkeypatch, fake_scraper)
    monkeypatch.setattr(
        browser_main.httpx,
        "AsyncClient",
        lambda: FakeAsyncClient(handler),
    )
    browser_main.scraping_tasks[session_id] = browser_main._new_scrape_task_info(1)

    try:
        await browser_main.scrape_urls_background(session_id, [result.url], False)

        status = browser_main.scraping_tasks[session_id]
        assert status["status"] == "completed_with_downstream_errors"
        assert status["backend_callback_failed"] == 1
        assert "Session not found" in status["downstream_errors"][0]["message"]
        assert fake_scraper.closed is True
    finally:
        browser_main.scraping_tasks.pop(session_id, None)


@pytest.mark.asyncio
async def test_ai_index_failure_is_visible_after_successful_callbacks(monkeypatch):
    session_id = "index-failure"
    result = ScrapeResult(
        url="https://www.iana.org/domains/reserved",
        status="success",
        title="IANA-managed Reserved Domains",
        content="Reserved domain content",
        status_code=200,
    )
    fake_scraper = FakeScraper([result])

    def handler(url, kwargs):
        if url.endswith("/callback/scrape-complete"):
            return _response(url, json={"status": "updated"})
        return _response(url, status_code=500, text="index failed")

    _use_background_scraper(monkeypatch, fake_scraper)
    monkeypatch.setattr(
        browser_main.httpx,
        "AsyncClient",
        lambda: FakeAsyncClient(handler),
    )
    monkeypatch.setenv("AI_ENGINE_URL", "http://ai.test")
    browser_main.scraping_tasks[session_id] = browser_main._new_scrape_task_info(1)

    try:
        await browser_main.scrape_urls_background(session_id, [result.url], False)

        status = browser_main.scraping_tasks[session_id]
        assert status["completed"] == 1
        assert status["success"] == 1
        assert status["status"] == "completed_with_downstream_errors"
        assert status["backend_callback_failed"] == 0
        assert status["ai_index_failed"] == 1
        assert status["downstream_errors"][0]["source"] == "ai_index"
        assert "index failed" in status["downstream_errors"][0]["message"]
        assert fake_scraper.closed is True
    finally:
        browser_main.scraping_tasks.pop(session_id, None)


@pytest.mark.asyncio
async def test_ai_index_http_detail_payload_is_visible(monkeypatch):
    session_id = "index-detail-failure"
    result = ScrapeResult(
        url="https://www.iana.org/domains/reserved",
        status="success",
        title="IANA-managed Reserved Domains",
        content="Reserved domain content",
        status_code=200,
    )
    fake_scraper = FakeScraper([result])

    def handler(url, kwargs):
        if url.endswith("/callback/scrape-complete"):
            return _response(url, json={"status": "updated"})
        return _response(
            url,
            status_code=500,
            json={"detail": "OPENROUTER_API_KEY is not configured"},
        )

    _use_background_scraper(monkeypatch, fake_scraper)
    monkeypatch.setattr(
        browser_main.httpx,
        "AsyncClient",
        lambda: FakeAsyncClient(handler),
    )
    monkeypatch.setenv("AI_ENGINE_URL", "http://ai.test")
    browser_main.scraping_tasks[session_id] = browser_main._new_scrape_task_info(1)

    try:
        await browser_main.scrape_urls_background(session_id, [result.url], False)

        status = browser_main.scraping_tasks[session_id]
        assert status["status"] == "completed_with_downstream_errors"
        assert status["ai_index_failed"] == 1
        assert "OPENROUTER_API_KEY is not configured" in (
            status["downstream_errors"][0]["message"]
        )
    finally:
        browser_main.scraping_tasks.pop(session_id, None)


@pytest.mark.asyncio
async def test_ai_index_batch_failure_counts_every_document(monkeypatch):
    # WI0 B5: one failed batched /index call leaves every document in that call
    # unindexed. ai_index_failed must reflect per-document blast radius and the
    # error entry must carry batch scope + docs_in_failed_call, not read as a
    # single 1-in-N blip.
    session_id = "index-batch-failure"
    results = [
        ScrapeResult(
            url=f"https://example.com/doc-{index}",
            status="success",
            title=f"Doc {index}",
            content=f"content {index}",
            status_code=200,
        )
        for index in range(3)
    ]
    fake_scraper = FakeScraper(results)

    def handler(url, kwargs):
        if url.endswith("/callback/scrape-complete"):
            return _response(url, json={"status": "updated"})
        return _response(url, status_code=500, text="index failed")

    _use_background_scraper(monkeypatch, fake_scraper)
    monkeypatch.setattr(
        browser_main.httpx,
        "AsyncClient",
        lambda: FakeAsyncClient(handler),
    )
    monkeypatch.setenv("AI_ENGINE_URL", "http://ai.test")
    browser_main.scraping_tasks[session_id] = browser_main._new_scrape_task_info(3)

    try:
        await browser_main.scrape_urls_background(
            session_id, [r.url for r in results], False
        )

        status = browser_main.scraping_tasks[session_id]
        assert status["success"] == 3
        assert status["status"] == "completed_with_downstream_errors"
        # One failed call, but three documents unindexed.
        assert status["downstream_error_count"] == 1
        assert status["ai_index_failed"] == 3
        error = status["downstream_errors"][0]
        assert error["source"] == "ai_index"
        assert error["scope"] == "batch"
        assert error["docs_in_failed_call"] == 3
    finally:
        browser_main.scraping_tasks.pop(session_id, None)


@pytest.mark.asyncio
async def test_background_batch_propagates_use_browser_to_scraper(monkeypatch):
    session_id = "browser-batch"
    result = ScrapeResult(
        url="https://example.net/",
        status="failed",
        error="browser launch unavailable",
    )
    fake_scraper = FakeScraper([result])

    _use_background_scraper(monkeypatch, fake_scraper)
    browser_main.scraping_tasks[session_id] = browser_main._new_scrape_task_info(1)

    try:
        await browser_main.scrape_urls_background(session_id, [result.url], True)

        assert fake_scraper.use_browser is True
    finally:
        browser_main.scraping_tasks.pop(session_id, None)


@pytest.mark.asyncio
async def test_overlapping_background_browser_batches_use_isolated_scrapers(
    monkeypatch,
):
    release = asyncio.Event()
    started = 0
    created = []
    active = 0
    max_active = 0

    class CoordinatedScraper:
        def __init__(self):
            self.closed = False
            self.use_browser = None
            self.urls = None

        async def scrape_batch(
            self,
            urls,
            session_id=None,
            callback=None,
            use_browser=False,
        ):
            nonlocal active, max_active, started
            self.urls = urls
            self.use_browser = use_browser
            active += 1
            max_active = max(max_active, active)
            started += 1
            if started == 2:
                release.set()
            await release.wait()
            active -= 1
            result = ScrapeResult(url=urls[0], status="failed", error="test")
            if callback:
                await callback(result)
            return [result]

        async def close(self):
            self.closed = True

    global_scraper = FakeScraper([])

    def factory():
        scraper = CoordinatedScraper()
        created.append(scraper)
        return scraper

    monkeypatch.setattr(browser_main, "scraper", global_scraper)
    monkeypatch.setattr(browser_main, "_new_scraper_engine", factory)
    monkeypatch.setattr(
        browser_main.httpx,
        "AsyncClient",
        lambda: FakeAsyncClient(lambda url, kwargs: _response(url)),
    )
    browser_main.scraping_tasks["session-a"] = browser_main._new_scrape_task_info(1)
    browser_main.scraping_tasks["session-b"] = browser_main._new_scrape_task_info(1)

    try:
        await asyncio.wait_for(
            asyncio.gather(
                browser_main.scrape_urls_background(
                    "session-a", ["https://one.example"], True
                ),
                browser_main.scrape_urls_background(
                    "session-b", ["https://two.example"], True
                ),
            ),
            timeout=1,
        )

        assert len(created) == 2
        assert [scraper.use_browser for scraper in created] == [True, True]
        assert all(scraper.closed for scraper in created)
        assert global_scraper.closed is False
        assert max_active == 2
    finally:
        browser_main.scraping_tasks.pop("session-a", None)
        browser_main.scraping_tasks.pop("session-b", None)


@pytest.mark.asyncio
async def test_scraper_get_browser_launches_once_for_concurrent_callers(monkeypatch):
    from services.browser_engine.app.scraper import engine

    launch_count = 0

    class FakeBrowser:
        async def close(self):
            return None

    class FakeChromium:
        async def launch(self, **kwargs):
            nonlocal launch_count
            launch_count += 1
            await asyncio.sleep(0)
            return FakeBrowser()

    class FakePlaywright:
        def __init__(self):
            self.chromium = FakeChromium()

        async def stop(self):
            return None

    class FakeStarter:
        async def start(self):
            await asyncio.sleep(0)
            return FakePlaywright()

    monkeypatch.setattr(engine, "async_playwright", lambda: FakeStarter())
    scraper = ScraperEngine(respect_robots=False)

    try:
        browsers = await asyncio.gather(
            scraper._get_browser(),
            scraper._get_browser(),
            scraper._get_browser(),
        )

        assert launch_count == 1
        assert len({id(browser) for browser in browsers}) == 1
    finally:
        await scraper.close()


@pytest.mark.asyncio
async def test_scraper_close_stops_playwright_when_browser_close_fails():
    stopped = False

    class FailingBrowser:
        async def close(self):
            raise RuntimeError("browser transport closed")

    class FakePlaywright:
        async def stop(self):
            nonlocal stopped
            stopped = True

    scraper = ScraperEngine(respect_robots=False)
    scraper._browser = FailingBrowser()
    scraper._playwright = FakePlaywright()

    with pytest.raises(RuntimeError, match="browser transport closed"):
        await scraper.close()

    assert stopped is True
    assert scraper._browser is None
    assert scraper._playwright is None


@pytest.mark.asyncio
async def test_scraper_batch_passes_use_browser_to_each_url():
    class RecordingScraper(ScraperEngine):
        def __init__(self):
            super().__init__(respect_robots=False)
            self.calls = []

        async def scrape_url(self, url, session_id=None, use_browser=False):
            self.calls.append(
                {
                    "url": url,
                    "session_id": session_id,
                    "use_browser": use_browser,
                }
            )
            return ScrapeResult(url=url, status="success", content="ok")

    scraper = RecordingScraper()

    results = await scraper.scrape_batch(
        ["https://one.example", "https://two.example"],
        session_id="session-1",
        use_browser=True,
    )

    assert [result.status for result in results] == ["success", "success"]
    assert scraper.calls == [
        {
            "url": "https://one.example",
            "session_id": "session-1",
            "use_browser": True,
        },
        {
            "url": "https://two.example",
            "session_id": "session-1",
            "use_browser": True,
        },
    ]


@pytest.mark.asyncio
async def test_scraper_batch_preserves_callback_failure_metadata():
    class StaticScraper(ScraperEngine):
        async def scrape_url(self, url, session_id=None, use_browser=False):
            return ScrapeResult(url=url, status="success", content="ok")

    async def failing_callback(result):
        raise RuntimeError("callback sink unavailable")

    scraper = StaticScraper(respect_robots=False)

    results = await scraper.scrape_batch(
        ["https://example.com"],
        session_id="session-callback-error",
        callback=failing_callback,
    )

    assert results[0].status == "success"
    assert results[0].metadata["callback_error"] == "callback sink unavailable"
