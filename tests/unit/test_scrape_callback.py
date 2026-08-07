"""Regression tests for backend scrape callback handling."""

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from services import url_safety
from services.backend_core.app.api import routes
from services.backend_core.app.api.routes import (
    ClusterRequest,
    ScrapeRequest,
    SearchRequest,
    URLInput,
    _ai_engine_headers,
    _browser_engine_headers,
    _require_backend_callback_auth,
    add_urls,
    get_pending_auth,
    get_scrape_status,
    scrape_complete_callback,
    search_tabs,
    session_manager,
    start_clustering,
    start_scraping,
    submit_credentials,
    trigger_scraping,
)
from services.backend_core.app.url_input.store import URLStore


def test_ai_engine_headers_use_configured_token(monkeypatch):
    monkeypatch.setenv("AI_ENGINE_API_TOKEN", "shared-token")

    assert _ai_engine_headers() == {"Authorization": "Bearer shared-token"}


def test_browser_engine_headers_send_only_browser_scope(monkeypatch):
    """Backend must send the browser scope, never a cross-scope fallback.

    Replaces the old `prefer_callback_token` assertion: browser-engine no
    longer ACCEPTS BACKEND_CALLBACK_TOKEN/AI_ENGINE_API_TOKEN, so sending them
    could only produce a misleading 401. Sending no header when the browser
    token is unset keeps the failure diagnosable.
    """
    monkeypatch.setenv("AI_ENGINE_API_TOKEN", "ai-token")
    monkeypatch.setenv("BACKEND_CALLBACK_TOKEN", "callback-token")
    monkeypatch.setenv("BROWSER_ENGINE_API_TOKEN", "browser-token")

    assert _browser_engine_headers() == {"Authorization": "Bearer browser-token"}

    monkeypatch.delenv("BROWSER_ENGINE_API_TOKEN")
    assert _browser_engine_headers() == {}


def test_backend_callback_auth_requires_shared_token(monkeypatch):
    monkeypatch.delenv("AI_ENGINE_API_TOKEN", raising=False)
    monkeypatch.delenv("BACKEND_CALLBACK_TOKEN", raising=False)

    with pytest.raises(HTTPException) as unconfigured:
        _require_backend_callback_auth(None)
    assert unconfigured.value.status_code == 401

    monkeypatch.setenv("BACKEND_CALLBACK_TOKEN", "callback-secret")
    with pytest.raises(HTTPException) as missing:
        _require_backend_callback_auth(None)
    assert missing.value.status_code == 401

    with pytest.raises(HTTPException) as wrong:
        _require_backend_callback_auth("Bearer wrong")
    assert wrong.value.status_code == 401

    assert _require_backend_callback_auth("Bearer callback-secret") is None


def test_scrape_request_accepts_browser_mode():
    request = ScrapeRequest(session_id="session-1", use_browser=True)

    assert request.use_browser is True


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",
        "http://localhost/admin",
        "http://127.0.0.1:8080/health",
        "http://172.16.0.10/private",
        "http://ai-engine:8090/health",
    ],
)
def test_url_store_rejects_unsafe_scrape_urls(url):
    store = URLStore()

    with pytest.raises(ValueError):
        store.add(url)


def test_url_store_rejects_hostname_resolving_to_private_address(monkeypatch):
    monkeypatch.setattr(
        url_safety.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (
                url_safety.socket.AF_INET,
                url_safety.socket.SOCK_STREAM,
                6,
                "",
                ("127.0.0.1", 0),
            )
        ],
    )

    store = URLStore()

    with pytest.raises(ValueError, match="private network"):
        store.add("https://public-looking.test/path")


def test_backend_add_urls_reports_unsafe_scrape_url_as_bad_request():
    session = session_manager.create_session("Unsafe URL Regression")

    try:
        with pytest.raises(HTTPException) as exc_info:
            add_urls(
                URLInput(
                    session_id=session.id,
                    urls=["http://127.0.0.1:8080/internal"],
                )
            )

        assert exc_info.value.status_code == 400
    finally:
        session_manager.delete_session(session.id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("browser_status", "url_status"),
    [
        ("success", "scraped"),
        ("failed", "failed"),
        ("auth_required", "auth_required"),
        ("timeout", "failed"),
        ("blocked", "failed"),
        ("unexpected", "failed"),
    ],
)
async def test_scrape_callback_maps_browser_statuses(browser_status, url_status):
    """Browser-engine result statuses should map to backend URL statuses."""
    session = session_manager.create_session(f"Callback {browser_status} Regression")
    url = f"https://example.com/{browser_status}"
    session_manager.add_urls_to_session(session.id, [url])

    try:
        scrape_complete_callback(
            {
                "session_id": session.id,
                "url": url,
                "status": browser_status,
                "content": "Example Domain content",
                "metadata": {"title": "Example Domain", "status_code": 200},
            }
        )

        record = session.url_store.get(url)
        assert record is not None
        assert record.status == url_status
        assert record.metadata["content"] == "Example Domain content"
    finally:
        session_manager.delete_session(session.id)


@pytest.mark.asyncio
async def test_scrape_callback_reports_missing_url():
    """A callback for an unknown URL should be visible to browser-engine."""
    session = session_manager.create_session("Missing URL Callback Regression")
    session_manager.add_urls_to_session(session.id, ["https://example.com/known"])

    try:
        result = scrape_complete_callback(
            {
                "session_id": session.id,
                "url": "https://example.com/missing",
                "status": "success",
                "content": "Missing content",
                "metadata": {"title": "Missing"},
            }
        )

        assert result == {
            "status": "error",
            "message": "URL not found in session",
        }
        record = session.url_store.get("https://example.com/known")
        assert record is not None
        assert record.status == "pending"
    finally:
        session_manager.delete_session(session.id)


@pytest.mark.asyncio
async def test_backend_service_urls_use_runtime_env(monkeypatch):
    """Backend proxies should honor configured service URLs, not hard-code hosts."""
    calls = []
    session = session_manager.create_session("Runtime URL Regression")
    session_manager.add_urls_to_session(session.id, ["https://example.com"])
    session.url_store.update_status(
        "https://example.com",
        "scraped",
        metadata={"content": "Example Domain content", "title": "Example Domain"},
    )

    monkeypatch.setenv("AI_ENGINE_URL", "http://ai.test/")
    monkeypatch.setenv("BROWSER_ENGINE_URL", "http://browser.test/")

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, url, **kwargs):
            calls.append(("POST", url, kwargs.get("json")))
            request = httpx.Request("POST", url)
            if url.endswith("/cluster"):
                return httpx.Response(
                    200, request=request, json={"clusters": [{"name": "Example"}]}
                )
            return httpx.Response(200, request=request, json={"ok": True})

        async def get(self, url, **kwargs):
            calls.append(("GET", url, None))
            request = httpx.Request("GET", url)
            return httpx.Response(200, request=request, json={"status": "running"})

    monkeypatch.setattr(
        "services.backend_core.app.api.routes.httpx.AsyncClient",
        lambda: FakeClient(),
    )

    try:
        await trigger_scraping(session.id, ["https://example.com"], use_browser=True)
        await start_clustering(ClusterRequest(session_id=session.id))
        await get_scrape_status(session.id)
        await get_pending_auth()
        await submit_credentials("example.com", {"username": "u"})

        assert [method_url[:2] for method_url in calls] == [
            ("POST", "http://browser.test/scrape"),
            ("POST", "http://ai.test/cluster"),
            ("GET", f"http://browser.test/scrape/status/{session.id}"),
            ("GET", "http://browser.test/auth/pending"),
            ("POST", "http://browser.test/auth/credentials"),
        ]
        assert calls[0][2]["use_browser"] is True
    finally:
        session_manager.delete_session(session.id)


def test_scrape_status_route_is_wired_to_get_scrape_status(monkeypatch):
    """Regression: GET /scrape/status/{id} must dispatch to get_scrape_status.

    The route decorator once landed on the private `_overlay_ingest_status`
    helper instead of on `get_scrape_status` (both sit next to each other in
    routes.py). FastAPI then read `_overlay_ingest_status`'s `payload: dict`
    parameter as a required JSON body on a GET, so every real caller got a
    422 instead of a status payload. A direct Python call to the helper can't
    catch that — this test goes over HTTP through the actual router so a
    future mis-bound decorator fails here.
    """
    monkeypatch.setenv("BROWSER_ENGINE_URL", "http://browser.test/")
    session = session_manager.create_session("HTTP Status Route Regression")
    session_manager.add_urls_to_session(session.id, ["https://example.com"])

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url, **kwargs):
            request = httpx.Request("GET", url)
            return httpx.Response(
                200,
                request=request,
                json={"status": "completed", "success": 1, "ai_index_failed": 0},
            )

    monkeypatch.setattr(
        "services.backend_core.app.api.routes.httpx.AsyncClient",
        lambda: FakeClient(),
    )

    app = FastAPI()
    app.include_router(routes.router, prefix="/api/v1")
    client = TestClient(app)

    try:
        response = client.get(f"/api/v1/scrape/status/{session.id}")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "completed"
        assert body["ai_index_failed"] == 0
        assert body["ai_index_pending"] == 0
    finally:
        session_manager.delete_session(session.id)


@pytest.mark.asyncio
async def test_start_scraping_reports_browser_engine_dispatch_failure(monkeypatch):
    session = session_manager.create_session("Dispatch Failure Regression")
    session_manager.add_urls_to_session(session.id, ["https://example.com"])

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, url, **_kwargs):
            request = httpx.Request("POST", url)
            return httpx.Response(
                401,
                request=request,
                json={"detail": "Not authenticated"},
            )

    monkeypatch.setattr(
        "services.backend_core.app.api.routes.httpx.AsyncClient",
        lambda: FakeClient(),
    )

    try:
        with pytest.raises(HTTPException) as exc_info:
            await start_scraping(ScrapeRequest(session_id=session.id))

        assert exc_info.value.status_code == 502
        assert "Browser Engine scrape dispatch failed" in exc_info.value.detail
        record = session.url_store.get("https://example.com")
        assert record.status == "failed"
        assert "dispatch_error" in record.metadata
    finally:
        session_manager.delete_session(session.id)


@pytest.mark.asyncio
async def test_clustering_reports_ai_engine_http_errors(monkeypatch):
    """AI engine HTTP errors should not be reported as empty successful clusters."""
    session = session_manager.create_session("Cluster Error Regression")
    session_manager.add_urls_to_session(session.id, ["https://example.com"])
    session.url_store.update_status(
        "https://example.com",
        "scraped",
        metadata={"content": "Example Domain content", "title": "Example Domain"},
    )

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, *args, **kwargs):
            request = httpx.Request("POST", "http://ai-engine:8090/cluster")
            return httpx.Response(
                500, request=request, json={"detail": "AI engine failed"}
            )

    monkeypatch.setattr(
        "services.backend_core.app.api.routes.httpx.AsyncClient",
        lambda: FakeClient(),
    )

    try:
        with pytest.raises(HTTPException) as exc_info:
            await start_clustering(ClusterRequest(session_id=session.id))

        assert exc_info.value.status_code == 500
        assert "Clustering failed" in exc_info.value.detail
    finally:
        session_manager.delete_session(session.id)


@pytest.mark.asyncio
async def test_hybrid_search_degrades_to_keyword_when_semantic_leg_fails(monkeypatch):
    # WI0 B4: a dead semantic leg must not 500 the hybrid default. Keyword hits
    # are returned with a `degraded` diagnostic instead.
    async def failing_semantic(session_id, query, top_k):
        response = httpx.Response(
            500,
            request=httpx.Request("POST", "http://ai-engine:8090/search"),
            json={"detail": "OPENROUTER_API_KEY is not configured"},
        )
        response.raise_for_status()

    monkeypatch.setattr(routes, "_semantic_search", failing_semantic)
    monkeypatch.setattr(
        routes.session_manager,
        "search_indexed_tabs",
        lambda session_id, query, limit: [
            {
                "url": "https://example.com/kw",
                "title": "Keyword Hit",
                "content": "widgets keyword content",
                "score": 1.0,
                "source": "keyword",
            }
        ],
    )

    result = await search_tabs(SearchRequest(query="widgets", mode="hybrid"))

    assert result["mode"] == "hybrid"
    assert result["count"] == 1
    assert result["results"][0]["url"] == "https://example.com/kw"
    assert result["degraded"].startswith("semantic_unavailable:")
    assert "OPENROUTER_API_KEY is not configured" in result["degraded"]


@pytest.mark.asyncio
async def test_semantic_only_search_raises_structured_error_when_leg_fails(monkeypatch):
    # Semantic-only has nothing to fall back to: surface a {code, cause, fix}
    # HTTP error instead of a bare 500.
    async def failing_semantic(session_id, query, top_k):
        response = httpx.Response(
            502,
            request=httpx.Request("POST", "http://ai-engine:8090/search"),
            text="ai-engine down",
        )
        response.raise_for_status()

    monkeypatch.setattr(routes, "_semantic_search", failing_semantic)

    with pytest.raises(HTTPException) as exc_info:
        await search_tabs(SearchRequest(query="widgets", mode="semantic"))

    assert exc_info.value.status_code == 502
    detail = exc_info.value.detail
    assert detail["code"] == "semantic_unavailable"
    assert "502" in detail["cause"]
    assert detail["fix"]


@pytest.mark.asyncio
async def test_hybrid_search_omits_degraded_field_when_semantic_leg_succeeds(
    monkeypatch,
):
    async def ok_semantic(session_id, query, top_k):
        return [
            {
                "url": "https://example.com/sem",
                "title": "Semantic Hit",
                "content": "semantic content",
                "score": 0.9,
                "source": "semantic",
            }
        ]

    monkeypatch.setattr(routes, "_semantic_search", ok_semantic)
    monkeypatch.setattr(
        routes.session_manager,
        "search_indexed_tabs",
        lambda session_id, query, limit: [],
    )

    result = await search_tabs(SearchRequest(query="widgets", mode="hybrid"))

    assert "degraded" not in result
    assert result["count"] == 1


def _route_dependency_calls(route) -> set:
    """Collect every dependency callable a FastAPI route resolves."""
    calls = set()
    stack = list(route.dependant.dependencies)
    while stack:
        dependency = stack.pop()
        if dependency.call is not None:
            calls.add(dependency.call)
        stack.extend(dependency.dependencies)
    return calls


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/auth/pending"),
        ("POST", "/auth/credentials"),
    ],
)
def test_backend_auth_proxy_requires_agent_token(method, path):
    """C3: the credential proxy must not be a confused deputy.

    These handlers attach the privileged Browser Engine service token
    server-side, so an unauthenticated caller could read the pending-auth
    queue and plant credentials for any domain.
    """
    matches = [
        route
        for route in routes.router.routes
        if getattr(route, "path", None) == path
        and method in getattr(route, "methods", set())
    ]
    assert matches, f"{method} {path} not registered"
    for route in matches:
        assert routes._require_backend_agent_auth in _route_dependency_calls(
            route
        ), f"{method} {path} is missing the backend agent auth dependency"
