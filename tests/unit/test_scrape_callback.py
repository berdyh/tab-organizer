"""Regression tests for backend scrape callback handling."""

import httpx
import pytest
from fastapi import HTTPException

from services.backend_core.app.api.routes import (
    ClusterRequest,
    ScrapeRequest,
    _ai_engine_headers,
    _require_backend_callback_auth,
    get_pending_auth,
    get_scrape_status,
    scrape_complete_callback,
    session_manager,
    start_clustering,
    submit_credentials,
    trigger_scraping,
)


def test_ai_engine_headers_use_configured_token(monkeypatch):
    monkeypatch.setenv("AI_ENGINE_API_TOKEN", "shared-token")

    assert _ai_engine_headers() == {"Authorization": "Bearer shared-token"}


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
