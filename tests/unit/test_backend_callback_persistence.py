"""Backend callback persistence regressions."""

import httpx
import pytest

from services.backend_core.app.api import routes
from services.backend_core.app.api.routes import ScrapeRequest, start_scraping
from services.backend_core.app.sessions.manager import SessionManager


def test_scrape_callback_persists_url_status_when_sqlite_enabled(tmp_path, monkeypatch):
    db_path = tmp_path / "backend.sqlite3"
    manager = SessionManager(db_path=str(db_path))
    session = manager.create_session("Callback Persistence")
    manager.add_urls_to_session(session.id, ["https://example.com/page"])
    monkeypatch.setattr(routes, "session_manager", manager)

    response = routes.scrape_complete_callback(
        {
            "session_id": session.id,
            "url": "https://example.com/page",
            "status": "success",
            "content": "Persisted callback content",
            "metadata": {"title": "Example"},
        }
    )

    assert response == {"status": "updated"}

    reloaded = SessionManager(db_path=str(db_path))
    restored = reloaded.get_session(session.id)
    assert restored is not None
    record = restored.url_store.get("https://example.com/page")
    assert record is not None
    assert record.status == "scraped"
    assert record.metadata == {
        "title": "Example",
        "content": "Persisted callback content",
    }


def test_scrape_callback_updates_single_url_row_without_clobbering_peers(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "backend.sqlite3"
    manager = SessionManager(db_path=str(db_path))
    session = manager.create_session("Callback Row Update")
    manager.add_urls_to_session(
        session.id,
        ["https://example.com/one", "https://example.com/two"],
    )
    monkeypatch.setattr(routes, "session_manager", manager)

    routes.scrape_complete_callback(
        {
            "session_id": session.id,
            "url": "https://example.com/one",
            "status": "success",
            "content": "First callback content",
            "metadata": {"title": "One"},
        }
    )
    routes.scrape_complete_callback(
        {
            "session_id": session.id,
            "url": "https://example.com/two",
            "status": "failed",
            "content": None,
            "metadata": {"title": "Two"},
        }
    )

    reloaded = SessionManager(db_path=str(db_path))
    restored = reloaded.get_session(session.id)
    assert restored is not None
    first = restored.url_store.get("https://example.com/one")
    second = restored.url_store.get("https://example.com/two")
    assert first is not None
    assert second is not None
    assert first.status == "scraped"
    assert first.metadata["content"] == "First callback content"
    assert second.status == "failed"
    assert second.metadata == {"title": "Two"}


@pytest.mark.asyncio
async def test_inline_scrape_urls_are_registered_then_callback_persists_and_indexes(
    tmp_path, monkeypatch
):
    # WI0 B3: POST /scrape with inline urls used to dispatch without registering
    # them, so every callback failed "URL not found in session" and the scraped
    # content vanished (no url record, no FTS row) while status read completed.
    db_path = tmp_path / "backend.sqlite3"
    manager = SessionManager(db_path=str(db_path))
    session = manager.create_session("Inline Scrape Registration")
    monkeypatch.setattr(routes, "session_manager", manager)

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, url, **kwargs):
            request = httpx.Request("POST", url)
            return httpx.Response(200, request=request, json={"status": "started"})

    monkeypatch.setattr(routes.httpx, "AsyncClient", lambda: FakeClient())

    url = "https://example.com/inline-widgets"
    dispatched = await start_scraping(ScrapeRequest(session_id=session.id, urls=[url]))
    assert dispatched["status"] == "started"

    # The inline url is now a registered record, not lost.
    registered = session.url_store.get(url)
    assert registered is not None

    response = routes.scrape_complete_callback(
        {
            "session_id": session.id,
            "url": url,
            "status": "success",
            "content": "Inline scrape body about widgets and gadgets",
            "metadata": {"title": "Inline Widgets"},
        }
    )
    assert response == {"status": "updated"}

    # url_record + FTS row survive a reload from disk.
    reloaded = SessionManager(db_path=str(db_path))
    restored = reloaded.get_session(session.id)
    assert restored is not None
    record = restored.url_store.get(url)
    assert record is not None
    assert record.status == "scraped"
    assert record.metadata["content"] == "Inline scrape body about widgets and gadgets"

    hits = reloaded.search_indexed_tabs(session.id, "widgets", 5)
    assert any(hit.get("url") == url for hit in hits)


@pytest.mark.asyncio
async def test_inline_scrape_urls_dedupe_against_existing_session_urls(
    tmp_path, monkeypatch
):
    # Re-registering an already-known url is a no-op: no duplicate rows.
    db_path = tmp_path / "backend.sqlite3"
    manager = SessionManager(db_path=str(db_path))
    session = manager.create_session("Inline Scrape Dedupe")
    url = "https://example.com/known"
    manager.add_urls_to_session(session.id, [url])
    monkeypatch.setattr(routes, "session_manager", manager)

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, url, **kwargs):
            request = httpx.Request("POST", url)
            return httpx.Response(200, request=request, json={"status": "started"})

    monkeypatch.setattr(routes.httpx, "AsyncClient", lambda: FakeClient())

    before = session.url_store.count()
    await start_scraping(ScrapeRequest(session_id=session.id, urls=[url]))
    assert session.url_store.count() == before
