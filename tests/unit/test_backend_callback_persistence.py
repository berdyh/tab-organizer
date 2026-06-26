"""Backend callback persistence regressions."""

from services.backend_core.app.api import routes
from services.backend_core.app.sessions.manager import SessionManager


def test_scrape_callback_persists_url_status_when_sqlite_enabled(
    tmp_path, monkeypatch
):
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
