"""Regression tests for backend session persistence."""

from services.backend_core.app.sessions.manager import SessionManager


def test_session_manager_persists_sessions_urls_and_clusters(tmp_path):
    """SQLite-backed managers should recover session state after restart."""
    db_path = tmp_path / "backend.sqlite3"
    manager = SessionManager(db_path=str(db_path))
    session = manager.create_session("Persistent Session")

    manager.add_urls_to_session(
        session.id,
        ["https://example.com/page", "https://www.iana.org/domains/reserved"],
    )
    assert manager.update_url_status(
        session.id,
        "https://example.com/page",
        "scraped",
        metadata={"title": "Example", "content": "Example content"},
    )
    assert manager.set_session_clusters(
        session.id,
        [{"name": "Reference Sites", "urls": ["https://example.com/page"]}],
    )

    reloaded = SessionManager(db_path=str(db_path))
    restored = reloaded.get_session(session.id)

    assert restored is not None
    assert restored.name == "Persistent Session"
    assert restored.url_store.count() == 2
    record = restored.url_store.get("https://example.com/page")
    assert record is not None
    assert record.status == "scraped"
    assert record.metadata["content"] == "Example content"
    assert restored.clusters == [
        {"name": "Reference Sites", "urls": ["https://example.com/page"]}
    ]


def test_session_delete_is_persisted(tmp_path):
    """Deleted sessions should stay deleted for a new manager instance."""
    db_path = tmp_path / "backend.sqlite3"
    manager = SessionManager(db_path=str(db_path))
    session = manager.create_session("Delete Me")
    manager.add_urls_to_session(session.id, ["https://example.com"])

    assert manager.delete_session(session.id)

    reloaded = SessionManager(db_path=str(db_path))
    assert reloaded.get_session(session.id) is None


def test_cleared_current_session_state_is_persisted(tmp_path):
    """Clearing current session should not select an archived session after restart."""
    db_path = tmp_path / "backend.sqlite3"
    manager = SessionManager(db_path=str(db_path))
    session = manager.create_session("Archive Me")
    assert manager.archive_session(session.id)

    reloaded = SessionManager(db_path=str(db_path))

    assert reloaded.get_current_session() is None
    new_session = reloaded.get_or_create_current_session()
    assert new_session.id != session.id
    assert new_session.status == "active"
