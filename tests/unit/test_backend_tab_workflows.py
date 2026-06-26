"""Contracts for agent-facing tab import and hybrid search backend state."""

import pytest

from services.backend_core.app.api import routes
from services.backend_core.app.sessions.manager import SessionManager


def test_agent_tab_api_requires_configured_bearer_token(monkeypatch):
    monkeypatch.delenv("BACKEND_AGENT_API_TOKEN", raising=False)

    with pytest.raises(routes.HTTPException) as unconfigured:
        routes._require_backend_agent_auth(None)
    assert unconfigured.value.status_code == 401

    monkeypatch.setenv("BACKEND_AGENT_API_TOKEN", "agent-secret")

    with pytest.raises(routes.HTTPException) as missing:
        routes._require_backend_agent_auth(None)
    assert missing.value.status_code == 401

    with pytest.raises(routes.HTTPException) as wrong:
        routes._require_backend_agent_auth("Bearer wrong")
    assert wrong.value.status_code == 401

    assert routes._require_backend_agent_auth("Bearer agent-secret") is None


def test_import_jobs_persist_and_reject_duplicate_active_jobs(tmp_path):
    db_path = tmp_path / "backend.sqlite3"
    manager = SessionManager(db_path=str(db_path))
    session = manager.create_session("Live Browser Import")

    job = manager.create_tab_import_job(
        session_id=session.id,
        cdp_url="http://localhost:9222",
    )
    manager.update_tab_import_job(
        job.id,
        status="running",
        total=1000,
        imported=100,
        indexed=50,
    )

    with pytest.raises(ValueError, match="active import"):
        manager.create_tab_import_job(
            session_id=session.id,
            cdp_url="http://localhost:9222",
        )

    reloaded = SessionManager(db_path=str(db_path))
    restored = reloaded.get_tab_import_job(job.id)

    assert restored is not None
    assert restored.session_id == session.id
    assert restored.cdp_url == "http://localhost:9222"
    assert restored.status == "running"
    assert restored.total == 1000
    assert restored.imported == 100
    assert restored.indexed == 50


def test_keyword_index_searches_persisted_tab_content(tmp_path):
    db_path = tmp_path / "backend.sqlite3"
    manager = SessionManager(db_path=str(db_path))
    session = manager.create_session("Keyword Search")
    manager.add_urls_to_session(
        session.id,
        [
            "https://example.com/research",
            "https://iana.org/domains/reserved",
        ],
    )
    assert manager.update_url_status(
        session.id,
        "https://example.com/research",
        "scraped",
        metadata={
            "title": "Browser research notes",
            "content": "Remote debugging and tab clustering strategy.",
            "source": "cdp",
        },
    )
    assert manager.update_url_status(
        session.id,
        "https://iana.org/domains/reserved",
        "scraped",
        metadata={
            "title": "Reserved domains",
            "content": "Documentation for example domains.",
        },
    )

    results = manager.search_indexed_tabs(session.id, "remote clustering", limit=5)

    assert [result["url"] for result in results] == [
        "https://example.com/research"
    ]
    assert results[0]["title"] == "Browser research notes"
    assert results[0]["source"] == "keyword"
    assert results[0]["score"] > 0
