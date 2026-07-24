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


def test_keyword_search_escapes_fts_punctuation(tmp_path):
    db_path = tmp_path / "backend.sqlite3"
    manager = SessionManager(db_path=str(db_path))
    session = manager.create_session("Keyword Punctuation")
    manager.add_urls_to_session(session.id, ["https://example.com/research"])
    manager.update_url_status(
        session.id,
        "https://example.com/research",
        "scraped",
        metadata={
            "title": "Remote clustering notes",
            "content": "remote-clustering docs are published on example.com",
        },
    )

    results = manager.search_indexed_tabs(
        session.id,
        "remote-clustering example.com",
        limit=5,
    )

    assert [result["url"] for result in results] == [
        "https://example.com/research"
    ]


def test_keyword_search_supports_global_sessionless_queries(tmp_path):
    db_path = tmp_path / "backend.sqlite3"
    manager = SessionManager(db_path=str(db_path))
    first = manager.create_session("First")
    second = manager.create_session("Second")
    manager.add_urls_to_session(first.id, ["https://example.com/first"])
    manager.add_urls_to_session(second.id, ["https://example.com/second"])
    manager.update_url_status(
        first.id,
        "https://example.com/first",
        "scraped",
        metadata={"title": "First", "content": "global keyword needle"},
    )
    manager.update_url_status(
        second.id,
        "https://example.com/second",
        "scraped",
        metadata={"title": "Second", "content": "global keyword needle"},
    )

    results = manager.search_indexed_tabs(None, "global needle", limit=10)

    assert {result["session_id"] for result in results} == {first.id, second.id}
    assert {result["url"] for result in results} == {
        "https://example.com/first",
        "https://example.com/second",
    }


def test_keyword_search_in_memory_preserves_session_scope():
    manager = SessionManager()
    first = manager.create_session("First")
    second = manager.create_session("Second")
    manager.add_urls_to_session(first.id, ["https://example.com/first"])
    manager.add_urls_to_session(second.id, ["https://example.com/second"])
    manager.update_url_status(
        first.id,
        "https://example.com/first",
        "scraped",
        metadata={"title": "First", "content": "memory keyword needle"},
    )
    manager.update_url_status(
        second.id,
        "https://example.com/second",
        "scraped",
        metadata={"title": "Second", "content": "memory keyword needle"},
    )

    scoped = manager.search_indexed_tabs(first.id, "memory needle", limit=10)
    global_results = manager.search_indexed_tabs(None, "memory needle", limit=10)
    missing = manager.search_indexed_tabs("missing-session", "memory needle", limit=10)

    assert [result["url"] for result in scoped] == ["https://example.com/first"]
    assert {result["url"] for result in global_results} == {
        "https://example.com/first",
        "https://example.com/second",
    }
    assert missing == []


class CapturingBackgroundTasks:
    def __init__(self):
        self.calls = []

    def add_task(self, func, *args):
        self.calls.append((func, args))


@pytest.mark.asyncio
async def test_backend_tab_import_endpoint_queues_durable_job(
    tmp_path,
    monkeypatch,
):
    manager = SessionManager(db_path=str(tmp_path / "backend.sqlite3"))
    monkeypatch.setattr(routes, "session_manager", manager)
    monkeypatch.setenv("BACKEND_AGENT_API_TOKEN", "agent-secret")
    background = CapturingBackgroundTasks()

    response = await routes.import_tabs_from_browser(
        routes.TabImportRequest(
            session_name="Imported Tabs",
            cdp_url="http://localhost:9222",
            max_tabs=25,
        ),
        background,
        _auth=routes._require_backend_agent_auth("Bearer agent-secret"),
    )

    assert response["status"] == "queued"
    assert response["session_id"]
    assert response["job_id"]
    assert manager.get_tab_import_job(response["job_id"]) is not None
    assert background.calls == [
        (
            routes.import_tabs_background,
            (response["job_id"], "http://localhost:9222", 25),
        )
    ]


@pytest.mark.asyncio
async def test_backend_search_endpoint_merges_keyword_and_semantic_results(
    tmp_path,
    monkeypatch,
):
    manager = SessionManager(db_path=str(tmp_path / "backend.sqlite3"))
    session = manager.create_session("Search Endpoint")
    manager.add_urls_to_session(session.id, ["https://example.com/local"])
    manager.update_url_status(
        session.id,
        "https://example.com/local",
        "scraped",
        metadata={"title": "Local Result", "content": "local browser tabs"},
    )
    monkeypatch.setattr(routes, "session_manager", manager)
    monkeypatch.setenv("BACKEND_AGENT_API_TOKEN", "agent-secret")

    async def fake_semantic_search(session_id, query, top_k):
        assert session_id == session.id
        assert query == "browser tabs"
        assert top_k == 5
        return [
            {
                "url": "https://example.com/semantic",
                "title": "Semantic Result",
                "content": "semantic browser tabs",
                "score": 0.9,
                "source": "semantic",
            }
        ]

    monkeypatch.setattr(routes, "_semantic_search", fake_semantic_search)

    response = await routes.search_tabs(
        routes.SearchRequest(
            session_id=session.id,
            query="browser tabs",
            mode="hybrid",
            top_k=5,
        ),
        _auth=routes._require_backend_agent_auth("Bearer agent-secret"),
    )

    assert response["mode"] == "hybrid"
    assert response["count"] == 2
    assert {result["url"] for result in response["results"]} == {
        "https://example.com/local",
        "https://example.com/semantic",
    }


@pytest.mark.asyncio
async def test_backend_keyword_search_endpoint_supports_global_query(
    tmp_path,
    monkeypatch,
):
    manager = SessionManager(db_path=str(tmp_path / "backend.sqlite3"))
    session = manager.create_session("Global Search Endpoint")
    manager.add_urls_to_session(session.id, ["https://example.com/global"])
    manager.update_url_status(
        session.id,
        "https://example.com/global",
        "scraped",
        metadata={"title": "Global Result", "content": "global browser tabs"},
    )
    monkeypatch.setattr(routes, "session_manager", manager)
    monkeypatch.setenv("BACKEND_AGENT_API_TOKEN", "agent-secret")

    response = await routes.search_tabs(
        routes.SearchRequest(query="global browser", mode="keyword", top_k=5),
        _auth=routes._require_backend_agent_auth("Bearer agent-secret"),
    )

    assert response["mode"] == "keyword"
    assert response["count"] == 1
    assert response["results"][0]["session_id"] == session.id
    assert response["results"][0]["url"] == "https://example.com/global"
