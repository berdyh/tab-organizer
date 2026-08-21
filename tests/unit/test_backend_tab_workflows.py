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


# ---------------------------------------------------------------------------
# Import-job accounting: progress survives a later failure, skips stay visible
# ---------------------------------------------------------------------------


class _StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _StubAsyncClient:
    """Stands in for httpx.AsyncClient, returning one canned browser payload."""

    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, *args, **kwargs):
        return _StubResponse(self._payload)


def _install_browser_payload(monkeypatch, payload):
    monkeypatch.setattr(
        routes.httpx, "AsyncClient", lambda *a, **k: _StubAsyncClient(payload)
    )
    monkeypatch.setattr(routes, "_browser_engine_url", lambda: "http://browser.test")
    monkeypatch.setattr(routes, "_browser_engine_request_headers", lambda: {})


@pytest.mark.asyncio
async def test_import_job_reports_skips_and_never_indexes_a_blank_document(
    tmp_path, monkeypatch
):
    """A blank document is dropped before indexing AND reported as a skip.

    This is the general form of the defect a 46-tab import surfaced: the
    embedding provider rejects the WHOLE batch when one entry is an empty
    string, so two blank tabs indexed zero of forty-six. Dropping them silently
    would fix the batch and hide the loss; the count and the reason both have
    to survive.
    """
    manager = SessionManager(db_path=str(tmp_path / "backend.sqlite3"))
    session = manager.create_session("Skips")
    job = manager.create_tab_import_job(
        session_id=session.id, cdp_url="http://localhost:9222"
    )
    monkeypatch.setattr(routes, "session_manager", manager)

    _install_browser_payload(
        monkeypatch,
        {
            "total": 3,
            "imported": 2,
            "skipped": 1,
            "failed": 0,
            "tabs": [
                {
                    "id": "https://good.example",
                    "url": "https://good.example",
                    "title": "Good",
                    "content": "real text",
                    "metadata": {},
                },
                {
                    "id": "https://blank.example",
                    "url": "https://blank.example",
                    "title": "Blank",
                    "content": "   ",
                    "metadata": {},
                },
            ],
            "skipped_tabs": [
                {
                    "url": "https://login.example",
                    "title": "Sign in",
                    "reason": "auth_wall",
                    "detail": "page is a sign-in prompt",
                }
            ],
            "errors": [],
        },
    )

    indexed_batches = []

    async def fake_index(session_id, documents):
        indexed_batches.append(list(documents))
        return len(documents)

    monkeypatch.setattr(routes, "_index_tab_documents", fake_index)

    await routes.import_tabs_background(job.id, "http://localhost:9222", 100)

    final = manager.get_tab_import_job(job.id)
    assert final.status == "completed"
    # The blank tab never reaches the embedding provider...
    assert [d["url"] for d in indexed_batches[0]] == ["https://good.example"]
    # ...but it is still counted and explained, alongside the harvester's skip.
    assert final.skipped == 2
    reasons = {s["reason"] for s in final.metadata["skipped_tabs"]}
    assert reasons == {"auth_wall", "blank"}


@pytest.mark.asyncio
async def test_import_job_keeps_its_counters_when_indexing_fails(tmp_path, monkeypatch):
    """A failure in the LAST leg must not erase what the earlier legs achieved.

    Observed on a real run: indexing failed, the job recorded
    `total=0 imported=0`, and the database held 43 fully captured pages. The
    status field said nothing had happened while the corpus said otherwise, so
    the operator's only true source was a manual sqlite query.
    """
    manager = SessionManager(db_path=str(tmp_path / "backend.sqlite3"))
    session = manager.create_session("Partial")
    job = manager.create_tab_import_job(
        session_id=session.id, cdp_url="http://localhost:9222"
    )
    monkeypatch.setattr(routes, "session_manager", manager)

    _install_browser_payload(
        monkeypatch,
        {
            "total": 2,
            "imported": 2,
            "skipped": 0,
            "failed": 0,
            "tabs": [
                {
                    "id": f"https://page{n}.example",
                    "url": f"https://page{n}.example",
                    "title": f"Page {n}",
                    "content": "captured text",
                    "metadata": {},
                }
                for n in (1, 2)
            ],
            "skipped_tabs": [],
            "errors": [],
        },
    )

    async def failing_index(session_id, documents):
        raise RuntimeError("embedding provider rejected the batch")

    monkeypatch.setattr(routes, "_index_tab_documents", failing_index)

    await routes.import_tabs_background(job.id, "http://localhost:9222", 100)

    final = manager.get_tab_import_job(job.id)
    assert final.status == "failed"
    assert "embedding provider rejected the batch" in final.error
    # The captures happened and are durable; the counters must say so.
    assert final.total == 2
    assert final.imported == 2
    assert final.indexed == 0


def test_downstream_error_text_keeps_the_response_body(monkeypatch):
    """`raise_for_status()` renders a status line; the cause is in the body.

    Every failure in this path comes from another service that explained itself
    in the response body -- "must be a local Chrome debugging endpoint",
    "expected string to have >=1 characters" -- and all of it was discarded,
    leaving an operator a status code and no cause.
    """

    class _Resp:
        text = "detail: must be a local Chrome debugging endpoint"

    error = RuntimeError("Client error '400 Bad Request' for url 'http://b/tabs/import'")
    error.response = _Resp()

    rendered = routes._downstream_error_text(error)
    assert "400 Bad Request" in rendered
    assert "must be a local Chrome debugging endpoint" in rendered

    plain = routes._downstream_error_text(RuntimeError("connection refused"))
    assert plain == "connection refused"
