"""Tests for the single versioned idempotent ingest endpoint.

Covers the replay/ordering rule (capture_id uniqueness + newest-wins
(fetched_at, attempt)), the single-writer forward to ai-engine /index, the
status overlay, in-memory/SQLite parity, and the legacy callback adapter.
"""

import uuid

import httpx
import pytest
from fastapi import BackgroundTasks, HTTPException

from services.backend_core.app.api import ingest, routes
from services.backend_core.app.sessions.manager import IngestCapture, SessionManager

T0 = "2026-07-24T09:00:00"
T1 = "2026-07-24T10:00:00"
T2 = "2026-07-24T11:00:00"


def _capture(
    session_id,
    url,
    *,
    capture_id=None,
    attempt=1,
    status="success",
    content="body content",
    metadata=None,
    auth_used=False,
    fetched_at=T1,
):
    return IngestCapture(
        capture_id=capture_id or str(uuid.uuid4()),
        attempt=attempt,
        session_id=session_id,
        url=url,
        status=status,
        content=content,
        metadata={"title": "Title"} if metadata is None else metadata,
        auth_used=auth_used,
        fetched_at=fetched_at,
    )


def _managers(tmp_path):
    """Return (in-memory manager, sqlite manager) for parity assertions."""
    return SessionManager(), SessionManager(db_path=str(tmp_path / "ingest.db"))


# ---------------------------------------------------------------------------
# 1. Late-clobber regression (WI0/Codex evidence)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("persist", [False, True])
def test_late_attempt_is_stale_and_does_not_clobber(tmp_path, persist):
    manager = SessionManager(db_path=str(tmp_path / "db")) if persist else SessionManager()
    session = manager.create_session("late clobber")
    url = "https://example.com/report"
    manager.add_urls_to_session(session.id, [url])

    newest = manager.ingest_scrape_result(
        _capture(session.id, url, attempt=2, fetched_at=T2, content="newest widgets")
    )
    late = manager.ingest_scrape_result(
        _capture(session.id, url, attempt=1, fetched_at=T1, content="stale gadgets")
    )

    assert newest.outcome == "applied"
    assert late.outcome == "stale"

    record = manager.get_session(session.id).url_store.get(url)
    assert record.status == "scraped"
    assert record.metadata["content"] == "newest widgets"

    hits = manager.search_indexed_tabs(session.id, "widgets", 5)
    assert any(hit["url"] == url for hit in hits)
    assert not manager.search_indexed_tabs(session.id, "gadgets", 5)


# ---------------------------------------------------------------------------
# 2. Replay rejection (duplicate capture_id)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("persist", [False, True])
def test_replay_same_capture_id_is_duplicate(tmp_path, persist):
    manager = SessionManager(db_path=str(tmp_path / "db")) if persist else SessionManager()
    session = manager.create_session("replay")
    url = "https://example.com/a"
    manager.add_urls_to_session(session.id, [url])
    cap = _capture(session.id, url, content="original body")

    first = manager.ingest_scrape_result(cap)
    second = manager.ingest_scrape_result(cap)

    assert first.outcome == "applied"
    assert second.outcome == "duplicate"
    record = manager.get_session(session.id).url_store.get(url)
    assert record.metadata["content"] == "original body"


# ---------------------------------------------------------------------------
# 3. Duplicate-heals-index (route + forward)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_receipt_reforwards_failed_index(tmp_path, monkeypatch):
    manager = SessionManager(db_path=str(tmp_path / "db"))
    session = manager.create_session("heal")
    url = "https://example.com/heal"
    manager.add_urls_to_session(session.id, [url])
    monkeypatch.setattr(routes, "session_manager", manager)

    posted = []
    outcomes = iter([500, 200])  # first forward fails, second succeeds

    def handler(url_, kwargs):
        posted.append(kwargs["json"]["documents"][0]["id"])
        status = next(outcomes)
        return httpx.Response(
            status,
            request=httpx.Request("POST", url_),
            json={"indexed": 1} if status == 200 else {"detail": "boom"},
        )

    monkeypatch.setattr(ingest.httpx, "AsyncClient", lambda: _FakeClient(handler))

    cap = _capture(session.id, url)
    bt = BackgroundTasks()
    ingest.apply_capture(cap, bt)
    await _run(bt)
    normalized = manager.get_session(session.id).url_store.get(url).normalized
    assert manager.capture_index_counts(session.id)["ai_index_failed"] == 1

    bt2 = BackgroundTasks()
    dup = ingest.apply_capture(cap, bt2)
    assert dup["status"] == "ignored"
    await _run(bt2)

    counts = manager.capture_index_counts(session.id)
    assert counts["ai_index_failed"] == 0
    assert counts["ai_index_pending"] == 0
    assert posted == [normalized, normalized]


# ---------------------------------------------------------------------------
# 4. Concurrent same-url ingest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ["increasing", "decreasing"])
def test_concurrent_same_url_final_record_is_max(tmp_path, order):
    import threading

    manager = SessionManager(db_path=str(tmp_path / "db"))
    session = manager.create_session("concurrent")
    url = "https://example.com/c"
    manager.add_urls_to_session(session.id, [url])

    older = _capture(session.id, url, attempt=1, fetched_at=T1, content="older body")
    newer = _capture(session.id, url, attempt=2, fetched_at=T2, content="newer body")
    caps = [older, newer] if order == "increasing" else [newer, older]

    results = {}

    def run(cap):
        results[cap.capture_id] = manager.ingest_scrape_result(cap)

    threads = [threading.Thread(target=run, args=(cap,)) for cap in caps]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    record = manager.get_session(session.id).url_store.get(url)
    assert record.metadata["content"] == "newer body"
    assert manager.is_latest_applied(session.id, record.normalized, newer.capture_id)
    assert not manager.is_latest_applied(session.id, record.normalized, older.capture_id)


# ---------------------------------------------------------------------------
# 5. In-memory / SQLite parity
# ---------------------------------------------------------------------------


def test_in_memory_and_sqlite_parity(tmp_path):
    mem, sql = _managers(tmp_path)
    url = "https://example.com/parity"

    def sequence(manager):
        session = manager.create_session("parity")
        manager.add_urls_to_session(session.id, [url])
        cap1 = _capture(session.id, url, attempt=1, fetched_at=T1, content="first body")
        cap2 = _capture(session.id, url, attempt=2, fetched_at=T2, content="second body")
        cap_old = _capture(
            session.id, url, attempt=1, fetched_at=T0, content="old body"
        )
        outcomes = [
            manager.ingest_scrape_result(cap1).outcome,
            manager.ingest_scrape_result(cap1).outcome,  # duplicate
            manager.ingest_scrape_result(cap2).outcome,  # applied (newer)
            manager.ingest_scrape_result(cap_old).outcome,  # stale
        ]
        record = manager.get_session(session.id).url_store.get(url)
        hits = [h["url"] for h in manager.search_indexed_tabs(session.id, "second", 5)]
        return outcomes, record.status, record.metadata["content"], hits

    assert sequence(mem) == sequence(sql)
    outcomes, status, content, hits = sequence(mem)
    assert outcomes == ["applied", "duplicate", "applied", "stale"]
    assert status == "scraped"
    assert content == "second body"
    assert url in hits


# ---------------------------------------------------------------------------
# 6. Legacy adapter
# ---------------------------------------------------------------------------


def test_legacy_adapter_applies_with_attempt_zero_and_loses_ties(tmp_path, monkeypatch):
    manager = SessionManager(db_path=str(tmp_path / "db"))
    session = manager.create_session("legacy")
    url = "https://example.com/legacy"
    manager.add_urls_to_session(session.id, [url])
    monkeypatch.setattr(routes, "session_manager", manager)

    events = []
    monkeypatch.setattr(
        routes, "log_event", lambda name, **kw: events.append(name)
    )

    # A v1 write at attempt 1 lands first with the same fetched_at.
    manager.ingest_scrape_result(
        _capture(session.id, url, attempt=1, fetched_at=T1, content="v1 body")
    )

    # Legacy adapter builds attempt=0 with the same fetched_at -> loses the tie.
    response = routes.scrape_complete_callback(
        {
            "session_id": session.id,
            "url": url,
            "status": "success",
            "content": "legacy body",
            "metadata": {"title": "Legacy"},
        }
    )
    # Legacy response shape is unchanged, and its own capture was recorded stale.
    assert response == {"status": "updated"}
    assert "ingest.legacy_callback_deprecated" in events
    record = manager.get_session(session.id).url_store.get(url)
    assert record.metadata["content"] == "v1 body"


def test_legacy_adapter_reports_unknown_session_and_url(tmp_path, monkeypatch):
    manager = SessionManager(db_path=str(tmp_path / "db"))
    session = manager.create_session("legacy errors")
    manager.add_urls_to_session(session.id, ["https://example.com/known"])
    monkeypatch.setattr(routes, "session_manager", manager)

    missing_session = routes.scrape_complete_callback(
        {"session_id": "nope", "url": "https://example.com/known", "status": "failed"}
    )
    assert missing_session == {"status": "error", "message": "Session not found"}

    missing_url = routes.scrape_complete_callback(
        {
            "session_id": session.id,
            "url": "https://example.com/unregistered",
            "status": "success",
            "content": "x",
        }
    )
    assert missing_url == {"status": "error", "message": "URL not found in session"}


# ---------------------------------------------------------------------------
# 7. Status overlay (B5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scrape_status_overlays_index_failures(tmp_path, monkeypatch):
    manager = SessionManager(db_path=str(tmp_path / "db"))
    session = manager.create_session("overlay")
    urls = ["https://example.com/x", "https://example.com/y"]
    manager.add_urls_to_session(session.id, urls)
    monkeypatch.setattr(routes, "session_manager", manager)

    def handler(url_, kwargs):
        return httpx.Response(
            500, request=httpx.Request("POST", url_), json={"detail": "index down"}
        )

    monkeypatch.setattr(ingest.httpx, "AsyncClient", lambda: _FakeClient(handler))

    for url in urls:
        bt = BackgroundTasks()
        ingest.apply_capture(_capture(session.id, url), bt)
        await _run(bt)

    # Browser-engine payload is merged, not replaced.
    browser_payload = {
        "session_id": session.id,
        "status": "completed",
        "success": 2,
        "ai_index_failed": 0,
        "downstream_errors": [{"source": "backend_callback", "message": "kept"}],
    }
    merged = routes._overlay_ingest_status(session.id, browser_payload)
    assert merged["success"] == 2
    assert merged["ai_index_failed"] == 2
    assert merged["ai_index_pending"] == 0
    sources = [e["source"] for e in merged["downstream_errors"]]
    assert "backend_callback" in sources
    assert sources.count("ai_index") == 2


# ---------------------------------------------------------------------------
# 8. auth_used propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auth_used_propagates_to_ledger_and_forward(tmp_path, monkeypatch):
    manager = SessionManager(db_path=str(tmp_path / "db"))
    session = manager.create_session("auth")
    url = "https://intranet.example/private"
    manager.add_urls_to_session(session.id, [url])
    monkeypatch.setattr(routes, "session_manager", manager)

    seen = {}

    def handler(url_, kwargs):
        seen.update(kwargs["json"]["documents"][0]["metadata"])
        return httpx.Response(
            200, request=httpx.Request("POST", url_), json={"indexed": 1}
        )

    monkeypatch.setattr(ingest.httpx, "AsyncClient", lambda: _FakeClient(handler))

    cap = _capture(session.id, url, auth_used=True)
    bt = BackgroundTasks()
    ingest.apply_capture(cap, bt)
    await _run(bt)

    assert seen["auth_used"] is True
    assert seen["capture_id"] == cap.capture_id
    with manager._connect() as conn:
        row = conn.execute(
            "SELECT auth_used FROM ingest_captures WHERE capture_id = ?",
            (cap.capture_id,),
        ).fetchone()
    assert row["auth_used"] == 1


# ---------------------------------------------------------------------------
# 9. No-content / non-success -> index skipped, no forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,content",
    [("success", None), ("failed", "irrelevant"), ("auth_required", None)],
)
def test_non_indexable_captures_skip_forward(tmp_path, monkeypatch, status, content):
    manager = SessionManager(db_path=str(tmp_path / "db"))
    session = manager.create_session("skip")
    url = "https://example.com/skip"
    manager.add_urls_to_session(session.id, [url])
    monkeypatch.setattr(routes, "session_manager", manager)

    outcome = manager.ingest_scrape_result(
        _capture(session.id, url, status=status, content=content)
    )
    assert outcome.outcome == "applied"
    assert outcome.should_forward is False
    assert outcome.index_state == "skipped"

    response = ingest.apply_capture(
        _capture(session.id, url, status=status, content=content), BackgroundTasks()
    )
    assert response["index"] == "skipped"


# ---------------------------------------------------------------------------
# 10. Superseded forward
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_superseded_forward_is_skipped(tmp_path, monkeypatch):
    manager = SessionManager(db_path=str(tmp_path / "db"))
    session = manager.create_session("supersede")
    url = "https://example.com/s"
    manager.add_urls_to_session(session.id, [url])
    monkeypatch.setattr(routes, "session_manager", manager)

    posted = []

    def handler(url_, kwargs):
        posted.append(kwargs["json"]["documents"][0]["content"])
        return httpx.Response(
            200, request=httpx.Request("POST", url_), json={"indexed": 1}
        )

    monkeypatch.setattr(ingest.httpx, "AsyncClient", lambda: _FakeClient(handler))

    older = _capture(session.id, url, attempt=1, fetched_at=T1, content="older body")
    newer = _capture(session.id, url, attempt=2, fetched_at=T2, content="newer body")

    bt_old = BackgroundTasks()
    ingest.apply_capture(older, bt_old)
    bt_new = BackgroundTasks()
    ingest.apply_capture(newer, bt_new)

    # Newer forward runs first and wins; the older forward finds itself no longer
    # the latest applied and is marked superseded without POSTing.
    await _run(bt_new)
    await _run(bt_old)

    assert posted == ["newer body"]
    with manager._connect() as conn:
        state = conn.execute(
            "SELECT index_state FROM ingest_captures WHERE capture_id = ?",
            (older.capture_id,),
        ).fetchone()["index_state"]
    assert state == "superseded"


# ---------------------------------------------------------------------------
# 11. Auth fail-closed + misdirected writes
# ---------------------------------------------------------------------------


def test_ingest_auth_is_fail_closed(monkeypatch):
    monkeypatch.delenv("BACKEND_CALLBACK_TOKEN", raising=False)
    monkeypatch.delenv("AI_ENGINE_API_TOKEN", raising=False)
    with pytest.raises(HTTPException) as unconfigured:
        ingest._require_backend_callback_auth(None)
    assert unconfigured.value.status_code == 401

    monkeypatch.setenv("BACKEND_CALLBACK_TOKEN", "secret")
    with pytest.raises(HTTPException) as wrong:
        ingest._require_backend_callback_auth("Bearer nope")
    assert wrong.value.status_code == 401
    assert ingest._require_backend_callback_auth("Bearer secret") is None


def test_ingest_rejects_unknown_session_and_url(tmp_path, monkeypatch):
    manager = SessionManager(db_path=str(tmp_path / "db"))
    session = manager.create_session("misdirected")
    manager.add_urls_to_session(session.id, ["https://example.com/known"])
    monkeypatch.setattr(routes, "session_manager", manager)

    with pytest.raises(HTTPException) as missing_session:
        ingest.apply_capture(
            _capture("no-such-session", "https://example.com/known"), BackgroundTasks()
        )
    assert missing_session.value.status_code == 404

    with pytest.raises(HTTPException) as missing_url:
        ingest.apply_capture(
            _capture(session.id, "https://example.com/unregistered"), BackgroundTasks()
        )
    assert missing_url.value.status_code == 409


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self, handler):
        self._handler = handler

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url, **kwargs):
        return self._handler(url, kwargs)


async def _run(background_tasks: BackgroundTasks) -> None:
    for task in background_tasks.tasks:
        await task()
