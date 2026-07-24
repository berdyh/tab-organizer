"""Tests for the single versioned idempotent ingest endpoint.

Covers the replay/ordering rule (capture_id uniqueness + newest-wins
(fetched_at, attempt)), the single-writer forward to ai-engine /index, the
status overlay, in-memory/SQLite parity, and the legacy callback adapter.
"""

import threading
import time
import uuid
from datetime import datetime, timedelta

import httpx
import pytest
from fastapi import BackgroundTasks, HTTPException
from pydantic import ValidationError

from services.backend_core.app.api import ingest, routes
from services.backend_core.app.sessions.manager import (
    IngestCapture,
    SessionManager,
    capture_order_key,
)

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
    manager = (
        SessionManager(db_path=str(tmp_path / "db")) if persist else SessionManager()
    )
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
    manager = (
        SessionManager(db_path=str(tmp_path / "db")) if persist else SessionManager()
    )
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
    assert not manager.is_latest_applied(
        session.id, record.normalized, older.capture_id
    )


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
        cap2 = _capture(
            session.id, url, attempt=2, fetched_at=T2, content="second body"
        )
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

        # Parity also covers the duplicate-outcome forward doc (built from stored
        # state) and the latest-only index counts.
        manager.update_capture_index_state(cap2.capture_id, "failed")
        dup = manager.ingest_scrape_result(cap2)
        doc = dict(dup.forward_document)
        doc["metadata"] = {
            k: v for k, v in doc["metadata"].items() if k != "capture_id"
        }
        counts = manager.capture_index_counts(session.id)
        # normalize the per-url downstream_errors url to the same key for compare
        return outcomes, record.status, record.metadata["content"], hits, doc, counts

    assert sequence(mem) == sequence(sql)
    outcomes, status, content, hits, doc, counts = sequence(mem)
    assert outcomes == ["applied", "duplicate", "applied", "stale"]
    assert status == "scraped"
    assert content == "second body"
    assert url in hits
    assert doc["content"] == "second body"
    assert doc["metadata"] == {"title": "Title", "auth_used": False}
    assert counts["ai_index_failed"] == 1
    assert counts["ai_index_pending"] == 0


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
    monkeypatch.setattr(routes, "log_event", lambda name, **kw: events.append(name))

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


def _ok_client(posted):
    def handler(url_, kwargs):
        posted.append(kwargs["json"]["documents"][0])
        return httpx.Response(
            200, request=httpx.Request("POST", url_), json={"indexed": 1}
        )

    return lambda: _FakeClient(handler)


# ---------------------------------------------------------------------------
# Round 2: ordering-race, stored-capture duplicate, fetched_at validation,
# legacy tier, durable outbox sweep, latest-only counts, search parity,
# snapshot rollback.
# ---------------------------------------------------------------------------


def test_two_writer_race_older_write_cannot_clobber(tmp_path):
    """Two SessionManager instances on one DB: the older writer holds the SQLite
    write lock (BEGIN IMMEDIATE) across the ordering check, so the newer write
    that lands afterward wins and is_latest_applied agrees with actual content.
    Pre-fix the check ran outside the transaction and the older write could
    clobber the newer one."""
    db = str(tmp_path / "race.db")
    setup = SessionManager(db_path=db)
    session = setup.create_session("race")
    url = "https://example.com/race"
    setup.add_urls_to_session(session.id, [url])

    mgr_old = SessionManager(db_path=db)
    mgr_new = SessionManager(db_path=db)

    older = _capture(session.id, url, attempt=1, fetched_at=T1, content="older body")
    newer = _capture(session.id, url, attempt=2, fetched_at=T2, content="newer body")

    started = threading.Event()
    original_lookup = mgr_old._latest_applied_lookup

    def slow_lookup(conn, session_id, normalized):
        started.set()
        time.sleep(0.4)
        return original_lookup(conn, session_id, normalized)

    mgr_old._latest_applied_lookup = slow_lookup

    def run_old():
        mgr_old.ingest_scrape_result(older)

    t_old = threading.Thread(target=run_old)
    t_old.start()
    assert started.wait(2.0)  # mgr_old now holds BEGIN IMMEDIATE and is sleeping

    t_new = threading.Thread(target=lambda: mgr_new.ingest_scrape_result(newer))
    t_new.start()
    t_old.join(5.0)
    t_new.join(5.0)

    fresh = SessionManager(db_path=db)
    record = fresh.get_session(session.id).url_store.get(url)
    assert record.metadata["content"] == "newer body"
    assert fresh.is_latest_applied(session.id, record.normalized, newer.capture_id)
    assert not fresh.is_latest_applied(session.id, record.normalized, older.capture_id)


@pytest.mark.parametrize("persist", [False, True])
def test_duplicate_replay_forwards_stored_capture_not_mutated_body(tmp_path, persist):
    """A duplicate delivery with mutated content/auth/url must forward the
    ORIGINALLY STORED body, never the replay's."""
    manager = (
        SessionManager(db_path=str(tmp_path / "dup.db"))
        if persist
        else SessionManager()
    )
    session = manager.create_session("dup")
    url_a = "https://example.com/a"
    url_b = "https://example.com/b"
    manager.add_urls_to_session(session.id, [url_a, url_b])

    cid = str(uuid.uuid4())
    cap = _capture(
        session.id, url_a, capture_id=cid, content="AAA stored", auth_used=False
    )
    applied = manager.ingest_scrape_result(cap)
    assert applied.outcome == "applied"
    manager.update_capture_index_state(cid, "failed")

    normalized_a = manager.get_session(session.id).url_store.get(url_a).normalized

    # Replay: same capture_id, but mutated body / flipped auth / different url.
    replay = _capture(
        session.id, url_b, capture_id=cid, content="BBB mutated", auth_used=True
    )
    dup = manager.ingest_scrape_result(replay)

    assert dup.outcome == "duplicate"
    assert dup.index_state == "failed"
    assert dup.should_forward is True
    assert dup.forward_document["content"] == "AAA stored"
    assert dup.forward_document["id"] == normalized_a
    assert dup.forward_document["metadata"]["auth_used"] is False
    # stored record untouched by the replay
    assert (
        manager.get_session(session.id).url_store.get(url_a).metadata["content"]
        == "AAA stored"
    )


@pytest.mark.parametrize("bad", ["zzz", "", "2026-13-45T00:00:00", "not-a-date"])
def test_ingest_v1_rejects_malformed_fetched_at_422(bad):
    """Unparseable fetched_at is a pydantic ValidationError (FastAPI 422)."""
    with pytest.raises(ValidationError):
        ingest.IngestResultV1(
            capture_id="c1",
            attempt=1,
            session_id="s1",
            url="https://example.com",
            status="success",
            fetched_at=bad,
        )


def test_fetched_at_offsets_normalized_and_compared_chronologically(tmp_path):
    """+02:00 input normalizes to UTC-naive; ordering is real-time not string
    order; a garbage fetched_at ledger row ranks below any parseable capture."""
    model = ingest.IngestResultV1(
        capture_id="c1",
        attempt=1,
        session_id="s1",
        url="https://example.com",
        status="success",
        fetched_at="2026-07-24T12:00:00+02:00",
    )
    assert model.fetched_at == "2026-07-24T10:00:00.000000"

    # 10:00 UTC (from +02:00) is EARLIER than 11:00 UTC, though its raw string
    # sorts lexicographically LATER — comparison must be chronological.
    assert capture_order_key("2026-07-24T12:00:00+02:00", 1) < capture_order_key(
        "2026-07-24T11:00:00", 1
    )
    # garbage loses to any real timestamp
    assert capture_order_key("zzz", 9) < capture_order_key("2026-07-24T00:00:00", 1)

    # end to end: a pre-seeded garbage applied row does not pin content — a real
    # (even old) capture outranks it and wins.
    manager = SessionManager(db_path=str(tmp_path / "poison.db"))
    session = manager.create_session("poison")
    url = "https://example.com/poison"
    manager.add_urls_to_session(session.id, [url])
    manager.ingest_scrape_result(
        _capture(session.id, url, attempt=1, fetched_at="garbage", content="poison")
    )
    real = manager.ingest_scrape_result(
        _capture(
            session.id, url, attempt=1, fetched_at="2020-01-01T00:00:00", content="real"
        )
    )
    assert real.outcome == "applied"
    assert manager.get_session(session.id).url_store.get(url).metadata["content"] == (
        "real"
    )


# ---------------------------------------------------------------------------
# 12. Future fetched_at clamp (security finding C5: permanent-pinning fix)
#
# `_valid_fetched_at` now rejects a `fetched_at` more than
# `ingest.MAX_FUTURE_CLOCK_SKEW` ahead of the server clock. Boundary
# assertions inject a fixed reference instant via `ingest._now_utc` (a
# monkeypatchable indirection point added for exactly this purpose) instead
# of comparing against the real wall clock, so these tests cannot go flaky.
# ---------------------------------------------------------------------------

FIXED_NOW = datetime(2026, 7, 24, 12, 0, 0)


@pytest.fixture
def frozen_now(monkeypatch):
    monkeypatch.setattr(ingest, "_now_utc", lambda: FIXED_NOW)
    return FIXED_NOW


def _build_v1(**overrides):
    fields = dict(
        capture_id=str(uuid.uuid4()),
        attempt=1,
        session_id="s1",
        url="https://example.com",
        status="success",
        content="body",
        fetched_at=T1,
    )
    fields.update(overrides)
    return ingest.IngestResultV1(**fields)


def test_ingest_v1_rejects_far_future_fetched_at_422():
    """The exploit payload from the C5 finding: an arbitrarily far-future
    fetched_at must be rejected at the pydantic boundary (422), not stored."""
    with pytest.raises(ValidationError):
        _build_v1(fetched_at="9999-12-31T23:59:59")


def test_fetched_at_exactly_at_skew_boundary_is_accepted(frozen_now):
    boundary = (frozen_now + ingest.MAX_FUTURE_CLOCK_SKEW).isoformat()
    model = _build_v1(fetched_at=boundary)
    assert model.fetched_at == boundary + ".000000"


def test_fetched_at_just_inside_skew_boundary_is_accepted(frozen_now):
    inside = (
        frozen_now + ingest.MAX_FUTURE_CLOCK_SKEW - timedelta(seconds=1)
    ).isoformat()
    model = _build_v1(fetched_at=inside)
    assert model.fetched_at == inside + ".000000"


def test_fetched_at_just_beyond_skew_boundary_is_rejected(frozen_now):
    beyond = (
        frozen_now + ingest.MAX_FUTURE_CLOCK_SKEW + timedelta(seconds=1)
    ).isoformat()
    with pytest.raises(ValidationError):
        _build_v1(fetched_at=beyond)


def test_fetched_at_at_current_time_is_unaffected(frozen_now):
    """A normal, non-future fetched_at (== "now") is untouched by the clamp."""
    model = _build_v1(fetched_at=frozen_now.isoformat())
    assert model.fetched_at == frozen_now.isoformat() + ".000000"


def test_ingest_v1_rejects_attempt_above_bound():
    """`attempt` is capped too (defense-in-depth tie-breaker bound, see
    MAX_INGEST_ATTEMPT's comment); a nonsensical attempt is rejected at 422
    just like an unbounded future fetched_at."""
    with pytest.raises(ValidationError):
        _build_v1(attempt=ingest.MAX_INGEST_ATTEMPT + 1)
    # the bound itself is still a legal attempt value
    assert _build_v1(attempt=ingest.MAX_INGEST_ATTEMPT).attempt == (
        ingest.MAX_INGEST_ATTEMPT
    )


def test_pinning_attack_rejected_end_to_end(tmp_path, monkeypatch, frozen_now):
    """Full regression for security finding C5. Pre-fix: a caller with the
    callback token could POST a far-future `fetched_at` (e.g. year 9999) that
    got applied, wrote to the FTS index / RAG corpus, and then permanently
    out-ranked every subsequent genuine recapture in `capture_order_key`
    forever (silently dropped as `ignored: stale_capture`, 200 OK). Post-fix:
    the poisoned payload is rejected before it ever reaches the ledger, so it
    never applies and never pins -- a following genuine capture (and the one
    after that) both apply normally."""
    manager = SessionManager(db_path=str(tmp_path / "pin.db"))
    session = manager.create_session("pin")
    url = "https://example.com/pin"
    manager.add_urls_to_session(session.id, [url])
    monkeypatch.setattr(routes, "session_manager", manager)

    def as_capture(result: "ingest.IngestResultV1") -> IngestCapture:
        return IngestCapture(
            capture_id=result.capture_id,
            attempt=result.attempt,
            session_id=result.session_id,
            url=result.url,
            status=result.status,
            content=result.content,
            metadata=result.metadata,
            auth_used=result.auth_used,
            fetched_at=result.fetched_at,
        )

    # 1. The attacker's payload (far-future fetched_at) is rejected at
    #    construction time -- it never reaches apply_capture, so it can never
    #    be applied and can never pin. `attempt` is deliberately an ordinary
    #    in-bound value here so this assertion isolates the fetched_at clamp
    #    specifically, independent of the separate attempt-bound check.
    with pytest.raises(ValidationError):
        _build_v1(
            session_id=session.id,
            url=url,
            content="attacker poison",
            fetched_at="9999-12-31T23:59:59",
            attempt=1,
        )
    record = manager.get_session(session.id).url_store.get(url)
    assert record.status == "pending"
    assert "content" not in record.metadata

    # 2. A genuine capture at (the frozen) "now" applies normally.
    genuine = as_capture(
        _build_v1(
            session_id=session.id,
            url=url,
            content="legit content",
            fetched_at=frozen_now.isoformat(),
        )
    )
    outcome = ingest.apply_capture(genuine, BackgroundTasks())
    assert outcome["status"] == "applied"
    assert (
        manager.get_session(session.id).url_store.get(url).metadata["content"]
        == "legit content"
    )

    # 3. And the pinning attack's whole point -- EVERY subsequent genuine
    #    recapture, forever -- also still applies: a second, later-still
    #    genuine capture supersedes the first.
    later = as_capture(
        _build_v1(
            session_id=session.id,
            url=url,
            content="even newer legit content",
            fetched_at=(frozen_now + timedelta(seconds=1)).isoformat(),
            attempt=2,
        )
    )
    outcome2 = ingest.apply_capture(later, BackgroundTasks())
    assert outcome2["status"] == "applied"
    assert (
        manager.get_session(session.id).url_store.get(url).metadata["content"]
        == "even newer legit content"
    )


@pytest.mark.parametrize("persist", [False, True])
def test_legacy_capture_never_outranks_v1_regardless_of_clock(tmp_path, persist):
    """Legacy (attempt=0) is its own losing tier: a far-future legacy fetched_at
    still loses to an older v1 capture, in both orders."""

    def fresh():
        manager = (
            SessionManager(db_path=str(tmp_path / f"legacy-{uuid.uuid4()}.db"))
            if persist
            else SessionManager()
        )
        session = manager.create_session("legacy tier")
        url = "https://example.com/legacy-tier"
        manager.add_urls_to_session(session.id, [url])
        return manager, session, url

    # legacy first, then v1
    m1, s1, u1 = fresh()
    m1.ingest_scrape_result(
        _capture(
            s1.id, u1, attempt=0, fetched_at="2099-01-01T00:00:00", content="legacy"
        )
    )
    v1 = m1.ingest_scrape_result(
        _capture(s1.id, u1, attempt=1, fetched_at="2026-07-24T10:00:00", content="v1")
    )
    assert v1.outcome == "applied"
    assert m1.get_session(s1.id).url_store.get(u1).metadata["content"] == "v1"

    # v1 first, then legacy
    m2, s2, u2 = fresh()
    m2.ingest_scrape_result(
        _capture(s2.id, u2, attempt=1, fetched_at="2026-07-24T10:00:00", content="v1")
    )
    legacy = m2.ingest_scrape_result(
        _capture(
            s2.id, u2, attempt=0, fetched_at="2099-01-01T00:00:00", content="legacy"
        )
    )
    assert legacy.outcome == "stale"
    assert m2.get_session(s2.id).url_store.get(u2).metadata["content"] == "v1"


@pytest.mark.asyncio
async def test_startup_reconcile_forwards_orphaned_pending(tmp_path, monkeypatch):
    """Crash between commit and forward leaves a pending ledger row with no task;
    a restart-time reconcile sweep re-forwards it."""
    db = str(tmp_path / "orphan.db")
    mgr = SessionManager(db_path=db)
    session = mgr.create_session("orphan")
    url = "https://example.com/orphan"
    mgr.add_urls_to_session(session.id, [url])
    cid = str(uuid.uuid4())
    monkeypatch.setattr(routes, "session_manager", mgr)

    # crash-equivalent: apply with no BackgroundTasks -> pending, no forward.
    ingest.apply_capture(_capture(session.id, url, capture_id=cid), None)
    assert mgr.get_capture_index_state(cid) == "pending"

    # restart-equivalent: a fresh manager on the same DB runs the sweep.
    fresh = SessionManager(db_path=db)
    monkeypatch.setattr(routes, "session_manager", fresh)
    posted = []
    monkeypatch.setattr(ingest.httpx, "AsyncClient", _ok_client(posted))

    swept = await ingest.reconcile_pending_forwards(None, ("pending", "failed"), 100)
    assert swept == 1
    assert len(posted) == 1
    assert fresh.get_capture_index_state(cid) == "indexed"


@pytest.mark.asyncio
async def test_opportunistic_sweep_recovers_pending_on_next_ingest(
    tmp_path, monkeypatch
):
    """An orphaned pending forward for url1 is healed when url2 is ingested."""
    mgr = SessionManager(db_path=str(tmp_path / "sweep.db"))
    session = mgr.create_session("sweep")
    url1 = "https://example.com/one"
    url2 = "https://example.com/two"
    mgr.add_urls_to_session(session.id, [url1, url2])
    monkeypatch.setattr(routes, "session_manager", mgr)

    cid1 = str(uuid.uuid4())
    ingest.apply_capture(_capture(session.id, url1, capture_id=cid1), None)  # orphan
    assert mgr.get_capture_index_state(cid1) == "pending"

    posted = []
    monkeypatch.setattr(ingest.httpx, "AsyncClient", _ok_client(posted))

    norm1 = mgr.get_session(session.id).url_store.get(url1).normalized
    bt = BackgroundTasks()
    ingest.apply_capture(_capture(session.id, url2, capture_id=str(uuid.uuid4())), bt)
    await _run(bt)

    posted_ids = [doc["id"] for doc in posted]
    assert norm1 in posted_ids
    assert mgr.get_capture_index_state(cid1) == "indexed"


@pytest.mark.asyncio
async def test_reconcile_marks_non_latest_superseded(tmp_path, monkeypatch):
    """A pending orphan superseded by a newer applied capture is marked
    superseded during the sweep and never POSTed."""
    mgr = SessionManager(db_path=str(tmp_path / "supersweep.db"))
    session = mgr.create_session("supersweep")
    url = "https://example.com/s"
    mgr.add_urls_to_session(session.id, [url])
    monkeypatch.setattr(routes, "session_manager", mgr)

    old_id = str(uuid.uuid4())
    new_id = str(uuid.uuid4())
    ingest.apply_capture(
        _capture(
            session.id,
            url,
            capture_id=old_id,
            attempt=1,
            fetched_at=T1,
            content="older",
        ),
        None,
    )
    ingest.apply_capture(
        _capture(
            session.id,
            url,
            capture_id=new_id,
            attempt=2,
            fetched_at=T2,
            content="newer",
        ),
        None,
    )

    posted = []
    monkeypatch.setattr(ingest.httpx, "AsyncClient", _ok_client(posted))
    await ingest.reconcile_pending_forwards(None, ("pending", "failed"), 100)

    assert mgr.get_capture_index_state(old_id) == "superseded"
    assert mgr.get_capture_index_state(new_id) == "indexed"
    assert [doc["content"] for doc in posted] == ["newer"]


@pytest.mark.asyncio
async def test_forward_skips_when_already_indexed_after_lock(tmp_path, monkeypatch):
    """Two forwards scheduled for the same capture: the second exits on the
    post-lock ledger-state re-check without a second POST."""
    mgr = SessionManager(db_path=str(tmp_path / "double.db"))
    session = mgr.create_session("double")
    url = "https://example.com/d"
    mgr.add_urls_to_session(session.id, [url])
    monkeypatch.setattr(routes, "session_manager", mgr)

    cid = str(uuid.uuid4())
    outcome = mgr.ingest_scrape_result(_capture(session.id, url, capture_id=cid))
    normalized = outcome.normalized
    doc = outcome.forward_document

    posted = []
    monkeypatch.setattr(ingest.httpx, "AsyncClient", _ok_client(posted))

    await ingest.forward_capture_index(session.id, normalized, cid, doc)
    await ingest.forward_capture_index(session.id, normalized, cid, doc)

    assert len(posted) == 1
    assert mgr.get_capture_index_state(cid) == "indexed"


@pytest.mark.parametrize("persist", [False, True])
def test_index_counts_reflect_only_latest_applied_capture(tmp_path, persist):
    """attempt 1 failed then attempt 2 indexed -> no stale failure reported."""
    manager = (
        SessionManager(db_path=str(tmp_path / "counts.db"))
        if persist
        else SessionManager()
    )
    session = manager.create_session("counts")
    url = "https://example.com/counts"
    manager.add_urls_to_session(session.id, [url])

    cid1 = str(uuid.uuid4())
    cid2 = str(uuid.uuid4())
    manager.ingest_scrape_result(
        _capture(
            session.id, url, capture_id=cid1, attempt=1, fetched_at=T1, content="v1"
        )
    )
    manager.update_capture_index_state(cid1, "failed")
    manager.ingest_scrape_result(
        _capture(
            session.id, url, capture_id=cid2, attempt=2, fetched_at=T2, content="v2"
        )
    )
    manager.update_capture_index_state(cid2, "indexed")

    counts = manager.capture_index_counts(session.id)
    assert counts["ai_index_failed"] == 0
    assert counts["ai_index_pending"] == 0
    assert counts["downstream_errors"] == []


@pytest.mark.parametrize("probe", ["findmepending", "zzuniquezz"])
def test_keyword_search_parity_pending_and_url_only_matches_excluded(tmp_path, probe):
    """In-memory keyword search matches the SQLite FTS row set exactly: a
    pending-status record and a URL-only term hit are excluded by both."""

    def build(manager):
        session = manager.create_session("parity search")
        url_pending = "https://example.com/pendingdoc"
        url_scraped = "https://zzuniquezz.example.com/page"
        manager.add_urls_to_session(session.id, [url_pending, url_scraped])
        # pending record with matching metadata but non-scraped status
        manager.update_url_status(
            session.id,
            url_pending,
            "pending",
            metadata={"title": "findmepending", "content": "findmepending body"},
        )
        # scraped record whose term appears ONLY in the URL (not title/content)
        manager.ingest_scrape_result(
            _capture(
                session.id,
                url_scraped,
                metadata={"title": "Regular", "content": "regular body about widgets"},
            )
        )
        return session

    mem = SessionManager()
    sql = SessionManager(db_path=str(tmp_path / "search.db"))
    s_mem = build(mem)
    s_sql = build(sql)

    mem_hits = [h["url"] for h in mem.search_indexed_tabs(s_mem.id, probe, 10)]
    sql_hits = [h["url"] for h in sql.search_indexed_tabs(s_sql.id, probe, 10)]
    assert mem_hits == sql_hits == []


def test_forced_write_failure_keeps_memory_and_sqlite_consistent(tmp_path):
    """A forced SQLite write failure rolls back BOTH the in-memory record and
    the transaction: status stays pending, no applied ledger row."""
    db = str(tmp_path / "rollback.db")
    mgr = SessionManager(db_path=db)
    session = mgr.create_session("rollback")
    url = "https://example.com/rollback"
    mgr.add_urls_to_session(session.id, [url])

    cid = str(uuid.uuid4())

    def boom(*args, **kwargs):
        raise RuntimeError("forced write failure")

    mgr._save_url_record = boom

    with pytest.raises(RuntimeError):
        mgr.ingest_scrape_result(_capture(session.id, url, capture_id=cid))

    # in-memory record restored to pre-call state
    assert mgr.get_session(session.id).url_store.get(url).status == "pending"

    # DB rolled back: url still pending, no applied ledger row
    fresh = SessionManager(db_path=db)
    assert fresh.get_session(session.id).url_store.get(url).status == "pending"
    with fresh._connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM ingest_captures WHERE capture_id = ?", (cid,)
        ).fetchone()["n"]
    assert count == 0
