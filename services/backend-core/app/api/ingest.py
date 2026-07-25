"""Single versioned idempotent ingest endpoint (``POST /api/v1/ingest/v1``).

New code lives outside the ``routes.py`` monolith; ``routes.py`` includes this
router and stays the compatibility aggregator. The endpoint is the single
writer of scraped content into backend persistence: browser-engine (and, at
cutover, the TS capture process) POST capture results here, and backend forwards
applied captures to ai-engine ``/index`` as the sole vector writer.

Contract (see the api/sessions MODULE cards):

* ``200 {"status": "applied", "capture_id", "index": "pending"|"skipped"}``
* ``200 {"status": "ignored", "reason": "duplicate_capture_id"|"stale_capture",
  "capture_id"}`` — the writer did nothing wrong; delivery is at-least-once and
  the receiver owns dedupe, so replay/stale rejection is a 200, never a 4xx that
  browser-engine would miscount as a callback failure.
* ``404`` session not found · ``409`` url not registered in session · ``422``
  malformed payload.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from services.observability import log_event

from ..sessions.manager import IngestCapture

router = APIRouter()

VALID_INGEST_STATUSES = {"success", "failed", "auth_required", "timeout", "blocked"}

# Maximum allowed clock skew for a `fetched_at` reported as being ahead of the
# server's clock. Ordering (`capture_order_key`) sorts newest `fetched_at`
# first, so an unbounded future timestamp would let one poisoned capture
# permanently pin content: it would outrank every subsequent *genuine*
# recapture forever, since a real scraper's `fetched_at` can never catch up to
# an arbitrary future date (security finding C5). Five minutes is generous
# slack for real clock drift between browser-engine and backend-core while
# keeping the pin window bounded and short.
MAX_FUTURE_CLOCK_SKEW = timedelta(minutes=5)

# Upper bound on the per-(session, url) attempt counter (Field `ge=1` below
# sets the lower bound). `attempt` is only a tie-breaker within an identical
# `fetched_at` (see `capture_order_key`), so on its own it is a much weaker
# pinning primitive than an unbounded `fetched_at` -- once `fetched_at` is
# clamped above, an inflated `attempt` can win ties only inside the same
# MAX_FUTURE_CLOCK_SKEW window, not forever. Bounding it anyway costs nothing:
# browser-engine's real counter (`_next_ingest_attempt` in browser-engine's
# main.py) starts at 1 and increments per retry of one URL within a process
# lifetime, so it never legitimately approaches this bound.
MAX_INGEST_ATTEMPT = 100_000


def _now_utc() -> datetime:
    """Indirection point for "now" so tests can inject a fixed, deterministic
    reference instant instead of asserting against wall-clock proximity."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# Per-(session_id, normalized) forward locks serialize LanceDB delete/add so a
# retry forward and a newer-attempt forward cannot interleave. The event loop is
# single-threaded, so a plain dict with get-or-create is race-free.
_forward_locks: dict[tuple[str, str], asyncio.Lock] = {}


class IngestResultV1(BaseModel):
    """Wire model for a single fetch-attempt result (v1 callers only).

    ``attempt >= 1`` is enforced here; the legacy callback adapter builds an
    ``IngestCapture`` with ``attempt=0`` directly, bypassing this validation so
    legacy writes deterministically lose ties against any v1 write.
    """

    capture_id: str
    attempt: int = Field(ge=1, le=MAX_INGEST_ATTEMPT)
    session_id: str
    url: str
    status: str
    content: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    auth_used: bool = False
    fetched_at: str

    @field_validator("status")
    @classmethod
    def _known_status(cls, value: str) -> str:
        if value not in VALID_INGEST_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_INGEST_STATUSES)}")
        return value

    @field_validator("fetched_at")
    @classmethod
    def _valid_fetched_at(cls, value: str) -> str:
        """Reject non-timestamps at the boundary (422) and normalize parseable
        ones to a canonical fixed-width UTC-naive ISO string so stored values are
        homogeneous. Ordering is still parse-based (never lexicographic); this
        just stops garbage like ``fetched_at="zzz"`` from ever entering the
        ledger and pinning poisoned content.

        Also rejects timestamps more than ``MAX_FUTURE_CLOCK_SKEW`` ahead of the
        server clock (422) -- see that constant's comment: an unbounded future
        ``fetched_at`` is the permanent-content-pinning primitive (security
        finding C5)."""
        try:
            parsed = datetime.fromisoformat(value)
        except (ValueError, TypeError):
            raise ValueError("fetched_at must be an RFC3339/ISO8601 timestamp")
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        if parsed > _now_utc() + MAX_FUTURE_CLOCK_SKEW:
            raise ValueError(
                "fetched_at must not be more than "
                f"{MAX_FUTURE_CLOCK_SKEW} ahead of the server clock"
            )
        return parsed.isoformat(timespec="microseconds")


def _require_backend_callback_auth(
    authorization: Optional[str] = Header(default=None),
):
    """Reuse routes.py's fail-closed callback auth (bearer BACKEND_CALLBACK_TOKEN
    with AI-token fallback), late-imported to avoid an import cycle."""
    from . import routes

    return routes._require_backend_callback_auth(authorization)


def _forward_lock(key: tuple[str, str]) -> asyncio.Lock:
    lock = _forward_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _forward_locks[key] = lock
    return lock


def _forward_error_message(response: httpx.Response) -> str:
    detail = None
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("message") or payload.get("error")
    if not detail:
        detail = response.text.strip()
    status = f"{response.status_code} {response.reason_phrase}".strip()
    return f"{status}: {detail}" if detail else status


async def forward_capture_index(
    session_id: str,
    normalized: str,
    capture_id: str,
    document: dict,
) -> None:
    """Forward one applied capture to ai-engine /index as the single writer.

    Runs after the ingest transaction commits (the response already returned
    ``index: pending``). Serializes per key, re-checks the capture is still the
    latest applied before POSTing (else marks it ``superseded`` and skips), and
    records ``indexed``/``failed`` back into the ledger for the status overlay.
    """
    from . import routes

    manager = routes.session_manager
    async with _forward_lock((session_id, normalized)):
        if not manager.is_latest_applied(session_id, normalized, capture_id):
            manager.update_capture_index_state(capture_id, "superseded")
            log_event(
                "ingest.index_superseded",
                session_id=session_id,
                capture_id=capture_id,
            )
            return
        # Re-read the ledger state AFTER acquiring the lock: a sweep racing the
        # original BackgroundTask (or a duplicate re-forward) must not POST twice
        # once the row is already indexed/superseded. ai-engine upserts by
        # document id, so a residual double-forward is at worst duplicate work,
        # never a stale vector — but this closes the common case.
        if manager.get_capture_index_state(capture_id) not in ("pending", "failed"):
            log_event(
                "ingest.index_already_settled",
                session_id=session_id,
                capture_id=capture_id,
            )
            return
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{routes._ai_engine_url()}/index",
                    json={"session_id": session_id, "documents": [document]},
                    headers=routes._ai_engine_request_headers(),
                    timeout=120.0,
                )
                if response.is_error:
                    raise RuntimeError(_forward_error_message(response))
            manager.update_capture_index_state(capture_id, "indexed")
            log_event(
                "ingest.index_forwarded",
                session_id=session_id,
                capture_id=capture_id,
            )
        except Exception as error:  # noqa: BLE001 - record, never crash the task
            manager.update_capture_index_state(capture_id, "failed", error=str(error))
            log_event(
                "ingest.index_failed",
                level=logging.ERROR,
                session_id=session_id,
                capture_id=capture_id,
                reason=str(error),
            )


async def reconcile_pending_forwards(
    session_id: Optional[str],
    states: tuple[str, ...] = ("pending",),
    limit: int = 100,
    exclude: Optional[str] = None,
) -> int:
    """Sweep the ingest ledger (the durable outbox) for un-forwarded captures.

    Recovers the crash-after-commit-before-forward orphan (a ``pending`` row
    whose BackgroundTask never ran because the process died) and gives ``failed``
    rows a bounded retry. Runs opportunistically at end of ``apply_capture``
    (session-scoped, pending-only) and once at startup (all sessions,
    pending+failed). Never raises out — ``forward_capture_index`` records its own
    failures — so it is safe as a fire-and-forget startup task.
    """
    from . import routes

    manager = routes.session_manager
    rows = manager.list_unforwarded_captures(session_id, states, limit)
    swept = 0
    for row in rows:
        if exclude is not None and row["capture_id"] == exclude:
            continue
        document = manager.build_forward_document(
            row["session_id"], row["normalized"], row["capture_id"]
        )
        if document is None:
            manager.update_capture_index_state(row["capture_id"], "superseded")
            continue
        await forward_capture_index(
            row["session_id"], row["normalized"], row["capture_id"], document
        )
        swept += 1
    if swept:
        log_event(
            "ingest.reconcile_swept",
            session_scope=session_id or "all",
            swept=swept,
            considered=len(rows),
        )
    return swept


def apply_capture(
    capture: IngestCapture,
    background_tasks: Optional[BackgroundTasks],
) -> dict:
    """Run the ingest decision and schedule any forward. Shared by the v1 route
    and the legacy callback adapter. Raises HTTPException for 404/409."""
    from . import routes

    outcome = routes.session_manager.ingest_scrape_result(capture)

    if outcome.outcome in ("session_not_found", "url_not_registered"):
        # WI0 B8: a misdirected ingest must be a visible structured record, not
        # just an HTTP status.
        log_event(
            "ingest.rejected",
            level=logging.WARNING,
            session_id=capture.session_id,
            capture_id=capture.capture_id,
            reason=outcome.outcome,
        )
        if outcome.outcome == "session_not_found":
            raise HTTPException(status_code=404, detail="Session not found")
        raise HTTPException(status_code=409, detail="URL not registered in session")

    if outcome.should_forward and background_tasks is not None:
        background_tasks.add_task(
            forward_capture_index,
            outcome.session_id or capture.session_id,
            outcome.normalized,
            outcome.capture_id,
            outcome.forward_document,
        )

    # Opportunistic outbox sweep: an applied/duplicate ingest is a cheap moment
    # to heal any OTHER orphaned pending forward in the same session (e.g. a row
    # whose BackgroundTask died with the process). Pending-only + session-scoped
    # + excluding the row we just scheduled -> no retry storm against a down
    # ai-engine (failed rows heal via duplicate-receipt retry + startup sweep).
    if background_tasks is not None and outcome.outcome in ("applied", "duplicate"):
        background_tasks.add_task(
            reconcile_pending_forwards,
            capture.session_id,
            ("pending",),
            10,
            outcome.capture_id,
        )

    if outcome.outcome == "stale":
        log_event(
            "ingest.ignored_stale",
            session_id=capture.session_id,
            capture_id=capture.capture_id,
        )
        return {
            "status": "ignored",
            "reason": "stale_capture",
            "capture_id": capture.capture_id,
        }
    if outcome.outcome == "duplicate":
        log_event(
            "ingest.ignored_duplicate",
            session_id=capture.session_id,
            capture_id=outcome.capture_id,
        )
        # ``index`` is an additive summary of the STORED capture's ledger state,
        # never an echo of the input. ``capture_id`` is the stored id.
        return {
            "status": "ignored",
            "reason": "duplicate_capture_id",
            "capture_id": outcome.capture_id,
            "index": outcome.index_state,
        }

    log_event(
        "ingest.applied",
        session_id=capture.session_id,
        capture_id=capture.capture_id,
        scrape_status=capture.status,
        index=outcome.index_state,
        content_length=len(capture.content) if capture.content else 0,
    )
    return {
        "status": "applied",
        "capture_id": capture.capture_id,
        "index": "pending" if outcome.should_forward else "skipped",
    }


@router.post("/ingest/v1")
def ingest_result_v1(
    result: IngestResultV1,
    background_tasks: BackgroundTasks,
    _auth=Depends(_require_backend_callback_auth),
):
    """Idempotent, versioned ingest of one capture result."""
    capture = IngestCapture(
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
    return apply_capture(capture, background_tasks)
