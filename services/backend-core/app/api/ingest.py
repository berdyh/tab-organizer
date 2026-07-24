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
from typing import Any, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from services.observability import log_event

from ..sessions.manager import IngestCapture

router = APIRouter()

VALID_INGEST_STATUSES = {"success", "failed", "auth_required", "timeout", "blocked"}

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
    attempt: int = Field(ge=1)
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
            capture_id=capture.capture_id,
        )
        return {
            "status": "ignored",
            "reason": "duplicate_capture_id",
            "capture_id": capture.capture_id,
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
