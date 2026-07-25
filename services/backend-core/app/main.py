"""Backend Core Service - Main Application."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.config_loader import get_ai_config
from services.cors import allowed_origins
from services.observability import RequestIDMiddleware, configure_logging, log_event

from .api import ingest
from .api.routes import router

configure_logging("backend-core")


def _log_reconcile_done(task: "asyncio.Task") -> None:
    """Startup reconcile is fire-and-forget: log its result, never propagate."""
    try:
        swept = task.result()
    except asyncio.CancelledError:
        return
    except Exception as error:  # noqa: BLE001 - observability only
        log_event("ingest.reconcile_failed", level=logging.ERROR, reason=str(error))
        return
    log_event("ingest.reconcile_startup_complete", swept=swept)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Refuse to start on malformed shared AI config (finding 27).

    Backend Core does not read this config itself, but it ships the same
    ai_models.yaml the AI Engine depends on; failing fast here surfaces a
    broken provider catalog before any request is served, not on first use.
    """
    errors = get_ai_config().validate_config()
    if errors:
        log_event("config.invalid", level=logging.CRITICAL, errors=errors)
        raise RuntimeError("AI model configuration is invalid: " + "; ".join(errors))

    # Durable-outbox recovery: after a crash between "ledger row committed" and
    # "POST to ai-engine", the pending row's BackgroundTask is gone. Sweep once
    # at startup (all sessions, pending+failed) to re-forward orphans and give
    # failed rows one bounded retry. Fire-and-forget so a booting/absent
    # ai-engine never blocks or fails startup.
    reconcile_task = asyncio.create_task(
        ingest.reconcile_pending_forwards(None, ("pending", "failed"), 100)
    )
    reconcile_task.add_done_callback(_log_reconcile_done)
    _app.state.ingest_reconcile_task = reconcile_task
    yield


app = FastAPI(
    title="Tab Organizer - Backend Core",
    description="Backend API and session management for Tab Organizer",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware. Scoped to the Web UI origin, credentials never allowed:
# `allow_origins=["*"]` + `allow_credentials=True` let any site the user had
# open read this service's responses, and the unauthenticated session/url reads
# here return stored page content. Streamlit calls us server-side, so no
# browser-side caller is lost. See services/cors.py.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Starlette wraps middleware in reverse add order (last added = outermost), so
# RequestIDMiddleware must be added last to wrap CORS -- otherwise a CORS
# preflight (OPTIONS) short-circuits inside CORSMiddleware before ever
# reaching this middleware and comes back with no X-Request-ID.
app.add_middleware(RequestIDMiddleware, service="backend-core")

# Include routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"service": "backend-core", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
