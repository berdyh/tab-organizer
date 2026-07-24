"""Browser Engine Service - Main Application."""

import asyncio
import hmac
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.config_loader import get_ai_config
from services.observability import (
    RequestIDMiddleware,
    configure_logging,
    get_request_id,
    log_event,
    request_id_headers,
    reset_request_id,
    set_request_id,
)
from services.url_safety import validate_scrape_url

from .auth.detector import AuthDetector
from .auth.queue import AuthQueue, CredentialStoreError
from .scraper.engine import ScraperEngine
from .tabs.cdp import DEFAULT_CDP_URL, CDPTabHarvester

configure_logging("browser-engine")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Refuse to start on malformed shared AI config (finding 27).

    Browser Engine does not read this config itself, but it ships the same
    ai_models.yaml the AI Engine depends on; failing fast here surfaces a
    broken provider catalog before any request is served, not on first use.
    """
    errors = get_ai_config().validate_config()
    if errors:
        log_event("config.invalid", level=logging.CRITICAL, errors=errors)
        raise RuntimeError("AI model configuration is invalid: " + "; ".join(errors))
    yield


app = FastAPI(
    title="Tab Organizer - Browser Engine",
    description="Web scraping and authentication handling",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Starlette wraps middleware in reverse add order (last added = outermost), so
# RequestIDMiddleware must be added last to wrap CORS -- otherwise a CORS
# preflight (OPTIONS) short-circuits inside CORSMiddleware before ever
# reaching this middleware and comes back with no X-Request-ID.
app.add_middleware(RequestIDMiddleware, service="browser-engine")

# Global instances
auth_detector = AuthDetector()
auth_queue = AuthQueue()


def _new_scraper_engine() -> ScraperEngine:
    engine = ScraperEngine(
        max_concurrent=int(os.getenv("MAX_CONCURRENT_SCRAPES", 10)),
        timeout=int(os.getenv("SCRAPE_TIMEOUT", 30)),
        respect_robots=os.getenv("RESPECT_ROBOTS", "true").lower() == "true",
    )
    engine.set_auth_queue(auth_queue)
    engine.set_auth_detector(auth_detector)
    return engine


scraper = _new_scraper_engine()

# Scraping state
scraping_tasks: dict[str, dict] = {}  # session_id → task info
MAX_RECORDED_DOWNSTREAM_ERRORS = 20

# Per-(session_id, url) attempt counter for the idempotent ingest endpoint.
# In-process only: it resets to 1 across restarts/re-dispatch, which is why the
# backend ordering rule keys on (fetched_at, attempt), not attempt alone.
_ingest_attempts: dict[tuple[str, str], int] = {}


def _next_ingest_attempt(session_id: str, url: str) -> int:
    key = (session_id, url)
    attempt = _ingest_attempts.get(key, 0) + 1
    _ingest_attempts[key] = attempt
    return attempt


# Request models
class ScrapeRequest(BaseModel):
    session_id: str
    urls: list[str]
    use_browser: bool = False


class CredentialsRequest(BaseModel):
    domain: str
    credentials: dict


class SingleScrapeRequest(BaseModel):
    url: str
    session_id: Optional[str] = None
    use_browser: bool = False


class TabImportRequest(BaseModel):
    cdp_url: str = DEFAULT_CDP_URL
    max_tabs: int = 2000
    max_concurrent: int = 25


class TabOpenRequest(BaseModel):
    urls: list[str]
    cdp_url: str = DEFAULT_CDP_URL


def _browser_engine_token() -> str:
    """Resolve the token required for Browser Engine control endpoints."""
    return (
        os.getenv("BROWSER_ENGINE_API_TOKEN", "").strip()
        or os.getenv("BACKEND_CALLBACK_TOKEN", "").strip()
        or os.getenv("AI_ENGINE_API_TOKEN", "").strip()
    )


def _require_browser_engine_auth(
    authorization: Optional[str] = Header(default=None),
):
    """Require a shared bearer token for browser control/auth endpoints."""
    expected = _browser_engine_token()
    if not expected:
        raise HTTPException(
            status_code=401,
            detail=(
                "Browser Engine token is not configured; set "
                "BROWSER_ENGINE_API_TOKEN or run scripts/cli.py start"
            ),
        )
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token.strip(), expected):
        raise HTTPException(status_code=401, detail="Invalid browser engine token")


def _validate_scrape_urls(urls: list[str]) -> None:
    """Reject unsafe outbound scrape targets before network dispatch."""
    for url in urls:
        try:
            validate_scrape_url(url)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))


# Health check
@app.get("/")
async def root():
    return {
        "service": "browser-engine",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


# Scraping endpoints
@app.post("/scrape")
async def start_scraping(
    request: ScrapeRequest,
    background_tasks: BackgroundTasks,
    _auth=Depends(_require_browser_engine_auth),
):
    """Start scraping URLs in the background."""
    session_id = request.session_id
    _validate_scrape_urls(request.urls)

    # Track scraping task
    scraping_tasks[session_id] = _new_scrape_task_info(len(request.urls))

    log_event(
        "scrape.received",
        session_id=session_id,
        url_count=len(request.urls),
        use_browser=request.use_browser,
    )

    # Start background scraping; carry the request id into the detached task so
    # its callbacks and index calls stay correlated with this request.
    background_tasks.add_task(
        scrape_urls_background,
        session_id,
        request.urls,
        request.use_browser,
        get_request_id(),
    )

    return {
        "status": "started",
        "session_id": session_id,
        "url_count": len(request.urls),
    }


def _new_scrape_task_info(total: int) -> dict:
    """Create scrape status state exposed by /scrape/status."""
    return {
        "total": total,
        "completed": 0,
        "success": 0,
        "failed": 0,
        "auth_required": 0,
        "status": "running",
        "backend_callback_failed": 0,
        "ai_index_failed": 0,
        "downstream_error_count": 0,
        "downstream_errors": [],
    }


def _record_downstream_error(
    task_info: dict,
    source: str,
    message: str,
    url: Optional[str] = None,
) -> None:
    """Record an ingest-callback failure without aborting the scrape batch.

    Browser-engine no longer writes vectors: it POSTs each result to the
    backend ingest endpoint and the backend is the single ai-engine /index
    writer. Ingest is per capture, so a failed callback affects exactly one
    document. AI-index failures now live in the backend ledger and surface via
    the backend `/scrape/status` overlay, not here.
    """
    task_info["downstream_error_count"] = task_info.get("downstream_error_count", 0) + 1

    if source == "backend_callback":
        task_info["backend_callback_failed"] = (
            task_info.get("backend_callback_failed", 0) + 1
        )

    errors = task_info.setdefault("downstream_errors", [])
    if len(errors) < MAX_RECORDED_DOWNSTREAM_ERRORS:
        error = {"source": source, "message": message}
        if url:
            error["url"] = url
        errors.append(error)
    else:
        task_info["downstream_errors_truncated"] = True


def _http_response_error_message(response: httpx.Response) -> str:
    """Preserve JSON/text error details from downstream service responses."""
    detail = None
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("message") or payload.get("error")
    elif payload is not None:
        detail = str(payload)

    if not detail:
        detail = response.text.strip()

    status = f"{response.status_code} {response.reason_phrase}".strip()
    if detail:
        return f"{status}: {detail}"
    return status


def _finalize_scrape_status(task_info: dict) -> None:
    """Mark final scrape status, preserving downstream failure visibility."""
    if task_info.get("downstream_error_count", 0):
        task_info["status"] = "completed_with_downstream_errors"
        task_info["error"] = (
            f"{task_info['downstream_error_count']} downstream operation(s) failed"
        )
    else:
        task_info["status"] = "completed"


def _result_domain(result) -> Optional[str]:
    """Best-effort host for auth-queue log lines (never the full URL/query)."""
    domain = result.metadata.get("domain") if result.metadata else None
    if domain:
        return domain
    try:
        from urllib.parse import urlparse

        return urlparse(result.url).hostname
    except Exception:
        return None


def _service_url(env_name: str, default: str) -> str:
    """Resolve service base URL and tolerate trailing slash env values."""
    return os.getenv(env_name, default).rstrip("/")


def _service_token_headers(*env_names: str) -> dict[str, str]:
    """Return bearer auth headers for downstream services with shared tokens."""
    token = ""
    for env_name in env_names:
        token = os.getenv(env_name, "").strip()
        if token:
            break
    return {"Authorization": f"Bearer {token}"} if token else {}


async def scrape_urls_background(
    session_id: str,
    urls: list[str],
    use_browser: bool,
    request_id: Optional[str] = None,
):
    """Background task for scraping URLs."""
    backend_url = _service_url("BACKEND_URL", "http://backend-core:8080")
    scraping_tasks.setdefault(session_id, _new_scrape_task_info(len(urls)))
    batch_scraper = _new_scraper_engine()
    request_token = set_request_id(request_id)

    async def on_result(result):
        """Callback for each scrape result."""
        task_info = scraping_tasks[session_id]
        task_info["completed"] = task_info.get("completed", 0) + 1

        if result.status == "success":
            task_info["success"] = task_info.get("success", 0) + 1
        elif result.status == "auth_required":
            task_info["auth_required"] = task_info.get("auth_required", 0) + 1
            log_event(
                "auth.queued",
                session_id=session_id,
                domain=_result_domain(result),
                auth_type=result.metadata.get("auth_type"),
            )
        else:
            task_info["failed"] = task_info.get("failed", 0) + 1

        # Deliver the result to the backend's single idempotent ingest endpoint.
        # Backend owns persistence AND the sole /index forward to ai-engine, so
        # browser-engine never writes vectors. A 200 "ignored" (duplicate/stale)
        # is a successful delivery, not a failure — only transport/4xx/5xx errors
        # count against backend_callback_failed.
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{backend_url}/api/v1/ingest/v1",
                    json={
                        "capture_id": str(uuid.uuid4()),
                        "attempt": _next_ingest_attempt(session_id, result.url),
                        "session_id": session_id,
                        "url": result.url,
                        "status": result.status,
                        "content": result.content,
                        "metadata": {
                            "title": result.title,
                            "status_code": result.status_code,
                            **result.metadata,
                        },
                        "auth_used": result.auth_used,
                        "fetched_at": result.scraped_at.isoformat(),
                    },
                    headers={
                        **_service_token_headers(
                            "BACKEND_CALLBACK_TOKEN", "AI_ENGINE_API_TOKEN"
                        ),
                        **request_id_headers(),
                    },
                    timeout=10.0,
                )
                if response.is_error:
                    raise RuntimeError(_http_response_error_message(response))
        except Exception as e:
            log_event(
                "callback.failed",
                level=logging.WARNING,
                session_id=session_id,
                scrape_status=result.status,
                reason=str(e),
            )
            _record_downstream_error(
                task_info,
                "backend_callback",
                str(e),
                url=result.url,
            )

    try:
        await batch_scraper.scrape_batch(
            urls=urls,
            session_id=session_id,
            callback=on_result,
            use_browser=use_browser,
        )

        _finalize_scrape_status(scraping_tasks[session_id])

    except Exception as e:
        scraping_tasks[session_id]["status"] = "failed"
        scraping_tasks[session_id]["error"] = str(e)
    finally:
        await batch_scraper.close()
        reset_request_id(request_token)


@app.post("/scrape/single")
async def scrape_single(
    request: SingleScrapeRequest,
    _auth=Depends(_require_browser_engine_auth),
):
    """Scrape a single URL synchronously."""
    _validate_scrape_urls([request.url])
    result = await scraper.scrape_url(
        url=request.url,
        session_id=request.session_id,
        use_browser=request.use_browser,
    )

    return {
        "url": result.url,
        "status": result.status,
        "title": result.title,
        "content": result.content[:5000] if result.content else None,
        "status_code": result.status_code,
        "error": result.error,
        "metadata": result.metadata,
    }


@app.post("/tabs/import")
async def import_tabs_from_browser(
    request: TabImportRequest,
    _auth=Depends(_require_browser_engine_auth),
):
    """Import live tabs from a user-started Chrome/Chromium CDP endpoint."""
    if request.max_tabs < 1 or request.max_tabs > 2000:
        raise HTTPException(
            status_code=400, detail="max_tabs must be between 1 and 2000"
        )
    if request.max_concurrent < 1 or request.max_concurrent > 100:
        raise HTTPException(
            status_code=400,
            detail="max_concurrent must be between 1 and 100",
        )

    try:
        harvester = CDPTabHarvester(
            cdp_url=request.cdp_url,
            max_concurrent=request.max_concurrent,
        )
        result = await harvester.harvest(max_tabs=request.max_tabs)
        return {"status": "completed", **result.to_dict()}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Tab import failed: {error}")


@app.post("/tabs/open")
async def open_tabs_in_browser(
    request: TabOpenRequest,
    _auth=Depends(_require_browser_engine_auth),
):
    """Open URLs in the attached Chrome/Chromium instance."""
    if not request.urls:
        raise HTTPException(status_code=400, detail="At least one URL is required")
    if len(request.urls) > 100:
        raise HTTPException(status_code=400, detail="Cannot open more than 100 URLs")

    try:
        harvester = CDPTabHarvester(cdp_url=request.cdp_url)
        opened = await harvester.open_urls(request.urls)
        return {"status": "completed", "opened": opened}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Tab open failed: {error}")


@app.get("/scrape/status/{session_id}")
async def get_scrape_status(
    session_id: str,
    _auth=Depends(_require_browser_engine_auth),
):
    """Get scraping status for a session."""
    task_info = scraping_tasks.get(session_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        **task_info,
        "pending_auth": auth_queue.get_pending_count(),
    }


# Auth endpoints
@app.get("/auth/pending")
async def get_pending_auth(_auth=Depends(_require_browser_engine_auth)):
    """Get all pending authentication requests."""
    return auth_queue.to_dict()


@app.get("/auth/pending/{session_id}")
async def get_pending_auth_for_session(
    session_id: str,
    _auth=Depends(_require_browser_engine_auth),
):
    """Get pending auth requests for a session."""
    requests = auth_queue.get_pending_for_session(session_id)
    return {
        "session_id": session_id,
        "pending": [
            {
                "id": r.id,
                "domain": r.domain,
                "url": r.url,
                "auth_type": r.auth_type,
                "form_fields": r.form_fields,
                "oauth_provider": r.oauth_provider,
            }
            for r in requests
        ],
    }


@app.post("/auth/credentials")
async def submit_credentials(
    request: CredentialsRequest,
    _auth=Depends(_require_browser_engine_auth),
):
    """Submit credentials for a domain."""
    try:
        success = await auth_queue.provide_credentials(
            domain=request.domain,
            credentials=request.credentials,
        )
    except CredentialStoreError as error:
        # Fail closed: never accept credentials we cannot encrypt securely.
        log_event(
            "auth.credentials_rejected",
            level=logging.WARNING,
            domain=request.domain,
            reason="credential_store_unavailable",
        )
        raise HTTPException(status_code=503, detail=error.to_dict())

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"No pending auth request for domain: {request.domain}",
        )

    # Log the queue transition only; never the submitted credential values.
    log_event("auth.credentials_stored", domain=request.domain)
    return {"status": "credentials_stored", "domain": request.domain}


@app.delete("/auth/pending/{domain}")
async def cancel_auth_request(
    domain: str,
    _auth=Depends(_require_browser_engine_auth),
):
    """Cancel a pending auth request."""
    success = await auth_queue.cancel_request(domain)
    if not success:
        raise HTTPException(status_code=404, detail="Request not found")
    return {"status": "cancelled", "domain": domain}


@app.post("/auth/expire")
async def expire_old_requests(
    max_age_seconds: int = 3600,
    _auth=Depends(_require_browser_engine_auth),
):
    """Expire old pending auth requests."""
    count = await auth_queue.expire_old_requests(max_age_seconds)
    return {"expired": count}


# Detection endpoint
@app.post("/detect-auth")
async def detect_auth(
    url: str,
    html: Optional[str] = None,
    _auth=Depends(_require_browser_engine_auth),
):
    """Detect if a URL requires authentication."""
    _validate_scrape_urls([url])
    result = auth_detector.detect(url=url, html=html)
    return {
        "url": url,
        "requires_auth": result.requires_auth,
        "auth_type": result.auth_type,
        "confidence": result.confidence,
        "form_fields": result.form_fields,
        "oauth_provider": result.oauth_provider,
    }


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    await scraper.close()
