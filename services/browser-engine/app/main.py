"""Browser Engine Service - Main Application."""

import asyncio
import os
from typing import Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .auth.detector import AuthDetector
from .auth.queue import AuthQueue
from .scraper.engine import ScraperEngine

app = FastAPI(
    title="Tab Organizer - Browser Engine",
    description="Web scraping and authentication handling",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
async def start_scraping(request: ScrapeRequest, background_tasks: BackgroundTasks):
    """Start scraping URLs in the background."""
    session_id = request.session_id

    # Track scraping task
    scraping_tasks[session_id] = _new_scrape_task_info(len(request.urls))

    # Start background scraping
    background_tasks.add_task(
        scrape_urls_background,
        session_id,
        request.urls,
        request.use_browser,
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
    """Record callback/index errors without aborting the scrape batch."""
    task_info["downstream_error_count"] = task_info.get("downstream_error_count", 0) + 1

    if source == "backend_callback":
        task_info["backend_callback_failed"] = (
            task_info.get("backend_callback_failed", 0) + 1
        )
    elif source == "ai_index":
        task_info["ai_index_failed"] = task_info.get("ai_index_failed", 0) + 1

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
):
    """Background task for scraping URLs."""
    backend_url = _service_url("BACKEND_URL", "http://backend-core:8080")
    scraping_tasks.setdefault(session_id, _new_scrape_task_info(len(urls)))
    batch_scraper = _new_scraper_engine()

    async def on_result(result):
        """Callback for each scrape result."""
        task_info = scraping_tasks[session_id]
        task_info["completed"] = task_info.get("completed", 0) + 1

        if result.status == "success":
            task_info["success"] = task_info.get("success", 0) + 1
        elif result.status == "auth_required":
            task_info["auth_required"] = task_info.get("auth_required", 0) + 1
        else:
            task_info["failed"] = task_info.get("failed", 0) + 1

        # Notify backend
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{backend_url}/api/v1/callback/scrape-complete",
                    json={
                        "session_id": session_id,
                        "url": result.url,
                        "status": result.status,
                        "content": result.content,
                        "metadata": {
                            "title": result.title,
                            "status_code": result.status_code,
                            **result.metadata,
                        },
                    },
                    headers=_service_token_headers(
                        "BACKEND_CALLBACK_TOKEN", "AI_ENGINE_API_TOKEN"
                    ),
                    timeout=10.0,
                )
                if response.is_error:
                    raise RuntimeError(_http_response_error_message(response))
                try:
                    payload = response.json()
                except ValueError:
                    payload = {}
                if payload.get("status") == "error":
                    raise RuntimeError(
                        payload.get("message") or "Backend callback returned error"
                    )
        except Exception as e:
            _record_downstream_error(
                task_info,
                "backend_callback",
                str(e),
                url=result.url,
            )

    try:
        results = await batch_scraper.scrape_batch(
            urls=urls,
            session_id=session_id,
            callback=on_result,
            use_browser=use_browser,
        )

        # Index successful results in AI engine
        ai_url = _service_url("AI_ENGINE_URL", "http://ai-engine:8090")
        documents = [
            {
                "id": r.url,
                "url": r.url,
                "title": r.title or "",
                "content": r.content or "",
                "metadata": r.metadata,
            }
            for r in results
            if r.status == "success" and r.content
        ]

        if documents:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{ai_url}/index",
                        json={
                            "session_id": session_id,
                            "documents": documents,
                        },
                        headers=_service_token_headers("AI_ENGINE_API_TOKEN"),
                        timeout=120.0,
                    )
                    if response.is_error:
                        raise RuntimeError(_http_response_error_message(response))
            except Exception as e:
                _record_downstream_error(
                    scraping_tasks[session_id],
                    "ai_index",
                    str(e),
                )

        _finalize_scrape_status(scraping_tasks[session_id])

    except Exception as e:
        scraping_tasks[session_id]["status"] = "failed"
        scraping_tasks[session_id]["error"] = str(e)
    finally:
        await batch_scraper.close()


@app.post("/scrape/single")
async def scrape_single(request: SingleScrapeRequest):
    """Scrape a single URL synchronously."""
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


@app.get("/scrape/status/{session_id}")
async def get_scrape_status(session_id: str):
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
async def get_pending_auth():
    """Get all pending authentication requests."""
    return auth_queue.to_dict()


@app.get("/auth/pending/{session_id}")
async def get_pending_auth_for_session(session_id: str):
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
async def submit_credentials(request: CredentialsRequest):
    """Submit credentials for a domain."""
    success = await auth_queue.provide_credentials(
        domain=request.domain,
        credentials=request.credentials,
    )

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"No pending auth request for domain: {request.domain}",
        )

    return {"status": "credentials_stored", "domain": request.domain}


@app.delete("/auth/pending/{domain}")
async def cancel_auth_request(domain: str):
    """Cancel a pending auth request."""
    success = await auth_queue.cancel_request(domain)
    if not success:
        raise HTTPException(status_code=404, detail="Request not found")
    return {"status": "cancelled", "domain": domain}


@app.post("/auth/expire")
async def expire_old_requests(max_age_seconds: int = 3600):
    """Expire old pending auth requests."""
    count = await auth_queue.expire_old_requests(max_age_seconds)
    return {"expired": count}


# Detection endpoint
@app.post("/detect-auth")
async def detect_auth(url: str, html: Optional[str] = None):
    """Detect if a URL requires authentication."""
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
