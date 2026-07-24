"""API routes for Backend Core service."""

import hmac
import logging
import os
from typing import Any, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from pydantic import BaseModel, Field, HttpUrl

from services.observability import log_event, request_id_headers

from ..export.exporter import Exporter
from ..platform.store import (
    AuthenticationError,
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    PlatformError,
    PlatformStore,
    PlatformValidationError,
)
from ..sessions.manager import SessionManager
from ..url_input.store import URLStore

# Global instances
session_manager = SessionManager()
exporter = Exporter()
platform_store = PlatformStore()

router = APIRouter()

SCRAPE_STATUS_TO_URL_STATUS = {
    "success": "scraped",
    "failed": "failed",
    "auth_required": "auth_required",
    "timeout": "failed",
    "blocked": "failed",
}

# Single-source defaults for inter-service URLs (finding 32). Every call site
# resolves through these two accessors — never a bare literal — and the
# fallback values here match docker-compose.yml's own defaults
# (`${AI_ENGINE_URL:-http://ai-engine:8090}` / `${BROWSER_ENGINE_URL:-...}`).
DEFAULT_AI_ENGINE_URL = "http://ai-engine:8090"
DEFAULT_BROWSER_ENGINE_URL = "http://browser-engine:8083"


def _ai_engine_url() -> str:
    """Resolve the AI engine base URL from runtime environment."""
    return os.getenv("AI_ENGINE_URL", DEFAULT_AI_ENGINE_URL).rstrip("/")


def _ai_engine_headers() -> dict[str, str]:
    """Return auth headers for AI Engine requests when host mode configures one."""
    token = os.getenv("AI_ENGINE_API_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _browser_engine_headers() -> dict[str, str]:
    """Return bearer auth headers for Browser Engine control endpoints."""
    token = (
        os.getenv("BROWSER_ENGINE_API_TOKEN", "").strip()
        or os.getenv("BACKEND_CALLBACK_TOKEN", "").strip()
        or os.getenv("AI_ENGINE_API_TOKEN", "").strip()
    )
    return {"Authorization": f"Bearer {token}"} if token else {}


def _ai_engine_request_headers() -> dict[str, str]:
    """AI Engine auth headers plus the current X-Request-ID for propagation."""
    return {**_ai_engine_headers(), **request_id_headers()}


def _browser_engine_request_headers() -> dict[str, str]:
    """Browser Engine auth headers plus the current X-Request-ID."""
    return {**_browser_engine_headers(), **request_id_headers()}


def _backend_callback_token() -> str:
    """Resolve the bearer token used by browser-engine callbacks."""
    return (
        os.getenv("BACKEND_CALLBACK_TOKEN", "").strip()
        or os.getenv("AI_ENGINE_API_TOKEN", "").strip()
    )


def _require_backend_callback_auth(
    authorization: Optional[str] = Header(default=None),
):
    """Require signed browser-engine callbacks before accepting scraped content."""
    expected = _backend_callback_token()
    if not expected:
        raise HTTPException(
            status_code=401,
            detail=(
                "Backend callback token is not configured; set BACKEND_CALLBACK_TOKEN "
                "or run scripts/cli.py start"
            ),
        )
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token.strip(), expected):
        raise HTTPException(status_code=401, detail="Invalid backend callback token")


def _backend_agent_token() -> str:
    """Resolve the bearer token for local agent-facing tab APIs."""
    return os.getenv("BACKEND_AGENT_API_TOKEN", "").strip()


def _require_backend_agent_auth(
    authorization: Optional[str] = Header(default=None),
):
    """Require a local agent bearer token for tab management APIs."""
    expected = _backend_agent_token()
    if not expected:
        raise HTTPException(
            status_code=401,
            detail=(
                "Backend agent token is not configured; set BACKEND_AGENT_API_TOKEN "
                "or run scripts/cli.py start"
            ),
        )
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token.strip(), expected):
        raise HTTPException(status_code=401, detail="Invalid backend agent token")


def _browser_engine_url() -> str:
    """Resolve the browser engine base URL from runtime environment."""
    return os.getenv("BROWSER_ENGINE_URL", DEFAULT_BROWSER_ENGINE_URL).rstrip("/")


# Request/Response models
class URLInput(BaseModel):
    urls: list[str]
    session_id: Optional[str] = None


class URLInputResponse(BaseModel):
    session_id: str
    added: int
    duplicates: int
    total: int


class SessionCreate(BaseModel):
    name: Optional[str] = None


class SessionResponse(BaseModel):
    id: str
    name: str
    total_urls: int
    status: str


class ExportRequest(BaseModel):
    session_id: str
    format: str = "markdown"


class ScrapeRequest(BaseModel):
    session_id: str
    urls: Optional[list[str]] = None
    use_browser: bool = False


class ClusterRequest(BaseModel):
    session_id: str


class TabImportRequest(BaseModel):
    session_id: Optional[str] = None
    session_name: Optional[str] = None
    cdp_url: str = "http://host.docker.internal:9222"
    max_tabs: int = 2000


class TabOpenRequest(BaseModel):
    urls: Optional[list[str]] = None
    session_id: Optional[str] = None
    cdp_url: str = "http://host.docker.internal:9222"


class SearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    mode: str = "hybrid"
    top_k: int = 10


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    top_k: int = 5


class PlatformSignupRequest(BaseModel):
    email: str
    password: str
    name: Optional[str] = None
    account_type: Optional[str] = None
    role: str = "user"
    company_name: Optional[str] = None
    maintainer_code: Optional[str] = None


class PlatformLoginRequest(BaseModel):
    email: str
    password: str


class PlatformApiTokenCreate(BaseModel):
    name: str
    scopes: Optional[list[str]] = None


class PlatformDashboardEventCreate(BaseModel):
    event_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlatformIssueCreate(BaseModel):
    title: str
    description: str
    severity: str = "medium"


def _platform_http_error(error: PlatformError) -> HTTPException:
    if isinstance(error, AuthenticationError):
        return HTTPException(status_code=401, detail=str(error))
    if isinstance(error, PermissionDeniedError):
        return HTTPException(status_code=403, detail=str(error))
    if isinstance(error, NotFoundError):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, ConflictError):
        return HTTPException(status_code=409, detail=str(error))
    if isinstance(error, PlatformValidationError):
        return HTTPException(status_code=400, detail=str(error))
    return HTTPException(status_code=500, detail=str(error))


def _bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    return token.strip()


def _require_platform_user(authorization: Optional[str]) -> dict[str, Any]:
    try:
        return platform_store.authenticate_session(_bearer_token(authorization))
    except PlatformError as error:
        raise _platform_http_error(error)


def _maintainer_signup_code() -> Optional[str]:
    return os.getenv("PLATFORM_MAINTAINER_SIGNUP_CODE") or os.getenv(
        "PLATFORM_MAINTAINER_CODE"
    )


def _platform_signup_role(request: PlatformSignupRequest) -> str:
    role = request.role
    account_type = (request.account_type or "").strip().lower()
    if account_type in {"business", "b2b", "team", "enterprise"}:
        role = "b2b"
    elif account_type == "maintainer":
        role = "maintainer"

    if role.strip().lower() == "maintainer":
        expected_code = _maintainer_signup_code()
        supplied_code = request.maintainer_code or ""
        if not expected_code or not hmac.compare_digest(supplied_code, expected_code):
            raise HTTPException(
                status_code=403,
                detail="Maintainer signup requires a valid local bootstrap code",
            )

    return role


# Health check
@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "backend-core",
        "persistence": {
            "enabled": session_manager.persistence_enabled,
            "type": "sqlite" if session_manager.persistence_enabled else "memory",
        },
    }


# Platform endpoints
@router.post("/platform/auth/signup")
def platform_signup(request: PlatformSignupRequest):
    try:
        user = platform_store.create_user(
            email=request.email,
            password=request.password,
            name=request.name,
            role=_platform_signup_role(request),
            company_name=request.company_name,
        )
        session = platform_store.create_session(user["id"])
        return {"user": user, **session}
    except PlatformError as error:
        raise _platform_http_error(error)


@router.post("/platform/auth/login")
def platform_login(request: PlatformLoginRequest):
    try:
        user = platform_store.login_user(request.email, request.password)
        session = platform_store.create_session(user["id"])
        return {"user": user, **session}
    except PlatformError as error:
        raise _platform_http_error(error)


@router.get("/platform/auth/session")
def platform_session(authorization: Optional[str] = Header(default=None)):
    user = _require_platform_user(authorization)
    return {"user": user}


@router.get("/platform/me")
def platform_me(authorization: Optional[str] = Header(default=None)):
    user = _require_platform_user(authorization)
    return {"user": user}


@router.get("/platform/companies")
def platform_list_companies(
    query: Optional[str] = None,
    limit: int = 25,
    authorization: Optional[str] = Header(default=None),
):
    _require_platform_user(authorization)
    return {"companies": platform_store.search_companies(query=query, limit=limit)}


@router.get("/platform/companies/search")
def platform_search_companies(
    q: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 25,
    authorization: Optional[str] = Header(default=None),
):
    _require_platform_user(authorization)
    return {"companies": platform_store.search_companies(query=q or query, limit=limit)}


@router.get("/platform/companies/{company_id}")
def platform_get_company(
    company_id: str,
    authorization: Optional[str] = Header(default=None),
):
    _require_platform_user(authorization)
    try:
        return {"company": platform_store.get_company(company_id)}
    except PlatformError as error:
        raise _platform_http_error(error)


@router.post("/platform/api-tokens")
def platform_create_api_token(
    request: PlatformApiTokenCreate,
    authorization: Optional[str] = Header(default=None),
):
    user = _require_platform_user(authorization)
    try:
        return platform_store.create_api_token(user["id"], request.name, request.scopes)
    except PlatformError as error:
        raise _platform_http_error(error)


@router.get("/platform/api-tokens")
def platform_list_api_tokens(authorization: Optional[str] = Header(default=None)):
    user = _require_platform_user(authorization)
    try:
        return {"tokens": platform_store.list_api_tokens(user["id"])}
    except PlatformError as error:
        raise _platform_http_error(error)


@router.delete("/platform/api-tokens/{token_id}")
def platform_revoke_api_token(
    token_id: str,
    authorization: Optional[str] = Header(default=None),
):
    user = _require_platform_user(authorization)
    try:
        return {"api_token": platform_store.revoke_api_token(user["id"], token_id)}
    except PlatformError as error:
        raise _platform_http_error(error)


@router.post("/platform/b2b/tokens")
def platform_create_b2b_token(
    request: PlatformApiTokenCreate,
    authorization: Optional[str] = Header(default=None),
):
    return platform_create_api_token(request, authorization)


@router.get("/platform/b2b/tokens")
def platform_list_b2b_tokens(
    authorization: Optional[str] = Header(default=None),
):
    return platform_list_api_tokens(authorization)


@router.delete("/platform/b2b/tokens/{token_id}")
def platform_revoke_b2b_token(
    token_id: str,
    authorization: Optional[str] = Header(default=None),
):
    return platform_revoke_api_token(token_id, authorization)


@router.get("/platform/b2b/first-call")
def platform_b2b_first_call(authorization: Optional[str] = Header(default=None)):
    user = _require_platform_user(authorization)
    try:
        return platform_store.first_api_call(user["id"])
    except PlatformError as error:
        raise _platform_http_error(error)


@router.get("/platform/v1/companies/search")
def platform_api_company_search(
    query: Optional[str] = None,
    limit: int = 25,
    authorization: Optional[str] = Header(default=None),
):
    try:
        return platform_store.search_companies_with_api_token(
            _bearer_token(authorization),
            query=query,
            limit=limit,
            path="/api/v1/platform/v1/companies/search",
        )
    except PlatformError as error:
        raise _platform_http_error(error)


@router.get("/platform/dashboard")
def platform_dashboard(authorization: Optional[str] = Header(default=None)):
    user = _require_platform_user(authorization)
    try:
        return platform_store.get_dashboard(user["id"])
    except PlatformError as error:
        raise _platform_http_error(error)


@router.post("/platform/dashboard/events")
def platform_create_dashboard_event(
    request: PlatformDashboardEventCreate,
    authorization: Optional[str] = Header(default=None),
):
    user = _require_platform_user(authorization)
    try:
        return {
            "event": platform_store.record_event(
                user["id"], request.event_type, request.metadata
            )
        }
    except PlatformError as error:
        raise _platform_http_error(error)


@router.post("/platform/issues")
def platform_create_issue(
    request: PlatformIssueCreate,
    authorization: Optional[str] = Header(default=None),
):
    user = _require_platform_user(authorization)
    try:
        return {
            "issue": platform_store.create_issue(
                user["id"],
                title=request.title,
                description=request.description,
                severity=request.severity,
            )
        }
    except PlatformError as error:
        raise _platform_http_error(error)


@router.get("/platform/maintainer/issues")
def platform_list_issues(
    status: Optional[str] = None,
    authorization: Optional[str] = Header(default=None),
):
    user = _require_platform_user(authorization)
    try:
        return {"issues": platform_store.list_issues(user["id"], status=status)}
    except PlatformError as error:
        raise _platform_http_error(error)


# Session endpoints
@router.post("/sessions", response_model=SessionResponse)
def create_session(request: SessionCreate):
    session = session_manager.create_session(request.name)
    return SessionResponse(
        id=session.id, name=session.name, total_urls=0, status=session.status
    )


@router.get("/sessions")
def list_sessions():
    sessions = session_manager.list_sessions()
    return [
        {
            "id": s.id,
            "name": s.name,
            "total_urls": s.url_store.count(),
            "status": s.status,
            "created_at": s.created_at.isoformat(),
            "updated_at": s.updated_at.isoformat(),
        }
        for s in sessions
    ]


@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    stats = session_manager.get_session_stats(session_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Session not found")
    return stats


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if not session_manager.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


# URL endpoints
@router.post("/urls", response_model=URLInputResponse)
def add_urls(request: URLInput):
    # Get or create session
    if request.session_id:
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = session_manager.get_or_create_current_session()

    try:
        added, duplicates, _ = session_manager.add_urls_to_session(
            session.id, request.urls
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    return URLInputResponse(
        session_id=session.id,
        added=added,
        duplicates=duplicates,
        total=session.url_store.count(),
    )


@router.get("/urls/{session_id}")
def get_urls(session_id: str, status: Optional[str] = None):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if status:
        records = session.url_store.get_by_status(status)
    else:
        records = session.url_store.get_all()

    return [
        {
            "original": r.original,
            "normalized": r.normalized,
            "status": r.status,
            "metadata": r.metadata,
        }
        for r in records
    ]


# Scraping endpoints
@router.post("/scrape")
async def start_scraping(request: ScrapeRequest, background_tasks: BackgroundTasks):
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get URLs to scrape
    if request.urls:
        urls = request.urls
        # WI0 B3: inline urls must be registered before dispatch. Without this
        # every scrape-complete callback fails "URL not found in session" and
        # the scraped content is dropped while batch status still reads
        # completed. add_urls_to_session dedupes against the existing session
        # url store, so re-registering already-known urls is a no-op.
        session_manager.add_urls_to_session(session.id, urls)
    else:
        pending = session.url_store.get_by_status("pending")
        urls = [r.original for r in pending]

    if not urls:
        return {"status": "no_urls", "message": "No URLs to scrape"}

    try:
        await trigger_scraping(session.id, urls, request.use_browser)
    except Exception as error:
        detail = f"Browser Engine scrape dispatch failed: {error}"
        log_event(
            "scrape.dispatch_failed",
            level=logging.ERROR,
            session_id=session.id,
            url_count=len(urls),
            reason=str(error),
        )
        for url in urls:
            session_manager.update_url_status(
                session.id,
                url,
                "failed",
                metadata={"dispatch_error": detail},
            )
        raise HTTPException(status_code=502, detail=detail) from error

    log_event(
        "scrape.dispatched",
        session_id=session.id,
        url_count=len(urls),
        use_browser=request.use_browser,
    )
    return {"status": "started", "session_id": session.id, "url_count": len(urls)}


async def trigger_scraping(
    session_id: str,
    urls: list[str],
    use_browser: bool = False,
):
    """Background task to trigger browser engine scraping."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{_browser_engine_url()}/scrape",
            json={
                "session_id": session_id,
                "urls": urls,
                "use_browser": use_browser,
            },
            headers=_browser_engine_request_headers(),
            timeout=30.0,
        )
        response.raise_for_status()


def _job_response(job) -> dict[str, Any]:
    """Return a JSON-safe tab import job response."""
    return {
        "job_id": job.id,
        "session_id": job.session_id,
        "cdp_url": job.cdp_url,
        "status": job.status,
        "total": job.total,
        "imported": job.imported,
        "indexed": job.indexed,
        "failed": job.failed,
        "error": job.error,
        "metadata": job.metadata,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
    }


def _tab_documents_from_import_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize browser-engine tab import output to document dictionaries."""
    documents = []
    for item in payload.get("tabs", []):
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("id")
        if not url:
            continue
        documents.append(
            {
                "id": item.get("id") or url,
                "url": url,
                "title": item.get("title") or url,
                "content": item.get("content") or "",
                "metadata": item.get("metadata") or {},
            }
        )
    return documents


def _chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


async def _index_tab_documents(session_id: str, documents: list[dict[str, Any]]) -> int:
    """Index imported tab documents in bounded AI Engine batches."""
    indexed = 0
    if not documents:
        return indexed

    async with httpx.AsyncClient() as client:
        for chunk in _chunked(documents, 100):
            try:
                response = await client.post(
                    f"{_ai_engine_url()}/index",
                    json={"session_id": session_id, "documents": chunk},
                    headers=_ai_engine_request_headers(),
                    timeout=120.0,
                )
                response.raise_for_status()
            except Exception as e:
                log_event(
                    "index.batch_failed",
                    level=logging.ERROR,
                    session_id=session_id,
                    document_count=len(chunk),
                    reason=str(e),
                )
                raise
            indexed += int(response.json().get("indexed", 0))
    log_event(
        "index.batch",
        session_id=session_id,
        document_count=len(documents),
        indexed=indexed,
    )
    return indexed


async def _semantic_search(
    session_id: Optional[str],
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Run semantic search through AI Engine."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{_ai_engine_url()}/search",
            json={"session_id": session_id, "query": query, "top_k": top_k},
            headers=_ai_engine_request_headers(),
            timeout=60.0,
        )
        if response.is_error:
            log_event(
                "search.semantic_failed",
                level=logging.WARNING,
                session_id=session_id,
                status_code=response.status_code,
            )
        response.raise_for_status()
        return response.json().get("results", [])


def _semantic_failure_reason(error: Exception) -> str:
    """Compact, log/response-safe reason for a failed semantic leg."""
    if isinstance(error, httpx.HTTPStatusError):
        response = error.response
        detail = None
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("message")
        reason = f"ai-engine returned HTTP {response.status_code}"
        return f"{reason}: {detail}" if detail else reason
    if isinstance(error, httpx.RequestError):
        return f"ai-engine unreachable: {error}"
    return str(error) or error.__class__.__name__


def _merge_search_results(
    keyword_results: list[dict[str, Any]],
    semantic_results: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Merge search results by URL, preserving the best source information."""
    merged: dict[str, dict[str, Any]] = {}
    for result in keyword_results + semantic_results:
        url = result.get("url")
        if not url:
            continue
        existing = merged.get(url)
        if existing is None:
            merged[url] = result
            continue
        existing["source"] = "hybrid"
        existing["score"] = max(
            float(existing.get("score", 0.0)),
            float(result.get("score", 0.0)),
        )
        if not existing.get("content") and result.get("content"):
            existing["content"] = result["content"]
        if not existing.get("title") and result.get("title"):
            existing["title"] = result["title"]
    return list(merged.values())[:limit]


async def import_tabs_background(job_id: str, cdp_url: str, max_tabs: int) -> None:
    """Import live browser tabs through Browser Engine and index them."""
    job = session_manager.update_tab_import_job(job_id, status="running")
    if not job:
        return

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{_browser_engine_url()}/tabs/import",
                json={"cdp_url": cdp_url, "max_tabs": max_tabs},
                headers=_browser_engine_request_headers(),
                timeout=180.0,
            )
            response.raise_for_status()
            payload = response.json()

        documents = _tab_documents_from_import_payload(payload)
        if documents:
            session_manager.add_urls_to_session(
                job.session_id, [d["url"] for d in documents]
            )
            for document in documents:
                session_manager.update_url_status(
                    job.session_id,
                    document["url"],
                    "scraped",
                    metadata={
                        **document.get("metadata", {}),
                        "title": document.get("title", ""),
                        "content": document.get("content", ""),
                    },
                )

        indexed = await _index_tab_documents(job.session_id, documents)
        failed = int(payload.get("failed", 0))
        final_status = "completed_with_errors" if failed else "completed"
        session_manager.update_tab_import_job(
            job_id,
            status=final_status,
            total=int(payload.get("total", len(documents))),
            imported=len(documents),
            indexed=indexed,
            failed=failed,
            metadata={"errors": payload.get("errors", [])},
        )
    except Exception as error:
        session_manager.update_tab_import_job(
            job_id,
            status="failed",
            error=str(error),
        )


# Agent-facing tab workflow endpoints
@router.post("/tabs/import")
async def import_tabs_from_browser(
    request: TabImportRequest,
    background_tasks: BackgroundTasks,
    _auth=Depends(_require_backend_agent_auth),
):
    """Start importing tabs from an attached Chrome/Chromium instance."""
    if request.max_tabs < 1 or request.max_tabs > 2000:
        raise HTTPException(
            status_code=400, detail="max_tabs must be between 1 and 2000"
        )

    if request.session_id:
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = session_manager.create_session(request.session_name or "Browser Tabs")

    try:
        job = session_manager.create_tab_import_job(session.id, request.cdp_url)
    except ValueError as error:
        detail = str(error)
        status_code = 409 if "active import" in detail.lower() else 400
        raise HTTPException(status_code=status_code, detail=detail)

    background_tasks.add_task(
        import_tabs_background,
        job.id,
        request.cdp_url,
        request.max_tabs,
    )
    return {"status": "queued", **_job_response(job)}


@router.get("/tabs/import/{job_id}")
def get_tab_import_job(
    job_id: str,
    _auth=Depends(_require_backend_agent_auth),
):
    """Return durable tab import status."""
    job = session_manager.get_tab_import_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Tab import job not found")
    return _job_response(job)


@router.post("/tabs/open")
async def open_tabs(
    request: TabOpenRequest,
    _auth=Depends(_require_backend_agent_auth),
):
    """Open supplied or session URLs in the attached browser."""
    urls = request.urls or []
    if not urls and request.session_id:
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        urls = [record.original for record in session.url_store.get_all()]
    if not urls:
        raise HTTPException(status_code=400, detail="At least one URL is required")
    if len(urls) > 100:
        raise HTTPException(status_code=400, detail="Cannot open more than 100 URLs")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{_browser_engine_url()}/tabs/open",
                json={"cdp_url": request.cdp_url, "urls": urls},
                headers=_browser_engine_request_headers(),
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Tab open failed: {error}")


@router.post("/search")
async def search_tabs(
    request: SearchRequest,
    _auth=Depends(_require_backend_agent_auth),
):
    """Search indexed tabs with semantic, keyword, or hybrid search."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    mode = request.mode.lower()
    if mode not in {"hybrid", "semantic", "keyword"}:
        raise HTTPException(status_code=400, detail="Unsupported search mode")
    limit = min(max(int(request.top_k), 1), 50)

    session_id = request.session_id
    if session_id and not session_manager.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    keyword_results: list[dict[str, Any]] = []
    semantic_results: list[dict[str, Any]] = []
    degraded: Optional[str] = None

    if mode in {"hybrid", "keyword"}:
        keyword_results = session_manager.search_indexed_tabs(session_id, query, limit)
    if mode in {"hybrid", "semantic"}:
        # WI0 B4: a dead semantic leg must not 500 the hybrid default. In
        # hybrid mode degrade to keyword-only and surface the reason; in
        # semantic-only mode there is nothing to fall back to, so raise a
        # structured error the caller can act on.
        try:
            semantic_results = await _semantic_search(session_id, query, limit)
        except Exception as error:  # noqa: BLE001 - degrade, don't propagate
            reason = _semantic_failure_reason(error)
            log_event(
                "search.semantic_degraded",
                level=logging.WARNING,
                session_id=session_id,
                mode=mode,
                reason=reason,
            )
            if mode == "semantic":
                raise HTTPException(
                    status_code=502,
                    detail={
                        "code": "semantic_unavailable",
                        "cause": reason,
                        "fix": (
                            "Verify the AI Engine embedding provider is "
                            "configured and reachable, or use mode=keyword."
                        ),
                    },
                ) from error
            degraded = f"semantic_unavailable: {reason}"

    if mode == "keyword":
        results = keyword_results
    elif mode == "semantic":
        results = semantic_results
    else:
        results = _merge_search_results(keyword_results, semantic_results, limit)

    response: dict[str, Any] = {
        "results": results[:limit],
        "count": len(results[:limit]),
        "mode": mode,
    }
    if degraded:
        response["degraded"] = degraded
    return response


# Chat endpoint (proxy to AI Engine; mirrors the /cluster proxy pattern)
@router.post("/chat")
async def chat(request: ChatRequest):
    """Proxy chat to AI Engine so it stays behind Backend Core (WI0 B7).

    Web UI previously called `{ai_url}/chat` directly, bypassing the "Backend
    Core is the only orchestrator" rule and duplicating AI Engine token
    handling client-side. This route keeps the bearer token server-side.
    """
    if request.session_id and not session_manager.get_session(request.session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{_ai_engine_url()}/chat",
                json={
                    "query": request.query,
                    "session_id": request.session_id,
                    "top_k": request.top_k,
                },
                headers=_ai_engine_request_headers(),
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        log_event(
            "chat.failed",
            level=logging.ERROR,
            session_id=request.session_id,
            reason=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


# Clustering endpoints
@router.post("/cluster")
async def start_clustering(request: ClusterRequest):
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get scraped URLs
    scraped = session.url_store.get_by_status("scraped")
    if not scraped:
        raise HTTPException(
            status_code=400, detail="No scraped content available for clustering"
        )

    # Trigger AI engine clustering
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{_ai_engine_url()}/cluster",
                json={
                    "session_id": session.id,
                    "urls": [
                        {
                            "url": r.original,
                            "content": r.metadata.get("content", ""),
                            "title": r.metadata.get("title", ""),
                        }
                        for r in scraped
                    ],
                },
                headers=_ai_engine_request_headers(),
                timeout=120.0,
            )
            response.raise_for_status()
            clusters = response.json().get("clusters", [])
            session_manager.set_session_clusters(session.id, clusters)

            return {"status": "completed", "clusters": clusters}
    except Exception as e:
        log_event(
            "cluster.failed",
            level=logging.ERROR,
            session_id=session.id,
            url_count=len(scraped),
            reason=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Clustering failed: {e}")


@router.get("/clusters/{session_id}")
async def get_clusters(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"clusters": session.clusters}


# Export endpoints
@router.post("/export")
def export_session(request: ExportRequest):
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        content = exporter.export(session, request.format)
        return {
            "format": request.format,
            "content": content,
            "filename": f"{session.name.replace(' ', '_')}.{request.format}",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/scrape/status/{session_id}")
async def get_scrape_status(session_id: str):
    """Get scraping status for a session from browser engine.

    Browser Engine's scrape state is in-process memory (documented stub): a
    404 there can mean "never scraped" OR "browser-engine restarted mid-batch"
    — those are not the same fact. Never fabricate not_started/completed from
    local URL-store counts when the real status is unknown (finding 28); report
    an explicit `status: "unknown"` with a `detail` reason instead, still
    including local counts as best-effort reference data.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    reason = None
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{_browser_engine_url()}/scrape/status/{session_id}",
                headers=_browser_engine_request_headers(),
                timeout=10.0,
            )
        if response.status_code == 404:
            reason = "browser-engine has no record of this scrape"
        else:
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        reason = f"browser-engine returned HTTP {e.response.status_code}"
    except httpx.RequestError as e:
        reason = f"browser-engine unreachable: {e}"

    log_event(
        "scrape.status_unavailable",
        level=logging.WARNING,
        session_id=session_id,
        reason=reason,
    )
    counts = session.url_store.count_by_status()
    total = session.url_store.count()
    completed = (
        counts.get("scraped", 0)
        + counts.get("failed", 0)
        + counts.get("auth_required", 0)
    )
    return {
        "session_id": session_id,
        "status": "unknown",
        "detail": f"status unavailable: {reason}",
        "total": total,
        "completed": completed,
        "success": counts.get("scraped", 0),
        "failed": counts.get("failed", 0),
        "auth_required": counts.get("auth_required", 0),
    }


# Auth queue endpoints (proxy to browser engine)
@router.get("/auth/pending")
async def get_pending_auth():
    """Get pending authentication requests from browser engine."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{_browser_engine_url()}/auth/pending",
                headers=_browser_engine_request_headers(),
                timeout=10.0,
            )
            return response.json()
    except Exception as e:
        return {"pending": [], "error": str(e)}


@router.post("/auth/credentials")
async def submit_credentials(domain: str, credentials: dict):
    """Submit credentials for a domain."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{_browser_engine_url()}/auth/credentials",
                json={"domain": domain, "credentials": credentials},
                headers=_browser_engine_request_headers(),
                timeout=10.0,
            )
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Callback endpoint for browser engine
@router.post("/callback/scrape-complete")
def scrape_complete_callback(
    data: dict,
    _auth=Depends(_require_backend_callback_auth),
):
    """Callback from browser engine when scraping is complete."""
    session_id = data.get("session_id")
    url = data.get("url")
    status = data.get("status")
    content = data.get("content")
    metadata = data.get("metadata", {})

    session = session_manager.get_session(session_id)
    if not session:
        log_event(
            "callback.scrape_complete_failed",
            level=logging.WARNING,
            session_id=session_id,
            reason="session_not_found",
        )
        return {"status": "error", "message": "Session not found"}

    url_status = SCRAPE_STATUS_TO_URL_STATUS.get(status, "failed")

    # Update URL record
    updated = session_manager.update_url_status(
        session.id,
        url,
        url_status,
        metadata={**metadata, "content": content} if content else metadata,
    )
    if not updated:
        log_event(
            "callback.scrape_complete_failed",
            level=logging.WARNING,
            session_id=session.id,
            scrape_status=status,
            reason="url_not_found_in_session",
        )
        return {"status": "error", "message": "URL not found in session"}

    log_event(
        "callback.scrape_complete",
        session_id=session.id,
        scrape_status=status,
        url_status=url_status,
        content_length=len(content) if content else 0,
    )
    return {"status": "updated"}
