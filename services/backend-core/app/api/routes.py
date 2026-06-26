"""API routes for Backend Core service."""

import hmac
import os
from typing import Any, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from pydantic import BaseModel, Field, HttpUrl

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


def _ai_engine_url() -> str:
    """Resolve the AI engine base URL from runtime environment."""
    return os.getenv("AI_ENGINE_URL", "http://ai-engine:8090").rstrip("/")


def _ai_engine_headers() -> dict[str, str]:
    """Return auth headers for AI Engine requests when host mode configures one."""
    token = os.getenv("AI_ENGINE_API_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


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


def _browser_engine_url() -> str:
    """Resolve the browser engine base URL from runtime environment."""
    return os.getenv("BROWSER_ENGINE_URL", "http://browser-engine:8083").rstrip("/")


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

    added, duplicates, _ = session_manager.add_urls_to_session(session.id, request.urls)

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
    else:
        pending = session.url_store.get_by_status("pending")
        urls = [r.original for r in pending]

    if not urls:
        return {"status": "no_urls", "message": "No URLs to scrape"}

    # Trigger browser engine scraping
    background_tasks.add_task(
        trigger_scraping,
        session.id,
        urls,
        request.use_browser,
    )

    return {"status": "started", "session_id": session.id, "url_count": len(urls)}


async def trigger_scraping(
    session_id: str,
    urls: list[str],
    use_browser: bool = False,
):
    """Background task to trigger browser engine scraping."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{_browser_engine_url()}/scrape",
                json={
                    "session_id": session_id,
                    "urls": urls,
                    "use_browser": use_browser,
                },
                timeout=30.0,
            )
    except Exception as e:
        print(f"Error triggering scraping: {e}")


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
                headers=_ai_engine_headers(),
                timeout=120.0,
            )
            response.raise_for_status()
            clusters = response.json().get("clusters", [])
            session_manager.set_session_clusters(session.id, clusters)

            return {"status": "completed", "clusters": clusters}
    except Exception as e:
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
    """Get scraping status for a session from browser engine."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{_browser_engine_url()}/scrape/status/{session_id}",
                timeout=10.0,
            )
            if response.status_code == 404:
                session = session_manager.get_session(session_id)
                if not session:
                    raise HTTPException(status_code=404, detail="Session not found")

                counts = session.url_store.count_by_status()
                total = session.url_store.count()
                completed = (
                    counts.get("scraped", 0)
                    + counts.get("failed", 0)
                    + counts.get("auth_required", 0)
                )
                status = (
                    "completed" if total > 0 and completed >= total else "not_started"
                )
                return {
                    "session_id": session_id,
                    "status": status,
                    "total": total,
                    "completed": completed,
                    "success": counts.get("scraped", 0),
                    "failed": counts.get("failed", 0),
                    "auth_required": counts.get("auth_required", 0),
                }
            response.raise_for_status()
            return response.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {e}")


# Auth queue endpoints (proxy to browser engine)
@router.get("/auth/pending")
async def get_pending_auth():
    """Get pending authentication requests from browser engine."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{_browser_engine_url()}/auth/pending", timeout=10.0
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
        return {"status": "error", "message": "URL not found in session"}

    return {"status": "updated"}
