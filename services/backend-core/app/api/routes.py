"""API routes for Backend Core service."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from typing import Optional
import httpx

from ..sessions.manager import SessionManager
from ..url_input.store import URLStore
from ..export.exporter import Exporter


# Global instances
session_manager = SessionManager()
exporter = Exporter()

router = APIRouter()


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


class ClusterRequest(BaseModel):
    session_id: str


# Health check
@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "backend-core"}


# Session endpoints
@router.post("/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreate):
    session = session_manager.create_session(request.name)
    return SessionResponse(
        id=session.id,
        name=session.name,
        total_urls=0,
        status=session.status
    )


@router.get("/sessions")
async def list_sessions():
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
async def get_session(session_id: str):
    stats = session_manager.get_session_stats(session_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Session not found")
    return stats


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if not session_manager.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


# URL endpoints
@router.post("/urls", response_model=URLInputResponse)
async def add_urls(request: URLInput):
    # Get or create session
    if request.session_id:
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = session_manager.get_or_create_current_session()
    
    added, duplicates, _ = session_manager.add_urls_to_session(
        session.id, request.urls
    )
    
    return URLInputResponse(
        session_id=session.id,
        added=added,
        duplicates=duplicates,
        total=session.url_store.count()
    )


@router.get("/urls/{session_id}")
async def get_urls(session_id: str, status: Optional[str] = None):
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
    background_tasks.add_task(trigger_scraping, session.id, urls)
    
    return {
        "status": "started",
        "session_id": session.id,
        "url_count": len(urls)
    }


async def trigger_scraping(session_id: str, urls: list[str]):
    """Background task to trigger browser engine scraping."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                "http://browser-engine:8083/scrape",
                json={"session_id": session_id, "urls": urls},
                timeout=30.0
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
            status_code=400, 
            detail="No scraped content available for clustering"
        )
    
    # Trigger AI engine clustering
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://ai-engine:8090/cluster",
                json={
                    "session_id": session.id,
                    "urls": [
                        {
                            "url": r.original,
                            "content": r.metadata.get("content", ""),
                            "title": r.metadata.get("title", ""),
                        }
                        for r in scraped
                    ]
                },
                timeout=120.0
            )
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
async def export_session(request: ExportRequest):
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        content = exporter.export(session, request.format)
        return {
            "format": request.format,
            "content": content,
            "filename": f"{session.name.replace(' ', '_')}.{request.format}"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Auth queue endpoints (proxy to browser engine)
@router.get("/auth/pending")
async def get_pending_auth():
    """Get pending authentication requests from browser engine."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://browser-engine:8083/auth/pending",
                timeout=10.0
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
                "http://browser-engine:8083/auth/credentials",
                json={"domain": domain, "credentials": credentials},
                timeout=10.0
            )
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Callback endpoint for browser engine
@router.post("/callback/scrape-complete")
async def scrape_complete_callback(data: dict):
    """Callback from browser engine when scraping is complete."""
    session_id = data.get("session_id")
    url = data.get("url")
    status = data.get("status")
    content = data.get("content")
    metadata = data.get("metadata", {})
    
    session = session_manager.get_session(session_id)
    if not session:
        return {"status": "error", "message": "Session not found"}
    
    # Update URL record
    session.url_store.update_status(
        url, 
        status,
        metadata={**metadata, "content": content} if content else metadata
    )
    
    return {"status": "updated"}
