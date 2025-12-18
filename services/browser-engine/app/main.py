"""Browser Engine Service - Main Application."""

import asyncio
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import httpx

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
scraper = ScraperEngine(
    max_concurrent=int(os.getenv("MAX_CONCURRENT_SCRAPES", 10)),
    timeout=int(os.getenv("SCRAPE_TIMEOUT", 30)),
    respect_robots=os.getenv("RESPECT_ROBOTS", "true").lower() == "true",
)
scraper.set_auth_queue(auth_queue)
scraper.set_auth_detector(auth_detector)

# Scraping state
scraping_tasks: dict[str, dict] = {}  # session_id → task info


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
    scraping_tasks[session_id] = {
        "total": len(request.urls),
        "completed": 0,
        "success": 0,
        "failed": 0,
        "auth_required": 0,
        "status": "running",
    }
    
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


async def scrape_urls_background(
    session_id: str,
    urls: list[str],
    use_browser: bool,
):
    """Background task for scraping URLs."""
    backend_url = os.getenv("BACKEND_URL", "http://backend-core:8080")
    
    async def on_result(result):
        """Callback for each scrape result."""
        task_info = scraping_tasks.get(session_id, {})
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
                await client.post(
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
                    timeout=10.0,
                )
        except Exception:
            pass
    
    try:
        results = await scraper.scrape_batch(
            urls=urls,
            session_id=session_id,
            callback=on_result,
        )
        
        scraping_tasks[session_id]["status"] = "completed"
        
        # Index successful results in AI engine
        ai_url = os.getenv("AI_ENGINE_URL", "http://ai-engine:8090")
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
                    await client.post(
                        f"{ai_url}/index",
                        json={
                            "session_id": session_id,
                            "documents": documents,
                        },
                        timeout=120.0,
                    )
            except Exception:
                pass
                
    except Exception as e:
        scraping_tasks[session_id]["status"] = "failed"
        scraping_tasks[session_id]["error"] = str(e)
    finally:
        await scraper.close()


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
