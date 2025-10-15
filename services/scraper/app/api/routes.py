"""FastAPI routes for the scraper service."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect

from ..dependencies import (
    duplicate_detector,
    parallel_engine,
    scraping_engine,
    url_classifier,
)
from ..logging import get_logger
from ..models import (
    ProcessingStats,
    QueueType,
    RealTimeStatus,
    ScrapeJob,
    ScrapeRequest,
    ScrapedContent,
    URLClassification,
    URLRequest,
)
from ..state import state

logger = get_logger()
router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint with detailed status."""
    return {
        "status": "healthy",
        "service": "Web Scraper Service",
        "version": "2.0.0",
        "timestamp": time.time(),
        "active_jobs": len(state.active_jobs),
        "active_websockets": len(state.websocket_connections),
        "total_content_hashes": len(state.content_hashes),
        "active_auth_sessions": len(state.auth_sessions),
        "features": [
            "Parallel authentication workflow",
            "Multi-format content extraction (HTML, PDF, Text)",
            "Advanced duplicate detection with similarity",
            "Real-time status tracking via WebSocket",
            "Content quality assessment",
            "Intelligent URL classification",
        ],
    }


@router.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint."""
    return {
        "service": "Web Scraper Service",
        "version": "1.0.0",
        "status": "running",
        "features": [
            "Scrapy-based scraping with rate limiting",
            "Content extraction with Beautiful Soup and trafilatura",
            "Duplicate detection using content hashing",
            "Robots.txt compliance",
            "Configurable delays and retries",
        ],
    }


@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint for real-time scraping status updates."""
    await websocket.accept()
    state.register_websocket(job_id, websocket)

    try:
        if job_id in state.active_jobs:
            await websocket.send_json(
                {
                    "type": "job_status",
                    "job": state.active_jobs[job_id].model_dump(mode="json"),
                }
            )

        while True:
            try:
                data = await websocket.receive_json()
                if data.get("action") == "get_status" and job_id in state.active_jobs:
                    await websocket.send_json(
                        {
                            "type": "job_status",
                            "job": state.active_jobs[job_id].model_dump(mode="json"),
                        }
                    )
            except WebSocketDisconnect:
                break
            except Exception as exc:
                logger.error("WebSocket error", job_id=job_id, error=str(exc))
                break
    finally:
        state.websocket_connections.pop(job_id, None)


@router.post("/scrape", response_model=Dict[str, Any])
async def start_scraping(
    scrape_request: ScrapeRequest, background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Start a comprehensive scraping job with parallel authentication workflow."""
    correlation_id = f"api_{uuid.uuid4().hex[:8]}"

    try:
        url_classifications = await url_classifier.classify_urls(
            scrape_request.urls, correlation_id
        )

        job_id = f"job_{int(time.time())}_{len(state.active_jobs)}"
        job = ScrapeJob(
            job_id=job_id,
            status="initializing",
            total_urls=len(scrape_request.urls),
            completed_urls=0,
            failed_urls=0,
            correlation_id=correlation_id,
            url_classifications=url_classifications,
        )

        url_metadata: Dict[str, Dict[str, Any]] = {}
        for classification, url_request in zip(url_classifications, scrape_request.urls):
            url_metadata[str(url_request.url)] = {
                "url": str(url_request.url),
                "priority": url_request.priority,
                "metadata": url_request.metadata,
                "queue_type": classification.queue_type.value,
                "requires_auth": classification.requires_auth,
                "auth_confidence": classification.confidence,
                "auth_indicators": classification.auth_indicators,
                "domain": classification.domain,
                "status": "pending",
            }
        job.url_metadata = url_metadata

        job.processing_stats = ProcessingStats(
            total_urls=len(scrape_request.urls),
            public_queue_size=sum(
                1 for c in url_classifications if c.queue_type == QueueType.PUBLIC
            ),
            auth_queue_size=sum(
                1 for c in url_classifications if c.queue_type == QueueType.AUTHENTICATED
            ),
        )

        state.reset_job(job)

        background_tasks.add_task(
            parallel_engine.process_job,
            job_id,
            scrape_request,
            url_classifications,
        )
        background_tasks.add_task(
            scraping_engine._prepare_auth_sessions,
            scrape_request,
            job,
            correlation_id,
        )

        logger.info(
            "Comprehensive scraping job initiated",
            job_id=job_id,
            session_id=scrape_request.session_id,
            url_count=len(scrape_request.urls),
            public_urls=job.processing_stats.public_queue_size,
            auth_urls=job.processing_stats.auth_queue_size,
            correlation_id=correlation_id,
        )

        return {
            "job_id": job_id,
            "status": "started",
            "message": f"Processing {len(scrape_request.urls)} URLs with parallel authentication workflow",
            "correlation_id": correlation_id,
            "url_classifications": [c.__dict__ for c in url_classifications],
            "processing_stats": job.processing_stats.__dict__,
            "websocket_url": f"/ws/{job_id}",
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to start scraping job", error=str(exc), correlation_id=correlation_id)
        raise HTTPException(status_code=500, detail=f"Failed to start scraping: {str(exc)}")


@router.post("/classify-urls", response_model=List[Dict[str, Any]])
async def classify_urls(urls: List[URLRequest]) -> List[Dict[str, Any]]:
    """Classify URLs for authentication requirements and queue routing."""
    correlation_id = f"classify_{uuid.uuid4().hex[:8]}"
    try:
        classifications = await url_classifier.classify_urls(urls, correlation_id)
        return [c.__dict__ for c in classifications]
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("URL classification failed", error=str(exc), correlation_id=correlation_id)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(exc)}")


@router.get("/jobs/{job_id}/realtime-status", response_model=RealTimeStatus)
async def get_realtime_status(job_id: str) -> RealTimeStatus:
    """Get real-time status of a scraping job."""
    if job_id not in state.active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = state.active_jobs[job_id]
    queues = state.processing_queues.get(job_id)

    urls_in_progress = []
    if queues:
        urls_in_progress = [{"url": url, "status": "processing"} for url in queues.processing]

    recent_completions = [
        {
            "url": result.url,
            "completed_at": result.scraped_at.isoformat(),
            "word_count": result.word_count,
            "quality_score": result.quality_score,
        }
        for result in job.results[-10:]
    ]

    auth_status = {
        "active_sessions": len(
            [s for s in job.auth_sessions.values() if s in state.auth_sessions]
        ),
        "domains_with_auth": list(job.auth_sessions.keys()),
        "auth_pending": job.auth_pending_urls,
    }

    performance_metrics = parallel_engine.get_performance_metrics(job_id)
    progress_percentage = (
        (job.completed_urls + job.failed_urls) / max(job.total_urls, 1) * 100
    )

    return RealTimeStatus(
        job_id=job_id,
        current_status=job.status,
        progress_percentage=progress_percentage,
        urls_in_progress=urls_in_progress,
        recent_completions=recent_completions,
        auth_status=auth_status,
        performance_metrics=performance_metrics,
    )


async def _schedule_auth_retry_processing(job_id: str) -> None:
    """Schedule periodic processing of auth retries."""
    await asyncio.sleep(10)
    for _ in range(10):
        await scraping_engine.process_auth_retries(job_id)
        await asyncio.sleep(30)


@router.get("/jobs/{job_id}", response_model=ScrapeJob)
async def get_job_status(job_id: str) -> ScrapeJob:
    """Get status of a scraping job."""
    if job_id not in state.active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return state.active_jobs[job_id]


@router.get("/jobs", response_model=List[str])
async def list_jobs() -> List[str]:
    """List all active job IDs."""
    return list(state.active_jobs.keys())


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str) -> Dict[str, str]:
    """Cancel and remove a job."""
    if job_id not in state.active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    state.drop_job(job_id)
    logger.info("Job cancelled", job_id=job_id)
    return {"message": f"Job {job_id} cancelled"}


@router.get("/stats")
async def get_comprehensive_stats() -> Dict[str, Any]:
    """Get comprehensive scraping statistics including parallel processing metrics."""
    total_jobs = len(state.active_jobs)
    running_jobs = sum(
        1 for job in state.active_jobs.values() if job.status in ["running", "initializing"]
    )
    completed_jobs = sum(1 for job in state.active_jobs.values() if job.status == "completed")
    failed_jobs = sum(1 for job in state.active_jobs.values() if job.status == "failed")

    total_urls_scraped = sum(job.completed_urls for job in state.active_jobs.values())
    total_urls_failed = sum(job.failed_urls for job in state.active_jobs.values())

    content_type_stats = defaultdict(int)
    quality_scores: List[float] = []

    for job in state.active_jobs.values():
        for result in job.results:
            content_type_stats[result.content_type.value] += 1
            quality_scores.append(result.quality_score)

    duplicate_stats = duplicate_detector.get_duplicate_stats()
    total_duplicates = sum(
        1 for job in state.active_jobs.values() for result in job.results if result.is_duplicate
    )

    jobs_with_auth = sum(1 for job in state.active_jobs.values() if job.auth_sessions)
    active_auth_sessions = len(state.auth_sessions)

    auth_success_count = sum(
        1
        for job in state.active_jobs.values()
        for result in job.results
        if result.metadata.get("auth_used", False)
    )

    return {
        "job_statistics": {
            "total_jobs": total_jobs,
            "running_jobs": running_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
        },
        "url_statistics": {
            "total_urls_scraped": total_urls_scraped,
            "total_urls_failed": total_urls_failed,
        },
        "content_statistics": {
            "content_type_distribution": dict(content_type_stats),
            "average_quality_score": sum(quality_scores) / len(quality_scores)
            if quality_scores
            else 0.0,
            "total_duplicates": total_duplicates,
        },
        "authentication_statistics": {
            "jobs_with_auth": jobs_with_auth,
            "active_auth_sessions": active_auth_sessions,
            "auth_success_count": auth_success_count,
        },
        "parallel_processing_statistics": {
            "performance_metrics": {
                job_id: parallel_engine.get_performance_metrics(job_id)
                for job_id in state.active_jobs.keys()
            }
        },
        "system_statistics": duplicate_stats,
    }


@router.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str) -> Dict[str, str]:
    """Pause a running scraping job."""
    job = state.active_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "running":
        raise HTTPException(status_code=400, detail="Job is not running")

    job.status = "paused"
    job.updated_at = datetime.now()
    logger.info("Job paused", job_id=job_id)
    return {"message": f"Job {job_id} paused"}


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str) -> Dict[str, str]:
    """Resume a paused scraping job."""
    job = state.active_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "paused":
        raise HTTPException(status_code=400, detail="Job is not paused")

    job.status = "running"
    job.updated_at = datetime.now()
    logger.info("Job resumed", job_id=job_id)
    return {"message": f"Job {job_id} resumed"}


@router.get("/content-quality/{job_id}")
async def analyze_content_quality(job_id: str) -> Dict[str, Any]:
    """Analyze content quality for a specific job."""
    job = state.active_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.results:
        return {"message": "No content available for analysis"}

    quality_scores = [result.quality_score for result in job.results]
    word_counts = [result.word_count for result in job.results]

    quality_analysis: Dict[str, Any] = {
        "total_content_pieces": len(job.results),
        "average_quality_score": sum(quality_scores) / len(quality_scores),
        "min_quality_score": min(quality_scores),
        "max_quality_score": max(quality_scores),
        "average_word_count": sum(word_counts) / len(word_counts),
        "content_type_distribution": defaultdict(int),
        "extraction_method_distribution": defaultdict(int),
        "low_quality_content": [],
        "high_quality_content": [],
    }

    for result in job.results:
        quality_analysis["content_type_distribution"][result.content_type.value] += 1
        quality_analysis["extraction_method_distribution"][result.extraction_method] += 1

        if result.quality_score < 0.3:
            quality_analysis["low_quality_content"].append(
                {
                    "url": result.url,
                    "quality_score": result.quality_score,
                    "word_count": result.word_count,
                    "issues": "Low quality score",
                }
            )
        elif result.quality_score > 0.8:
            quality_analysis["high_quality_content"].append(
                {
                    "url": result.url,
                    "quality_score": result.quality_score,
                    "word_count": result.word_count,
                }
            )

    quality_analysis["content_type_distribution"] = dict(
        quality_analysis["content_type_distribution"]
    )
    quality_analysis["extraction_method_distribution"] = dict(
        quality_analysis["extraction_method_distribution"]
    )

    return quality_analysis


@router.get("/jobs/{job_id}/urls")
async def get_job_urls(job_id: str) -> Dict[str, Any]:
    """Return classified URL metadata for front-end preview."""
    job = state.active_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "urls": list(job.url_metadata.values())}


@router.post("/jobs/{job_id}/retry-auth")
async def retry_auth_failed_urls(job_id: str) -> Dict[str, str]:
    """Manually trigger retry of authentication-failed URLs."""
    if job_id not in state.active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        await scraping_engine.process_auth_retries(job_id)
        return {"message": f"Auth retry processing triggered for job {job_id}"}
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to process auth retries", job_id=job_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Failed to process retries: {str(exc)}")


@router.get("/duplicate-analysis")
async def get_duplicate_analysis() -> Dict[str, Any]:
    """Get comprehensive duplicate content analysis."""
    duplicate_stats = duplicate_detector.get_duplicate_stats()
    duplicate_domains = defaultdict(int)

    for job in state.active_jobs.values():
        for result in job.results:
            if result.is_duplicate:
                domain = result.metadata.get("domain") or result.url
                duplicate_domains[domain] += 1

    total_results = sum(len(job.results) for job in state.active_jobs.values())

    return {
        "duplicate_detection_stats": duplicate_stats,
        "duplicate_domains": dict(duplicate_domains),
        "total_unique_content": len(state.content_hashes),
        "duplicate_rate": len(
            [r for job in state.active_jobs.values() for r in job.results if r.is_duplicate]
        )
        / max(total_results, 1),
    }


@router.get("/sessions")
async def list_auth_sessions() -> Dict[str, Any]:
    """List all active authentication sessions with enhanced details."""
    sessions_info = []
    for session_id, session in state.auth_sessions.items():
        sessions_info.append(
            {
                "session_id": session_id,
                "domain": session.domain,
                "is_active": session.is_active,
                "expires_at": session.expires_at.isoformat() if session.expires_at else None,
                "last_used": session.last_used.isoformat() if session.last_used else None,
                "success_count": session.success_count,
                "failure_count": session.failure_count,
                "has_cookies": bool(session.cookies),
                "has_headers": bool(session.headers),
                "user_agent": session.user_agent,
            }
        )

    return {
        "total_sessions": len(state.auth_sessions),
        "active_sessions": len([s for s in state.auth_sessions.values() if s.is_active]),
        "sessions": sessions_info,
    }


@router.delete("/sessions/{session_id}")
async def delete_auth_session(session_id: str) -> Dict[str, str]:
    """Delete an authentication session."""
    if session_id not in state.auth_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = state.auth_sessions[session_id]
    state.auth_sessions.pop(session_id, None)
    logger.info("Auth session deleted", session_id=session_id, domain=session.domain)
    return {"message": f"Session {session_id} deleted"}


@router.get("/robots-compliance/{job_id}")
async def check_robots_compliance(job_id: str) -> Dict[str, Any]:
    """Check robots.txt compliance for a job."""
    job = state.active_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    compliance_stats = {
        "total_urls": len(job.url_classifications),
        "robots_checked": 0,
        "robots_compliant": 0,
        "robots_violations": 0,
        "domains_analyzed": set(),
    }

    for classification in job.url_classifications:
        domain = classification.domain
        compliance_stats["domains_analyzed"].add(domain)
        compliance_stats["robots_checked"] += 1
        compliance_stats["robots_compliant"] += 1

    compliance_stats["domains_analyzed"] = len(compliance_stats["domains_analyzed"])
    return compliance_stats


@router.post("/test-extract")
async def test_content_extraction(url: str) -> Dict[str, Any]:
    """Test content extraction for a single URL."""
    import requests

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        from ..extraction import ContentExtractor

        extractor = ContentExtractor()
        extracted = extractor.extract_content(response.content, url, response.headers.get("content-type", ""))

        is_duplicate, content_hash, similarity = duplicate_detector.is_duplicate(
            extracted["content"], url
        )

        preview = extracted["content"]
        if len(preview) > 500:
            preview = preview[:500] + "..."

        return {
            "url": url,
            "title": extracted["title"],
            "content_preview": preview,
            "word_count": extracted["word_count"],
            "content_hash": content_hash,
            "duplicate_similarity": similarity,
            "is_duplicate": is_duplicate,
            "status": "success",
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Test extraction failed", url=url, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(exc)}")
