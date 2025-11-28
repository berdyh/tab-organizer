"""In-memory state management for the scraper service."""

from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

from fastapi import WebSocket

from .models import AuthSession, ScrapeJob

if TYPE_CHECKING:
    from .queues import ProcessingQueues


class InMemoryState:
    """Holds in-memory state for active scraping jobs and shared resources."""

    def __init__(self) -> None:
        self.active_jobs: Dict[str, ScrapeJob] = {}
        self.content_hashes: Dict[str, str] = {}
        self.auth_sessions: Dict[str, AuthSession] = {}
        self.processing_queues: Dict[str, "ProcessingQueues"] = {}
        self.websocket_connections: Dict[str, WebSocket] = {}

    def reset_job(self, job: ScrapeJob) -> None:
        """Register a new job and prepare its processing queues."""
        from .queues import ProcessingQueues  # Local import to avoid circular dependency

        self.active_jobs[job.job_id] = job
        self.processing_queues[job.job_id] = ProcessingQueues()

    def drop_job(self, job_id: str) -> None:
        """Remove job-specific state."""
        self.active_jobs.pop(job_id, None)
        self.processing_queues.pop(job_id, None)
        self.websocket_connections.pop(job_id, None)

    def register_websocket(self, job_id: str, websocket: WebSocket) -> None:
        """Add a websocket connection for a job."""
        self.websocket_connections[job_id] = websocket

    def get_websocket(self, job_id: str) -> Optional[WebSocket]:
        """Return websocket connection for job if any."""
        return self.websocket_connections.get(job_id)


state = InMemoryState()
