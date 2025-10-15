"""Entry point for the scraper service."""

from __future__ import annotations

import uvicorn
from urllib.robotparser import RobotFileParser

from .app import app, create_app  # noqa: F401
from .app.auth_client import AuthenticationServiceClient
from .app.classification import URLClassifier
from .app.dependencies import (
    duplicate_detector as duplicate_detector_instance,
    parallel_engine,
    scraping_engine,
    url_classifier,
)
from .app.duplicates import DuplicateDetector
from .app.extraction import ContentExtractor, trafilatura
from .app.models import (
    AuthSession,
    ContentType,
    QueueType,
    RealTimeStatus,
    ScrapeJob,
    ScrapeRequest,
    ScrapedContent,
    URLClassification,
    URLRequest,
)
from .app.engine import ScrapingEngine
from .app.processing import ParallelProcessingEngine
from .app.queues import ProcessingQueues
from .app.spider import ContentSpider
from .app.robots import RobotsChecker
from .app.state import state

# Re-export mutable state for compatibility with existing tests/tooling
active_jobs = state.active_jobs
content_hashes = state.content_hashes
auth_sessions = state.auth_sessions
processing_queues = state.processing_queues
websocket_connections = state.websocket_connections

# Provide access to shared singletons
duplicate_detector = duplicate_detector_instance

__all__ = [
    "app",
    "create_app",
    "URLClassifier",
    "ContentExtractor",
    "DuplicateDetector",
    "ParallelProcessingEngine",
    "ProcessingQueues",
    "URLClassification",
    "ScrapeRequest",
    "ScrapeJob",
    "ScrapedContent",
    "URLRequest",
    "ContentType",
    "QueueType",
    "AuthSession",
    "RealTimeStatus",
    "RobotsChecker",
    "AuthenticationServiceClient",
    "ScrapingEngine",
    "ContentSpider",
    "RobotFileParser",
    "active_jobs",
    "content_hashes",
    "auth_sessions",
    "processing_queues",
    "websocket_connections",
    "url_classifier",
    "duplicate_detector",
    "parallel_engine",
    "scraping_engine",
    "trafilatura",
]

# Maintain backward compatibility for modules importing `main`
import sys as _sys

_sys.modules.setdefault("main", _sys.modules[__name__])


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
