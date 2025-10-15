"""Data models and schemas for the scraper service."""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl


class URLStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    SCRAPING = "scraping"
    COMPLETED = "completed"
    FAILED = "failed"
    AUTH_REQUIRED = "auth_required"
    AUTH_PENDING = "auth_pending"
    RETRYING = "retrying"


class ContentType(str, Enum):
    HTML = "html"
    PDF = "pdf"
    TEXT = "text"
    JSON = "json"
    XML = "xml"
    UNKNOWN = "unknown"


class QueueType(str, Enum):
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    RETRY = "retry"


@dataclass
class URLClassification:
    url: str
    queue_type: QueueType
    requires_auth: bool
    confidence: float
    auth_indicators: List[str] = field(default_factory=list)
    domain: str = ""
    priority: int = 1


@dataclass
class ProcessingStats:
    total_urls: int = 0
    public_queue_size: int = 0
    auth_queue_size: int = 0
    retry_queue_size: int = 0
    completed: int = 0
    failed: int = 0
    auth_pending: int = 0
    processing_rate: float = 0.0
    avg_response_time: float = 0.0


class URLRequest(BaseModel):
    url: HttpUrl
    priority: int = Field(default=1, ge=1, le=10)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    force_auth_check: bool = False
    content_type_hint: Optional[ContentType] = None


class ScrapeRequest(BaseModel):
    urls: List[URLRequest]
    session_id: str
    rate_limit_delay: float = Field(default=1.0, ge=0.1, le=10.0)
    respect_robots: bool = True
    max_retries: int = Field(default=3, ge=1, le=10)
    parallel_auth: bool = True
    max_concurrent_workers: int = Field(default=5, ge=1, le=20)
    content_quality_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    enable_pdf_extraction: bool = True


class ScrapedContent(BaseModel):
    url: str
    title: str
    content: str
    content_hash: str
    content_type: ContentType
    metadata: Dict[str, Any]
    scraped_at: datetime
    word_count: int
    quality_score: float
    is_duplicate: bool = False
    extraction_method: str = ""
    response_time_ms: int = 0


class AuthSession(BaseModel):
    session_id: str
    domain: str
    cookies: Dict[str, Any] = Field(default_factory=dict)
    headers: Dict[str, str] = Field(default_factory=dict)
    user_agent: Optional[str] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True
    last_used: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0


class ScrapeJob(BaseModel):
    job_id: str
    status: str
    total_urls: int
    completed_urls: int
    failed_urls: int
    auth_pending_urls: int = 0
    results: List[ScrapedContent] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    auth_sessions: Dict[str, str] = Field(default_factory=dict)
    correlation_id: str = ""
    started_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    processing_stats: ProcessingStats = Field(default_factory=ProcessingStats)
    url_classifications: List[URLClassification] = Field(default_factory=list)
    url_metadata: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    job_token: str = Field(default_factory=lambda: uuid.uuid4().hex)


class RealTimeStatus(BaseModel):
    job_id: str
    current_status: str
    progress_percentage: float
    urls_in_progress: List[Dict[str, Any]]
    recent_completions: List[Dict[str, Any]]
    auth_status: Dict[str, Any]
    performance_metrics: Dict[str, Any]
