"""Domain models for the export service."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExportFormat(str, Enum):
    """Supported export formats."""

    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"
    WORD = "word"
    NOTION = "notion"
    OBSIDIAN = "obsidian"
    PDF = "pdf"


class ExportStatus(str, Enum):
    """State of an export job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ExportFilter(BaseModel):
    """Filtering options applied when retrieving session data."""

    cluster_ids: Optional[List[int]] = None
    date_range: Optional[Dict[str, str]] = None
    content_types: Optional[List[str]] = None
    domains: Optional[List[str]] = None
    min_score: Optional[float] = None
    keywords: Optional[List[str]] = None


class ExportTemplate(BaseModel):
    """Representation of a template stored on disk."""

    name: str
    format: ExportFormat
    template_content: str
    variables: Dict[str, Any] = Field(default_factory=dict)


class ExportRequest(BaseModel):
    """Request body accepted by export endpoints."""

    session_id: str
    format: ExportFormat
    template_name: Optional[str] = None
    custom_template: Optional[str] = None
    filters: Optional[ExportFilter] = None
    include_metadata: bool = True
    include_clusters: bool = True
    include_embeddings: bool = False
    batch_size: Optional[int] = 1000
    notion_token: Optional[str] = None
    notion_database_id: Optional[str] = None


class ExportJob(BaseModel):
    """In-memory representation of an export job."""

    job_id: str
    session_id: str
    format: ExportFormat
    status: ExportStatus
    progress: float = 0.0
    total_items: int = 0
    processed_items: int = 0
    file_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


__all__ = [
    "ExportFormat",
    "ExportStatus",
    "ExportFilter",
    "ExportTemplate",
    "ExportRequest",
    "ExportJob",
]
