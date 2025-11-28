"""Pydantic models and enums used by the Session service."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    """Lifecycle state for a session."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    SHARED = "shared"
    DELETED = "deleted"


class ProcessingStats(BaseModel):
    """Counters that summarise session processing activity."""

    urls_processed: int = 0
    content_analyzed: int = 0
    clusters_generated: int = 0
    embeddings_created: int = 0
    last_processing_time: Optional[datetime] = None


class ModelUsageHistory(BaseModel):
    """Tracks model usage for auditing and optimisation."""

    llm_models_used: List[str] = []
    embedding_models_used: List[str] = []
    model_switches: int = 0
    last_model_switch: Optional[datetime] = None


class SessionConfiguration(BaseModel):
    """User-configurable parameters for a session."""

    clustering_params: Dict[str, Any] = {}
    embedding_model: Optional[str] = None
    llm_model: Optional[str] = None
    processing_options: Dict[str, Any] = {}


class SessionModel(BaseModel):
    """The persisted representation of a session."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    owner_id: Optional[str] = None
    shared_with: List[str] = []
    qdrant_collection_name: str = Field(default_factory=lambda: f"session_{uuid.uuid4().hex}")
    configuration: SessionConfiguration = Field(default_factory=SessionConfiguration)
    processing_stats: ProcessingStats = Field(default_factory=ProcessingStats)
    model_usage_history: ModelUsageHistory = Field(default_factory=ModelUsageHistory)
    tags: List[str] = []
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateSessionRequest(BaseModel):
    """Payload for creating a new session."""

    name: str
    description: Optional[str] = None
    owner_id: Optional[str] = None
    configuration: Optional[SessionConfiguration] = None
    tags: List[str] = []
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpdateSessionRequest(BaseModel):
    """Patch-style updates for an existing session."""

    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[SessionStatus] = None
    configuration: Optional[SessionConfiguration] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class ShareSessionRequest(BaseModel):
    """Grant other users access to a session."""

    user_ids: List[str]
    permissions: List[str] = ["read"]


class SessionExportData(BaseModel):
    """Export package that bundles session metadata and optional Qdrant points."""

    session: SessionModel
    collection_data: Optional[Dict[str, Any]] = None
    export_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MergeSessionsRequest(BaseModel):
    """Payload describing how to merge multiple sessions into a new one."""

    source_session_ids: List[str]
    target_name: Optional[str] = None
    target_description: Optional[str] = None
    owner_id: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    archive_sources: bool = True


class SplitSessionPart(BaseModel):
    """Definition of a new session created from a subset of points."""

    name: str
    point_ids: List[str]
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SplitSessionRequest(BaseModel):
    """Payload describing how to split an existing session."""

    parts: List[SplitSessionPart]
    archive_original: bool = False
    remove_points: bool = True


class RetentionPolicy(BaseModel):
    """Retention policy used for automated cleanup operations."""

    max_age_days: Optional[int] = None
    max_sessions_per_user: Optional[int] = None
    auto_archive_inactive_days: Optional[int] = 30
    auto_delete_archived_days: Optional[int] = 90


__all__ = [
    "SessionStatus",
    "ProcessingStats",
    "ModelUsageHistory",
    "SessionConfiguration",
    "SessionModel",
    "CreateSessionRequest",
    "UpdateSessionRequest",
    "ShareSessionRequest",
    "SessionExportData",
    "MergeSessionsRequest",
    "SplitSessionRequest",
    "SplitSessionPart",
    "RetentionPolicy",
]
