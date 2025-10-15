"""Session Management Service entrypoint."""

from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from app import create_app, session_service as _session_service  # type: ignore
    from app.models import (  # type: ignore
        CreateSessionRequest,
        ModelUsageHistory,
        ProcessingStats,
        RetentionPolicy,
        SessionConfiguration,
        SessionExportData,
        SessionModel,
        SessionStatus,
        ShareSessionRequest,
        UpdateSessionRequest,
    )
    from app.services import SessionService  # type: ignore
    from app.storage import session_repository  # type: ignore
    from app.qdrant import session_vector_store as _session_vector_store  # type: ignore
else:
    from .app import create_app, session_service as _session_service
    from .app.models import (
        CreateSessionRequest,
        ModelUsageHistory,
        ProcessingStats,
        RetentionPolicy,
        SessionConfiguration,
        SessionExportData,
        SessionModel,
        SessionStatus,
        ShareSessionRequest,
        UpdateSessionRequest,
    )
    from .app.services import SessionService
    from .app.storage import session_repository
    from .app.qdrant import session_vector_store as _session_vector_store

# Create FastAPI application
app: FastAPI = create_app()

# Compatibility exports for existing tests/imports
SessionService = SessionService
SessionModel = SessionModel
SessionStatus = SessionStatus
ProcessingStats = ProcessingStats
ModelUsageHistory = ModelUsageHistory
SessionConfiguration = SessionConfiguration
CreateSessionRequest = CreateSessionRequest
UpdateSessionRequest = UpdateSessionRequest
ShareSessionRequest = ShareSessionRequest
SessionExportData = SessionExportData
RetentionPolicy = RetentionPolicy

# Provide direct access to in-memory storage for tests expecting the old dict.
sessions_storage = session_repository.data
session_vector_store = _session_vector_store
session_service = _session_service


__all__ = [
    "app",
    "session_service",
    "SessionService",
    "SessionModel",
    "SessionStatus",
    "ProcessingStats",
    "ModelUsageHistory",
    "SessionConfiguration",
    "CreateSessionRequest",
    "UpdateSessionRequest",
    "ShareSessionRequest",
    "SessionExportData",
    "RetentionPolicy",
    "sessions_storage",
    "session_vector_store",
]
