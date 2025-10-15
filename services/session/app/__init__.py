"""Application factory and shared objects for the Session service."""

from __future__ import annotations

from fastapi import FastAPI

from .logging import configure_logging
from .services import SessionService
from .storage import session_repository
from .qdrant import session_vector_store

# Configure structlog once at import time so every module shares the same setup.
configure_logging()

# Global service instance that routers obtain via dependency injection.
session_service = SessionService(
    repository=session_repository,
    vector_store=session_vector_store,
)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from .api import router as api_router

    app = FastAPI(
        title="Session Management Service",
        description="Handles persistent storage and incremental processing for analysis sessions",
        version="1.0.0",
    )

    app.include_router(api_router)
    return app


__all__ = ["create_app", "session_service"]
