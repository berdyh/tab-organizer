"""Application factory for the export service."""

from __future__ import annotations

from fastapi import FastAPI

from .config import ExportSettings
from .engine import ExportEngine
from .logging import configure_logging
from .routes import router
from .state import export_jobs


def create_app(
    settings: ExportSettings,
    engine: ExportEngine,
    job_store,
    logger,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    configure_logging()

    app = FastAPI(
        title=settings.service_name,
        description="Multi-format export system for web scraping and clustering results",
        version=settings.version,
    )

    app.state.settings = settings
    app.state.export_engine = engine
    app.state.export_jobs = job_store
    app.state.logger = logger

    app.include_router(router)

    return app


__all__ = ["create_app"]
