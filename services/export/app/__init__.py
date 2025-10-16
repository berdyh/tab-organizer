"""Export service package initialisation."""

from __future__ import annotations

from notion_client import Client as NotionClient

from .clients import create_qdrant_client
from .config import ExportSettings
from .engine import ExportEngine
from .factory import create_app
from .logging import configure_logging
from .models import (
    ExportFilter,
    ExportFormat,
    ExportJob,
    ExportRequest,
    ExportStatus,
    ExportTemplate,
)
from .state import export_jobs
from .tasks import process_export_job as _process_export_job

settings = ExportSettings()
logger = configure_logging()
qdrant_client = create_qdrant_client(settings)
export_engine = ExportEngine(
    qdrant_client=qdrant_client,
    templates_dir=settings.templates_dir,
    notion_client_factory=NotionClient,
)

app = create_app(settings=settings, engine=export_engine, job_store=export_jobs, logger=logger)


async def process_export_job(job_id: str, request: ExportRequest) -> None:
    """Compatibility wrapper matching the previous module-level signature."""
    await _process_export_job(job_id, request, export_engine, export_jobs, settings, logger)


__all__ = [
    "app",
    "create_app",
    "export_engine",
    "export_jobs",
    "process_export_job",
    "ExportFilter",
    "ExportFormat",
    "ExportJob",
    "ExportRequest",
    "ExportStatus",
    "ExportTemplate",
    "settings",
    "logger",
    "qdrant_client",
    "NotionClient",
]
