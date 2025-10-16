"""Compatibility layer for the refactored export service package."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:  # Support execution as a script.
    sys.path.append(str(Path(__file__).resolve().parent))
    from app import (  # type: ignore[import-not-found]
        ExportFilter,
        ExportFormat,
        ExportJob,
        ExportRequest,
        ExportStatus,
        ExportTemplate,
        app,
        create_app,
        export_engine,
        export_jobs,
        logger,
        process_export_job,
        qdrant_client,
        settings,
    )
else:
    from .app import (  # pragma: no cover - exercised through imports
        ExportFilter,
        ExportFormat,
        ExportJob,
        ExportRequest,
        ExportStatus,
        ExportTemplate,
        app,
        create_app,
        export_engine,
        export_jobs,
        logger,
        process_export_job,
        qdrant_client,
        settings,
    )

# Preserve legacy module-level constants for downstream imports.
EXPORT_DIR = settings.export_dir
TEMPLATES_DIR = settings.templates_dir

__all__ = [
    "app",
    "create_app",
    "export_engine",
    "export_jobs",
    "process_export_job",
    "ExportFormat",
    "ExportStatus",
    "ExportFilter",
    "ExportTemplate",
    "ExportRequest",
    "ExportJob",
    "EXPORT_DIR",
    "TEMPLATES_DIR",
    "settings",
    "logger",
    "qdrant_client",
]


if __name__ == "__main__":  # pragma: no cover - manual execution support
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
