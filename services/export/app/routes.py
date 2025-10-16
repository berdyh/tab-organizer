"""FastAPI routes for the export service."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import FileResponse

from .models import (
    ExportFilter,
    ExportFormat,
    ExportJob,
    ExportRequest,
    ExportStatus,
    ExportTemplate,
)
from .tasks import process_export_job as process_export_job_task

router = APIRouter()


def get_engine(request: Request):
    return request.app.state.export_engine


def get_jobs(request: Request):
    return request.app.state.export_jobs


def get_settings(request: Request):
    return request.app.state.settings


def get_logger(request: Request):
    return request.app.state.logger


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "export", "timestamp": time.time()}


@router.get("/")
async def root(request: Request) -> Dict[str, Any]:
    """Root endpoint returning metadata."""
    settings = get_settings(request)
    return {
        "service": settings.service_name,
        "version": settings.version,
        "status": "running",
        "supported_formats": [format.value for format in ExportFormat],
    }


@router.get("/formats", response_model=List[Dict[str, Any]])
async def get_supported_formats() -> List[Dict[str, Any]]:
    """Return supported export formats and their metadata."""
    return [
        {
            "format": export_format.value,
            "description": description,
            "file_extension": extension,
            "supports_templates": supports_templates,
        }
        for export_format, description, extension, supports_templates in [
            (ExportFormat.JSON, "JavaScript Object Notation - structured data format", "json", True),
            (ExportFormat.CSV, "Comma-Separated Values - spreadsheet compatible format", "csv", False),
            (ExportFormat.MARKDOWN, "Markdown format - human-readable text format", "md", True),
            (ExportFormat.OBSIDIAN, "Obsidian-compatible Markdown with internal linking", "md", True),
            (ExportFormat.WORD, "Microsoft Word document format", "docx", False),
            (ExportFormat.NOTION, "Export to Notion database", None, False),
            (ExportFormat.PDF, "Portable Document Format - professional document format", "pdf", False),
        ]
    ]


@router.post("/export", response_model=Dict[str, Any])
async def create_export(
    export_request: ExportRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> Dict[str, Any]:
    """Create a new export job."""
    import uuid

    job_id = str(uuid.uuid4())
    logger = get_logger(request)
    job = ExportJob(
        job_id=job_id,
        session_id=export_request.session_id,
        format=export_request.format,
        status=ExportStatus.PENDING,
        created_at=datetime.now(),
    )

    jobs = get_jobs(request)
    jobs[job_id] = job

    background_tasks.add_task(
        process_export_job_task,
        job_id,
        export_request,
        get_engine(request),
        jobs,
        get_settings(request),
        logger,
    )

    logger.info("export_job_created", job_id=job_id, session_id=export_request.session_id, format=export_request.format.value)

    return {"job_id": job_id, "status": job.status, "message": "Export job created successfully"}


@router.get("/export/{job_id}/status", response_model=ExportJob)
async def get_export_status(job_id: str, request: Request) -> ExportJob:
    """Fetch the status of an export job."""
    jobs = get_jobs(request)
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Export job not found")
    return jobs[job_id]


@router.get("/export/{job_id}/download")
async def download_export(job_id: str, request: Request) -> FileResponse:
    """Download a completed export."""
    jobs = get_jobs(request)
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Export job not found")

    job = jobs[job_id]
    if job.status != ExportStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Export job not completed")

    if not job.file_path or not Path(job.file_path).exists():
        raise HTTPException(status_code=404, detail="Export file not found")

    filename = f"export_{job.session_id}_{job.format.value}_{job_id[:8]}"
    return FileResponse(path=job.file_path, filename=filename, media_type="application/octet-stream")


@router.post("/export/batch", response_model=Dict[str, Any])
async def create_batch_export(
    export_requests: List[ExportRequest],
    background_tasks: BackgroundTasks,
    request: Request,
) -> Dict[str, Any]:
    """Create multiple export jobs."""
    import uuid

    jobs = get_jobs(request)
    logger = get_logger(request)
    job_ids: List[str] = []

    for export_request in export_requests:
        job_id = str(uuid.uuid4())
        job = ExportJob(
            job_id=job_id,
            session_id=export_request.session_id,
            format=export_request.format,
            status=ExportStatus.PENDING,
            created_at=datetime.now(),
        )
        jobs[job_id] = job
        job_ids.append(job_id)

        background_tasks.add_task(
            process_export_job_task,
            job_id,
            export_request,
            get_engine(request),
            jobs,
            get_settings(request),
            logger,
        )

    logger.info("export_batch_created", job_count=len(job_ids))

    return {
        "job_ids": job_ids,
        "total_jobs": len(job_ids),
        "message": "Batch export jobs created successfully",
    }


@router.get("/export/jobs", response_model=List[ExportJob])
async def list_export_jobs(
    request: Request,
    session_id: Optional[str] = Query(None),
    status: Optional[ExportStatus] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
) -> List[ExportJob]:
    """List export jobs with optional filters."""
    jobs = list(get_jobs(request).values())

    if session_id:
        jobs = [job for job in jobs if job.session_id == session_id]

    if status:
        jobs = [job for job in jobs if job.status == status]

    jobs.sort(key=lambda job: job.created_at, reverse=True)
    return jobs[:limit]


@router.delete("/export/{job_id}")
async def delete_export_job(job_id: str, request: Request) -> Dict[str, str]:
    """Delete an export job and its associated file."""
    jobs = get_jobs(request)
    logger = get_logger(request)

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Export job not found")

    job = jobs[job_id]
    if job.file_path and Path(job.file_path).exists():
        Path(job.file_path).unlink(missing_ok=True)

    del jobs[job_id]
    logger.info("export_job_deleted", job_id=job_id)
    return {"message": "Export job deleted successfully"}


@router.post("/templates", response_model=Dict[str, Any])
async def create_template(template: ExportTemplate, request: Request) -> Dict[str, Any]:
    """Create a custom template."""
    templates_dir = get_settings(request).templates_dir
    template_path = templates_dir / f"{template.name}_{template.format.value}.j2"

    if template_path.exists():
        raise HTTPException(status_code=400, detail="Template already exists")

    template_path.write_text(template.template_content)

    get_logger(request).info("template_created", name=template.name, format=template.format.value)
    return {
        "message": "Template created successfully",
        "template_name": template.name,
        "template_path": str(template_path),
    }


@router.get("/templates", response_model=List[Dict[str, Any]])
async def list_templates(request: Request) -> List[Dict[str, Any]]:
    """List available templates."""
    templates_dir = get_settings(request).templates_dir
    templates: List[Dict[str, Any]] = []

    for template_file in templates_dir.glob("*.j2"):
        name_parts = template_file.stem.split("_")
        if len(name_parts) >= 2:
            name = "_".join(name_parts[:-1])
            format_name = name_parts[-1]
        else:
            name = template_file.stem
            format_name = "unknown"

        templates.append(
            {
                "name": name,
                "format": format_name,
                "file_path": str(template_file),
                "created_at": datetime.fromtimestamp(template_file.stat().st_mtime).isoformat(),
            }
        )

    return templates


@router.get("/templates/{template_name}")
async def get_template(template_name: str, format: ExportFormat, request: Request) -> Dict[str, Any]:
    """Return the content of a specific template."""
    template_path = get_settings(request).templates_dir / f"{template_name}_{format.value}.j2"

    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Template not found")

    content = template_path.read_text()
    return {
        "name": template_name,
        "format": format.value,
        "content": content,
        "file_path": str(template_path),
    }


@router.post("/export/preview", response_model=Dict[str, Any])
async def preview_export(
    session_id: str,
    format: ExportFormat,
    request: Request,
    template_content: Optional[str] = None,
    filters: Optional[ExportFilter] = None,
    limit: int = Query(5, ge=1, le=50),
) -> Dict[str, Any]:
    """Preview export output with limited data."""
    engine = get_engine(request)

    data = await engine.get_session_data(session_id, filters)
    data["items"] = data["items"][:limit]
    data["clusters"] = [
        {**cluster, "items": cluster["items"][:limit]}
        for cluster in data["clusters"][:3]
    ]
    data["total_items"] = len(data["items"])
    data["total_clusters"] = len(data["clusters"])

    if format == ExportFormat.JSON:
        preview = await engine.export_to_json(data, template_content)
    elif format == ExportFormat.CSV:
        preview = await engine.export_to_csv(data)
    elif format == ExportFormat.MARKDOWN:
        preview = await engine.export_to_markdown(data, template_content)
    elif format == ExportFormat.OBSIDIAN:
        preview = await engine.export_to_obsidian(data, template_content)
    else:
        preview = "Preview not available for this format"

    return {
        "preview": preview[:2000],
        "truncated": len(preview) > 2000,
        "sample_items": len(data["items"]),
        "sample_clusters": len(data["clusters"]),
    }
