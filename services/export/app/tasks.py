"""Background tasks for processing export jobs."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict

from fastapi import HTTPException
from structlog.stdlib import BoundLogger

from .config import ExportSettings
from .engine import ExportEngine
from .models import ExportFormat, ExportJob, ExportRequest, ExportStatus


async def process_export_job(
    job_id: str,
    request: ExportRequest,
    engine: ExportEngine,
    job_store: Dict[str, ExportJob],
    settings: ExportSettings,
    logger: BoundLogger,
) -> None:
    """Execute an export job and update its lifecycle state."""
    job = job_store[job_id]

    try:
        job.status = ExportStatus.PROCESSING
        logger.info("export_job_started", job_id=job_id, format=request.format.value)

        data = await engine.get_session_data(request.session_id, request.filters)
        job.total_items = data["total_items"]

        template_content = None
        templates_dir = settings.templates_dir
        if request.custom_template:
            template_content = request.custom_template
        elif request.template_name:
            template_path = templates_dir / f"{request.template_name}_{request.format.value}.j2"
            if template_path.exists():
                template_content = template_path.read_text()

        file_extension = request.format.value
        output_filename = f"export_{request.session_id}_{request.format.value}_{job_id[:8]}.{file_extension}"
        output_path = settings.export_dir / output_filename

        if request.format == ExportFormat.JSON:
            content = await engine.export_to_json(data, template_content)
            output_path.write_text(content)
        elif request.format == ExportFormat.CSV:
            content = await engine.export_to_csv(data)
            output_path.write_text(content)
        elif request.format == ExportFormat.MARKDOWN:
            content = await engine.export_to_markdown(data, template_content)
            output_path.write_text(content)
        elif request.format == ExportFormat.OBSIDIAN:
            content = await engine.export_to_obsidian(data, template_content)
            output_path = output_path.with_suffix(".md")
            output_path.write_text(content)
        elif request.format == ExportFormat.WORD:
            doc_io = await engine.export_to_word(data)
            output_path = output_path.with_suffix(".docx")
            output_path.write_bytes(doc_io.getvalue())
        elif request.format == ExportFormat.PDF:
            pdf_io = await engine.export_to_pdf(data)
            output_path = output_path.with_suffix(".pdf")
            output_path.write_bytes(pdf_io.getvalue())
        elif request.format == ExportFormat.NOTION:
            if not request.notion_token or not request.notion_database_id:
                raise HTTPException(
                    status_code=400,
                    detail="Notion export requires notion_token and notion_database_id parameters",
                )
            result = await engine.export_to_notion(
                data,
                request.notion_token,
                request.notion_database_id,
            )
            output_path = output_path.with_suffix(".json")
            output_path.write_text(json.dumps(result, indent=2))
        else:  # pragma: no cover - safeguarded by Enum
            raise ValueError(f"Unsupported export format: {request.format}")

        job.status = ExportStatus.COMPLETED
        job.processed_items = job.total_items
        job.progress = 100.0
        job.file_path = str(output_path)
        job.completed_at = datetime.now()
        logger.info("export_job_completed", job_id=job_id, file_path=str(output_path))
    except Exception as exc:  # pragma: no cover - defensive logging
        job.status = ExportStatus.FAILED
        job.error_message = str(exc)
        logger.error("export_job_failed", job_id=job_id, error=str(exc))


__all__ = ["process_export_job"]
