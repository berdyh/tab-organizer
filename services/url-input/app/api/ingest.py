"""Ingestion endpoints for uploading and submitting URLs."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from ..deduplication import URLDeduplicator
from ..enrichment import URLEnricher
from ..logging import get_logger
from ..models import URLEntry
from ..parser import URLParser
from ..services import create_url_input
from ..validators import URLValidator


router = APIRouter(prefix="/api/input")
logger = get_logger(__name__)


def _store_input(
    urls: List[URLEntry],
    source_type: str,
    source_metadata: Dict[str, Any],
) -> tuple[str, Dict[str, Any]]:
    source_metadata = dict(source_metadata)
    url_input, stats = create_url_input(urls, source_type, source_metadata)

    logger.info(
        "URL input stored",
        input_id=url_input.input_id,
        source_type=source_type,
        total_urls=stats["total_urls"],
        valid_urls=stats["valid_urls"],
        duplicate_urls=stats["duplicate_urls"],
    )

    response = {
        "input_id": url_input.input_id,
        "source_type": source_type,
        "enriched": source_metadata.get("enriched", True),
        **stats,
    }

    if "filename" in source_metadata:
        response["filename"] = source_metadata["filename"]

    return url_input.input_id, response


@router.post("/upload/text")
async def upload_text_file(
    file: UploadFile = File(...),
    enrich: bool = Query(True, description="Enable URL enrichment and deduplication"),
):
    """Upload and parse plain text file with URLs."""
    logger.info("Processing text file upload", filename=file.filename, enrich=enrich)

    try:
        content_bytes = await file.read()
        content_str = content_bytes.decode("utf-8")

        urls = URLParser.parse_text_file(content_str, enrich=enrich)
        _, response = _store_input(
            urls,
            "text",
            {
                "filename": file.filename,
                "file_size": len(content_bytes),
                "total_lines": len(content_str.split("\n")),
                "enriched": enrich,
            },
        )
        return response

    except Exception as exc:
        logger.error("Error processing text file", error=str(exc), filename=file.filename)
        raise HTTPException(status_code=400, detail=f"Error processing text file: {exc}") from exc


@router.post("/upload/json")
async def upload_json_file(
    file: UploadFile = File(...),
    enrich: bool = Query(True, description="Enable URL enrichment and deduplication"),
):
    """Upload and parse JSON file with URLs."""
    logger.info("Processing JSON file upload", filename=file.filename, enrich=enrich)

    try:
        content_bytes = await file.read()
        content_str = content_bytes.decode("utf-8")

        urls = URLParser.parse_json_file(content_str, enrich=enrich)
        _, response = _store_input(
            urls,
            "json",
            {
                "filename": file.filename,
                "file_size": len(content_bytes),
                "enriched": enrich,
            },
        )
        return response

    except Exception as exc:
        logger.error("Error processing JSON file", error=str(exc), filename=file.filename)
        raise HTTPException(status_code=400, detail=f"Error processing JSON file: {exc}") from exc


@router.post("/upload/csv")
async def upload_csv_file(
    file: UploadFile = File(...),
    enrich: bool = Query(True, description="Enable URL enrichment and deduplication"),
):
    """Upload and parse CSV file with URLs."""
    logger.info("Processing CSV file upload", filename=file.filename, enrich=enrich)

    try:
        content_bytes = await file.read()
        content_str = content_bytes.decode("utf-8")

        urls = URLParser.parse_csv_file(content_str, enrich=enrich)
        _, response = _store_input(
            urls,
            "csv",
            {
                "filename": file.filename,
                "file_size": len(content_bytes),
                "enriched": enrich,
            },
        )
        return response

    except Exception as exc:
        logger.error("Error processing CSV file", error=str(exc), filename=file.filename)
        raise HTTPException(status_code=400, detail=f"Error processing CSV file: {exc}") from exc


@router.post("/upload/excel")
async def upload_excel_file(
    file: UploadFile = File(...),
    enrich: bool = Query(True, description="Enable URL enrichment and deduplication"),
):
    """Upload and parse Excel file with URLs."""
    logger.info("Processing Excel file upload", filename=file.filename, enrich=enrich)

    try:
        content_bytes = await file.read()
        workbook = pd.read_excel(BytesIO(content_bytes))
        csv_content = workbook.to_csv(index=False)

        urls = URLParser.parse_csv_file(csv_content, enrich=enrich)
        _, response = _store_input(
            urls,
            "excel",
            {
                "filename": file.filename,
                "file_size": len(content_bytes),
                "sheet_shape": workbook.shape,
                "enriched": enrich,
            },
        )
        return response

    except Exception as exc:
        logger.error("Error processing Excel file", error=str(exc), filename=file.filename)
        raise HTTPException(status_code=400, detail=f"Error processing Excel file: {exc}") from exc


@router.post("/urls")
async def input_urls_direct(
    urls: List[str],
    enrich: bool = Query(True, description="Enable URL enrichment and deduplication"),
):
    """Direct URL list input."""
    logger.info("Processing direct URL input", url_count=len(urls), enrich=enrich)

    try:
        entries: List[URLEntry] = []
        for index, url in enumerate(urls):
            is_valid, error = URLValidator.validate_url(url)
            entry = URLEntry(
                url=url,
                source_metadata={"index": index},
                validated=is_valid,
                validation_error=error,
            )
            if enrich and is_valid:
                entry = URLEnricher.enrich_url_entry(entry)
            entries.append(entry)

        if enrich:
            entries = URLDeduplicator.mark_duplicates(entries)

        _, response = _store_input(
            entries,
            "direct",
            {"input_method": "api_direct", "enriched": enrich},
        )
        return response

    except Exception as exc:
        logger.error("Error processing direct URL input", error=str(exc))
        raise HTTPException(status_code=400, detail=f"Error processing URLs: {exc}") from exc


@router.post("/form")
async def input_urls_form(
    urls_text: str = Form(...),
    enrich: bool = Form(True, description="Enable URL enrichment and deduplication"),
):
    """Web form URL input."""
    logger.info("Processing form URL input", enrich=enrich)

    try:
        urls = URLParser.parse_text_file(urls_text, enrich=enrich)
        _, response = _store_input(
            urls,
            "form",
            {
                "input_method": "web_form",
                "text_length": len(urls_text),
                "enriched": enrich,
            },
        )
        return response

    except Exception as exc:
        logger.error("Error processing form URL input", error=str(exc))
        raise HTTPException(status_code=400, detail=f"Error processing form URLs: {exc}") from exc


__all__ = ["router"]
