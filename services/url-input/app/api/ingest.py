"""Ingestion endpoints for uploading and submitting URLs."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from ..deduplication import URLDeduplicator
from ..detector import InputFormatDetector
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


def _decode_text(content_bytes: bytes) -> str:
    """Decode raw bytes into text, ignoring undecodable characters."""
    return content_bytes.decode("utf-8", errors="ignore")


def _normalize_format(format_hint: Optional[str]) -> str:
    """Normalize a user-supplied format hint."""
    if not format_hint:
        return ""
    fmt = format_hint.strip().lower()
    if fmt in {"xlsx", "xls"}:
        return "excel"
    if fmt in {"csv", "tsv"}:
        return fmt
    if fmt in {"json", "text", "excel"}:
        return fmt
    return fmt


def _guess_format_from_filename(filename: Optional[str]) -> str:
    """Guess file format based on the filename extension."""
    if not filename:
        return ""
    lower = filename.lower()
    if lower.endswith((".xlsx", ".xls")):
        return "excel"
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".tsv"):
        return "tsv"
    if lower.endswith(".json"):
        return "json"
    if lower.endswith(".txt"):
        return "text"
    return ""


def _parse_uploaded_content(
    format_key: str,
    content_bytes: bytes,
    enrich: bool,
) -> Tuple[List[URLEntry], Dict[str, Any]]:
    """Parse uploaded file content into URL entries and metadata."""
    fmt = _normalize_format(format_key) or "text"
    metadata: Dict[str, Any] = {"detected_format": fmt}

    try:
        if fmt == "excel":
            dataframe = pd.read_excel(BytesIO(content_bytes))
            metadata["sheet_shape"] = tuple(dataframe.shape)
            metadata["total_rows"] = int(dataframe.shape[0])
            csv_content = dataframe.to_csv(index=False)
            urls = URLParser.parse_csv_file(csv_content, enrich=enrich)
        elif fmt in {"csv", "tsv"}:
            text = _decode_text(content_bytes)
            metadata["total_lines"] = sum(1 for line in text.splitlines() if line.strip())
            urls = URLParser.parse_csv_file(text, enrich=enrich)
        elif fmt == "json":
            text = _decode_text(content_bytes)
            metadata["character_count"] = len(text)
            urls = URLParser.parse_json_file(text, enrich=enrich)
        else:  # Treat as plain text
            text = _decode_text(content_bytes)
            metadata["total_lines"] = sum(1 for line in text.splitlines() if line.strip())
            urls = URLParser.parse_text_file(text, enrich=enrich)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    except Exception as exc:
        raise ValueError(f"Unable to parse uploaded file: {exc}") from exc

    return urls, metadata


@router.post("/upload/text")
async def upload_text_file(
    file: UploadFile = File(...),
    enrich: bool = Query(True, description="Enable URL enrichment and deduplication"),
):
    """Upload and parse plain text file with URLs."""
    logger.info("Processing text file upload", filename=file.filename, enrich=enrich)

    try:
        content_bytes = await file.read()
        urls, extra_metadata = _parse_uploaded_content("text", content_bytes, enrich=enrich)
        _, response = _store_input(
            urls,
            "text",
            {
                "filename": file.filename,
                "file_size": len(content_bytes),
                **extra_metadata,
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
        urls, extra_metadata = _parse_uploaded_content("json", content_bytes, enrich=enrich)
        _, response = _store_input(
            urls,
            "json",
            {
                "filename": file.filename,
                "file_size": len(content_bytes),
                **extra_metadata,
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
        urls, extra_metadata = _parse_uploaded_content("csv", content_bytes, enrich=enrich)
        _, response = _store_input(
            urls,
            "csv",
            {
                "filename": file.filename,
                "file_size": len(content_bytes),
                **extra_metadata,
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
        urls, extra_metadata = _parse_uploaded_content("excel", content_bytes, enrich=enrich)
        _, response = _store_input(
            urls,
            "excel",
            {
                "filename": file.filename,
                "file_size": len(content_bytes),
                **extra_metadata,
                "enriched": enrich,
            },
        )
        return response

    except Exception as exc:
        logger.error("Error processing Excel file", error=str(exc), filename=file.filename)
        raise HTTPException(status_code=400, detail=f"Error processing Excel file: {exc}") from exc


@router.post("/upload")
async def upload_file_auto(
    file: UploadFile = File(...),
    enrich: bool = Query(True, description="Enable URL enrichment and deduplication"),
    format_hint: Optional[str] = Query(None, description="Optional format hint (text, json, csv, excel)"),
):
    """Upload a file and automatically detect its format."""
    logger.info(
        "Processing auto-detected file upload",
        filename=file.filename,
        enrich=enrich,
        format_hint=format_hint,
    )

    content_bytes = await file.read()
    if not content_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    normalized_hint = _normalize_format(format_hint)
    detected_format = normalized_hint or _guess_format_from_filename(file.filename)
    detection_method = "hint" if normalized_hint else ("filename" if detected_format else "content")

    if not detected_format or detected_format not in {"text", "json", "csv", "tsv", "excel"}:
        preview_text = ""
        try:
            preview_text = _decode_text(content_bytes[:4096])
        except Exception:
            preview_text = ""
        detected_format = _normalize_format(
            InputFormatDetector.detect_file_type(file.filename or "upload", preview_text)
        ) or "text"
        detection_method = "content"

    try:
        urls, extra_metadata = _parse_uploaded_content(detected_format, content_bytes, enrich=enrich)
    except ValueError as exc:
        logger.error("Error processing uploaded file", error=str(exc), filename=file.filename)
        raise HTTPException(status_code=400, detail=f"Error processing file: {exc}") from exc

    if not urls:
        fallback_text = _decode_text(content_bytes)
        fallback_urls = URLParser.parse_text_file(fallback_text, enrich=enrich)
        if fallback_urls:
            urls = fallback_urls
            extra_metadata["fallback_format"] = extra_metadata.get("detected_format")
            extra_metadata["detected_format"] = "text"

    extra_metadata.update(
        {
            "filename": file.filename,
            "file_size": len(content_bytes),
            "enriched": enrich,
            "format_hint": normalized_hint or None,
            "detection_method": detection_method,
        }
    )

    source_type = extra_metadata.get("detected_format", detected_format)
    if source_type == "tsv":
        source_type = "csv"

    _, response = _store_input(urls, source_type, extra_metadata)
    return response


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
