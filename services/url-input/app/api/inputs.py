"""Input management endpoints for the URL Input service."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Query

from ..api.dependencies import get_url_input_or_404
from ..batching import BatchProcessor
from ..logging import get_logger
from ..models import URLEntry, URLInput
from ..storage import url_input_storage
from ..utils import ensure_entry_ids, find_entry_by_id, parse_input_identifier


router = APIRouter(prefix="/api/input")
logger = get_logger(__name__)


def _ensure_entry_id(entry: URLEntry) -> str:
    if "entry_id" not in entry.source_metadata or not entry.source_metadata.get("entry_id"):
        entry.source_metadata["entry_id"] = str(uuid.uuid4())
    return entry.source_metadata["entry_id"]


def _entry_status(url_input: URLInput, url_entry: URLEntry) -> str:
    if url_entry.validation_error:
        return "failed"
    if url_input.validated or url_entry.validated:
        return "completed"
    if url_entry.enriched and url_entry.similarity_group:
        return "auth_required"
    return "pending"


def _entry_domain(url_entry: URLEntry) -> str | None:
    if url_entry.metadata and url_entry.metadata.domain:
        return url_entry.metadata.domain
    try:
        return urlparse(url_entry.url).netloc
    except Exception:  # pragma: no cover - defensive fallback
        return None


@router.get("/list")
async def list_url_inputs(
    session_id: str = Query(..., description="Filter by session identifier"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
):
    """List all URL entries with pagination."""
    logger.info("Listing URL inputs", skip=skip, limit=limit)

    entries: List[tuple[datetime, Dict[str, Any]]] = []

    for url_input in url_input_storage.values():
        if url_input.session_id != session_id:
            continue
        ensure_entry_ids(url_input.urls)
        created_at = url_input.created_at

        for url_entry in url_input.urls:
            entry_id = _ensure_entry_id(url_entry)
            status = _entry_status(url_input, url_entry)
            domain = _entry_domain(url_entry)

            entries.append(
                (
                    created_at,
                    {
                        "id": f"{url_input.input_id}:{entry_id}",
                        "input_id": url_input.input_id,
                        "url": url_entry.url,
                        "status": status,
                        "title": url_entry.source_metadata.get("title") or "",
                        "created_at": created_at,
                        "domain": domain,
                        "source_type": url_input.source_type,
                        "validated": url_entry.validated,
                        "validation_error": url_entry.validation_error,
                        "category": url_entry.category,
                        "priority": url_entry.priority,
                    },
                )
            )

    entries.sort(key=lambda item: item[0], reverse=True)

    total_entries = len(entries)
    paginated_entries = entries[skip : skip + limit]

    response_data = []
    for created_at, entry_data in paginated_entries:
        entry = dict(entry_data)
        entry["created_at"] = created_at.isoformat()
        entry["session_id"] = session_id
        response_data.append(entry)

    return {
        "data": response_data,
        "total": total_entries,
        "skip": skip,
        "limit": limit,
        "has_more": skip + limit < total_entries,
    }


@router.get("/{input_identifier}")
async def get_url_input(
    input_identifier: str,
    include_metadata: bool = Query(True, description="Include URL metadata in the response"),
    session_id: str = Query(..., description="Session identifier for access control"),
):
    """Get detailed information about a specific URL input or entry."""
    logger.info("Getting URL input details", input_identifier=input_identifier)

    input_id, entry_id = parse_input_identifier(input_identifier)
    url_input = get_url_input_or_404(input_id)
    if url_input.session_id != session_id:
        raise HTTPException(status_code=404, detail="URL input not found for session")
    ensure_entry_ids(url_input.urls)

    if entry_id:
        entry = find_entry_by_id(url_input, entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail="URL entry not found")

        status = _entry_status(url_input, entry)
        domain = _entry_domain(entry)

        metadata_payload = asdict(entry.metadata) if entry.metadata else None
        return {
            "id": f"{input_id}:{entry_id}",
            "input_id": input_id,
            "url": entry.url,
            "status": status,
            "title": entry.source_metadata.get("title") or "",
            "domain": domain,
            "created_at": url_input.created_at.isoformat(),
            "validated": entry.validated,
            "validation_error": entry.validation_error,
            "category": entry.category,
            "priority": entry.priority,
            "notes": entry.notes,
            "metadata": metadata_payload,
            "source_metadata": entry.source_metadata,
        }

    stats = BatchProcessor.get_processing_stats(url_input.urls)
    stats_copy = dict(stats)

    urls_data = []
    for url_entry in url_input.urls:
        entry_id_value = _ensure_entry_id(url_entry)
        entry_dict = asdict(url_entry)
        entry_dict["id"] = f"{input_id}:{entry_id_value}"

        if not include_metadata:
            metadata = url_entry.metadata
            entry_dict["metadata"] = (
                {
                    "domain": metadata.domain if metadata else None,
                    "path_depth": metadata.path_depth if metadata else 0,
                }
                if metadata
                else None
            )

        urls_data.append(entry_dict)

    return {
        "input_id": input_id,
        "source_type": url_input.source_type,
        "source_metadata": url_input.source_metadata,
        "created_at": url_input.created_at.isoformat(),
        "validated": url_input.validated,
        "session_id": session_id,
        "stats": stats_copy,
        "processing_stats": stats_copy,
        "urls": urls_data,
    }


@router.delete("/{input_identifier}")
async def delete_url_input(
    input_identifier: str,
    session_id: str = Query(..., description="Session identifier for access control"),
):
    """Delete a URL input or a specific URL entry."""
    logger.info("Deleting URL input", input_identifier=input_identifier)

    input_id, entry_id = parse_input_identifier(input_identifier)
    url_input = get_url_input_or_404(input_id)
    if url_input.session_id != session_id:
        raise HTTPException(status_code=404, detail="URL input not found for session")

    if entry_id is None:
        url_input_storage.delete(input_id)
        return {
            "message": "URL input deleted successfully",
            "input_id": input_id,
            "session_id": session_id,
        }

    entry = find_entry_by_id(url_input, entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="URL entry not found")

    url_input.urls = [e for e in url_input.urls if e.source_metadata.get("entry_id") != entry_id]

    if not url_input.urls:
        url_input_storage.delete(input_id)

    return {
        "message": "URL entry deleted successfully",
        "input_id": input_id,
        "entry_id": entry_id,
        "session_id": session_id,
    }


@router.put("/{input_identifier}")
async def update_url_input(
    input_identifier: str,
    update_data: Dict[str, Any],
    session_id: str = Query(..., description="Session identifier for access control"),
):
    """Update URL input metadata or a specific URL entry."""
    logger.info("Updating URL input", input_identifier=input_identifier)

    input_id, entry_id = parse_input_identifier(input_identifier)
    url_input = get_url_input_or_404(input_id)
    if url_input.session_id != session_id:
        raise HTTPException(status_code=404, detail="URL input not found for session")

    if entry_id is None:
        if "source_metadata" in update_data and isinstance(update_data["source_metadata"], dict):
            url_input.source_metadata.update(update_data["source_metadata"])
        if "validated" in update_data:
            url_input.validated = bool(update_data["validated"])
        return {
            "message": "URL input updated successfully",
            "input_id": input_id,
            "session_id": session_id,
        }

    entry = find_entry_by_id(url_input, entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="URL entry not found")

    if "category" in update_data:
        entry.category = update_data["category"]
    if "priority" in update_data:
        entry.priority = update_data["priority"]
    if "notes" in update_data:
        entry.notes = update_data["notes"]
    if "title" in update_data:
        entry.source_metadata["title"] = update_data["title"]
    if "validated" in update_data:
        entry.validated = bool(update_data["validated"])
        if entry.validated:
            entry.validation_error = None
    if "validation_error" in update_data:
        entry.validation_error = update_data["validation_error"]

    return {
        "message": "URL entry updated successfully",
        "input_id": input_id,
        "entry_id": entry_id,
        "session_id": session_id,
    }


@router.get("")
async def list_inputs(session_id: str = Query(..., description="Filter by session identifier")):
    """List all URL inputs."""
    return {
        "inputs": [
            {
                "input_id": input_id,
                "source_type": url_input.source_type,
                "total_urls": len(url_input.urls),
                "created_at": url_input.created_at.isoformat(),
                "validated": url_input.validated,
                "session_id": url_input.session_id,
            }
            for input_id, url_input in url_input_storage.items()
            if url_input.session_id == session_id
        ]
    }


__all__ = ["router"]
