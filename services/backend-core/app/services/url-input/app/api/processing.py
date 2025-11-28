"""Processing and analytics endpoints for URL inputs."""

from __future__ import annotations

from collections import Counter
import os
from typing import Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query

from ..api.dependencies import get_url_input_or_404
from ..batching import BatchProcessor
from ..deduplication import URLDeduplicator
from ..enrichment import URLEnricher
from ..logging import get_logger
from ..models import URLEntry
from ..services import create_url_input
from ..utils import ensure_entry_ids
from ..validators import URLValidator


router = APIRouter(prefix="/api/input")
logger = get_logger(__name__)

async def _update_session_stats(session_id: str, urls_processed: int, source_type: str) -> None:
    if not session_id or urls_processed <= 0 or not SESSION_SERVICE_URL:
        return
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.put(
                f"{SESSION_SERVICE_URL}/sessions/{session_id}/stats",
                json={
                    "urls_processed": urls_processed,
                    "metadata": {"last_ingest_source": source_type},
                },
            )
    except Exception as exc:  # pragma: no cover - non-critical coupling
        logger.warning(
            "Failed to update session statistics from processing module",
            session_id=session_id,
            error=str(exc),
        )

SESSION_SERVICE_URL = os.getenv("SESSION_SERVICE_URL", "http://session-service:8087")


@router.post("/enrich/{input_id}")
async def enrich_urls(input_id: str):
    """Enrich existing URL input with metadata and categorization."""
    url_input = get_url_input_or_404(input_id)

    try:
        enriched_urls: List[URLEntry] = []
        for url_entry in url_input.urls:
            if url_entry.validated and not url_entry.enriched:
                enriched_urls.append(URLEnricher.enrich_url_entry(url_entry))
            else:
                enriched_urls.append(url_entry)

        enriched_urls = URLDeduplicator.mark_duplicates(enriched_urls)
        url_input.urls = ensure_entry_ids(enriched_urls)

        stats = BatchProcessor.get_processing_stats(url_input.urls)
        url_input.source_metadata["processing_stats"] = stats
        url_input.source_metadata["enriched"] = True

        logger.info("URLs enriched", input_id=input_id, **stats)

        return {"input_id": input_id, "enriched": True, **stats}

    except Exception as exc:
        logger.error("Error enriching URLs", error=str(exc), input_id=input_id)
        raise HTTPException(status_code=500, detail=f"Error enriching URLs: {exc}") from exc


@router.get("/duplicates/{input_id}")
async def get_duplicates(input_id: str):
    """Get duplicate URLs for an input."""
    url_input = get_url_input_or_404(input_id)

    duplicate_groups = URLDeduplicator.find_duplicates(url_input.urls)
    duplicates = []
    for normalized, entries in duplicate_groups.items():
        duplicates.append(
            {
                "normalized_url": normalized,
                "count": len(entries),
                "urls": [
                    {
                        "url": entry.url,
                        "category": entry.category,
                        "source_metadata": entry.source_metadata,
                    }
                    for entry in entries
                ],
            }
        )

    return {
        "input_id": input_id,
        "duplicate_groups": len(duplicates),
        "total_duplicates": sum(group["count"] - 1 for group in duplicates),
        "duplicates": duplicates,
    }


@router.get("/categories/{input_id}")
async def get_categories(input_id: str):
    """Get URL categories and domain groups for an input."""
    url_input = get_url_input_or_404(input_id)

    similarity_groups = URLDeduplicator.get_similarity_groups(url_input.urls)
    categories = Counter(entry.category for entry in url_input.urls if entry.category)
    domains = Counter(
        entry.metadata.domain
        for entry in url_input.urls
        if entry.metadata and entry.validated
    )

    return {
        "input_id": input_id,
        "categories": dict(categories.most_common()),
        "domains": dict(domains.most_common(20)),
        "domain_groups": {domain: len(entries) for domain, entries in similarity_groups.items()},
    }


@router.post("/batch-process")
async def batch_process_urls(
    urls: List[str],
    batch_size: int = Query(100, description="Batch size for processing"),
    session_id: str = Query(..., description="Session to associate the processed URLs with"),
):
    """Process large URL lists in batches."""
    logger.info("Processing batch URL input", url_count=len(urls), batch_size=batch_size)

    try:
        url_entries: List[URLEntry] = []
        for index, url in enumerate(urls):
            is_valid, error = URLValidator.validate_url(url)
            url_entries.append(
                URLEntry(
                    url=url,
                    source_metadata={"index": index},
                    validated=is_valid,
                    validation_error=error,
                )
            )

        processed_urls = BatchProcessor.process_urls_batch(url_entries, batch_size=batch_size)
        url_input, stats = create_url_input(
            processed_urls,
            "batch",
            {
                "input_method": "batch_processing",
                "batch_size": batch_size,
                "enriched": True,
            },
            session_id=session_id,
        )

        await _update_session_stats(session_id, stats.get("total_urls", 0), "batch")

        logger.info("Batch URLs processed", input_id=url_input.input_id, batch_size=batch_size, **stats)

        return {
            "input_id": url_input.input_id,
            "source_type": "batch",
            "batch_size": batch_size,
            "enriched": True,
            "session_id": session_id,
            **stats,
        }

    except Exception as exc:
        logger.error("Error processing batch URLs", error=str(exc))
        raise HTTPException(status_code=400, detail=f"Error processing batch URLs: {exc}") from exc


@router.get("/validate/{input_id}")
async def validate_input(input_id: str):
    """Validate parsed URL input."""
    url_input = get_url_input_or_404(input_id)

    for url_entry in url_input.urls:
        is_valid, error = URLValidator.validate_url(url_entry.url)
        url_entry.validated = is_valid
        url_entry.validation_error = error

    url_input.validated = True

    valid_count = sum(1 for url in url_input.urls if url.validated)
    invalid_count = len(url_input.urls) - valid_count

    logger.info(
        "Input validated",
        input_id=input_id,
        total_urls=len(url_input.urls),
        valid_urls=valid_count,
        invalid_urls=invalid_count,
    )

    return {
        "input_id": input_id,
        "validated": True,
        "total_urls": len(url_input.urls),
        "valid_urls": valid_count,
        "invalid_urls": invalid_count,
        "validation_errors": [
            {"url": url.url, "error": url.validation_error}
            for url in url_input.urls
            if not url.validated
        ],
    }


@router.get("/preview/{input_id}")
async def preview_input(
    input_id: str,
    limit: int = Query(10, description="Number of URLs to preview"),
    show_duplicates: bool = Query(False, description="Include duplicate URLs in preview"),
    category_filter: Optional[str] = Query(None, description="Filter by category"),
):
    """Preview parsed URL list with enriched data."""
    url_input = get_url_input_or_404(input_id)

    filtered_urls = list(url_input.urls)

    if not show_duplicates:
        filtered_urls = [url for url in filtered_urls if not url.duplicate_of]

    if category_filter:
        filtered_urls = [url for url in filtered_urls if url.category == category_filter]

    preview_urls = filtered_urls[:limit]
    stats = BatchProcessor.get_processing_stats(url_input.urls)

    return {
        "input_id": input_id,
        "source_type": url_input.source_type,
        "total_urls": len(url_input.urls),
        "filtered_urls": len(filtered_urls),
        "preview_count": len(preview_urls),
        "created_at": url_input.created_at.isoformat(),
        "source_metadata": url_input.source_metadata,
        "processing_stats": stats,
        "filters": {"show_duplicates": show_duplicates, "category_filter": category_filter},
        "preview_urls": [
            {
                "url": url.url,
                "category": url.category,
                "priority": url.priority,
                "notes": url.notes,
                "validated": url.validated,
                "validation_error": url.validation_error,
                "enriched": url.enriched,
                "duplicate_of": url.duplicate_of,
                "similarity_group": url.similarity_group,
                "metadata": {
                    "domain": url.metadata.domain if url.metadata else None,
                    "subdomain": url.metadata.subdomain if url.metadata else None,
                    "path": url.metadata.path if url.metadata else None,
                    "parameter_count": url.metadata.parameter_count if url.metadata else 0,
                    "path_depth": url.metadata.path_depth if url.metadata else 0,
                    "tld": url.metadata.tld if url.metadata else None,
                }
                if url.metadata
                else None,
                "source_metadata": url.source_metadata,
            }
            for url in preview_urls
        ],
    }


__all__ = ["router"]
