"""Batch processing helpers for URL entries."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from .deduplication import URLDeduplicator
from .enrichment import URLEnricher
from .models import URLEntry


class BatchProcessor:
    """Handle batch processing of large URL lists."""

    @classmethod
    def process_urls_batch(cls, url_entries: List[URLEntry], batch_size: int = 100) -> List[URLEntry]:
        """Process URLs in batches for large datasets."""
        processed_entries: List[URLEntry] = []

        for i in range(0, len(url_entries), batch_size):
            batch = url_entries[i : i + batch_size]
            enriched_batch = []
            for entry in batch:
                enriched_batch.append(URLEnricher.enrich_url_entry(entry))
            processed_entries.extend(enriched_batch)

        processed_entries = URLDeduplicator.mark_duplicates(processed_entries)
        return processed_entries

    @classmethod
    def get_processing_stats(cls, url_entries: List[URLEntry]) -> Dict[str, Any]:
        """Get comprehensive statistics about processed URLs."""
        total_urls = len(url_entries)
        valid_urls = sum(1 for entry in url_entries if entry.validated)
        enriched_urls = sum(1 for entry in url_entries if entry.enriched)
        duplicate_urls = sum(1 for entry in url_entries if entry.duplicate_of)

        categories = Counter(entry.category for entry in url_entries if entry.category)
        domains = Counter(
            entry.metadata.domain for entry in url_entries if entry.metadata and entry.validated
        )
        priorities = Counter(entry.priority for entry in url_entries if entry.priority)

        return {
            "total_urls": total_urls,
            "valid_urls": valid_urls,
            "invalid_urls": total_urls - valid_urls,
            "enriched_urls": enriched_urls,
            "duplicate_urls": duplicate_urls,
            "unique_urls": valid_urls - duplicate_urls,
            "categories": dict(categories.most_common()),
            "top_domains": dict(domains.most_common(10)),
            "priorities": dict(priorities),
            "processing_complete": enriched_urls == valid_urls,
        }


__all__ = ["BatchProcessor"]
