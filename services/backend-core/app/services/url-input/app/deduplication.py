"""URL deduplication and similarity helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List
from urllib.parse import parse_qs, urlparse

from .models import URLEntry


class URLDeduplicator:
    """URL deduplication and similarity detection."""

    @classmethod
    def normalize_url(cls, url: str) -> str:
        """Normalize URL for deduplication comparison."""
        try:
            parsed = urlparse(url.lower().strip())

            tracking_params = {
                "utm_source",
                "utm_medium",
                "utm_campaign",
                "utm_term",
                "utm_content",
                "fbclid",
                "gclid",
                "ref",
                "source",
                "campaign_id",
                "ad_id",
            }

            params = parse_qs(parsed.query)
            filtered_params = {k: v for k, v in params.items() if k.lower() not in tracking_params}

            query_parts = []
            for key, values in sorted(filtered_params.items()):
                for value in sorted(values):
                    query_parts.append(f"{key}={value}")

            normalized_query = "&".join(query_parts)

            path = parsed.path.rstrip("/")
            if not path:
                path = "/"

            normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
            if normalized_query:
                normalized += f"?{normalized_query}"

            return normalized

        except Exception:
            return url.lower().strip()

    @classmethod
    def find_duplicates(cls, url_entries: List[URLEntry]) -> Dict[str, List[URLEntry]]:
        """Find duplicate URLs and group them."""
        normalized_groups = defaultdict(list)

        for entry in url_entries:
            if entry.validated:
                normalized = cls.normalize_url(entry.url)
                normalized_groups[normalized].append(entry)

        return {norm_url: entries for norm_url, entries in normalized_groups.items() if len(entries) > 1}

    @classmethod
    def mark_duplicates(cls, url_entries: List[URLEntry]) -> List[URLEntry]:
        """Mark duplicate URLs in the list."""
        duplicate_groups = cls.find_duplicates(url_entries)

        for entries in duplicate_groups.values():
            primary = entries[0]
            for duplicate in entries[1:]:
                duplicate.duplicate_of = primary.url

        return url_entries

    @classmethod
    def get_similarity_groups(cls, url_entries: List[URLEntry]) -> Dict[str, List[URLEntry]]:
        """Group URLs by domain similarity."""
        domain_groups = defaultdict(list)

        for entry in url_entries:
            if entry.validated and entry.metadata:
                domain_groups[entry.metadata.domain].append(entry)

        return dict(domain_groups)


__all__ = ["URLDeduplicator"]
