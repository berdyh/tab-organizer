"""Parsing helpers for ingesting URLs from multiple formats."""

from __future__ import annotations

import json
from io import StringIO
from typing import Any, List

import pandas as pd

from .deduplication import URLDeduplicator
from .enrichment import URLEnricher
from .models import URLEntry
from .validators import URLValidator


class URLParser:
    """Parse URLs from various input sources."""

    @classmethod
    def parse_text_file(cls, content: str, enrich: bool = True) -> List[URLEntry]:
        """Extract URLs from plain text files."""
        urls: List[URLEntry] = []
        lines = content.strip().split("\n")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            url = line.split()[0] if line.split() else line

            is_valid, error = URLValidator.validate_url(url)
            entry = URLEntry(
                url=url,
                source_metadata={"line_number": line_num, "original_line": line},
                validated=is_valid,
                validation_error=error,
            )

            if enrich and is_valid:
                entry = URLEnricher.enrich_url_entry(entry)

            urls.append(entry)

        if enrich:
            urls = URLDeduplicator.mark_duplicates(urls)

        return urls

    @classmethod
    def parse_json_file(cls, content: str, enrich: bool = True) -> List[URLEntry]:
        """Parse structured JSON with URL lists."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON format: {exc}") from exc

        urls: List[URLEntry] = []

        if isinstance(data, list):
            for index, item in enumerate(data):
                if isinstance(item, str):
                    is_valid, error = URLValidator.validate_url(item)
                    entry = URLEntry(
                        url=item,
                        source_metadata={"index": index},
                        validated=is_valid,
                        validation_error=error,
                    )
                elif isinstance(item, dict) and "url" in item:
                    url = item["url"]
                    is_valid, error = URLValidator.validate_url(url)
                    entry = URLEntry(
                        url=url,
                        category=item.get("category"),
                        priority=item.get("priority"),
                        notes=item.get("notes"),
                        source_metadata={
                            "index": index,
                            **{k: v for k, v in item.items() if k != "url"},
                        },
                        validated=is_valid,
                        validation_error=error,
                    )
                else:
                    continue

                if enrich and entry.validated:
                    entry = URLEnricher.enrich_url_entry(entry)
                urls.append(entry)

        elif isinstance(data, dict) and "urls" in data:
            url_list = data["urls"]
            metadata = {k: v for k, v in data.items() if k != "urls"}

            for index, item in enumerate(url_list):
                if isinstance(item, str):
                    is_valid, error = URLValidator.validate_url(item)
                    entry = URLEntry(
                        url=item,
                        source_metadata={"index": index, **metadata},
                        validated=is_valid,
                        validation_error=error,
                    )
                elif isinstance(item, dict) and "url" in item:
                    url = item["url"]
                    is_valid, error = URLValidator.validate_url(url)
                    entry = URLEntry(
                        url=url,
                        category=item.get("category"),
                        priority=item.get("priority"),
                        notes=item.get("notes"),
                        source_metadata={
                            "index": index,
                            **metadata,
                            **{k: v for k, v in item.items() if k != "url"},
                        },
                        validated=is_valid,
                        validation_error=error,
                    )
                else:
                    continue

                if enrich and entry.validated:
                    entry = URLEnricher.enrich_url_entry(entry)
                urls.append(entry)

        if enrich:
            urls = URLDeduplicator.mark_duplicates(urls)

        return urls

    @classmethod
    def parse_csv_file(cls, content: str, enrich: bool = True) -> List[URLEntry]:
        """Extract URLs from CSV/TSV content."""
        try:
            df = pd.read_csv(StringIO(content))
        except Exception as exc:
            raise ValueError(f"Invalid CSV format: {exc}") from exc

        urls: List[URLEntry] = []

        url_column = None
        for column in df.columns:
            if column.lower() in {"url", "urls", "link", "links", "website", "site"}:
                url_column = column
                break

        if url_column is None:
            url_column = df.columns[0]

        for index, row in df.iterrows():
            raw_value = row[url_column]
            if pd.isna(raw_value):
                continue

            url = str(raw_value).strip()
            if not url or url.lower() == "nan":
                continue

            is_valid, error = URLValidator.validate_url(url)

            metadata = {}
            for column in df.columns:
                if column != url_column and not pd.isna(row[column]):
                    metadata[column.lower()] = row[column]

            entry = URLEntry(
                url=url,
                category=metadata.get("category"),
                priority=metadata.get("priority"),
                notes=metadata.get("notes"),
                source_metadata={"row_index": index, **metadata},
                validated=is_valid,
                validation_error=error,
            )

            if enrich and entry.validated:
                entry = URLEnricher.enrich_url_entry(entry)

            urls.append(entry)

        if enrich:
            urls = URLDeduplicator.mark_duplicates(urls)

        return urls

    @classmethod
    def parse_json_input(cls, content: Any, enrich: bool = True) -> List[URLEntry]:
        """Compatibility wrapper that accepts strings or file-like objects."""
        if hasattr(content, "read"):
            content = content.read()
        if isinstance(content, bytes):
            content = content.decode()
        return cls.parse_json_file(content or "{}", enrich=enrich)

    @classmethod
    def parse_csv_input(cls, content: Any, enrich: bool = True) -> List[URLEntry]:
        """Compatibility wrapper that accepts strings or file-like objects."""
        if hasattr(content, "read"):
            content = content.read()
        if isinstance(content, bytes):
            content = content.decode()
        return cls.parse_csv_file(content or "", enrich=enrich)


__all__ = ["URLParser"]
