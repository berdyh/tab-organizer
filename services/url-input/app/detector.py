"""Detect input formats and extract URLs from unstructured text."""

from __future__ import annotations

import json
import re
from typing import List


class InputFormatDetector:
    """Auto-detect input file format and extract URLs."""

    @classmethod
    def detect_file_type(cls, filename: str, content: str) -> str:
        """Auto-detect file type based on filename and content."""
        filename_lower = filename.lower()

        if filename_lower.endswith(".json"):
            return "json"
        if filename_lower.endswith((".csv", ".tsv")):
            return "csv"
        if filename_lower.endswith((".xlsx", ".xls")):
            return "excel"
        if filename_lower.endswith(".txt"):
            return "text"

        try:
            json.loads(content)
            return "json"
        except Exception:
            pass

        if "," in content or "\t" in content:
            return "csv"

        return "text"

    @classmethod
    def extract_url_patterns(cls, text: str) -> List[str]:
        """Find URLs in unstructured text."""
        url_pattern = re.compile(
            r"https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?",
            re.IGNORECASE,
        )
        return url_pattern.findall(text)


__all__ = ["InputFormatDetector"]
