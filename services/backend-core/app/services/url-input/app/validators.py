"""URL validation utilities."""

from __future__ import annotations

import re
from typing import Optional, Tuple


class URLValidator:
    """URL validation utilities."""

    URL_PATTERN = re.compile(
        r"^https?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )

    @classmethod
    def is_valid_url(cls, url: str) -> bool:
        """Check if URL is valid format."""
        if not url or not isinstance(url, str):
            return False
        return bool(cls.URL_PATTERN.match(url.strip()))

    @classmethod
    def validate_url(cls, url: str) -> Tuple[bool, Optional[str]]:
        """Validate URL and return validation result with error message."""
        url = url.strip() if url else ""

        if not url:
            return False, "URL cannot be empty"

        if not cls.is_valid_url(url):
            return False, "Invalid URL format"

        return True, None


__all__ = ["URLValidator"]
