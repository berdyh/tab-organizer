"""Utility helpers for managing URL entries and identifiers."""

from __future__ import annotations

import uuid
from typing import List, Optional, Tuple

from .models import URLInput, URLEntry


def ensure_entry_ids(entries: List[URLEntry]) -> List[URLEntry]:
    """Ensure each URL entry has a stable entry_id for client operations."""
    for entry in entries:
        if entry.source_metadata is None:
            entry.source_metadata = {}
        if "entry_id" not in entry.source_metadata:
            entry.source_metadata["entry_id"] = str(uuid.uuid4())
    return entries


def parse_input_identifier(identifier: str) -> Tuple[str, Optional[str]]:
    """Split identifiers of the form '<input_id>:<entry_id>'."""
    if ":" in identifier:
        input_id, entry_id = identifier.split(":", 1)
        entry_id = entry_id or None
        return input_id, entry_id
    return identifier, None


def find_entry_by_id(url_input: URLInput, entry_id: str) -> Optional[URLEntry]:
    """Locate a URL entry by its entry_id within a URLInput."""
    for entry in url_input.urls:
        if entry.source_metadata.get("entry_id") == entry_id:
            return entry
    return None


__all__ = ["ensure_entry_ids", "parse_input_identifier", "find_entry_by_id"]
