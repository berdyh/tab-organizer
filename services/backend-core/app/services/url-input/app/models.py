"""Core data models for the URL Input service."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class URLMetadata:
    """URL metadata extracted from URL components."""

    domain: str
    subdomain: Optional[str] = None
    path: str = "/"
    parameters: Dict[str, List[str]] = field(default_factory=dict)
    fragment: Optional[str] = None
    port: Optional[int] = None
    scheme: str = "https"
    tld: Optional[str] = None
    path_segments: List[str] = field(default_factory=list)
    parameter_count: int = 0
    path_depth: int = 0
    url_hash: str = ""


@dataclass
class URLEntry:
    """Represents a single URL and its associated metadata and state."""

    url: str
    category: Optional[str] = None
    priority: Optional[str] = None
    notes: Optional[str] = None
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    validated: bool = False
    validation_error: Optional[str] = None
    metadata: Optional[URLMetadata] = None
    enriched: bool = False
    duplicate_of: Optional[str] = None
    similarity_group: Optional[str] = None


@dataclass
class URLInput:
    """Represents an uploaded or provided collection of URLs."""

    input_id: str
    urls: List[URLEntry]
    session_id: str
    source_type: str  # text, json, csv, excel, form, direct
    source_metadata: Dict[str, Any]
    created_at: datetime
    validated: bool = False


__all__ = ["URLMetadata", "URLEntry", "URLInput"]
