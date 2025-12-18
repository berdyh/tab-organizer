"""URL Store with set-like deduplication."""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse


@dataclass
class URLRecord:
    """Record for a stored URL."""
    original: str
    normalized: str
    content_hash: Optional[str] = None
    embedding_id: Optional[str] = None
    scraped_at: Optional[datetime] = None
    status: str = "pending"  # pending, scraped, failed, auth_required
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "gclsrc", "dclid", "zanpid", "msclkid",
    "ref", "ref_src", "referer", "source", "mc_cid", "mc_eid",
    "yclid", "_ga", "_gl", "igshid", "s_kwcid", "ef_id",
}


class URLStore:
    """Store URLs like a set - no duplicates, fast lookup."""
    
    def __init__(self):
        self._urls: dict[str, URLRecord] = {}  # normalized_url → record
        self._content_hashes: dict[str, str] = {}  # content_hash → normalized_url
        self._original_to_normalized: dict[str, str] = {}  # original → normalized
    
    def normalize(self, url: str) -> str:
        """Convert URL to canonical form."""
        url = url.strip()
        
        # Parse URL
        parsed = urlparse(url.lower())
        
        # Ensure scheme
        scheme = parsed.scheme or "https"
        
        # Clean netloc (remove www. prefix for consistency)
        netloc = parsed.netloc
        if netloc.startswith("www."):
            netloc = netloc[4:]
        
        # Remove trailing slash from path
        path = parsed.path.rstrip("/") or "/"
        
        # Remove tracking parameters from query
        if parsed.query:
            query_params = parse_qs(parsed.query, keep_blank_values=True)
            clean_params = {
                k: v for k, v in query_params.items()
                if k.lower() not in TRACKING_PARAMS
            }
            # Sort params for consistency
            query = urlencode(sorted(clean_params.items()), doseq=True)
        else:
            query = ""
        
        # Reconstruct URL
        normalized = urlunparse((scheme, netloc, path, "", query, ""))
        return normalized
    
    def add(self, url: str, metadata: Optional[dict] = None) -> tuple[bool, URLRecord]:
        """
        Add a URL to the store.
        
        Returns:
            Tuple of (is_new, record)
        """
        normalized = self.normalize(url)
        
        if normalized in self._urls:
            return False, self._urls[normalized]
        
        record = URLRecord(
            original=url,
            normalized=normalized,
            metadata=metadata or {}
        )
        self._urls[normalized] = record
        self._original_to_normalized[url] = normalized
        
        return True, record
    
    def add_batch(self, urls: list[str]) -> tuple[int, int, list[URLRecord]]:
        """
        Add multiple URLs.
        
        Returns:
            Tuple of (added_count, duplicate_count, new_records)
        """
        added = 0
        duplicates = 0
        new_records = []
        
        for url in urls:
            is_new, record = self.add(url)
            if is_new:
                added += 1
                new_records.append(record)
            else:
                duplicates += 1
        
        return added, duplicates, new_records
    
    def get(self, url: str) -> Optional[URLRecord]:
        """Get a URL record by original or normalized URL."""
        normalized = self.normalize(url)
        return self._urls.get(normalized)
    
    def update_status(self, url: str, status: str, **kwargs) -> bool:
        """Update the status of a URL record."""
        record = self.get(url)
        if not record:
            return False
        
        record.status = status
        for key, value in kwargs.items():
            if hasattr(record, key):
                setattr(record, key, value)
        
        return True
    
    def set_content_hash(self, url: str, content: str) -> tuple[bool, Optional[str]]:
        """
        Set content hash for a URL.
        
        Returns:
            Tuple of (is_unique_content, duplicate_url_if_any)
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        record = self.get(url)
        
        if not record:
            return False, None
        
        # Check for content duplicate
        if content_hash in self._content_hashes:
            duplicate_url = self._content_hashes[content_hash]
            return False, duplicate_url
        
        record.content_hash = content_hash
        self._content_hashes[content_hash] = record.normalized
        
        return True, None
    
    def get_by_status(self, status: str) -> list[URLRecord]:
        """Get all URLs with a specific status."""
        return [r for r in self._urls.values() if r.status == status]
    
    def get_all(self) -> list[URLRecord]:
        """Get all URL records."""
        return list(self._urls.values())
    
    def count(self) -> int:
        """Get total count of URLs."""
        return len(self._urls)
    
    def count_by_status(self) -> dict[str, int]:
        """Get count of URLs by status."""
        counts: dict[str, int] = {}
        for record in self._urls.values():
            counts[record.status] = counts.get(record.status, 0) + 1
        return counts
    
    def remove(self, url: str) -> bool:
        """Remove a URL from the store."""
        normalized = self.normalize(url)
        record = self._urls.get(normalized)
        
        if not record:
            return False
        
        # Clean up all references
        del self._urls[normalized]
        if record.original in self._original_to_normalized:
            del self._original_to_normalized[record.original]
        if record.content_hash and record.content_hash in self._content_hashes:
            del self._content_hashes[record.content_hash]
        
        return True
    
    def clear(self) -> int:
        """Clear all URLs. Returns count of removed URLs."""
        count = len(self._urls)
        self._urls.clear()
        self._content_hashes.clear()
        self._original_to_normalized.clear()
        return count
    
    def __contains__(self, url: str) -> bool:
        """Check if URL exists in store."""
        normalized = self.normalize(url)
        return normalized in self._urls
    
    def __len__(self) -> int:
        return len(self._urls)
