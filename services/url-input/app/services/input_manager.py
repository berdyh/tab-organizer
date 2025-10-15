"""Helpers for creating and persisting URL inputs."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple

from ..batching import BatchProcessor
from ..models import URLEntry, URLInput
from ..storage import url_input_storage
from ..utils import ensure_entry_ids


def create_url_input(
    urls: List[URLEntry],
    source_type: str,
    source_metadata: Dict[str, Any],
    session_id: str,
) -> Tuple[URLInput, Dict[str, Any]]:
    """Create a URLInput object, persist it, and return the instance with processing stats."""
    stats = BatchProcessor.get_processing_stats(urls)
    metadata = dict(source_metadata)
    metadata.setdefault("session_id", session_id)
    metadata.setdefault("processing_stats", stats)

    url_input = URLInput(
        input_id=str(uuid.uuid4()),
        urls=ensure_entry_ids(urls),
        session_id=session_id,
        source_type=source_type,
        source_metadata=metadata,
        created_at=datetime.now(),
    )

    url_input_storage.set(url_input.input_id, url_input)
    return url_input, stats


__all__ = ["create_url_input"]
