"""Shared API dependencies and helpers."""

from __future__ import annotations

from fastapi import HTTPException

from ..models import URLInput
from ..storage import url_input_storage


def get_url_input_or_404(input_id: str) -> URLInput:
    """Return a stored URL input or raise a 404 error."""
    url_input = url_input_storage.get(input_id)
    if not url_input:
        raise HTTPException(status_code=404, detail="Input not found")
    return url_input


__all__ = ["get_url_input_or_404"]
