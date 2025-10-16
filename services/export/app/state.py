"""Shared in-memory state for the export service."""

from __future__ import annotations

from typing import Dict

from .models import ExportJob

export_jobs: Dict[str, ExportJob] = {}

__all__ = ["export_jobs"]
