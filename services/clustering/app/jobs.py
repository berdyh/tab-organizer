"""
Shared job registry for long-running background tasks.
"""

from __future__ import annotations

from typing import Any, Dict

active_jobs: Dict[str, Dict[str, Any]] = {}

__all__ = ["active_jobs"]
