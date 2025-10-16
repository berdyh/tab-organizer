"""
Thread pool executor shared across CPU-bound tasks in the clustering service.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .config import MAX_WORKERS

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

__all__ = ["executor"]
