"""Shared singletons for the scraper service."""

from __future__ import annotations

from .classification import URLClassifier
from .duplicates import DuplicateDetector
from .engine import ScrapingEngine
from .processing import ParallelProcessingEngine

url_classifier = URLClassifier()
duplicate_detector = DuplicateDetector()
parallel_engine = ParallelProcessingEngine()
scraping_engine = ScrapingEngine()

__all__ = [
    "url_classifier",
    "duplicate_detector",
    "parallel_engine",
    "scraping_engine",
]
