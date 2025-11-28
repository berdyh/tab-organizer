"""Queue management utilities for scraper jobs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Optional, Set

from .models import ProcessingStats, QueueType, URLClassification, URLRequest


@dataclass
class QueueItem:
    """Represents an item scheduled for processing."""

    classification: URLClassification
    request: URLRequest
    queued_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0


class ProcessingQueues:
    """Holds queues for public, authenticated, and retry URLs."""

    def __init__(self) -> None:
        self.public_queue: Deque[QueueItem] = deque()
        self.auth_queue: Deque[QueueItem] = deque()
        self.retry_queue: Deque[QueueItem] = deque()
        self.processing: Set[str] = set()
        self.completed: Set[str] = set()
        self.failed: Set[str] = set()

    def add_to_queue(self, classification: URLClassification, url_request: URLRequest) -> None:
        """Add URL to the appropriate queue."""
        item = QueueItem(classification=classification, request=url_request)

        if classification.queue_type == QueueType.PUBLIC:
            self.public_queue.append(item)
        elif classification.queue_type == QueueType.AUTHENTICATED:
            self.auth_queue.append(item)
        else:
            self.retry_queue.append(item)

    def get_next_public(self) -> Optional[QueueItem]:
        """Retrieve the next public queue item."""
        return self.public_queue.popleft() if self.public_queue else None

    def get_next_auth(self) -> Optional[QueueItem]:
        """Retrieve the next authenticated queue item."""
        return self.auth_queue.popleft() if self.auth_queue else None

    def get_next_retry(self) -> Optional[QueueItem]:
        """Retrieve the next retry queue item."""
        return self.retry_queue.popleft() if self.retry_queue else None

    def get_stats(self) -> ProcessingStats:
        """Return current queue statistics."""
        return ProcessingStats(
            total_urls=len(self.public_queue)
            + len(self.auth_queue)
            + len(self.retry_queue)
            + len(self.processing)
            + len(self.completed)
            + len(self.failed),
            public_queue_size=len(self.public_queue),
            auth_queue_size=len(self.auth_queue),
            retry_queue_size=len(self.retry_queue),
            completed=len(self.completed),
            failed=len(self.failed),
        )
