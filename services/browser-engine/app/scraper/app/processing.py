"""Parallel processing engine for scraper jobs."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from .logging import get_logger
from .models import QueueType, ScrapeJob, ScrapeRequest, ScrapedContent
from .queues import ProcessingQueues, QueueItem
from .state import state

logger = get_logger()


class ParallelProcessingEngine:
    """Engine for parallel processing of authenticated and non-authenticated URLs."""

    def __init__(self, max_workers: int = 5) -> None:
        self.max_workers = max_workers
        self.public_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.auth_executor = ThreadPoolExecutor(max_workers=max_workers // 2 + 1)
        self.retry_executor = ThreadPoolExecutor(max_workers=2)
        self.active_tasks: Dict[str, Set[asyncio.Task[Any]]] = defaultdict(set)
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)

    async def process_job(
        self,
        job_id: str,
        scrape_request: ScrapeRequest,
        url_classifications: List[Any],
    ) -> None:
        """Process a scraping job with parallel authentication workflow."""
        try:
            state.processing_queues[job_id] = ProcessingQueues()
            queues = state.processing_queues[job_id]

            for index, classification in enumerate(url_classifications):
                url_request = scrape_request.urls[index]
                queues.add_to_queue(classification, url_request)

            tasks: List[asyncio.Task[Any]] = []

            if queues.public_queue:
                task = asyncio.create_task(
                    self._process_queue(job_id, scrape_request, QueueType.PUBLIC)
                )
                tasks.append(task)
                self.active_tasks[job_id].add(task)

            if queues.auth_queue:
                task = asyncio.create_task(
                    self._process_queue(job_id, scrape_request, QueueType.AUTHENTICATED)
                )
                tasks.append(task)
                self.active_tasks[job_id].add(task)

            if scrape_request.max_retries > 0:
                task = asyncio.create_task(
                    self._process_queue(job_id, scrape_request, QueueType.RETRY)
                )
                tasks.append(task)
                self.active_tasks[job_id].add(task)

            if tasks:
                await asyncio.gather(*tasks)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Parallel processing failed", job_id=job_id, error=str(exc))

    async def _process_queue(
        self, job_id: str, scrape_request: ScrapeRequest, queue_type: QueueType
    ) -> None:
        """Process items from the selected queue."""
        queues = state.processing_queues.get(job_id)
        if not queues:
            return

        get_next_map = {
            QueueType.PUBLIC: queues.get_next_public,
            QueueType.AUTHENTICATED: queues.get_next_auth,
            QueueType.RETRY: queues.get_next_retry,
        }
        get_next = get_next_map[queue_type]

        while True:
            if self._should_stop_processing(job_id):
                break

            item = get_next()
            if item is None:
                if queue_type == QueueType.RETRY:
                    break
                await asyncio.sleep(0.1)
                continue

            try:
                await self._process_single_url(job_id, item, scrape_request)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(
                    "Queue processing failed",
                    job_id=job_id,
                    url=str(item.request.url),
                    error=str(exc),
                )

    async def _process_single_url(
        self, job_id: str, item: QueueItem, scrape_request: ScrapeRequest
    ) -> None:
        """Process a single URL with timing and error handling."""
        start_time = time.time()
        url = str(item.request.url)

        queues = state.processing_queues[job_id]
        queues.processing.add(url)
        await self._update_realtime_status(job_id, url, "processing")

        try:
            result = await self._scrape_url(job_id, item, scrape_request)
            response_time = (time.time() - start_time) * 1000
            self.performance_metrics[job_id].append(response_time)

            if result:
                await self._update_job_success(job_id, result, response_time)
                queues.completed.add(url)
            else:
                await self._mark_url_failed(job_id, item, "Scraping failed")
                queues.failed.add(url)
        except Exception as exc:  # pragma: no cover - defensive
            await self._mark_url_failed(job_id, item, str(exc))
            queues.failed.add(url)
        finally:
            queues.processing.discard(url)

    async def _scrape_url(
        self, job_id: str, item: QueueItem, scrape_request: ScrapeRequest
    ) -> Optional[ScrapedContent]:
        """Perform actual URL scraping with authentication support."""
        # Placeholder for integration with Scrapy or HTTP client.
        return None

    async def _has_auth_session(self, job_id: str, domain: str) -> bool:
        """Check if we have an active auth session for domain."""
        job = state.active_jobs.get(job_id)
        if not job:
            return False

        session_id = job.auth_sessions.get(domain)
        if not session_id:
            return False

        session = state.auth_sessions.get(session_id)
        return bool(session and session.is_active)

    async def _update_job_success(
        self, job_id: str, content: ScrapedContent, response_time: float
    ) -> None:
        """Update job with successful result."""
        job = state.active_jobs.get(job_id)
        if not job:
            return

        job.results.append(content)
        job.completed_urls += 1
        job.updated_at = datetime.now()

        job.processing_stats.completed = job.completed_urls
        if self.performance_metrics[job_id]:
            job.processing_stats.avg_response_time = sum(
                self.performance_metrics[job_id]
            ) / len(self.performance_metrics[job_id])

        total_processed = job.completed_urls + job.failed_urls
        if total_processed >= job.total_urls:
            job.status = "completed"

        await self._broadcast_job_update(job_id, job)

    async def _mark_url_failed(self, job_id: str, item: QueueItem, error: str) -> None:
        """Mark URL as failed and update job."""
        job = state.active_jobs.get(job_id)
        if not job:
            return

        job.errors.append(
            {
                "url": str(item.request.url),
                "error": error,
                "timestamp": datetime.now().isoformat(),
                "retry_count": item.retry_count,
            }
        )
        job.failed_urls += 1
        job.updated_at = datetime.now()

        job.processing_stats.failed = job.failed_urls
        total_processed = job.completed_urls + job.failed_urls
        if total_processed >= job.total_urls:
            job.status = "completed"

        await self._broadcast_job_update(job_id, job)

    async def _update_realtime_status(self, job_id: str, url: str, status: str) -> None:
        """Send real-time status updates to websocket clients."""
        websocket = state.get_websocket(job_id)
        if not websocket:
            return

        job = state.active_jobs.get(job_id)
        timestamp = datetime.now().isoformat()

        try:
            await websocket.send_json(
                {
                    "type": "url_status_update",
                    "job_id": job_id,
                    "url": url,
                    "status": status,
                    "timestamp": timestamp,
                }
            )
        except Exception:
            state.websocket_connections.pop(job_id, None)

    async def _broadcast_job_update(self, job_id: str, job: ScrapeJob) -> None:
        """Broadcast job updates to websocket clients."""
        websocket = state.get_websocket(job_id)
        if not websocket:
            return

        try:
            await websocket.send_json(
                {"type": "job_update", "job": job.model_dump(mode="json")}
            )
        except Exception:
            state.websocket_connections.pop(job_id, None)

    def _should_stop_processing(self, job_id: str) -> bool:
        """Check if processing should stop for a job."""
        job = state.active_jobs.get(job_id)
        if not job:
            return True

        total_processed = job.completed_urls + job.failed_urls
        return total_processed >= job.total_urls or job.status == "cancelled"

    def get_performance_metrics(self, job_id: str) -> Dict[str, Any]:
        """Return performance metrics for a job."""
        response_times = self.performance_metrics.get(job_id)
        if not response_times:
            return {}

        return {
            "avg_response_time": sum(response_times) / len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "total_requests": len(response_times),
        }

    async def shutdown(self) -> None:
        """Shutdown the processing engine."""
        self.public_executor.shutdown(wait=True)
        self.auth_executor.shutdown(wait=True)
        self.retry_executor.shutdown(wait=True)
