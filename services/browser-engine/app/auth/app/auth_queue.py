"""Background queue for handling authentication tasks."""

import asyncio
import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple

import structlog

from .interactive import InteractiveAuthenticator
from .models import AuthenticationTask


logger = structlog.get_logger()


class AuthenticationQueue:
    """Manages queued authentication tasks with parallel processing."""

    def __init__(self, authenticator: Optional[InteractiveAuthenticator] = None, max_workers: int = 3) -> None:
        self.task_queue: "queue.PriorityQueue[Tuple[int, str]]" = queue.PriorityQueue()
        self.active_tasks: Dict[str, AuthenticationTask] = {}
        self.completed_tasks: Dict[str, AuthenticationTask] = {}
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.authenticator = authenticator or InteractiveAuthenticator()
        self._running = False
        self._worker_threads: list[threading.Thread] = []

    def start_processing(self) -> None:
        """Start the authentication queue processing."""
        if self._running:
            return

        self._running = True
        for _ in range(self.max_workers):
            thread = threading.Thread(target=self._worker_loop, daemon=True)
            thread.start()
            self._worker_threads.append(thread)

        logger.info("Authentication queue processing started", workers=self.max_workers)

    def stop_processing(self) -> None:
        """Stop the authentication queue processing."""
        self._running = False
        logger.info("Authentication queue processing stopped")

    def queue_authentication(
        self,
        domain: str,
        auth_method: str,
        credentials: Dict[str, Any],
        login_url: str,
        priority: int = 1,
    ) -> str:
        """Queue an authentication task."""
        task_id = str(uuid.uuid4())

        task = AuthenticationTask(
            task_id=task_id,
            domain=domain,
            auth_method=auth_method,
            credentials=credentials,
            priority=priority,
        )

        task.credentials["login_url"] = login_url

        self.active_tasks[task_id] = task
        self.task_queue.put((-priority, task_id))

        logger.info("Authentication task queued", task_id=task_id, domain=domain)
        return task_id

    def get_task_status(self, task_id: str) -> Optional[AuthenticationTask]:
        """Get the status of a queued task."""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        return None

    def _worker_loop(self) -> None:
        """Worker thread loop for processing authentication tasks."""
        while self._running:
            try:
                try:
                    _, task_id = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if task_id not in self.active_tasks:
                    continue

                task = self.active_tasks[task_id]
                task.status = "processing"

                logger.info("Processing authentication task", task_id=task_id, domain=task.domain)

                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    result = loop.run_until_complete(
                        self.authenticator.authenticate_with_popup(
                            domain=task.domain,
                            auth_method=task.auth_method,
                            credentials=task.credentials,
                            login_url=task.credentials["login_url"],
                        )
                    )

                    task.result = result
                    task.status = "completed" if result.get("success") else "failed"

                    if not result.get("success"):
                        task.error_message = result.get("message", "Authentication failed")

                    logger.info("Authentication task completed", task_id=task_id, success=result.get("success"))
                except Exception as exc:
                    task.status = "failed"
                    task.error_message = str(exc)
                    logger.error("Authentication task failed", task_id=task_id, error=str(exc))
                finally:
                    self.completed_tasks[task_id] = task
                    if task_id in self.active_tasks:
                        del self.active_tasks[task_id]
                    self.task_queue.task_done()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Worker thread error", error=str(exc))


__all__ = ["AuthenticationQueue"]
