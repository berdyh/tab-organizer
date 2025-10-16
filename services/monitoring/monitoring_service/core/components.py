"""
Lifecycle management for monitoring service components.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional

from ..alerts.manager import AlertManager
from ..config import MonitoringSettings
from ..health.monitor import HealthMonitor
from ..logging import get_logger
from ..metrics.collector import MetricsCollector
from ..performance.tracker import PerformanceTracker
from ..tracing.distributed import DistributedTracer

logger = get_logger("components")


@dataclass
class MonitoringComponents:
    """Manage lifecycle of the monitoring subsystems."""

    settings: MonitoringSettings = field(default_factory=MonitoringSettings)
    metrics: Optional[MetricsCollector] = None
    health: Optional[HealthMonitor] = None
    alerts: Optional[AlertManager] = None
    performance: Optional[PerformanceTracker] = None
    tracing: Optional[DistributedTracer] = None
    _tasks: List[asyncio.Task] = field(default_factory=list, init=False)

    async def startup(self) -> None:
        """Initialise subsystems and background workers."""
        logger.info("Initialising monitoring components")

        self.metrics = MetricsCollector(self.settings)
        self.health = HealthMonitor(self.settings)
        self.alerts = AlertManager(self.settings)
        self.performance = PerformanceTracker(self.settings)
        self.tracing = DistributedTracer(self.settings)

        if self.metrics:
            self._tasks.append(asyncio.create_task(self.metrics.start_collection()))
        if self.health:
            self._tasks.append(asyncio.create_task(self.health.start_monitoring()))
        if self.alerts:
            self._tasks.append(asyncio.create_task(self.alerts.start_alert_processing()))
        if self.performance and self.settings.performance_tracking_enabled:
            self._tasks.append(asyncio.create_task(self.performance.start_tracking()))
        if self.tracing and self.settings.distributed_tracing_enabled:
            self._tasks.append(asyncio.create_task(self.tracing.start_tracing()))

        logger.info("Monitoring components started", tasks=len(self._tasks))

    async def shutdown(self) -> None:
        """Stop background workers and release resources."""
        logger.info("Shutting down monitoring components", tasks=len(self._tasks))

        for task in self._tasks:
            task.cancel()

        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                continue
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Background task shutdown error", error=str(exc))

        self._tasks.clear()

        if self.metrics:
            await self.metrics.close()
        if self.health:
            await self.health.close()
        if self.alerts:
            await self.alerts.close()
        if self.performance:
            await self.performance.close()
        if self.tracing:
            await self.tracing.close()

        logger.info("Monitoring components shutdown complete")

    async def reload(self) -> None:
        """Reload dynamic configuration across subsystems."""
        logger.info("Reloading monitoring configuration")
        self.settings = MonitoringSettings()

        if self.metrics:
            await self.metrics.reload_config()
        if self.health:
            await self.health.reload_config()
        if self.alerts:
            await self.alerts.reload_config()
        if self.performance:
            await self.performance.reload_config()
        if self.tracing:
            await self.tracing.reload_config()

        logger.info("Monitoring configuration reloaded")
