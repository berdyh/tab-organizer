"""
System-level helpers for collecting host metrics.
"""

from __future__ import annotations

import psutil

from ..logging import get_logger

logger = get_logger("system_metrics")


async def collect_system_metrics() -> dict:
    """Gather CPU, memory, disk, and network metrics."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        network = psutil.net_io_counters()

        return {
            "cpu": {
                "percent": cpu_percent,
                "count": psutil.cpu_count(),
                "count_logical": psutil.cpu_count(logical=True),
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used,
                "free": memory.free,
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100 if disk.total else 0,
            },
            "network": {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv,
            },
        }
    except Exception as exc:  # pragma: no cover - relies on host metrics
        logger.error("Failed to collect system metrics", error=str(exc))
        return {}
