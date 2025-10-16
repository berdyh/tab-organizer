"""Performance monitoring utilities."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil
import structlog

from .schemas import ModelPerformanceMetrics


class PerformanceMonitor:
    """Monitor AI model performance, resource usage, and availability."""

    def __init__(self) -> None:
        self.logger = structlog.get_logger("performance_monitor")
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.resource_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000

    def record_model_performance(
        self,
        model_id: str,
        model_type: str,
        success: bool,
        response_time: float,
        tokens_per_second: Optional[float] = None,
        resource_usage: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record performance metrics for a model."""
        if model_id not in self.metrics:
            self.metrics[model_id] = {
                "model_type": model_type,
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "response_times": [],
                "tokens_per_second_history": [],
                "last_used": None,
                "resource_usage_history": [],
            }

        metrics = self.metrics[model_id]
        metrics["total_requests"] += 1
        metrics["last_used"] = datetime.now()

        if success:
            metrics["successful_requests"] += 1
            metrics["response_times"].append(response_time)

            if tokens_per_second is not None:
                metrics["tokens_per_second_history"].append(tokens_per_second)

            if resource_usage:
                metrics["resource_usage_history"].append({"timestamp": datetime.now().isoformat(), **resource_usage})
        else:
            metrics["failed_requests"] += 1

        for key in ["response_times", "tokens_per_second_history", "resource_usage_history"]:
            if len(metrics[key]) > self.max_history_size:
                metrics[key] = metrics[key][-self.max_history_size :]

    def get_model_metrics(self, model_id: str) -> Optional[ModelPerformanceMetrics]:
        """Get performance metrics for a specific model."""
        if model_id not in self.metrics:
            return None

        metrics = self.metrics[model_id]

        avg_response_time = (
            sum(metrics["response_times"]) / len(metrics["response_times"]) if metrics["response_times"] else 0.0
        )

        avg_tokens_per_second = (
            sum(metrics["tokens_per_second_history"]) / len(metrics["tokens_per_second_history"])
            if metrics["tokens_per_second_history"]
            else None
        )

        resource_usage: Dict[str, float] = {}
        if metrics["resource_usage_history"]:
            resource_keys = set()
            for usage in metrics["resource_usage_history"]:
                resource_keys.update(usage.keys())
            resource_keys.discard("timestamp")

            for key in resource_keys:
                values = [usage.get(key, 0) for usage in metrics["resource_usage_history"] if key in usage]
                if values:
                    resource_usage[key] = sum(values) / len(values)

        return ModelPerformanceMetrics(
            model_id=model_id,
            model_type=metrics["model_type"],
            total_requests=metrics["total_requests"],
            successful_requests=metrics["successful_requests"],
            failed_requests=metrics["failed_requests"],
            average_response_time=avg_response_time,
            average_tokens_per_second=avg_tokens_per_second,
            last_used=metrics["last_used"],
            resource_usage=resource_usage,
        )

    def get_all_metrics(self) -> List[ModelPerformanceMetrics]:
        """Get performance metrics for all models."""
        return [metrics for model_id in self.metrics for metrics in [self.get_model_metrics(model_id)] if metrics]

    def record_system_resources(self) -> None:
        """Record current system resource usage."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)

            resource_snapshot: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "ram_usage_percent": memory.percent,
                "available_ram_gb": memory.available / (1024**3),
                "cpu_usage_percent": cpu_percent,
            }

            try:  # pragma: no cover - GPU introspection optional
                import torch

                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_stats()
                    resource_snapshot.update(
                        {
                            "gpu_memory_allocated_gb": gpu_memory.get("allocated_bytes.all.current", 0) / (1024**3),
                            "gpu_memory_reserved_gb": gpu_memory.get("reserved_bytes.all.current", 0) / (1024**3),
                        }
                    )
            except Exception:
                pass

            self.resource_history.append(resource_snapshot)

            if len(self.resource_history) > self.max_history_size:
                self.resource_history = self.resource_history[-self.max_history_size :]

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Error recording system resources", error=str(exc))

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of recent resource usage."""
        if not self.resource_history:
            return {}

        recent_history = self.resource_history[-100:]

        avg_ram_usage = sum(r["ram_usage_percent"] for r in recent_history) / len(recent_history)
        avg_cpu_usage = sum(r["cpu_usage_percent"] for r in recent_history) / len(recent_history)

        summary: Dict[str, Any] = {
            "average_ram_usage_percent": avg_ram_usage,
            "average_cpu_usage_percent": avg_cpu_usage,
            "current_available_ram_gb": recent_history[-1]["available_ram_gb"],
            "measurements_count": len(recent_history),
        }

        gpu_measurements = [r for r in recent_history if "gpu_memory_allocated_gb" in r]
        if gpu_measurements:
            summary["average_gpu_memory_allocated_gb"] = (
                sum(r["gpu_memory_allocated_gb"] for r in gpu_measurements) / len(gpu_measurements)
            )

        return summary


__all__ = ["PerformanceMonitor"]
