"""Hardware detection utilities."""

from __future__ import annotations

import builtins
from typing import Any, Dict

import psutil
import structlog


logger = structlog.get_logger(__name__)


class HardwareDetector:
    """Detect and monitor system hardware capabilities."""

    def __init__(self) -> None:
        self.logger = structlog.get_logger("hardware_detector")

    def detect_hardware(self) -> Dict[str, Any]:
        """Detect comprehensive hardware capabilities."""
        try:
            torch = self._import_torch()
            ram_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count()

            has_gpu = False
            gpu_memory_gb = 0.0
            gpu_name = "None"

            if torch is not None:
                try:
                    cuda = getattr(torch, "cuda", None)
                    cuda_available = bool(cuda and cuda.is_available())
                except Exception as exc:  # pragma: no cover - defensive logging
                    self.logger.warning("GPU availability check failed", error=str(exc))
                    cuda_available = False

                if cuda_available:
                    try:
                        props = cuda.get_device_properties(0)  # type: ignore[union-attr]
                        total_memory = getattr(props, "total_memory", 0)
                        if callable(total_memory):
                            total_memory = total_memory()
                        if hasattr(total_memory, "item"):
                            total_memory = total_memory.item()

                        total_memory_bytes = float(total_memory)
                        if total_memory_bytes > 0:
                            gpu_memory_gb = total_memory_bytes / (1024.0**3)
                            gpu_name = str(cuda.get_device_name(0))
                            has_gpu = True
                    except Exception as exc:  # pragma: no cover - defensive logging
                        self.logger.warning("GPU detection failed", error=str(exc))
                        has_gpu = False
                        gpu_memory_gb = 0.0
                        gpu_name = "None"

            memory = psutil.virtual_memory()
            available_ram_gb = memory.available / (1024**3)

            return {
                "ram_gb": ram_gb,
                "cpu_count": cpu_count,
                "has_gpu": has_gpu,
                "gpu_memory_gb": gpu_memory_gb,
                "gpu_name": gpu_name,
                "available_ram_gb": available_ram_gb,
                "ram_usage_percent": memory.percent,
                "cpu_usage_percent": psutil.cpu_percent(interval=0.1),
            }
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.error("Hardware detection failed", error=str(exc))
            return {
                "ram_gb": 8.0,
                "cpu_count": 4,
                "has_gpu": False,
                "gpu_memory_gb": 0.0,
                "gpu_name": "Unknown",
                "available_ram_gb": 4.0,
                "ram_usage_percent": 50.0,
                "cpu_usage_percent": 20.0,
            }

    @staticmethod
    def _import_torch():
        """Import torch lazily to honor runtime monkeypatching."""
        try:  # pragma: no cover - optional dependency
            return builtins.__import__("torch")
        except Exception:
            return None


__all__ = ["HardwareDetector", "logger"]
