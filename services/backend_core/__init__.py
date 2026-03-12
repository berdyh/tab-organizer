"""Compatibility package mapping `services.backend_core` to `services/backend-core`."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent / "backend-core")]
