"""Configuration helpers for the export service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class ExportSettings:
    """Runtime configuration for the export service.

    The service primarily relies on environment variables so we initialise
    everything up-front and make sure required directories exist. For local
    development we gracefully fall back to repository paths when the configured
    location is missing.
    """

    def __init__(
        self,
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        export_dir: Optional[Path] = None,
        templates_dir: Optional[Path] = None,
    ) -> None:
        self.qdrant_host = qdrant_host or os.getenv("QDRANT_HOST", "qdrant")
        self.qdrant_port = int(qdrant_port or os.getenv("QDRANT_PORT", "6333"))

        # Resolve export directory with sensible fallbacks for local execution.
        export_dir_candidate = Path(export_dir or os.getenv("EXPORT_DIR", "/app/exports"))
        templates_dir_candidate = Path(templates_dir or os.getenv("TEMPLATES_DIR", "/app/templates"))

        self.export_dir = self._ensure_directory(export_dir_candidate, Path("./exports"))
        self.templates_dir = self._ensure_directory(templates_dir_candidate, Path("./templates"))

        self.service_name = "Export Service"
        self.version = "1.0.0"

    @staticmethod
    def _ensure_directory(primary: Path, fallback: Path) -> Path:
        """Guarantee a directory exists, falling back when necessary."""
        path = primary
        if not path.exists():
            path = fallback
        path.mkdir(parents=True, exist_ok=True)
        return path


__all__ = ["ExportSettings"]
