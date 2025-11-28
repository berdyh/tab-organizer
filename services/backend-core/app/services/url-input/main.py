"""URL Input service entrypoint."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

if __package__ in (None, ""):
    # Allow running as a stand-alone script ("python services/url-input/main.py")
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from app.api import router as api_router  # type: ignore
    from app.batching import BatchProcessor  # type: ignore
    from app.deduplication import URLDeduplicator  # type: ignore
    from app.detector import InputFormatDetector  # type: ignore
    from app.enrichment import URLEnricher  # type: ignore
    from app.logging import configure_logging, get_logger  # type: ignore
    from app.models import URLInput, URLEntry, URLMetadata  # type: ignore
    from app.parser import URLParser  # type: ignore
    from app.storage import url_input_storage  # type: ignore
    from app.utils import ensure_entry_ids, find_entry_by_id, parse_input_identifier  # type: ignore
    from app.validators import URLValidator  # type: ignore
else:
    from .app.api import router as api_router
    from .app.batching import BatchProcessor
    from .app.deduplication import URLDeduplicator
    from .app.detector import InputFormatDetector
    from .app.enrichment import URLEnricher
    from .app.logging import configure_logging, get_logger
    from .app.models import URLInput, URLEntry, URLMetadata
    from .app.parser import URLParser
    from .app.storage import url_input_storage
    from .app.utils import ensure_entry_ids, find_entry_by_id, parse_input_identifier
    from .app.validators import URLValidator


configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="URL Input Service",
    description="Handles URL parsing and validation from various input sources",
    version="1.0.0",
)

app.include_router(api_router)


# Backwards compatibility exports -------------------------------------------------
# Tests and other services import these symbols directly from main.py.
url_inputs = url_input_storage.data

__all__ = [
    "app",
    "url_inputs",
    "URLMetadata",
    "URLEntry",
    "URLInput",
    "URLValidator",
    "InputFormatDetector",
    "URLEnricher",
    "URLDeduplicator",
    "BatchProcessor",
    "URLParser",
    "ensure_entry_ids",
    "parse_input_identifier",
    "find_entry_by_id",
]


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
