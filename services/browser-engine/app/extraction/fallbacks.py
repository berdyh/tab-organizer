"""Fallback extractors for tabs whose text is not in the DOM.

The harvester's normal path (readable-HTML, then `document.body.innerText`)
covers ordinary pages. A few kinds of tab render their content outside the DOM
entirely and come back blank; a blank document is not merely useless, it makes
the embedding provider reject the whole batch it travels in.

This is a REGISTRY rather than a PDF special case, because "not only PDFs" is
the general shape of the problem -- but it deliberately ships with exactly one
entry. The evidence available is two blank tabs out of 46, both PDFs. Adding
speculative extractors for formats nobody has seen fail would be inventing
work; the part that must be general is the SKIP path, so anything still blank
after this registry runs is reported with a reason rather than passed on as an
empty string.

To add a format: write `async def _extract_x(page, url) -> str` and register it
with the predicate that recognises it. Raise `ExtractionFailed` to record a
visible failure; return `""` to record a visible blank.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Optional
from urllib.parse import urlparse

from .pdf import PDFExtractionError, extract_pdf_text, fetch_pdf_bytes


class ExtractionFailed(RuntimeError):
    """A fallback extractor recognised the tab but could not read it."""


def _is_pdf_url(url: str, content_type: Optional[str]) -> bool:
    if content_type and "application/pdf" in content_type.lower():
        return True
    path = (urlparse(url).path or "").lower()
    if path.endswith(".pdf"):
        return True
    # arXiv serves PDFs from /pdf/<id> with no extension, which is exactly the
    # case that produced this bug.
    return "/pdf/" in path


async def _extract_pdf(page, url: str) -> str:
    try:
        data = await fetch_pdf_bytes(page, url)
        return extract_pdf_text(data or b"")
    except PDFExtractionError as exc:
        raise ExtractionFailed(str(exc)) from exc
    except Exception as exc:  # transport, protocol, browser teardown
        raise ExtractionFailed(f"{type(exc).__name__}: {exc}") from exc


# (name, predicate, extractor)
FALLBACK_EXTRACTORS: tuple[
    tuple[str, Callable[[str, Optional[str]], bool], Callable[..., Awaitable[str]]],
    ...,
] = (("pdf", _is_pdf_url, _extract_pdf),)


def select_extractor(url: str, content_type: Optional[str] = None):
    """Return ``(name, extractor)`` for this tab, or ``(None, None)``."""
    for name, predicate, extractor in FALLBACK_EXTRACTORS:
        if predicate(url, content_type):
            return name, extractor
    return None, None
