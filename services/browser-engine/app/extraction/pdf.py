"""PDF text extraction for tabs whose content is not in the DOM.

Chrome renders a PDF in PDFium, a separate plugin process, so the page's DOM is
an empty viewer shell: `page.content()` returns ~275 bytes of chrome and
`document.body.innerText` returns "". Both of the harvester's existing
extraction paths therefore yield an empty string for a PDF tab, and an empty
string is what the AI Engine forwards to the embedding provider -- which
rejects the WHOLE batch (`Too small: expected string to have >=1 characters`).
Two arXiv PDFs among 46 real tabs blocked all 46 from being indexed.

`Page.getResourceContent` does not help: the PDF is not in the page's resource
tree (verified -- it returns an empty body). The bytes are fetched instead
through the page's own `context.request`, which is Playwright's
`APIRequestContext` bound to the browser context and therefore carries that
context's cookie jar. That matters beyond convenience: a paywalled or
authenticated PDF the user is reading extracts correctly, where a server-side
`httpx` fetch would silently get a login page or a 403 and store THAT as the
document -- the same "logged-out capture read as an authenticated one" failure
the scraper module already had to fix once.
"""

from __future__ import annotations

import io
from typing import Optional

# Pages beyond this are not read. A thesis or a scanned book would otherwise
# spend minutes of CPU inside a tab import; the first pages carry more than
# enough signal for retrieval, and the cap is recorded in the result metadata
# so a truncated extraction is never mistaken for a complete one.
MAX_PDF_PAGES = 50

# Above this the download is refused outright rather than buffered.
MAX_PDF_BYTES = 40 * 1024 * 1024

PDF_MAGIC = b"%PDF-"


class PDFExtractionError(RuntimeError):
    """The bytes could not be read as a PDF."""


def looks_like_pdf(data: bytes) -> bool:
    """Whether these bytes start with the PDF magic number."""
    return data[:5] == PDF_MAGIC


def extract_pdf_text(data: bytes, *, max_pages: int = MAX_PDF_PAGES) -> str:
    """Return the text of a PDF, or raise ``PDFExtractionError``.

    An image-only (scanned) PDF legitimately yields no text. That is reported
    as an empty string rather than an error, so the caller records it as a
    visible skip with a reason instead of a failure -- there is nothing wrong
    with the document, it simply has no text layer to extract.
    """
    if not data:
        raise PDFExtractionError("no bytes to extract")
    if not looks_like_pdf(data):
        raise PDFExtractionError(f"not a PDF (first bytes: {data[:8]!r})")
    if len(data) > MAX_PDF_BYTES:
        raise PDFExtractionError(
            f"PDF is {len(data)} bytes, over the {MAX_PDF_BYTES} limit"
        )

    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise PDFExtractionError(f"pypdf unavailable: {exc}") from exc

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = reader.pages[:max_pages]
        chunks = []
        for page in pages:
            try:
                chunks.append(page.extract_text() or "")
            except Exception:
                # One unreadable page must not lose the other forty-nine.
                continue
    except PDFExtractionError:
        raise
    except Exception as exc:
        raise PDFExtractionError(f"could not read PDF: {exc}") from exc

    return "\n".join(chunk for chunk in chunks if chunk.strip())


async def fetch_pdf_bytes(page, url: str) -> Optional[bytes]:
    """Fetch ``url`` through the page's browser context (cookies included)."""
    response = await page.context.request.get(url)
    if response.status != 200:
        raise PDFExtractionError(f"fetch returned HTTP {response.status}")
    body = await response.body()
    if len(body) > MAX_PDF_BYTES:
        raise PDFExtractionError(
            f"PDF is {len(body)} bytes, over the {MAX_PDF_BYTES} limit"
        )
    return body
