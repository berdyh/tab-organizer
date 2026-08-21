"""Contracts for tab content extraction and the visible-skip path.

These freeze the behaviour a real 46-tab import surfaced: two arXiv PDFs
harvested as empty strings, which made the embedding provider reject the WHOLE
batch (`Too small: expected string to have >=1 characters`) so that 0 of 46 tabs
were indexed. The fix has two halves, and both are asserted here:

* a tab whose text is not in the DOM gets a format-specific extractor, and
* anything still without text is REPORTED as a skip with a reason, never
  forwarded as an empty document.

The second half is the load-bearing one. A silent empty document reads as a
success at every layer except the one that rejects the batch.
"""

import sys
import types

import pytest


def _install_playwright_stub() -> None:
    """Let these tests import browser-engine code without browser binaries."""
    playwright_module = types.ModuleType("playwright")
    async_api = types.ModuleType("playwright.async_api")
    async_api.Browser = object
    async_api.Page = object
    async_api.TimeoutError = TimeoutError
    async_api.async_playwright = lambda: None
    playwright_module.async_api = async_api
    sys.modules["playwright"] = playwright_module
    sys.modules["playwright.async_api"] = async_api


try:
    import playwright.async_api  # noqa: F401
except ModuleNotFoundError:
    _install_playwright_stub()

from services.browser_engine.app.extraction.fallbacks import (  # noqa: E402
    ExtractionFailed,
    select_extractor,
)
from services.browser_engine.app.extraction.pdf import (  # noqa: E402
    MAX_PDF_BYTES,
    PDFExtractionError,
    extract_pdf_text,
    looks_like_pdf,
)
from services.browser_engine.app.tabs.cdp import (  # noqa: E402
    CDPTabHarvester,
    HarvestedTab,
    SkippedTab,
    TabHarvestResult,
)

# A minimal single-page PDF carrying one line of extractable text. Built by hand
# rather than with a writer library so the fixture has no dependency of its own
# and the bytes under test are visible in the source.
SAMPLE_PDF_TEXT = "Lumbosacral radiculoplexus neuropathy"


def _build_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R"
        b"/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length " + str(len(stream)).encode() + b">>stream\n" + stream + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj".encode() + body + b"endobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer<</Size {len(objects) + 1}/Root 1 0 R>>\n".encode()
    out += f"startxref\n{xref_at}\n%%EOF\n".encode()
    return bytes(out)


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------


def test_extract_pdf_text_reads_a_real_pdf():
    text = extract_pdf_text(_build_pdf(SAMPLE_PDF_TEXT))
    assert SAMPLE_PDF_TEXT.split()[0] in text


def test_looks_like_pdf_checks_the_magic_number():
    assert looks_like_pdf(b"%PDF-1.7 ...")
    assert not looks_like_pdf(b"<html><body>login</body></html>")


def test_extract_pdf_text_refuses_non_pdf_bytes():
    """An HTML login page fetched instead of a PDF must not be stored as one."""
    with pytest.raises(PDFExtractionError) as excinfo:
        extract_pdf_text(b"<html><body>Please sign in</body></html>")
    assert "not a PDF" in str(excinfo.value)


def test_extract_pdf_text_refuses_empty_input():
    with pytest.raises(PDFExtractionError):
        extract_pdf_text(b"")


def test_extract_pdf_text_refuses_oversized_input():
    oversized = b"%PDF-" + b"0" * MAX_PDF_BYTES
    with pytest.raises(PDFExtractionError) as excinfo:
        extract_pdf_text(oversized)
    assert "over the" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Extractor selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://arxiv.org/pdf/2608.02878",  # no extension: the real failing case
        "https://example.com/paper.pdf",
        "https://example.com/a/PAPER.PDF",
    ],
)
def test_select_extractor_recognises_pdf_urls(url):
    name, extractor = select_extractor(url)
    assert name == "pdf" and extractor is not None


def test_select_extractor_recognises_pdf_content_type():
    name, _ = select_extractor("https://example.com/download", "application/pdf")
    assert name == "pdf"


def test_select_extractor_returns_nothing_for_ordinary_pages():
    assert select_extractor("https://example.com/article") == (None, None)


# ---------------------------------------------------------------------------
# Harvester doubles
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    async def body(self):
        return self._body


class _FakeRequest:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self._status = status
        self.requested: list[str] = []

    async def get(self, url):
        self.requested.append(url)
        return _FakeResponse(self._body, self._status)


class _FakeContext:
    def __init__(self, body: bytes = b"", status: int = 200):
        self.request = _FakeRequest(body, status)


class _FakePage:
    """Playwright page double whose DOM text and fetched bytes are separate."""

    def __init__(self, url, title, body_html="", inner_text="", pdf_bytes=b""):
        self.url = url
        self._title = title
        self._body_html = body_html
        self._inner_text = inner_text
        self.context = _FakeContext(pdf_bytes)

    async def title(self):
        return self._title

    async def content(self):
        return (
            f"<html><head><title>{self._title}</title></head>"
            f"<body>{self._body_html}</body></html>"
        )

    async def evaluate(self, script):
        assert "innerText" in script
        return self._inner_text


def _harvest(page):
    import asyncio

    harvester = CDPTabHarvester(cdp_url="http://127.0.0.1:9222")
    return asyncio.run(harvester._harvest_page(page))


# ---------------------------------------------------------------------------
# Harvest outcomes
# ---------------------------------------------------------------------------


def test_pdf_tab_is_extracted_rather_than_harvested_blank():
    """The exact tab that broke a 46-tab import: a PDF with an empty DOM."""
    page = _FakePage(
        "https://arxiv.org/pdf/2608.02878",
        "arXiv PDF",
        body_html="",
        inner_text="",
        pdf_bytes=_build_pdf(SAMPLE_PDF_TEXT),
    )
    result = _harvest(page)
    assert isinstance(result, HarvestedTab), f"expected a tab, got {result}"
    assert SAMPLE_PDF_TEXT.split()[0] in result.content
    assert result.metadata["extracted_via"] == "pdf"


def test_pdf_tab_that_cannot_be_read_is_a_visible_skip_not_a_blank_document():
    page = _FakePage(
        "https://example.com/paper.pdf",
        "Broken PDF",
        pdf_bytes=b"<html>not a pdf at all</html>",
    )
    result = _harvest(page)
    assert isinstance(result, SkippedTab)
    assert result.reason == "extraction_failed"
    assert "pdf" in result.detail


def test_genuinely_blank_tab_is_skipped_with_a_reason():
    page = _FakePage("https://example.com/empty", "Empty")
    result = _harvest(page)
    assert isinstance(result, SkippedTab)
    assert result.reason == "blank"
    assert result.detail, "a skip must carry a reason a human can act on"


def test_sign_in_page_is_skipped_rather_than_embedded_as_content():
    """A login wall's text is the login form, not the page the user has open.

    Embedding it files the sign-in prompt under the tab's title, so the
    logged-OUT page answers searches for the logged-IN one.
    """
    page = _FakePage(
        "https://elevenreader.io/reader/library/u:abc",
        "Sign in",
        body_html=(
            '<form action="/login" method="post">'
            '<input type="email" name="email">'
            '<input type="password" name="password">'
            '<button>Sign in with Google</button></form>'
        ),
        inner_text="Sign in to continue",
    )
    result = _harvest(page)
    assert isinstance(result, SkippedTab), f"expected a skip, got {result}"
    assert result.reason == "auth_wall"


def test_ordinary_page_is_harvested_normally():
    page = _FakePage(
        "https://example.com/article",
        "An article",
        body_html="<p>Some genuine article text here.</p>",
        inner_text="Some genuine article text here.",
    )
    result = _harvest(page)
    assert isinstance(result, HarvestedTab)
    assert "genuine article text" in result.content
    assert "extracted_via" not in result.metadata


# ---------------------------------------------------------------------------
# Result accounting
# ---------------------------------------------------------------------------


def test_result_counts_skips_and_lists_them():
    """`total` must include skips, and each skip must be individually visible.

    Before this, `total` was `len(tabs)` -- so a skipped tab vanished from the
    arithmetic as well as from the payload, and an operator reading
    `total=44 imported=44` had no way to learn that two tabs were dropped.
    """
    result = TabHarvestResult(
        tabs=[
            HarvestedTab(url="https://a.example", title="A", content="text"),
        ],
        skipped=[
            SkippedTab(url="https://b.example", title="B", reason="blank", detail="d"),
            SkippedTab(
                url="https://c.example", title="C", reason="auth_wall", detail="d"
            ),
        ],
    )
    payload = result.to_dict()
    assert payload["total"] == 3
    assert payload["imported"] == 1
    assert payload["skipped"] == 2
    assert {s["reason"] for s in payload["skipped_tabs"]} == {"blank", "auth_wall"}
    assert {s["url"] for s in payload["skipped_tabs"]} == {
        "https://b.example",
        "https://c.example",
    }
