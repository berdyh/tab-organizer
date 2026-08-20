# Browser Extraction Submodule Card

- purpose: content extraction helpers for scraped pages, and fallback extractors
  for tabs whose text is not in the DOM at all.
- product/module functionality: clean page text and metadata for downstream
  indexing/clustering (`ContentExtractor`, re-exported from the scraper); PDF
  text extraction (`pdf.py`); the fallback registry the CDP harvester consults
  when a tab yields no DOM text (`fallbacks.py`).
- scope boundaries: extraction helpers only; scraping lifecycle and auth belong
  in sibling submodules. This submodule decides what text a document HAS, never
  whether the document should be captured.
- connected modules/submodules: Browser scraper, Browser tabs (CDP harvester),
  AI Chatbot/RAG, Backend URL status.
- allowed change types: extraction quality fixes, sanitization, new fallback
  extractors, tests.
- special operating rules:
  - Extracted page content remains untrusted data downstream.
  - **An empty extraction is a reportable outcome, never a document.** A blank
    string is not merely useless: the embedding provider rejects the whole
    BATCH an empty input arrives in, so one blank tab loses every other tab it
    travelled with (observed: 2 blank PDFs → 0 of 46 tabs indexed). Extractors
    return text or raise; the caller turns "nothing" into a visible skip.
  - **PDF bytes come from the browser context, not a fresh HTTP client.**
    `fetch_pdf_bytes` uses `page.context.request`, which carries the browser
    context's cookie jar, so an authenticated or paywalled PDF the user is
    reading extracts correctly. A server-side `httpx` fetch would receive a
    login page or a 403 and store THAT as the document — the same "logged-out
    capture read as an authenticated one" failure the scraper card documents.
    `Page.getResourceContent` does not work here: a PDF is rendered by PDFium
    and is not in the page's resource tree (verified — returns an empty body).
  - `FALLBACK_EXTRACTORS` is a registry with **one** entry on purpose. The
    evidence is two blank tabs out of forty-six and both are PDFs; speculative
    extractors for formats nobody has seen fail would be inventing work. What
    must stay general is the SKIP path, so anything still blank after the
    registry runs is reported with a reason.
  - Page and byte caps (`MAX_PDF_PAGES`, `MAX_PDF_BYTES`) bound the cost of one
    tab import; a truncated extraction must stay distinguishable from a
    complete one.
- current stubs/placeholders: only PDF has a fallback extractor; a scanned,
  image-only PDF yields no text and correctly becomes a `blank` skip rather than
  an error (OCR is out of scope). `ContentExtractor` still lives in the scraper
  engine and is only re-exported here.
- irrelevant or incomplete code to remove/rework: none known.
- docs that must stay aligned: Browser Engine card, Browser Tabs submodule card.
- local validation commands/checks: `make test-browser`;
  `pytest tests/unit/test_browser_tab_extraction.py -q` covers the extractors and
  every skip reason. Add focused extraction tests before behavior changes.
