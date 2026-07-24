# Browser Extraction Submodule Card

- purpose: content extraction helpers for scraped pages.
- product/module functionality: clean page text and metadata for downstream indexing/clustering.
- scope boundaries: extraction helpers only; scraping lifecycle and auth belong in sibling submodules.
- connected modules/submodules: Browser scraper, AI Chatbot/RAG, Backend URL status.
- allowed change types: extraction quality fixes, sanitization, tests.
- special operating rules: extracted page content remains untrusted data downstream.
- current stubs/placeholders: module is currently small and mostly package structure.
- irrelevant or incomplete code to remove/rework: none known.
- docs that must stay aligned: Browser Engine card.
- local validation commands/checks: `make test-browser`; add focused extraction tests before behavior changes.
