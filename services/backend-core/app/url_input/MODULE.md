# Backend URL Input Submodule Card

- purpose: URL normalization, deduplication, and URL record storage.
- product/module functionality: add URLs, detect duplicates, update scrape statuses, group/count records.
- scope boundaries: owns URL identity and status; does not own scraping, extraction, embeddings, or UI forms.
- connected modules/submodules: Backend API, sessions, Browser Engine callbacks, Web UI URL input.
- allowed change types: normalization fixes, status mapping updates, storage bug fixes, focused tests.
- special operating rules: preserve tracking-parameter stripping, duplicate behavior, and default rejection of unsafe local/private scrape targets.
- current stubs/placeholders: none known.
- irrelevant or incomplete code to remove/rework: none known.
- docs that must stay aligned: Backend Core card and `README.md` feature descriptions.
- local validation commands/checks: `make test-backend`; focused file `tests/unit/test_url_store.py`.
