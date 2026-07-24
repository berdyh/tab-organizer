# Browser Auth Submodule Card

- purpose: detect authentication requirements and manage credential requests.
- product/module functionality: URL/response/HTML auth detection, pending auth queue, credential handoff.
- scope boundaries: owns auth classification and queue state; does not own credential storage UX or backend sessions.
- connected modules/submodules: Browser scraper, Browser routes, Web UI scraping page.
- allowed change types: detection heuristic fixes, queue behavior, credential validation, tests.
- special operating rules: do not log credentials; callback errors are best-effort unless the contract changes.
- current stubs/placeholders: auth callback failure handling is intentionally tolerant.
- irrelevant or incomplete code to remove/rework: none known.
- docs that must stay aligned: Browser Engine card and scraping docs.
- local validation commands/checks: `make test-browser`; focused file `tests/unit/test_auth_detector.py`.
