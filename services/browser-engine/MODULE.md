# Browser Engine Module Card

- purpose: scrape public/authenticated pages and attach to local browser tabs for import/open workflows.
- product/module functionality: scrape API, CDP tab import/open API, auth detection, auth queue, HTTP/browser scraping, backend callbacks, AI indexing handoff.
- scope boundaries: owns scraping and extraction runtime; does not own backend persistence or AI indexing internals.
- connected modules/submodules: Backend Core callbacks and tab APIs, AI Engine indexing, Web UI scraping page, Ops CLI/MCP tools, Test Harness.
- allowed change types: scraper lifecycle fixes, auth detection, callback/error contract work, route split/refactor, tests.
- special operating rules: preserve bearer auth on control/auth endpoints, local-only CDP attach, default rejection of private/local scrape targets, batch-local scraper instances, browser launch lock, Playwright shutdown in `finally`, callback token headers, and downstream error reporting.
- current stubs/placeholders: callback failures are best-effort and must remain visible in batch status; private-network scraping is an explicit local opt-in via `SCRAPE_ALLOW_PRIVATE_NETWORKS`; managed browser launch/profile lifecycle is deferred.
- irrelevant or incomplete code to remove/rework: `app/main.py` mixes routes, background task state, callbacks, and downstream accounting; split behind compatible routes later.
- docs that must stay aligned: `docs/SCRAPER_AGENTIC_ANALYSIS.md`, `docs/ARCHITECTURE.md`, `docs/TESTING.md`.
- local validation commands/checks: `make test-browser`; focused file `tests/unit/test_browser_tab_harvester.py`; integration smoke for `/scrape/single` and callback status changes.
