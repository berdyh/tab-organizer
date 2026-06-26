# Browser Engine Module Card

- purpose: scrape public/authenticated pages and report extracted content/statuses.
- product/module functionality: scrape API, auth detection, auth queue, HTTP/browser scraping, backend callbacks, AI indexing handoff.
- scope boundaries: owns scraping and extraction runtime; does not own backend persistence or AI indexing internals.
- connected modules/submodules: Backend Core callbacks, AI Engine indexing, Web UI scraping page, Test Harness.
- allowed change types: scraper lifecycle fixes, auth detection, callback/error contract work, route split/refactor, tests.
- special operating rules: preserve bearer auth on control/auth endpoints, default rejection of private/local scrape targets, batch-local scraper instances, browser launch lock, Playwright shutdown in `finally`, callback token headers, and downstream error reporting.
- current stubs/placeholders: callback failures are best-effort and must remain visible in batch status; private-network scraping is an explicit local opt-in via `SCRAPE_ALLOW_PRIVATE_NETWORKS`.
- irrelevant or incomplete code to remove/rework: `app/main.py` mixes routes, background task state, callbacks, and downstream accounting; split behind compatible routes later.
- docs that must stay aligned: `docs/SCRAPER_AGENTIC_ANALYSIS.md`, `docs/ARCHITECTURE.md`, `docs/TESTING.md`.
- local validation commands/checks: `make test-browser`; integration smoke for `/scrape/single` and callback status changes.
