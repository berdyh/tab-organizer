# Browser Scraper Submodule Card

- purpose: page fetching, content extraction orchestration, robots compliance, and browser lifecycle.
- product/module functionality: HTTP scrape, browser scrape, authenticated scrape, batch scrape, callback-aware scraping, resource cleanup.
- scope boundaries: owns scraper behavior; request/task API state belongs in Browser Engine routes.
- connected modules/submodules: Browser auth, extraction, Backend callback API, AI indexing handoff.
- allowed change types: resource lifecycle fixes, extraction behavior, robots policy, callback accounting, focused tests.
- special operating rules: do not reintroduce shared scraper races; `_get_browser()` must remain concurrency-safe and `close()` must stop Playwright even if browser close fails; outbound targets must pass the shared scrape URL policy.
- current stubs/placeholders: callback exceptions are intentionally best-effort but must be recorded in result metadata until a stronger contract is added.
- irrelevant or incomplete code to remove/rework: `engine.py` is broad and should be split by HTTP/browser/batch/callback concerns later.
- docs that must stay aligned: Browser Engine card and `docs/SCRAPER_AGENTIC_ANALYSIS.md`.
- local validation commands/checks: `make test-browser`; focused file `tests/unit/test_browser_engine_callbacks.py`.
