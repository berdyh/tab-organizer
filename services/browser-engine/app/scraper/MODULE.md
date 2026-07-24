# Browser Scraper Submodule Card

- purpose: page fetching, content extraction orchestration, robots compliance, and browser lifecycle.
- product/module functionality: HTTP scrape, browser scrape, authenticated scrape, batch scrape, callback-aware scraping, resource cleanup.
- scope boundaries: owns scraper behavior; request/task API state belongs in Browser Engine routes.
- connected modules/submodules: Browser auth, extraction, Backend ingest API.
- allowed change types: resource lifecycle fixes, extraction behavior, robots policy, callback accounting, focused tests.
- special operating rules: do not reintroduce shared scraper races; `_get_browser()` must remain concurrency-safe and `close()` must stop Playwright even if browser close fails; outbound targets must pass the shared scrape URL policy. `ScrapeResult.auth_used` is set `True` ONLY in the three credential-store branches (`_scrape_basic_auth`, `_scrape_cookie_auth`, `_scrape_form_auth`); the unknown-auth-type fallback to plain httpx stays `False`. It rides capture → backend ledger → `/index` document metadata (decision-37 hook). KNOWN GAP: ambient auth (cookies already in a Playwright/browser context, session reuse, auth-wall pages queued as `auth_required`) is invisible today, so `auth_used=False` there is an under-report — do not treat `False` as proof no credentials were involved.
- current stubs/placeholders: callback exceptions are intentionally best-effort but must be recorded in result metadata until a stronger contract is added; `auth_used` under-reports ambient/session auth (see special rules).
- irrelevant or incomplete code to remove/rework: `engine.py` is broad and should be split by HTTP/browser/batch/callback concerns later.
- docs that must stay aligned: Browser Engine card and `docs/archive/2026-07-24-SCRAPER_AGENTIC_ANALYSIS.md`.
- local validation commands/checks: `make test-browser`; focused file `tests/unit/test_browser_engine_callbacks.py`.
