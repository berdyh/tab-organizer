# Web UI Module Card

- purpose: Streamlit user interface for tab intake, scraping, clustering, chat, settings, and platform workflows.
- product/module functionality: navigation, page rendering, session state, sync API client calls, user-facing error display.
- scope boundaries: owns UI state and presentation; backend/AI/browser behavior belongs in services.
- connected modules/submodules: Backend Core, AI Engine, Browser Engine, Shared Config docs, Test Harness web UI tests.
- allowed change types: page split/refactor, API client facade changes, error display fixes, user-flow tests.
- special operating rules: preserve account-scoped state clearing, raw API token handling, provider settings behavior, and safe rendering of LLM/user content. `chat()` in `src/api/client.py` calls Backend Core (`{backend_url}/chat`), never AI Engine directly (WI0 B7) — `search()`/`summarize_session()` still call `ai_url` directly, a known, untouched remaining gap. `get_scrape_status()` must never fabricate `not_started`/`completed` from local session counts on a failed backend call — return an honest `{"status": "unknown", "detail": ...}` instead (finding 28). `check_health()` passes through each service's real status string (`backend_status`/`ai_engine_status`/`browser_engine_status`) via `_probe_health()`, not just a collapsed boolean (finding 33). `DEFAULT_BACKEND_URL`/`DEFAULT_AI_ENGINE_URL`/`DEFAULT_BROWSER_ENGINE_URL` are the single named source for those fallbacks (finding 32) — web-ui builds from its own Docker context and cannot import `routes.py`'s constants, so keep the literal values in step by convention.
- current stubs/placeholders: platform page is large and should be split by panel after tests remain green.
- irrelevant or incomplete code to remove/rework: obsolete Node/Jest helper scripts were removed; keep remaining scripts as repo-level pytest wrappers. `services/web-ui/.env.example` (dead React/Node vars Streamlit never read) was deleted (finding 31).
- docs that must stay aligned: Web UI README, root README workflow, `docs/TESTING.md`, page/API cards.
- local validation commands/checks: `make test-web`; focused files `tests/unit/test_web_ui_text.py`, `tests/unit/test_web_ui_platform.py`; browser/manual smoke for visible UI behavior changes.
