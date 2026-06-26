# Backend Core Module Card

- purpose: backend API and local persistence owner for tab workflows and platform accounts.
- product/module functionality: sessions, URL intake, tab import jobs, hybrid search, scrape orchestration, AI clustering proxy, exports, platform auth/B2B/maintainer APIs.
- scope boundaries: owns HTTP contracts under `/api/v1`, backend SQLite persistence, and service-to-service calls; does not own AI provider logic, browser scraping internals, or Streamlit UI state.
- connected modules/submodules: Web UI API client, AI Engine HTTP APIs, Browser Engine scrape/CDP APIs, Ops CLI/MCP tools, Test Harness integration tests.
- allowed change types: route split/refactor, persistence fixes, request/response validation, orchestration error handling, docs/cards, focused tests.
- special operating rules: preserve bearer callback auth, `BACKEND_AGENT_API_TOKEN` on agent tab APIs, Browser Engine control-token forwarding, platform session/API-token behavior, maintainer signup code checks, `/api/v1/*` response shapes, parameterized FTS queries, and underscore import compatibility.
- current stubs/placeholders: backend-to-browser scrape trigger failures are best-effort and need a documented failure contract before behavior changes; tab import cancellation/resume is not implemented.
- irrelevant or incomplete code to remove/rework: broad `app/api/routes.py` should remain a compatibility aggregator while routes move into submodule routers.
- docs that must stay aligned: `docs/MODULE_INDEX.md`, `docs/ARCHITECTURE.md`, `docs/TESTING.md`, `README.md`, submodule cards under `app/*`.
- local validation commands/checks: `make test-backend`; focused file `tests/unit/test_backend_tab_workflows.py`; then `./scripts/cli.py test --type integration` for API contract changes.
