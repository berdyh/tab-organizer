# Web UI API Client Submodule Card

- purpose: synchronous facade from Streamlit pages to backend, AI, and browser service APIs.
- product/module functionality: URL/session/scrape/cluster/export calls, AI provider/settings calls, platform auth/token/company/dashboard/issue calls.
- scope boundaries: HTTP client shape only; no page rendering or service-domain logic.
- connected modules/submodules: Web UI pages, Backend Core API, AI Engine API, Browser Engine API.
- allowed change types: request/response shape fixes, auth header handling, timeout/error propagation, tests.
- special operating rules: preserve shared AI token header behavior and do not leak raw API tokens outside creation/selected UI state.
- current stubs/placeholders: direct AI/Browser calls are accepted current architecture and must be documented when changed.
- irrelevant or incomplete code to remove/rework: split platform-specific client methods only with tests.
- docs that must stay aligned: Web UI card, Backend/API cards, `docs/AI_CONFIG.md`.
- local validation commands/checks: `make test-web`; focused files `tests/unit/test_web_ui_platform.py` and `tests/unit/test_web_ui_text.py`.
