# Web UI Module Card

- purpose: Streamlit user interface for tab intake, scraping, clustering, chat, settings, and platform workflows.
- product/module functionality: navigation, page rendering, session state, sync API client calls, user-facing error display.
- scope boundaries: owns UI state and presentation; backend/AI/browser behavior belongs in services.
- connected modules/submodules: Backend Core, AI Engine, Browser Engine, Shared Config docs, Test Harness web UI tests.
- allowed change types: page split/refactor, API client facade changes, error display fixes, user-flow tests.
- special operating rules: preserve account-scoped state clearing, raw API token handling, provider settings behavior, and safe rendering of LLM/user content.
- current stubs/placeholders: platform page is large and should be split by panel after tests remain green.
- irrelevant or incomplete code to remove/rework: obsolete Node/Jest helper scripts were removed; keep remaining scripts as repo-level pytest wrappers.
- docs that must stay aligned: Web UI README, root README workflow, `docs/TESTING.md`, page/API cards.
- local validation commands/checks: `make test-web`; browser/manual smoke for visible UI behavior changes.
