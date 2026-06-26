# Web UI Pages Submodule Card

- purpose: Streamlit page composition for user workflows.
- product/module functionality: URL input, scraping progress, clustering, chatbot, settings, and platform account/B2B/maintainer panels.
- scope boundaries: rendering and page-local state only; API requests go through `src/api/client.py`.
- connected modules/submodules: Web UI API client, Backend Core, AI Engine, Browser Engine.
- allowed change types: page splits, form/state fixes, copy updates, safe rendering fixes, tests.
- special operating rules: do not render untrusted LLM labels as unsafe HTML; keep platform sign-out clearing account-scoped state.
- current stubs/placeholders: `platform.py` is intentionally broad until split by panel.
- irrelevant or incomplete code to remove/rework: legacy/hardcoded local test helper scripts belong outside pages and can be removed.
- docs that must stay aligned: Web UI card and root README web workflow.
- local validation commands/checks: `make test-web`; browser/manual smoke for visible behavior changes.
