# Browser Tabs Submodule Card

- purpose: attach to a user-owned Chromium/Chrome debugging endpoint and import/open tabs.
- product/module functionality: CDP endpoint validation, live tab inventory, readable tab extraction, attached-browser tab opening.
- scope boundaries: owns browser control-plane interaction only; backend persistence and AI indexing contracts belong to connected services.
- connected modules/submodules: Browser routes, Backend tab import callbacks, AI Engine indexing, Ops CLI/MCP tools.
- allowed change types: CDP attach behavior, local endpoint validation, import/open result contracts, focused tests.
- special operating rules: attach-only v1; never close the user's browser/profile; only local CDP endpoints are allowed; imported page URLs still pass scrape URL safety unless explicitly opted in later.
- current stubs/placeholders: browser launch mode, rofi/fzf, and TUI flows are isolate-for-later.
- irrelevant or incomplete code to remove/rework: no `ichrome` dependency is used in v1; Playwright CDP attach is the documented contract.
- docs that must stay aligned: Browser Engine card, Backend Core card, Ops Tooling card, README, architecture/testing docs.
- local validation commands/checks: `make test-browser`; focused file `tests/unit/test_browser_tab_harvester.py`.
