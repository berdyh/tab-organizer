# Backend API Submodule Card

- purpose: FastAPI route wiring and public backend HTTP contract.
- product/module functionality: health, session CRUD, URL input, scrape start/callbacks, clustering proxy, auth proxy, export, platform route mounting.
- scope boundaries: route definitions and dependency wiring only; domain persistence belongs in `sessions`, `url_input`, `export`, and `platform`.
- connected modules/submodules: Backend platform/session/url/export submodules, AI Engine, Browser Engine, Web UI API client.
- allowed change types: split routers, move request models beside route groups, add dependency helpers, tighten validation.
- special operating rules: preserve `/api/v1` paths and current JSON shapes; do not weaken callback bearer auth.
- current stubs/placeholders: scrape trigger errors are currently not persisted as a durable backend status.
- irrelevant or incomplete code to remove/rework: mixed route groups should be split into focused router modules without changing URLs.
- docs that must stay aligned: Backend Core card, Web UI API card, `docs/ARCHITECTURE.md`, `docs/TESTING.md`.
- local validation commands/checks: `make test-backend`; integration smoke for route changes.
