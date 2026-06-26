# Module Index

This index is the top-level context for module/submodule work. Load this file
first, then the relevant local `MODULE.md`, then only the source/tests linked by
that card.

## Module Tree

| Module | Local card | Purpose | Main connections |
| --- | --- | --- | --- |
| Backend Core | `services/backend-core/MODULE.md` | API gateway, sessions, URL state, exports, platform accounts, orchestration | Web UI, AI Engine, Browser Engine |
| AI Engine | `services/ai-engine/MODULE.md` | Provider runtime, embeddings, RAG, clustering, local CLI providers | Backend Core, Browser Engine, shared config, LanceDB/Ollama |
| Browser Engine | `services/browser-engine/MODULE.md` | HTTP/browser scraping, auth detection, callbacks, downstream indexing | Backend Core, AI Engine |
| Web UI | `services/web-ui/MODULE.md` | Streamlit app, page state, user workflows, API facade | Backend Core, AI Engine, Browser Engine |
| Ops Tooling | `scripts/MODULE.md` | CLI lifecycle, host AI mode, init, Docker/test orchestration | Docker Compose, tests, all services |
| Shared Config | `config/MODULE.md` | AI model/provider catalog and config loader | AI Engine, CLI/init, Web UI settings |
| Test Harness | `tests/MODULE.md` | Unit, integration, E2E, load, and contract checks | All modules |

## Current Boundary Rules

- Preserve the four runtime services and Docker ports; this rework is not a
  service split or Git submodule conversion.
- Keep underscore compatibility packages such as `services/ai_engine` and
  `services/backend_core`; tests and CLI imports depend on them.
- Make behavior changes only after module-local tests exist for the affected
  boundary.
- Keep service contracts stable unless the relevant module card and connected
  cards are updated in the same change.
- Treat local CLI provider execution, bearer tokens, maintainer signup, API
  tokens, scrape callbacks, and browser lifecycle as security-sensitive areas.

## Validation Ladder

Use the smallest relevant check first:

1. Module target such as `make test-backend`, `make test-ai`,
   `make test-browser`, `make test-web`, or `make test-ops`.
2. Cross-service smoke: `./scripts/cli.py test --type integration`.
3. Full gate: `./scripts/cli.py test --type all`.
4. Browser/manual smoke when UI behavior or scraping behavior changed.

## Stub And Cleanup Ledger

| Item | Classification | Owner module | Decision |
| --- | --- | --- | --- |
| Legacy skipped gateway E2E file | remove | Test Harness | Removed; current direct-service coverage lives in `tests/e2e/test_workflow.py`. |
| Old Web UI Node/Jest scripts | remove | Web UI | Removed obsolete Docker/debug scripts; retained wrappers call repo-level pytest targets. |
| Backend fire-and-forget scrape trigger errors | replace-with-contract | Backend Core / Browser Engine | Document and test the status/error contract before behavioral changes. |
| Browser callback best-effort failures | replace-with-contract | Browser Engine | Keep best-effort behavior explicit in the card and tests. |
| Backend platform store monolith | isolate-for-later | Backend Core platform | Split by auth/tokens/companies/dashboard/issues in a behavior-preserving pass. |
| Web platform page monolith | isolate-for-later | Web UI pages | Split render sections after page-level tests are green. |
