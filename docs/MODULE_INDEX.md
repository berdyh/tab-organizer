# Module Index

This index is the top-level context for module/submodule work. Load this file
first, then the relevant local `MODULE.md`, then only the source/tests linked by
that card.

## Module Tree

| Module | Local card | Purpose | Main connections |
| --- | --- | --- | --- |
| Backend Core | `services/backend-core/MODULE.md` | API gateway, sessions, URL/tab state, agent tab APIs, exports, platform accounts, orchestration | Web UI, AI Engine, Browser Engine, Ops Tooling |
| AI Engine | `services/ai-engine/MODULE.md` | Provider runtime, embeddings, chunked RAG, clustering, local CLI providers | Backend Core, Browser Engine, shared config, LanceDB/Ollama |
| Browser Engine | `services/browser-engine/MODULE.md` | HTTP/browser scraping, CDP tab harvesting, auth detection, callbacks, downstream indexing | Backend Core, AI Engine, Ops Tooling |
| Web UI | `services/web-ui/MODULE.md` | Streamlit app, page state, user workflows, API facade | Backend Core, AI Engine, Browser Engine |
| Ops Tooling | `scripts/MODULE.md` | CLI lifecycle, host AI mode, tab CLI/MCP wrappers, init, Docker/test orchestration | Docker Compose, tests, all services |
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
- Protect Backend Core agent tab APIs with `BACKEND_AGENT_API_TOKEN`; do not
  route agent browser-control calls through unauthenticated paths.
- Keep CDP tab harvesting attach-only and local-endpoint-only until a browser
  lifecycle module explicitly owns profile launch/cleanup behavior.

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
| Concept-diagram `ichrome` tab harvester | replace-with-contract | Browser Engine tabs | V1 contract is Playwright CDP attach through `CDPTabHarvester`; add `ichrome` only behind the tabs card and tests. |
| Browser launch/profile management | isolate-for-later | Browser Engine tabs | Current implementation attaches to a user-started local Chrome/Chromium and never closes the profile. |
| Textual TUI and rofi/fzf quick-pick | isolate-for-later | Ops Tooling | CLI/MCP wrappers are the active agent surface; richer local navigation UIs should stay behind `scripts/` boundaries. |
| Full MCP SDK server package | replace-with-contract | Ops Tooling | Current contract is Python-callable wrappers plus JSON-lines stdio adapter; promote to SDK server only with MCP validation. |
| BERTopic topic modeling | isolate-for-later | AI Engine clustering | Current cluster engine remains UMAP + HDBSCAN + LLM labels; add topic modeling behind the clustering submodule. |
| Throwaway credential-store key | replaced | Browser Engine auth | Fail-closed: OS keyring or `CREDENTIAL_ENCRYPTION_KEY`, else `store()` raises `CredentialStoreError`; never invents a key. Covered by `tests/unit/test_credential_store.py`. |
| Unbounded cluster recursion | replaced | AI Engine clustering | `cluster()` bounded by `max_subcluster_depth` + no-progress guard. Covered by `tests/unit/test_clustering.py`. |
| Small-N clustering 500 (WI0 B9) | replaced | AI Engine clustering | UMAP params keep `n_components + 1 < n_samples`; below `min_cluster_corpus` a single "All Tabs" cluster is returned. Covered by `tests/unit/test_clustering.py`. |
| Frozen security-invariant suite (SECSUITE 1.0.0) | replace-with-contract | Test Harness | Black-box probes SEC-1..39 in `tests/security/` freeze URL safety, CDP local-only, token scopes/fail-closed auth, credential isolation, agent subprocess hardening, prompt envelope, CI secret scanning (`.gitleaks.toml` + `secret-scan` job), and WI0-B6 auth fixtures. FROZEN: edits need a ledger row here + a `SECSUITE_VERSION` bump. Excludes `/platform/*` (decision 41). Challenge fixtures xfail until the B6 fix deletes the marker. |
