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
| Backend fire-and-forget scrape trigger errors | replaced | Backend Core / Browser Engine | `POST /scrape` dispatch failures now surface as HTTP 502 and mark URLs `failed` (commit 3f1dcd8); inline `urls` are registered via `add_urls_to_session` before dispatch so callbacks persist instead of failing "URL not found" (WI0 B3). Covered by `tests/unit/test_scrape_callback.py`, `tests/unit/test_backend_callback_persistence.py`. |
| Browser callback best-effort failures | replace-with-contract | Browser Engine | Keep best-effort behavior explicit in the card and tests. Downstream `*_failed` counters now track per-document consequence, not call count (WI0 B5). |
| Inline `/scrape` urls lost before registration (WI0 B3) | replaced | Backend Core | `start_scraping` calls `session_manager.add_urls_to_session(session.id, urls)` before dispatching inline urls (dedupes against the existing store), so each scrape-complete callback finds its record and writes the url_record + FTS row. Covered by `tests/unit/test_backend_callback_persistence.py::test_inline_scrape_urls_*`. |
| Hybrid search 500 on dead semantic leg (WI0 B4) | replaced | Backend Core | `search_tabs` catches a failing `_semantic_search`; hybrid mode degrades to keyword hits plus a `degraded: "semantic_unavailable: <reason>"` field, semantic-only mode raises HTTP 502 with `{code, cause, fix}` detail. `search.semantic_degraded` logged via `log_event`. Covered by `tests/unit/test_scrape_callback.py::test_*_search_*`. |
| Batch `/index` counters under-report failure (WI0 B5) | replaced | Browser Engine | `_record_downstream_error(..., docs_affected, scope)` counts every document in a failed batched `/index` call toward `ai_index_failed` and tags the error entry `scope="batch"` + `docs_in_failed_call`; backend `/scrape/status` passes the browser-engine payload through verbatim. Covered by `tests/unit/test_browser_engine_callbacks.py::test_ai_index_batch_failure_counts_every_document`. |
| CDP attach impossible in Docker (WI0 B2) | partially-replaced | Browser Engine | Interim: `resolve_cdp_connect_url` resolves the input host (e.g. `host.docker.internal`, which Chrome's debug port rejects as a Host header) to its IP before connecting, keeping `validate_cdp_url`'s allowlist for INPUT; connect failures raise `CDPConnectionError` with a `{code, cause, fix}` message naming the two real requirements (Chrome reachable from Docker; host-side `socat` bridge). Durable fix is host-side capture in the TS migration (plan decision 24). Covered by `tests/unit/test_browser_tab_harvester.py`. |
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
| No runtime logging / request IDs (finding 29, WI0 B8) | replaced | Backend Core / AI Engine / Browser Engine | Shared `services/observability.py` gives stdlib JSON-lines logging + `RequestIDMiddleware` (accept/echo `X-Request-ID`) and outbound propagation on all inter-service httpx calls. Copied into all three service images (Dockerfiles updated; ai-engine newly copies the `services/` package). Seams logged: scrape dispatch, per-callback success/failure, index batch counts, provider switch, cluster failure, auth-queue events — never tokens/credentials/page bodies. Covered by `tests/unit/test_observability.py`. web-ui excluded. |
| Frozen security-invariant suite (SECSUITE 1.0.0) | replace-with-contract | Test Harness | Black-box probes SEC-1..39 in `tests/security/` freeze URL safety, CDP local-only, token scopes/fail-closed auth, credential isolation, agent subprocess hardening, prompt envelope, CI secret scanning (`.gitleaks.toml` + `secret-scan` job), and WI0-B6 auth fixtures. FROZEN: edits need a ledger row here + a `SECSUITE_VERSION` bump. Excludes `/platform/*` (decision 41). Challenge fixtures xfail until the B6 fix deletes the marker. |
| Never-called `validate_config()` + silent unusable-provider health (finding 27, WI0 B1) | replaced | Backend Core / AI Engine / Browser Engine / Shared Config | Each service's FastAPI `lifespan` now calls `config.config_loader.get_ai_config().validate_config()` and refuses to start (raises `RuntimeError`, logs `config.invalid`) on a malformed `ai_models.yaml`. Backend Core and Browser Engine Dockerfiles/`requirements.txt` now carry `config/` + `PyYAML` to support this (previously ai-engine only). ai-engine's lifespan additionally checks `llm_client.get_runtime_health()`; an unusable selected LLM/embedding provider logs a structured `provider.unusable_at_startup` event and reports `GET /health` as `{"status": "degraded", "runtime": {...reason...}}` — it never crashes the service (degraded-visible beats dead). Covered by `tests/unit/test_startup_config_validation.py`. |
| Config divergence: `.env.example` vs compose/yaml defaults (finding 30) | replaced | Shared Config | `.env.example` `AI_PROVIDER`/`EMBEDDING_PROVIDER` now default to `openrouter`, matching `docker-compose.yml` and `ai_models.yaml`, with a comment on opting into local Ollama. `ai_models.yaml`'s ollama `base_url: http://localhost:11434` is commented as the host-process default; containers always override via `OLLAMA_HOST` (docker-compose passthrough already existed). README updated to match. |
| Silent status synthesis on browser-engine outage (finding 28) | replaced | Backend Core / Web UI | `GET /api/v1/scrape/status/{id}` (routes.py) and `SyncAPIClient.get_scrape_status()` (web-ui client.py) no longer fabricate `not_started`/`completed` from local URL-store counts when browser-engine 404s, errors, or is unreachable — both return an explicit `{"status": "unknown", "detail": "status unavailable: ..."}` (backend still includes local counts as best-effort reference data). `scraping.py` renders the new status. Covered by `tests/unit/test_scrape_callback.py`, `tests/unit/test_web_ui_text.py`. |
| Hardcoded inter-service URL literals + boolean-only health (finding 32, 33) | replaced | Backend Core / Web UI | `routes.py` and `client.py` each resolve `AI_ENGINE_URL`/`BROWSER_ENGINE_URL`/`BACKEND_URL` through a single named `DEFAULT_*` constant per URL instead of an inline literal (web-ui builds from its own Docker context and cannot import routes.py's constants directly, so the two constant sets are kept in step by convention, not by import). `SyncAPIClient.check_health()` now passes through each service's real status string (`backend_status`/`ai_engine_status`/`browser_engine_status`, including `http_error:<code>`/`unreachable`) instead of collapsing to booleans only; `settings.py` renders it. Covered by `tests/unit/test_scrape_callback.py::test_backend_service_urls_use_runtime_env`, `tests/unit/test_web_ui_text.py`. |
| Web UI bypasses Backend Core for chat (WI0 B7) | replaced | Backend Core / Web UI | New `POST /api/v1/chat` in `routes.py` proxies to AI Engine `/chat` (mirrors the `/cluster` proxy pattern), attaching the AI Engine bearer token server-side. `web-ui/src/api/client.py`'s `chat()` now calls Backend Core, not `ai_url`, directly. Web UI's `search()`/`summarize_session()` still call ai-engine directly — a known, untouched remaining gap (not fixed in this change). Covered by `tests/unit/test_web_ui_platform.py::test_ai_client_request_shapes_include_shared_token`, `tests/unit/test_web_ui_text.py`. |
| Dead `services/web-ui/.env.example` (finding 31) | removed | Web UI | Deleted; held React/Node vars (`REACT_APP_API_URL`, `NODE_ENV`) the Streamlit app never reads. |
| No `make smoke-test` target (finding 34) | replaced | Test Harness | Added `smoke-test` target running `pytest -m smoke`; `tests/unit/test_startup_config_validation.py::test_lifespan_starts_cleanly_on_valid_config` is now marked `@pytest.mark.smoke`. |
