> ARCHIVED 2026-07-24 — superseded by the reviewed architecture plan (see docs/ARCHITECTURE_PLAN.md). Retained for reference.

# Repository Improvement Review (Post-pass)

This review is grounded in the current code. Each item names the concrete
location(s) and the specific change to make. Unless noted otherwise, all
operational state is **in-memory** and the only persistent store is the embedded
**LanceDB** vector database (`services/ai-engine/app/chatbot/rag.py`, backed by the
`lancedb-data` Docker volume) — there is no Qdrant, Redis, or Postgres in this
system.

## High Priority

1. **Persistence layer for operational state**
   - All session, URL, scrape-task, and credential state lives in module-level
     Python dicts and is lost on restart:
     - Sessions & URLs: `SessionManager` (`services/backend-core/app/sessions/manager.py:25-30`),
       instantiated as a module global in `services/backend-core/app/api/routes.py`;
       URL/dedup store in `services/backend-core/app/url_input/store.py:52-58`.
     - Scrape task state: `scraping_tasks: dict` in `services/browser-engine/app/main.py`.
     - Auth/credential state: `CredentialStore._credentials` /`AuthQueue`
       (`services/browser-engine/app/auth/queue.py`).
   - Move session/task/auth state to a persistent backend (e.g. SQLite for
     local-first, or Postgres). LanceDB already persists vectors, so scope this to
     the operational stores only.

2. **Configuration consistency + startup validation**
   - Config is read ad-hoc via `os.getenv(...)` scattered across services (no shared
     settings module), with a separate YAML loader for model metadata
     (`config/config_loader.py`, `config/ai_models.yaml`). Known divergences:
     - Default provider mismatch: `.env.example` ships `AI_PROVIDER=ollama` while the
       code, `ai_models.yaml`, and `docker-compose.yml` default to `openrouter`.
     - Ollama host uses `OLLAMA_HOST` while other providers use
       `LLM_BASE_URL`/`EMBEDDING_BASE_URL`; `ai_models.yaml` points ollama at
       `localhost:11434` while compose/`.env.example` use `ollama:11434`.
     - Leftover React-style `services/web-ui/.env.example` (`REACT_APP_API_URL`,
       `NODE_ENV`) that the Streamlit app never reads (it uses `BACKEND_URL`,
       `AI_ENGINE_URL`, `BROWSER_ENGINE_URL`, `UI_API_TIMEOUT`).
   - `validate_config()` **already exists** (`config/config_loader.py:233`) but is
     never called. Wire it into each FastAPI startup/lifespan hook so a mismatched
     `EMBEDDING_DIMENSIONS`, missing API key, or bad provider fails fast at boot
     instead of at first request.

3. **Contract-driven APIs**
   - Add shared OpenAPI/JSON schema checks between web-ui, backend-core,
     browser-engine, and ai-engine to prevent drift in fields like scrape-status
     payloads.
   - Related config smell to fix here: backend-core mixes env-var-derived URLs with
     **hardcoded** `http://browser-engine:8083` and `http://ai-engine:8090` in
     `services/backend-core/app/api/routes.py`. Route all inter-service calls through
     the same env-configured base URLs.

4. **Error propagation**
   - Several cross-service calls swallow exceptions and hide failures:
     - `routes.py` scrape trigger prints and drops the error after already returning
       `{"status": "started"}`; `get_pending_auth()` returns `{"pending": []}` on any
       exception, masking a backend outage as "no pending auth."
     - `services/browser-engine/app/main.py` wraps the `/callback/scrape-complete`
       POST and the ai-engine `/index` POST in `except Exception: pass`, so scrape
       results and RAG indexing can fail silently.
     - `services/browser-engine/app/auth/queue.py` swallows decrypt and callback
       errors.
   - Also flag the **silent status synthesis**: when browser-engine is down/404s,
     both `services/web-ui/src/api/client.py` and `routes.py` fabricate a status
     payload from local counts, so an outage looks like a valid
     "not_started"/"completed".
   - Return structured error envelopes (code + message, not bare `detail=str(e)`) and
     surface them in the UI, which today only renders raw
     `except Exception as e: st.error(...)` strings.

## Medium Priority

1. **Test coverage expansion**
   - Current integration tests (`tests/integration/test_api.py`) cover health and
     backend CRUD only. Add integration tests for the untested paths:
     - Provider switching (`POST /providers/switch` — only `GET /providers` is tested).
     - Scrape-status proxy and its 404 fallback logic.
     - Auth credential flow (`POST /auth/credentials` encrypt/retrieve — only
       `GET /auth/pending` is tested).
     - RAG lifecycle (`/index`, `/chat`, `/search`) and cross-service callbacks.

2. **Observability**
   - There is essentially **no runtime logging** anywhere (a repo-wide search finds a
     single `print`), no request/correlation IDs, and no metrics.
   - Add a structured-logging baseline, propagate an `X-Request-ID` across all
     internal HTTP calls, and export scrape success/error counters and latency
     metrics (the counts already tracked in `browser-engine` are only used for the
     status endpoint today).

3. **Security hardening**
   - Credential encryption: when `CREDENTIAL_ENCRYPTION_KEY` is unset,
     `CredentialStore` (`services/browser-engine/app/auth/queue.py`) generates a
     **random ephemeral** Fernet key — credentials become undecryptable after a
     restart, and there is no production guard. `.env.example` and
     `docker-compose.yml` ship the key blank. Enforce a persistent, non-empty key in
     production and fail fast when it is missing.
   - Secret scanning: CI (`.github/workflows/ci-cd.yml`) runs `bandit` + `safety`
     only. Add a secret-scanning step (gitleaks/trufflehog) and `.env` guidance.

4. **UI reliability**
   - `check_health()` (`services/web-ui/src/api/client.py`) swallows all three service
     checks and returns booleans, with no distinction between "service down", "bad
     request", and "server error".
   - Add user-facing diagnostics, partial-availability indicators, and retry hints
     when backend services are unavailable.

## Low Priority

1. **Developer experience**
   - Add a `make smoke-test` target for a quick local validation loop. It does not
     exist today (the closest is `make health`), even though a `smoke` pytest marker
     is already defined.

2. **Code organization**
   - Consolidate API client logic and add typed response models for the Streamlit
     pages.
   - Clean up the duplicate service directories: the underscored
     `services/{ai_engine,backend_core,browser_engine,web_ui}` dirs are import shims
     pointing at the real hyphenated dirs — consolidate or document them.

## Suggested Next Sprint Scope

- Persistent state layer (sessions + scrape tasks + auth credentials).
- Observability baseline (structured logs + request IDs + scrape metrics).
- Integration test suite for service-to-service contracts (provider switch, scrape
  status, auth flow, RAG lifecycle).
- Startup preflight checks by wiring the existing `validate_config()` into each
  service's startup hook (provider + embedding-dimension + vector-DB config).
