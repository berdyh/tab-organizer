# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo layout note

This worktree (`main/`) is a checkout inside a bare-repo container. Do repo-level Git operations (`git worktree ...`, `git fetch`) in the parent directory; do all code work here. See the container's `CLAUDE.md`/`AGENTS.md`.

## Module-card workflow (read this first)

This repo encodes its own boundaries in markdown "module cards" and expects agents to load them in order:

1. `docs/MODULE_INDEX.md` — module tree, boundary rules, validation ladder, stub/cleanup ledger.
2. The local `MODULE.md` next to the code you're touching (one per service, plus one per submodule such as `services/ai-engine/app/providers/MODULE.md`).
3. Only then the source/tests that card points at.

Each card lists `special operating rules` (invariants you must preserve), `current stubs/placeholders`, `docs that must stay aligned`, and `local validation commands`. When you change a boundary, update that card in the same commit.

## Common commands

Everything runs in Docker; `scripts/cli.py` is the wrapper CI also uses.

```bash
./scripts/cli.py init --build --models     # copy .env.example, build images, pull Ollama models
./scripts/cli.py start -d                  # start default stack (detached)
./scripts/cli.py start -d --dev            # dev profile
./scripts/cli.py start -d --host-ai        # route containers at a host-run AI Engine
./scripts/cli.py stop / status / logs -f <svc> / restart / clean
./scripts/cli.py host-ai --provider claude_code   # run AI Engine on the host for subscription CLI providers
./scripts/cli.py check-provider --provider codex_acp --generate
```

`start` / `host-ai` generate the local bearer tokens (`AI_ENGINE_API_TOKEN`, `BACKEND_CALLBACK_TOKEN`, `BACKEND_AGENT_API_TOKEN`) — services fail closed without them, so prefer these over raw `docker compose up`.

### Tests

Use the smallest check that covers the change (the "validation ladder" in `docs/MODULE_INDEX.md`):

```bash
make test-backend | test-ai | test-browser | test-web | test-ops   # focused unit subsets per module
./scripts/cli.py test --type unit         # full unit suite w/ coverage
./scripts/cli.py test --type integration  # boots the default stack first
./scripts/cli.py test --type e2e
./scripts/cli.py test --type all
```

Single test / `-k` filter (unit tests only need the test container):

```bash
docker compose --profile test-unit run --rm test-unit \
  pytest tests/unit/test_backend_tab_workflows.py -k keyword_search -q
```

`pytest.ini` sets `testpaths = services tests`, `--strict-markers`, a 300s timeout, and coverage reporting; markers include `unit`, `integration`, `e2e`, `requires_ollama`, `requires_lancedb`. Artifacts land in `coverage/` and `test-results/`. Load tests (`tests/load/locustfile.py`) are manual — there is no `--type performance`.

### Quality gates

```bash
make lint            # flake8 (E9,F63,F7,F82) + pylint --exit-zero
make format          # black + isort (line-length 88, py312)
make format-check    # what CI enforces
make type-check      # mypy --ignore-missing-imports
make security        # bandit + safety
make quality         # all of the above
```

These run in throwaway `python:3.12-slim` containers, so no local toolchain is needed.

## Architecture

Four FastAPI/Streamlit services on a shared Docker network, plus Ollama. Backend Core is the only orchestrator — the UI and agent tooling talk to it, and it fans out to AI Engine and Browser Engine.

| Service | Port | Owns |
| --- | --- | --- |
| `web-ui` (Streamlit) | 8089 | Pages + session state only; no domain logic |
| `backend-core` | 8080 | `/api/v1/*` contracts, SQLite persistence, orchestration, platform/B2B accounts |
| `ai-engine` | 8090 | Providers, embeddings, LanceDB, RAG, UMAP+HDBSCAN clustering |
| `browser-engine` | 8083 | HTTP/Playwright scraping, auth detection, CDP tab attach |

Two separate persistence stores, each owned by exactly one service:
- **SQLite** (`BACKEND_DB_PATH`, volume `backend-data`) — sessions, URL records, tab-import jobs, clusters, callback metadata, platform data, and an **FTS5 table** (`tab_search_fts`) for keyword search. Managed in `services/backend-core/app/sessions/manager.py`, which also has an in-memory fallback path that must stay behaviorally equivalent.
- **LanceDB** (`VECTOR_DB_PATH`, volume `lancedb-data`) — embedded *inside* the ai-engine container. There is no separate vector-DB service.

### Key cross-service flows

- **Scrape**: UI → `POST /api/v1/scrape` (backend) → `POST /scrape` (browser-engine, background task) → per-result it calls back to `POST /api/v1/callback/scrape-complete` (bearer `BACKEND_CALLBACK_TOKEN`) **and** pushes content to ai-engine `POST /index`. Callback/index failures are best-effort but are counted and surfaced in batch status — keep them visible.
- **Search**: `POST /api/v1/search` merges SQLite FTS keyword hits with ai-engine `/search` semantic hits in backend-core.
- **Cluster / chat / export**: backend proxies to ai-engine `/cluster`, `/chat`, `/summarize`; export rendering lives in `services/backend-core/app/export/`.
- **Tab import**: `POST /api/v1/tabs/import` (bearer `BACKEND_AGENT_API_TOKEN`) → browser-engine `/tabs/import` → Playwright CDP **attach** to a user-started local Chrome.

### AI provider system

`services/ai-engine/app/providers/` holds one adapter per provider (ollama, openai, anthropic, deepseek, gemini, openrouter, plus `agent_cli.py` for `claude_code` / `codex_cli` / `codex_acp`). The static catalog of providers, default models, dimensions, and capabilities lives in `config/ai_models.yaml` + `config/models.json`, read through `config/config_loader.py`. Providers can be hot-swapped via `POST /providers/switch`.

LLM-only providers (`claude_code`, `codex_cli`, `codex_acp`) shell out to locally-authenticated CLIs; the stock Docker image ships none of those binaries, so they require `./scripts/cli.py host-ai` + `start --host-ai`. `EMBEDDING_PROVIDER` must stay on an embedding-capable provider, and `EMBEDDING_DIMENSIONS` must match the embedding model or ai-engine refuses to write to the LanceDB table.

## Conventions and invariants

- **Underscore compatibility packages**: `services/ai_engine/`, `services/backend_core/`, `services/browser_engine/`, `services/web_ui/` are `__init__.py` shims that repoint `__path__` at the hyphenated directories. Tests and CLI import via `services.backend_core.app...` — do not delete them, and add one if you add a hyphenated service.
- **Docker build context**: backend-core, ai-engine, and browser-engine build from the *repo root* (they copy `services/url_safety.py` / `config/`); web-ui builds from its own directory. Adding a shared module means updating those Dockerfiles.
- **Outbound URL safety**: all scrape targets go through `services/url_safety.py` (scheme allowlist, private/loopback rejection, DNS-rebinding-safe resolution). `SCRAPE_ALLOW_PRIVATE_NETWORKS=true` is the only escape hatch and defaults off.
- **Auth is fail-closed**: ai-engine endpoints (except `/health`) require `AI_ENGINE_API_TOKEN`; browser-engine scrape/auth control endpoints require `BROWSER_ENGINE_API_TOKEN` (with callback/AI token fallback); backend agent tab APIs require `BACKEND_AGENT_API_TOKEN`. Never route agent browser-control calls through unauthenticated paths, and never print generated tokens from `scripts/`.
- **CDP is attach-only and local-only** — never launch or close the user's browser/profile; reject non-local CDP endpoints.
- **Known monoliths** (`backend-core/app/api/routes.py`, both `app/main.py` files, `platform/store.py`, `web-ui/src/pages/platform.py`) are slated for behavior-preserving splits behind compatible routes — keep `routes.py` working as a compatibility aggregator.
- Conventional commits (`feat(scope): ...`, `fix(scope): ...`).

## Docs map

`docs/ARCHITECTURE.md` (system design), `docs/AI_CONFIG.md` (providers/models), `docs/TESTING.md` (authoritative test guide), `docs/DEVELOPMENT.md` (workflow), `docs/MANUAL_SETUP.md`, `docs/REQUIREMENTS.md`. Module cards list which of these must be updated alongside a given change.
