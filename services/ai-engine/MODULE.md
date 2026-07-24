# AI Engine Module Card

- purpose: AI runtime for embeddings, provider switching, clustering, RAG, indexing, and chat/search APIs.
- product/module functionality: provider health/config, generation, embeddings, LanceDB persistence, chunked document indexing/search, cluster labeling.
- scope boundaries: owns AI behavior and vector data; does not own backend session persistence, scraping internals, or Streamlit state.
- connected modules/submodules: Backend Core, Browser Engine, Shared Config, Ops Tooling host-AI mode, Test Harness.
- allowed change types: provider adapter updates, auth/config hardening, RAG/clustering fixes, route split/refactor, focused tests.
- special operating rules: preserve bearer auth, `AI_ENGINE_ALLOW_UNAUTHENTICATED` fail-closed default, provider-state locking, embedding dimension checks, and local CLI guardrails. Structured logging goes through shared `services/observability.py`: `RequestIDMiddleware` binds/echoes `X-Request-ID`; log provider switches, index counts, and cluster failures via `log_event`. The Dockerfile now copies `services/__init__.py` + `services/observability.py` (previously it copied only `app/` and `config/`) — keep that in sync when the shared module changes. Never log tokens, credentials, or document bodies (counts/lengths only).
- current stubs/placeholders: unavailable local subscription CLI providers should report unhealthy rather than silently falling back.
- irrelevant or incomplete code to remove/rework: `app/main.py` is a broad route/runtime file and should be split behind compatible routers.
- docs that must stay aligned: `docs/AI_CONFIG.md`, `docs/ARCHITECTURE.md`, `README.md`, provider and RAG submodule cards.
- local validation commands/checks: `make test-ai`; focused file `tests/unit/test_rag_lancedb_persistence.py`; then integration smoke for `/health`, `/providers`, indexing, and chat/search changes.
