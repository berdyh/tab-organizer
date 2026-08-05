# Shared Config Module Card

- purpose: shared AI provider and model configuration.
- product/module functionality: provider catalog, default models, dimensions, provider capabilities, runtime config loading.
- scope boundaries: owns static config and loader behavior; provider execution belongs in AI Engine providers.
- connected modules/submodules: AI Engine, Ops Tooling init/check-provider/configure-provider, Web UI settings, docs.
- allowed change types: provider catalog updates, default model/dimension fixes, loader validation, tests.
- special operating rules: keep LLM-only providers separate from embedding-capable providers. `validate_config()` (schema-level: providers/models missing required fields or referencing an unknown provider) now runs in every service's FastAPI `lifespan` (backend-core, ai-engine, browser-engine — finding 27), not just ai-engine; backend-core and browser-engine now also copy `config/` and pin `PyYAML` to support this even though they don't otherwise read this config. `ai_models.yaml`'s `ollama.base_url` (`http://localhost:11434`) is the host-process default only — containers always override via `OLLAMA_HOST` (see the comment in the file); keep that comment accurate if the override path changes.
- current stubs/placeholders: local subscription providers may be unavailable at runtime and must remain selectable/reportable.
- irrelevant or incomplete code to remove/rework: none known.
- docs that must stay aligned: `docs/AI_CONFIG.md`, README environment variables, `.env.example` (defaults to `openrouter`, matching `docker-compose.yml`/`ai_models.yaml` — finding 30).
- local validation commands/checks: `make test-ai`, `make test-backend`, `make test-browser` (all three now exercise `validate_config()` via `tests/unit/test_startup_config_validation.py`), and `make test-ops`.
