# Shared Config Module Card

- purpose: shared AI provider and model configuration.
- product/module functionality: provider catalog, default models, dimensions, provider capabilities, runtime config loading.
- scope boundaries: owns static config and loader behavior; provider execution belongs in AI Engine providers.
- connected modules/submodules: AI Engine, Ops Tooling init/check-provider, Web UI settings, docs.
- allowed change types: provider catalog updates, default model/dimension fixes, loader validation, tests.
- special operating rules: keep LLM-only providers separate from embedding-capable providers.
- current stubs/placeholders: local subscription providers may be unavailable at runtime and must remain selectable/reportable.
- irrelevant or incomplete code to remove/rework: none known.
- docs that must stay aligned: `docs/AI_CONFIG.md`, README environment variables, `.env.example`.
- local validation commands/checks: `make test-ai` and `make test-ops`.
