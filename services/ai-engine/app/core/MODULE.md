# AI Core Submodule Card

- purpose: provider-neutral LLM client and runtime provider selection logic.
- product/module functionality: provider registration, capability checks, runtime config, provider switching, default model selection.
- scope boundaries: owns provider abstraction; concrete provider subprocess or HTTP implementation belongs in `providers`.
- connected modules/submodules: AI routes, providers, Shared Config, Web UI settings.
- allowed change types: capability validation, provider registry updates, config parsing fixes, tests.
- special operating rules: do not allow unsupported capability/provider combinations; preserve provider availability semantics.
- current stubs/placeholders: unavailable CLI providers are valid configured options but must be marked unavailable.
- irrelevant or incomplete code to remove/rework: route-specific state should move out during AI route split.
- docs that must stay aligned: AI Engine card and `docs/AI_CONFIG.md`.
- local validation commands/checks: `make test-ai`; focused files `tests/unit/test_ai_provider_switch.py` and `tests/unit/test_subscription_cli_providers.py`.
