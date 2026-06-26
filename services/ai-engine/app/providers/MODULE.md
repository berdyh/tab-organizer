# AI Providers Submodule Card

- purpose: concrete AI provider adapters for cloud, local, and subscription CLI runtimes.
- product/module functionality: OpenRouter/OpenAI/Anthropic/DeepSeek/Gemini/Ollama adapters plus Claude Code, Codex CLI, and Codex ACP local CLI adapters.
- scope boundaries: owns adapter-specific request execution; shared provider selection belongs in AI Core.
- connected modules/submodules: AI Core, Shared Config, Ops Tooling host-AI mode.
- allowed change types: provider API fixes, command construction, environment isolation, availability checks, tests.
- special operating rules: never pass app/cloud secrets into local CLI subprocesses; keep Codex CLI read-only by default and reject untrusted scraped context unless explicitly overridden; keep Codex ACP deny-all default.
- current stubs/placeholders: stock Docker image documents CLI providers as unavailable unless host-AI/custom image supplies binaries.
- irrelevant or incomplete code to remove/rework: `agent_cli.py` is large; split by base class and provider subclasses later with no behavior change.
- docs that must stay aligned: AI Engine card, `docs/AI_CONFIG.md`, `.env.example`, `docker-compose.yml`.
- local validation commands/checks: `make test-ai`; focused file `tests/unit/test_subscription_cli_providers.py`.
