# AI Model Configuration Guide

This guide explains the centralized AI model configuration system that makes it easy to manage, update, and switch between different AI providers and models.

## Overview

The Tab Organizer uses a centralized configuration system located in `config/ai_models.yaml` that defines:
- Available AI providers (OpenRouter, Ollama, OpenAI, Anthropic, Claude Code, Codex CLI, Codex ACP, DeepSeek, Gemini)
- All supported models with their metadata
- Default configurations for each provider
- Use case recommendations

OpenRouter is the docker-compose default (one API key, many models). Ollama is the `.env.example` default for running fully offline. Claude Code and Codex CLI are LLM-only options that use local CLI subscription login state instead of API keys. `codex_acp` is an LLM-only ACP harness route through `acpx` and the Codex ACP adapter. The other providers are wired in the same registry and can be swapped at runtime.

## Configuration Structure

### Providers Section

Defines each AI provider with their capabilities and default models:

```yaml
providers:
  openrouter:
    type: cloud
    base_url: "https://openrouter.ai/api/v1"
    api_key_env: "OPENROUTER_API_KEY"
    supports:
      llm: true
      embeddings: true
    default_models:
      llm: "openai/gpt-4o-mini"
      embedding: "nvidia/llama-nemotron-embed-vl-1b-v2:free"

  ollama:
    type: local
    base_url: "http://localhost:11434"
    supports:
      llm: true
      embeddings: true
    default_models:
      llm: "llama3.2:3b"
      embedding: "nomic-embed-text"

  claude_code:
    type: local_cli
    command_env: "CLAUDE_CODE_COMMAND"
    supports:
      llm: true
      embeddings: false
    default_models:
      llm: "sonnet"
      embedding: null

  codex_cli:
    type: local_cli
    command_env: "CODEX_CLI_COMMAND"
    supports:
      llm: true
      embeddings: false
    default_models:
      llm: "codex-default"
      embedding: null

  codex_acp:
    type: local_acp
    command_env: "CODEX_ACP_COMMAND"
    supports:
      llm: true
      embeddings: false
    default_models:
      llm: "codex-acp-default"
      embedding: null
```

### Models Section

Contains detailed information about each model:

```yaml
models:
  claude-3-5-sonnet-latest:
    provider: anthropic
    type: llm
    description: "Anthropic balanced model"
    context_length: 200000
    recommended: true
```

### Defaults Section

Provides recommended configurations for different use cases:

```yaml
defaults:
  provider: "openrouter"
  use_cases:
    reasoning:
      provider: "openrouter"
      model: "openai/gpt-4o-mini"
    coding:
      provider: "claude_code"
      model: "sonnet"
    embeddings:
      provider: "openrouter"
      model: "nvidia/llama-nemotron-embed-vl-1b-v2:free"
```

## Using the Configuration System

### In Python Code

```python
from config.config_loader import get_ai_config

# Get AI configuration
ai_config = get_ai_config()

# Get provider models
models = ai_config.get_provider_models("openai", "llm")

# Get model information
model_info = ai_config.get_model_info("openai/gpt-4o-mini")

# Get use case recommendation
recommendation = ai_config.get_use_case_config("reasoning")
```

### In LLM Client

```python
from services.ai_engine.app.core.llm_client import LLMClient

# Create client with automatic configuration
client = LLMClient()

# Get available models for UI
available_models = client.get_available_models()

# Get current provider info
info = client.get_provider_info()
```

### In Initialization Script

```bash
# Interactive setup using config
./scripts/init.py --provider ollama       # local Ollama setup
./scripts/init.py --provider claude       # Anthropic Claude plus a separate embedding provider

# The script will automatically:
# 1. Load models from config
# 2. Present choices with descriptions
# 3. Configure environment variables in .env

# Or use the CLI wrapper, which copies .env.example, builds images, and pulls Ollama models:
./scripts/cli.py init --build --models
```

`scripts/init.py` accepts `ollama`, `claude`, `openrouter`, `claude_code`,
`codex_cli`, and `codex_acp`. Configure OpenAI, Gemini, or DeepSeek by editing
`.env`/runtime provider settings rather than passing them to `--provider`.

## Adding New Models

To add a new model, edit `config/ai_models.yaml`:

1. Add the model to the `models` section:
```yaml
models:
  new-model-name:
    provider: openai
    type: llm
    description: "Description of the model"
    context_length: 128000
    # ... other metadata
```

2. Optionally update provider defaults:
```yaml
providers:
  openai:
    default_models:
      llm: "new-model-name"  # Set as default
```

## Adding New Providers

To add a new AI provider:

1. Add provider configuration:
```yaml
providers:
  new-provider:
    type: cloud
    base_url: "https://api.new-provider.com"
    api_key_env: "NEW_PROVIDER_API_KEY"
    supports:
      llm: true
      embeddings: false
    default_models:
      llm: "new-provider-model"
```

2. Add model definitions:
```yaml
models:
  new-provider-model:
    provider: new-provider
    type: llm
    description: "New provider model"
    # ... metadata
```

3. Implement provider classes in `services/ai-engine/app/providers/`

## Environment Variables

The system uses these environment variables:

- `AI_PROVIDER`: Default AI provider (openrouter, ollama, openai, anthropic, claude_code, codex_cli, codex_acp, deepseek, gemini)
- `EMBEDDING_PROVIDER`: Default embedding provider
- `LLM_MODEL`: Override default LLM model
- `EMBEDDING_MODEL`: Override default embedding model
- `EMBEDDING_DIMENSIONS`: Override embedding dimensions (must match the model)
- `LLM_BASE_URL` / `EMBEDDING_BASE_URL`: Explicit runtime endpoint overrides
- `OLLAMA_HOST`: Ollama endpoint selected by init for local or Docker mode. For Ollama, the runtime uses `LLM_BASE_URL`/`EMBEDDING_BASE_URL` when explicitly set, then `OLLAMA_HOST`, then the YAML default.
- `OPENROUTER_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY` / `GOOGLE_API_KEY`: API keys for cloud providers
- `CLAUDE_CODE_COMMAND`: Claude Code CLI command, defaults to `claude`
- `CLAUDE_CODE_TIMEOUT`: Claude Code request timeout in seconds, defaults to `300`
- `CLAUDE_CODE_DISABLE_TOOLS`: Disable Claude Code tools for app LLM calls, defaults to `true`
- `CODEX_CLI_COMMAND`: Codex CLI command, defaults to `codex`
- `CODEX_CLI_TIMEOUT`: Codex CLI request timeout in seconds, defaults to `300`
- `CODEX_CLI_SANDBOX`: Codex sandbox mode, defaults to `read-only`
- `CODEX_CLI_ALLOW_UNTRUSTED_CONTEXT`: Allow `codex_cli` for scraped-content prompts. Defaults to disabled because `codex exec` is not a tool-free LLM-only mode.
- `CODEX_ACP_COMMAND`: ACPX command for Codex ACP harness routing, defaults to `acpx`
- `CODEX_ACP_TIMEOUT`: Codex ACP request timeout in seconds, defaults to `300`
- `CODEX_ACP_PERMISSION_MODE`: ACPX permission mode, defaults to `deny-all` (`approve-reads` and `approve-all` are also accepted for trusted local experiments)
- `CODEX_ACP_NON_INTERACTIVE_PERMISSIONS`: ACPX policy for non-interactive permission prompts, defaults to `fail` (`deny` is also accepted)
- `CODEX_ACP_QUEUE_TTL_SECONDS`: ACPX queue-owner TTL for each prompt turn, defaults to `0.1`
- `CODEX_ACP_SESSION_NAME`: Optional persistent ACP session name. If unset, each app LLM call creates and closes a unique ACP session to avoid cross-request context bleed.
- `AGENT_CLI_WORKDIR`: Working directory for local agent CLI calls, defaults to `/tmp/tab-organizer-agent-cli`
- `AI_ENGINE_API_TOKEN`: Bearer token required by generation, indexing, clustering, chat, search, summarization, document deletion, and provider-switch endpoints. `scripts/cli.py start` and `scripts/cli.py host-ai` generate `data/host-ai-token` automatically.
- `BACKEND_CALLBACK_TOKEN`: Bearer token required for browser-engine scrape callbacks into backend-core. Defaults operationally to the same generated local token when started through `scripts/cli.py`.
- `BACKEND_AGENT_API_TOKEN`: Bearer token required for local agent/CLI tab-management endpoints in backend-core. Defaults operationally to the same generated local token when started through `scripts/cli.py`.

### Subscription CLI Providers

`claude_code` invokes `claude -p` and uses the local Claude Code login state. `codex_cli` invokes `codex exec` and uses local Codex/ChatGPT login state. They do not require Anthropic or OpenAI API keys, but they only work where the AI engine process can execute those commands. The stock Docker image does not install these CLIs or mount their auth state; run the AI engine on the host or build a custom image for Docker-based subscription CLI routing.

`codex_cli` is not ACP mode: it is a one-shot `codex exec --ephemeral --json -` provider. For ACP semantics use `codex_acp`, which invokes `acpx` and drives the Codex harness with the ACP session lifecycle (`sessions ensure`, `prompt --file -`, and session cleanup). This is still a repo-local LLM provider, not an OpenClaw `sessions_spawn(runtime: "acp")` orchestrator.

The local CLI/ACP providers are LLM-only. Keep `EMBEDDING_PROVIDER` on `ollama`, `openrouter`, `openai`, or `gemini`.

For a host-run local subscription mode:

```bash
./scripts/init.py --provider codex_acp --subscription-embedding-provider ollama
./scripts/cli.py host-ai --provider codex_acp
./scripts/cli.py start -d --host-ai
./scripts/cli.py check-provider --provider codex_acp --generate
```

Use `--provider codex_cli` for one-shot Codex CLI or `--provider claude_code` for Claude Code print mode. `--host-ai` sets the Docker services to call `http://host.docker.internal:8090`, while the AI Engine process itself runs on the host and can access your authenticated CLI state. The CLI creates a local `data/host-ai-token` and passes it as `AI_ENGINE_API_TOKEN` so containers can call the host AI Engine without exposing unauthenticated generation and provider-switching endpoints.

## Model Metadata

Each model can include these metadata fields:

- `provider`: The AI provider
- `type`: "llm" or "embedding"
- `description`: Human-readable description
- `size`: Model size (for local models)
- `context_length`: Maximum context window
- `dimensions`: Embedding dimensions (for embedding models)
- `max_tokens`: Maximum tokens (for embedding models)
- `input_price`: Cost per 1M input tokens
- `output_price`: Cost per 1M output tokens
- `recommended`: Mark as recommended choice

## Use Case Configurations

The system provides recommended configurations for common use cases:

- `reasoning`: Complex reasoning tasks
- `coding`: Code generation and analysis
- `chat`: General conversation
- `embeddings`: Text embeddings
- `multilingual`: Multi-language support
- `cost_optimized`: Best value for money
- `high_performance`: Maximum capability

## UI Configuration

The `ui_options` section controls how models are presented in the UI:

```yaml
ui_options:
  group_by_provider: true
  show_metadata: true
  filters:
    - "type"
    - "provider"
    - "size"
    - "price"
  sort_options:
    - "name"
    - "size"
    - "price"
    - "recommended"
```

## Validation

The configuration loader includes validation:

```python
from config.config_loader import get_ai_config

ai_config = get_ai_config()
errors = ai_config.validate_config()

if errors:
    for error in errors:
        print(f"Config error: {error}")
```

## Reloading Configuration

To reload configuration without restarting:

```python
from config.config_loader import reload_config

reload_config()
```

## Best Practices

1. **Keep descriptions informative**: Include model size, capabilities, and use cases
2. **Mark recommended models**: Use `recommended: true` for best choices
3. **Maintain consistency**: Use consistent naming across providers
4. **Document pricing**: Include token prices for cloud models
5. **Test new models**: Verify models work before adding to config
6. **Version control**: Track configuration changes in git

## Troubleshooting

### Model not found
- Check if model is in `config/ai_models.yaml`
- Verify provider is correct
- Reload configuration with `reload_config()`

### Provider not supported
- Ensure provider is in `providers` section
- Check `supports` field for required capabilities
- Implement provider classes if missing

### API key errors
- Verify environment variable name matches `api_key_env`
- Check that API key is set in environment
- Ensure key has required permissions

### Embedding provider mismatch
- Some providers (like Anthropic) don't support embeddings
- System will automatically fallback to default provider
- Configure `EMBEDDING_PROVIDER` explicitly if needed
