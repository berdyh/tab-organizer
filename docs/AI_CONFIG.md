# AI Model Configuration Guide

This guide explains the centralized AI model configuration system that makes it easy to manage, update, and switch between different AI providers and models.

## Overview

The Tab Organizer uses a centralized configuration system located in `config/ai_models.yaml` that defines:
- Available AI providers (OpenRouter, Ollama, OpenAI, Anthropic, DeepSeek, Gemini)
- All supported models with their metadata
- Default configurations for each provider
- Use case recommendations

OpenRouter is the docker-compose default (one API key, many models). Ollama is the `.env.example` default for running fully offline. The other providers are wired in the same registry and can be swapped at runtime.

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
      provider: "anthropic"
      model: "claude-3-5-sonnet-latest"
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
./scripts/init.py --provider openrouter   # or ollama, anthropic, openai, deepseek, gemini

# The script will automatically:
# 1. Load models from config
# 2. Present choices with descriptions
# 3. Configure environment variables in .env

# Or use the CLI wrapper, which copies .env.example, builds images, and pulls Ollama models:
./scripts/cli.py init --build --models
```

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

- `AI_PROVIDER`: Default AI provider (openrouter, ollama, openai, anthropic, deepseek, gemini)
- `EMBEDDING_PROVIDER`: Default embedding provider
- `LLM_MODEL`: Override default LLM model
- `EMBEDDING_MODEL`: Override default embedding model
- `EMBEDDING_DIMENSIONS`: Override embedding dimensions (must match the model)
- `OPENROUTER_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY` / `GOOGLE_API_KEY`: API keys for cloud providers

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
