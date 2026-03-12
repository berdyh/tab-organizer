# AI Model Configuration Guide

This guide explains the centralized AI model configuration system that makes it easy to manage, update, and switch between different AI providers and models.

## Overview

The Tab Organizer uses a centralized configuration system located in `config/ai_models.yaml` that defines:
- Available AI providers (Ollama, OpenAI, Anthropic, DeepSeek, Gemini)
- All supported models with their metadata
- Default configurations for each provider
- Use case recommendations

## Configuration Structure

### Providers Section

Defines each AI provider with their capabilities and default models:

```yaml
providers:
  openai:
    type: cloud
    base_url: "https://api.openai.com/v1"
    api_key_env: "OPENAI_API_KEY"
    supports:
      llm: true
      embeddings: true
    default_models:
      llm: "gpt-5.1"
      embedding: "text-embedding-3-small"
```

### Models Section

Contains detailed information about each model:

```yaml
models:
  claude-sonnet-4-5-20250929:
    provider: anthropic
    type: llm
    description: "Balanced Claude 3.5 Sonnet"
    context_length: 200000
    input_price: "3/1M tokens"
    output_price: "15/1M tokens"
    recommended: true
```

### Defaults Section

Provides recommended configurations for different use cases:

```yaml
defaults:
  use_cases:
    reasoning:
      provider: "anthropic"
      model: Model name (e.g., 'claude-sonnet-4-5-20250929')
    coding:
      provider: "deepseek"
      model: "deepseek-coder"
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
model_info = ai_config.get_model_info("gpt-5.1")

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
./scripts/init.py --provider claude

# The script will automatically:
# 1. Load models from config
# 2. Present choices with descriptions
# 3. Configure environment variables
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

- `AI_PROVIDER`: Default AI provider (ollama, openai, anthropic, deepseek, gemini)
- `EMBEDDING_PROVIDER`: Default embedding provider
- `LLM_MODEL`: Override default LLM model
- `EMBEDDING_MODEL`: Override default embedding model
- `EMBEDDING_DIMENSIONS`: Override embedding dimensions
- `{PROVIDER}_API_KEY`: API key for cloud providers

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
