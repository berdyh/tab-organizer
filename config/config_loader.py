"""Configuration loader for AI models and providers."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import yaml

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

# How a `supports:` capability claim was established. Only 'called' means the
# thing serving the capability was actually invoked; everything else is
# unverified. See `supports_evidence:` in config/ai_models.yaml.
CAPABILITY_EVIDENCE_CLASSES = frozenset({"called", "listing", "none"})


class AIModelConfig:
    """Configuration manager for AI models and providers."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_file: Path to config file. Defaults to config/ai_models.yaml
        """
        self.config_file = Path(config_file) if config_file else CONFIG_DIR / "ai_models.yaml"
        self._config: Optional[Dict] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_file, 'r') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()
    
    @property
    def config(self) -> Dict:
        """Get the full configuration dictionary."""
        if self._config is None:
            self._load_config()
        return self._config
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            
        Returns:
            Provider configuration dictionary
        """
        providers = self.config.get('providers', {})
        if provider not in providers:
            raise ValueError(f"Unknown provider: {provider}")
        return providers[provider]
    
    def get_model_config(self, model: str) -> Dict[str, Any]:
        """Get configuration for a specific model.
        
        Args:
            model: Model name (e.g., 'claude-3-5-sonnet-20241022')
            
        Returns:
            Model configuration dictionary
        """
        models = self.config.get('models', {})
        if model not in models:
            raise ValueError(f"Unknown model: {model}")
        return models[model]
    
    def get_provider_models(self, provider: str, model_type: Optional[str] = None) -> List[str]:
        """Get all models for a provider, optionally filtered by type.
        
        Args:
            provider: Provider name
            model_type: Filter by 'llm' or 'embedding'
            
        Returns:
            List of model names
        """
        models = self.config.get('models', {})
        provider_models = []
        
        for model_name, model_config in models.items():
            if model_config.get('provider') == provider:
                if model_type is None or model_config.get('type') == model_type:
                    provider_models.append(model_name)
        
        return provider_models
    
    def get_default_model(self, provider: str, model_type: str) -> Optional[str]:
        """Get the default model for a provider and type.
        
        Args:
            provider: Provider name
            model_type: 'llm' or 'embedding'
            
        Returns:
            Default model name or None if not found
        """
        provider_config = self.get_provider_config(provider)
        default_models = provider_config.get('default_models', {})
        return default_models.get(model_type)
    
    def get_use_case_config(self, use_case: str) -> Dict[str, str]:
        """Get recommended configuration for a specific use case.
        
        Args:
            use_case: Use case name (e.g., 'reasoning', 'coding')
            
        Returns:
            Dictionary with 'provider' and 'model' keys
        """
        use_cases = self.config.get('defaults', {}).get('use_cases', {})
        if use_case not in use_cases:
            raise ValueError(f"Unknown use case: {use_case}")
        return use_cases[use_case]
    
    def get_all_providers(self) -> List[str]:
        """Get list of all configured providers."""
        return list(self.config.get('providers', {}).keys())
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get detailed information about a model.
        
        Args:
            model: Model name
            
        Returns:
            Model information including provider config
        """
        model_config = self.get_model_config(model)
        provider_name = model_config.get('provider')
        provider_config = self.get_provider_config(provider_name)
        
        # Merge model and provider information
        info = {
            'name': model,
            'provider': provider_name,
            'type': model_config.get('type'),
            **model_config,
            'provider_config': provider_config,
        }
        
        return info
    
    def get_model_cost_model(self, model: str) -> str:
        """Get the cost model for a model, resolved through its PROVIDER.

        Cost differs per provider route, not per model -- `gpt-5.6-luna` on
        codex_cli is `subscription` and `openai/gpt-5.6-luna` on openrouter is
        `metered`, same weights. So it is stored once, on the provider, and
        resolved here. Never copy it onto a model entry: a copy is a second
        source of truth that drifts the moment a provider's billing changes.

        Args:
            model: Model name (the catalog key, which is the wire id)

        Returns:
            The provider's `cost_model`, or 'unknown' if it declares none
        """
        provider = self.get_model_config(model).get('provider')
        if not provider:
            return 'unknown'
        return self.get_provider_config(provider).get('cost_model', 'unknown')

    def get_model_family(self, model: str) -> str:
        """Get the logical family a model route belongs to.

        Two entries sharing a family are the same underlying weights reached
        through different providers under different ids. Falls back to the
        model key so a family-less entry is its own family rather than a
        KeyError; `validate_config()` reports the missing field separately.
        """
        return self.get_model_config(model).get('model_family') or model

    def describe_model(self, model: str) -> Dict[str, Any]:
        """Describe one model ROUTE: which provider serves it, at what cost.

        This is the answer to "show me the provider and the model together".
        Everything here is derived from the catalog on each call, so nothing
        can announce a provider or a price that the catalog no longer says.

        Args:
            model: Model name (the catalog key, which is the wire id)

        Returns:
            Dict with model/provider/provider_model_id/cost_model/type/family
            plus tier, description, dimensions and recommended where present.
        """
        model_config = self.get_model_config(model)
        provider = model_config.get('provider')
        return {
            'model': model,
            # The catalog key IS the provider-side wire id. Stated explicitly
            # rather than left implicit, because it is the field a caller needs
            # in order to talk to the provider.
            'provider_model_id': model,
            'provider': provider,
            'cost_model': self.get_model_cost_model(model),
            'type': model_config.get('type'),
            'family': self.get_model_family(model),
            'tier': model_config.get('tier'),
            'description': model_config.get('description', ''),
            'dimensions': model_config.get('dimensions'),
            'dimensions_configurable': bool(
                model_config.get('dimensions_configurable')
            ),
            'recommended': bool(model_config.get('recommended')),
            'superseded_by': model_config.get('superseded_by'),
        }

    def get_family_routes(self, family: str) -> List[Dict[str, Any]]:
        """Get every provider route that serves a given model family.

        This is what makes "the same weights at two prices" visible as data
        rather than as a comment: `get_family_routes('gpt-5.6-luna')` returns
        both the codex_cli subscription route and the openrouter metered one.
        """
        routes = []
        for model_name in self.config.get('models', {}):
            if self.get_model_family(model_name) == family:
                routes.append(self.describe_model(model_name))
        return sorted(routes, key=lambda route: route['model'])

    def format_model_description(self, model: str) -> str:
        """Render the description slot of a menu row: route first, prose after.

        Menus render as ``<model> — <this>``, so putting the route at the front
        of the description is what guarantees the provider and the cost travel
        with the model name in every list a human sees. Without it,
        `gpt-5.6-luna` and `openai/gpt-5.6-luna` appear as two unrelated names
        and nothing on screen says one is billed and the other is not.
        """
        route = self.describe_model(model)
        line = f"via {route['provider']} ({route['cost_model']})"
        if route['superseded_by']:
            line = f"{line} — superseded by {route['superseded_by']}"
        if route['description']:
            line = f"{line} — {route['description']}"
        return line

    def format_model_choice(self, model: str) -> str:
        """Render one model for a human, provider and cost always attached."""
        return f"{model} — {self.format_model_description(model)}"

    def is_provider_supported(self, provider: str, capability: str) -> bool:
        """Check if a provider supports a specific capability.
        
        Args:
            provider: Provider name
            capability: 'llm' or 'embeddings'
            
        Returns:
            True if supported
        """
        provider_config = self.get_provider_config(provider)
        supports = provider_config.get('supports', {})
        return supports.get(capability, False)

    def get_capability_evidence(
        self, provider: str, capability: str
    ) -> Optional[str]:
        """Return how a `supports:` claim was established, or None if unrecorded.

        See the `supports_evidence:` block on gemini_cli in ai_models.yaml for
        the vocabulary. This is deliberately separate from
        `is_provider_supported`: the flag says what the catalog claims, this
        says what the claim is worth.
        """
        provider_config = self.get_provider_config(provider)
        evidence = provider_config.get('supports_evidence', {})
        if not isinstance(evidence, dict):
            return None
        value = evidence.get(capability)
        return value if isinstance(value, str) else None

    def is_capability_verified(self, provider: str, capability: str) -> bool:
        """True only when the capability was established by CALLING it.

        Absence is not proof, so an unrecorded claim answers False. That is the
        whole point: `openrouter.supports.embeddings: false` rotted into
        "verified" precisely because nothing distinguished a claim nobody had
        tested from one somebody had.
        """
        return self.get_capability_evidence(provider, capability) == 'called'


    def get_api_key_env(self, provider: str) -> Optional[str]:
        """Get the environment variable name for API key.
        
        Args:
            provider: Provider name
            
        Returns:
            Environment variable name or None for local providers
        """
        provider_config = self.get_provider_config(provider)
        return provider_config.get('api_key_env')
    
    def get_base_url(self, provider: str) -> str:
        """Get the base URL for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Base URL
        """
        provider_config = self.get_provider_config(provider)
        return provider_config.get('base_url', '')
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration for model selection.
        
        Returns:
            UI configuration dictionary
        """
        return self.config.get('ui_options', {})
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values.
        
        Returns:
            Defaults dictionary
        """
        return self.config.get('defaults', {})
    
    def search_models(self, query: str, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for models by name or description.
        
        Args:
            query: Search query
            provider: Optional provider filter
            
        Returns:
            List of matching model configurations
        """
        models = self.config.get('models', {})
        results = []
        query_lower = query.lower()
        
        for model_name, model_config in models.items():
            if provider and model_config.get('provider') != provider:
                continue
            
            # Search in name and description
            if (query_lower in model_name.lower() or 
                query_lower in model_config.get('description', '').lower()):
                results.append({
                    'name': model_name,
                    **model_config
                })
        
        return results
    
    def _validate_capability_evidence(
        self, provider_name: str, provider_config: Dict[str, Any]
    ) -> List[str]:
        """Keep `supports_evidence:` honest: real capabilities, known vocabulary.

        It cannot check that somebody actually made the call -- nothing in a
        file can. What it can do is stop the record from drifting away from the
        claim it annotates (an evidence key for a capability that no longer
        exists), and stop the vocabulary from being widened by hand into
        something that reads as proof ("assumed-verified", "probably").
        """
        errors: List[str] = []
        evidence = provider_config.get('supports_evidence')
        if evidence is None:
            return errors
        if not isinstance(evidence, dict):
            return [
                f"Provider {provider_name} has a non-mapping "
                "'supports_evidence' field"
            ]

        supports = provider_config.get('supports') or {}
        for capability, value in evidence.items():
            if capability not in supports:
                errors.append(
                    f"Provider {provider_name} records supports_evidence for "
                    f"'{capability}', which is not a declared capability"
                )
            if value not in CAPABILITY_EVIDENCE_CLASSES:
                errors.append(
                    f"Provider {provider_name} supports_evidence.{capability} "
                    f"is {value!r}; expected one of "
                    f"{sorted(CAPABILITY_EVIDENCE_CLASSES)}"
                )
        return errors

    def validate_config(self) -> List[str]:
        """Validate configuration for common issues.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check providers
        providers = self.config.get('providers', {})
        for provider_name, provider_config in providers.items():
            if 'supports' not in provider_config:
                errors.append(f"Provider {provider_name} missing 'supports' field")
            if 'default_models' not in provider_config:
                errors.append(f"Provider {provider_name} missing 'default_models' field")
            errors.extend(
                self._validate_capability_evidence(provider_name, provider_config)
            )


        # Check models
        models = self.config.get('models', {})
        for model_name, model_config in models.items():
            if 'provider' not in model_config:
                errors.append(f"Model {model_name} missing 'provider' field")
            if 'type' not in model_config:
                errors.append(f"Model {model_name} missing 'type' field")
            if 'model_family' not in model_config:
                # Without this the "same weights, two providers" link exists
                # only in prose, which is how the two gpt-5.6-luna entries got
                # to look unrelated in the first place.
                errors.append(f"Model {model_name} missing 'model_family' field")

            # Check if provider exists
            provider = model_config.get('provider')
            if provider and provider not in providers:
                errors.append(f"Model {model_name} references unknown provider {provider}")

            # A `superseded_by` pointer must resolve to a real subscription
            # route, or it silently stops protecting anything.
            superseded_by = model_config.get('superseded_by')
            if superseded_by:
                target = models.get(superseded_by)
                if target is None:
                    errors.append(
                        f"Model {model_name} is superseded_by unknown model "
                        f"{superseded_by}"
                    )
                else:
                    target_provider = providers.get(target.get('provider'), {})
                    if target_provider.get('cost_model') != 'subscription':
                        errors.append(
                            f"Model {model_name} is superseded_by "
                            f"{superseded_by}, which is not on a subscription "
                            "provider"
                        )

        return errors


# Global configuration instance
_ai_config: Optional[AIModelConfig] = None


def get_ai_config() -> AIModelConfig:
    """Get the global AI configuration instance.
    
    Returns:
        AIModelConfig instance
    """
    global _ai_config
    if _ai_config is None:
        _ai_config = AIModelConfig()
    return _ai_config


def reload_config() -> None:
    """Reload the global configuration."""
    global _ai_config
    if _ai_config is not None:
        _ai_config.reload()
    else:
        _ai_config = AIModelConfig()


# Convenience functions for common operations
def get_provider_config(provider: str) -> Dict[str, Any]:
    """Get provider configuration."""
    return get_ai_config().get_provider_config(provider)


def get_model_config(model: str) -> Dict[str, Any]:
    """Get model configuration."""
    return get_ai_config().get_model_config(model)


def get_default_model(provider: str, model_type: str) -> Optional[str]:
    """Get default model for provider."""
    return get_ai_config().get_default_model(provider, model_type)


def get_use_case_config(use_case: str) -> Dict[str, str]:
    """Get use case configuration."""
    return get_ai_config().get_use_case_config(use_case)
