"""Configuration loader for AI models and providers."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import yaml

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


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
        
        # Check models
        models = self.config.get('models', {})
        for model_name, model_config in models.items():
            if 'provider' not in model_config:
                errors.append(f"Model {model_name} missing 'provider' field")
            if 'type' not in model_config:
                errors.append(f"Model {model_name} missing 'type' field")
            
            # Check if provider exists
            provider = model_config.get('provider')
            if provider and provider not in providers:
                errors.append(f"Model {model_name} references unknown provider {provider}")
        
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
