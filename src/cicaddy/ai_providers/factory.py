"""Provider factory following Llama Stack patterns."""

from typing import Any, Dict

from cicaddy.utils.logger import get_logger

from .base import BaseProvider
from .claude import ClaudeProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider

logger = get_logger(__name__)

# Default AI provider
DEFAULT_AI_PROVIDER = "gemini"

# Default model mappings for each provider
DEFAULT_MODELS = {
    DEFAULT_AI_PROVIDER: "gemini-3-flash-preview",
    "openai": "gpt-4o",
    "claude": "claude-3-5-sonnet-latest",
    "anthropic": "claude-3-5-sonnet-latest",
}


def get_default_model(provider: str) -> str:
    """Get the default model for a given AI provider.

    Args:
        provider: AI provider name (e.g., "gemini", "openai", "claude")

    Returns:
        Default model name for the provider
    """
    return DEFAULT_MODELS.get(provider.lower(), "gpt-4o")


def create_provider(provider_name: str, config: Dict[str, Any]) -> BaseProvider:
    """Create provider instance based on provider name and config."""

    provider_name = provider_name.lower()
    # Honor effective provider possibly adjusted in config (e.g., fallback when key missing)
    effective_provider = (config.get("ai_provider") or provider_name).lower()

    if effective_provider == DEFAULT_AI_PROVIDER:
        return GeminiProvider(config)
    elif effective_provider == "openai":
        return OpenAIProvider(config)
    elif effective_provider == "claude" or effective_provider == "anthropic":
        return ClaudeProvider(config)
    else:
        # Fallback to default provider
        logger.warning(
            f"Unknown provider '{effective_provider}', falling back to {DEFAULT_AI_PROVIDER}"
        )
        return GeminiProvider(config)


def get_provider_config(settings) -> Dict[str, Any]:
    """Build provider config from settings following Llama Stack patterns."""

    provider = settings.ai_provider.lower() or DEFAULT_AI_PROVIDER

    config = {
        "ai_provider": provider,
        "model_id": settings.ai_model or get_default_model(provider),
        "temperature": float(settings.ai_temperature)
        if settings.ai_temperature
        else 0.0,
    }

    # Provider-specific configurations
    if provider == "gemini":
        config["api_key"] = settings.gemini_api_key
        # If no Gemini API key available (e.g., in tests), fall back to OpenAI to avoid external failures
        if not config.get("api_key"):
            logger.warning(
                "GEMINI_API_KEY missing; falling back to OpenAI provider for tests"
            )
            provider = "openai"
            config["ai_provider"] = provider
            config["model_id"] = get_default_model(provider)
            config["api_key"] = settings.openai_api_key
    elif provider == "openai":
        config["api_key"] = settings.openai_api_key
        config["base_url"] = None  # Use default OpenAI endpoint
    elif provider in ["claude", "anthropic"]:
        config["api_key"] = settings.anthropic_api_key

    logger.info(
        f"Created provider config for {provider} with model {config['model_id']}"
    )
    return config
