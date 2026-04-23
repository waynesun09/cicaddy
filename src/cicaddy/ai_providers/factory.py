"""Provider factory following Llama Stack patterns."""

from typing import Any, Dict, Optional

from cicaddy.utils.logger import get_logger

from .base import BaseProvider
from .claude import ClaudeProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider

logger = get_logger(__name__)

# Default AI provider
DEFAULT_AI_PROVIDER = "gemini"

# Default Vertex AI region
DEFAULT_VERTEX_REGION = "us-east5"

# Default model mappings for each provider
DEFAULT_MODELS = {
    DEFAULT_AI_PROVIDER: "gemini-3-flash-preview",
    "gemini-vertex": "gemini-3-flash-preview",
    "openai": "gpt-5.4",
    "claude": "claude-sonnet-4-6",
    "anthropic": "claude-sonnet-4-6",
    "anthropic-vertex": "claude-sonnet-4-6",
}


def get_default_model(provider: str) -> str:
    """Get the default model for a given AI provider.

    Args:
        provider: AI provider name (e.g., "gemini", "openai", "claude")

    Returns:
        Default model name for the provider
    """
    return DEFAULT_MODELS.get(provider.lower(), "gpt-5.4")


def create_provider(provider_name: str, config: Dict[str, Any]) -> BaseProvider:
    """Create provider instance based on provider name and config."""

    provider_name = provider_name.lower()
    # Honor effective provider possibly adjusted in config (e.g., fallback when key missing)
    effective_provider = (config.get("ai_provider") or provider_name).lower()

    if effective_provider in (DEFAULT_AI_PROVIDER, "gemini-vertex"):
        return GeminiProvider(config)
    elif effective_provider == "openai":
        return OpenAIProvider(config)
    elif effective_provider in ("claude", "anthropic", "anthropic-vertex"):
        return ClaudeProvider(config)
    else:
        # Fallback to default provider
        logger.warning(
            f"Unknown provider '{effective_provider}', falling back to {DEFAULT_AI_PROVIDER}"
        )
        return GeminiProvider(config)


def _require_api_key(raw_key: Optional[str], provider_label: str, env_var: str) -> str:
    """Validate and return a stripped API key, or raise ValueError."""
    key = raw_key.strip() if isinstance(raw_key, str) else raw_key
    if not key:
        raise ValueError(
            f"{provider_label} API key not provided. "
            f"Set the {env_var} environment variable."
        )
    return key


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
        api_key = (
            settings.gemini_api_key.strip()
            if isinstance(settings.gemini_api_key, str)
            else settings.gemini_api_key
        )
        if api_key:
            config["api_key"] = api_key
        else:
            # Auto-fallback to Vertex AI when no API key but project is set
            project = (
                settings.google_cloud_project.strip()
                if isinstance(settings.google_cloud_project, str)
                else settings.google_cloud_project
            )
            if project:
                logger.info(
                    "No GEMINI_API_KEY found; falling back to Vertex AI with ADC"
                )
                config["ai_provider"] = "gemini-vertex"
                config["vertexai"] = True
                config["google_cloud_project"] = project
                location = (
                    settings.google_cloud_location.strip()
                    if isinstance(settings.google_cloud_location, str)
                    else settings.google_cloud_location
                )
                config["google_cloud_location"] = location or "us-central1"
            else:
                raise ValueError(
                    "Gemini API key not provided. "
                    "Set GEMINI_API_KEY for API key auth, or set "
                    "GOOGLE_CLOUD_PROJECT for Vertex AI with ADC."
                )
    elif provider == "gemini-vertex":
        project = (
            settings.google_cloud_project.strip()
            if isinstance(settings.google_cloud_project, str)
            else settings.google_cloud_project
        )
        if not project:
            raise ValueError(
                "Google Cloud project not provided for gemini-vertex. "
                "Set the GOOGLE_CLOUD_PROJECT environment variable."
            )
        config["vertexai"] = True
        config["google_cloud_project"] = project
        location = (
            settings.google_cloud_location.strip()
            if isinstance(settings.google_cloud_location, str)
            else settings.google_cloud_location
        )
        config["google_cloud_location"] = location or "us-central1"
    elif provider == "openai":
        config["api_key"] = _require_api_key(
            settings.openai_api_key, "OpenAI", "OPENAI_API_KEY"
        )
        config["base_url"] = None  # Use default OpenAI endpoint
    elif provider in ["claude", "anthropic"]:
        config["api_key"] = _require_api_key(
            settings.anthropic_api_key, "Anthropic", "ANTHROPIC_API_KEY"
        )
    elif provider == "anthropic-vertex":
        project_id = (
            settings.anthropic_vertex_project_id.strip()
            if isinstance(settings.anthropic_vertex_project_id, str)
            else settings.anthropic_vertex_project_id
        )
        if not project_id:
            raise ValueError(
                "Anthropic Vertex project ID not provided. "
                "Set the ANTHROPIC_VERTEX_PROJECT_ID environment variable."
            )
        config["vertex_project_id"] = project_id
        region = (
            settings.cloud_ml_region.strip()
            if isinstance(settings.cloud_ml_region, str)
            else settings.cloud_ml_region
        )
        config["region"] = region or DEFAULT_VERTEX_REGION

    logger.info(
        f"Created provider config for {provider} with model {config['model_id']}"
    )
    return config
