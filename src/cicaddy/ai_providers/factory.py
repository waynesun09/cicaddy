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

# Default Vertex AI location (shared by all Vertex providers)
DEFAULT_VERTEX_LOCATION = "global"

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


def _safe_strip(value: Optional[str]) -> Optional[str]:
    """Strip whitespace from a string value, passing through None."""
    return value.strip() if isinstance(value, str) else value


def _resolve_setting(settings: Any, *attrs: str) -> Optional[str]:
    """Resolve a setting by trying attributes in priority order."""
    for attr in attrs:
        value = _safe_strip(getattr(settings, attr, None))
        if value:
            return value
    return None


def _require_api_key(raw_key: Optional[str], provider_label: str, env_var: str) -> str:
    """Validate and return a stripped API key, or raise ValueError."""
    key = _safe_strip(raw_key)
    if not key:
        raise ValueError(
            f"{provider_label} API key not provided. "
            f"Set the {env_var} environment variable."
        )
    return key


def _apply_gemini_vertex_config(
    config: Dict[str, Any], project: str, settings: Any
) -> None:
    """Apply Gemini Vertex AI config (shared by explicit and auto-fallback paths)."""
    config["vertexai"] = True
    config["google_cloud_project"] = project
    location = _resolve_setting(settings, "google_cloud_location")
    config["google_cloud_location"] = location or DEFAULT_VERTEX_LOCATION


def _configure_gemini(config: Dict[str, Any], settings: Any) -> None:
    """Configure Gemini provider (API key or auto-fallback to Vertex AI)."""
    api_key = _safe_strip(settings.gemini_api_key)
    if api_key:
        config["api_key"] = api_key
        return
    project = _resolve_setting(settings, "google_cloud_project")
    if not project:
        raise ValueError(
            "Gemini API key not provided. "
            "Set GEMINI_API_KEY for API key auth, or set "
            "GOOGLE_CLOUD_PROJECT for Vertex AI with ADC."
        )
    logger.warning(
        "GEMINI_API_KEY not set; falling back to Vertex AI with ADC "
        "(set AI_PROVIDER=gemini-vertex to silence this warning)"
    )
    config["ai_provider"] = "gemini-vertex"
    _apply_gemini_vertex_config(config, project, settings)


def _configure_gemini_vertex(config: Dict[str, Any], settings: Any) -> None:
    """Configure explicit Gemini Vertex AI provider."""
    project = _resolve_setting(settings, "google_cloud_project")
    if not project:
        raise ValueError(
            "Google Cloud project not provided for gemini-vertex. "
            "Set the GOOGLE_CLOUD_PROJECT environment variable."
        )
    _apply_gemini_vertex_config(config, project, settings)


def _configure_openai(config: Dict[str, Any], settings: Any) -> None:
    """Configure OpenAI provider."""
    config["api_key"] = _require_api_key(
        settings.openai_api_key, "OpenAI", "OPENAI_API_KEY"
    )
    config["base_url"] = None


def _configure_anthropic(config: Dict[str, Any], settings: Any) -> None:
    """Configure Anthropic (Claude) provider."""
    config["api_key"] = _require_api_key(
        settings.anthropic_api_key, "Anthropic", "ANTHROPIC_API_KEY"
    )


def _configure_anthropic_vertex(config: Dict[str, Any], settings: Any) -> None:
    """Configure Anthropic Vertex AI provider."""
    project_id = _safe_strip(getattr(settings, "anthropic_vertex_project_id", None))
    if not project_id:
        project_id = _safe_strip(getattr(settings, "google_cloud_project", None))
        if project_id:
            logger.warning(
                "ANTHROPIC_VERTEX_PROJECT_ID not set; using GOOGLE_CLOUD_PROJECT "
                "(set ANTHROPIC_VERTEX_PROJECT_ID to silence this warning)"
            )
        else:
            raise ValueError(
                "Anthropic Vertex project ID not provided. "
                "Set ANTHROPIC_VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT."
            )
    config["vertex_project_id"] = project_id
    # Region: prefer CLOUD_ML_REGION, fall back to GOOGLE_CLOUD_LOCATION
    region = _safe_strip(getattr(settings, "cloud_ml_region", None))
    if not region:
        region = _safe_strip(getattr(settings, "google_cloud_location", None))
        if region:
            logger.warning(
                "CLOUD_ML_REGION not set; using GOOGLE_CLOUD_LOCATION "
                "(set CLOUD_ML_REGION to silence this warning)"
            )
    config["region"] = region or DEFAULT_VERTEX_LOCATION


_PROVIDER_CONFIGURATORS = {
    "gemini": _configure_gemini,
    "gemini-vertex": _configure_gemini_vertex,
    "openai": _configure_openai,
    "claude": _configure_anthropic,
    "anthropic": _configure_anthropic,
    "anthropic-vertex": _configure_anthropic_vertex,
}


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

    configurator = _PROVIDER_CONFIGURATORS.get(provider)
    if configurator:
        configurator(config, settings)

    logger.info(
        f"Created provider config for {config['ai_provider']} with model {config['model_id']}"
    )
    return config
