"""Tests for AI provider factory configuration."""

from unittest.mock import MagicMock

import pytest

from cicaddy.ai_providers.claude import ClaudeProvider
from cicaddy.ai_providers.factory import create_provider, get_provider_config


def _make_settings(**overrides):
    """Create a mock settings object with sensible defaults."""
    defaults = {
        "ai_provider": "gemini",
        "ai_model": None,
        "ai_temperature": "0.0",
        "gemini_api_key": None,
        "openai_api_key": None,
        "anthropic_api_key": None,
        "anthropic_vertex_project_id": None,
        "cloud_ml_region": None,
        "google_cloud_project": None,
        "google_cloud_location": "global",
    }
    defaults.update(overrides)
    settings = MagicMock()
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


class TestGetProviderConfigMissingKey:
    """When the configured provider's API key is missing, raise ValueError."""

    def test_gemini_missing_key_no_project_raises(self):
        settings = _make_settings(ai_provider="gemini", gemini_api_key=None)
        with pytest.raises(ValueError, match=r"GEMINI_API_KEY.*GOOGLE_CLOUD_PROJECT"):
            get_provider_config(settings)

    def test_gemini_empty_key_no_project_raises(self):
        settings = _make_settings(ai_provider="gemini", gemini_api_key="")
        with pytest.raises(ValueError, match=r"GEMINI_API_KEY.*GOOGLE_CLOUD_PROJECT"):
            get_provider_config(settings)

    def test_gemini_whitespace_key_no_project_raises(self):
        settings = _make_settings(ai_provider="gemini", gemini_api_key="   ")
        with pytest.raises(ValueError, match=r"GEMINI_API_KEY.*GOOGLE_CLOUD_PROJECT"):
            get_provider_config(settings)

    def test_openai_missing_key_raises(self):
        settings = _make_settings(ai_provider="openai", openai_api_key=None)
        with pytest.raises(ValueError, match="OpenAI API key not provided"):
            get_provider_config(settings)

    def test_claude_missing_key_raises(self):
        settings = _make_settings(ai_provider="claude", anthropic_api_key=None)
        with pytest.raises(ValueError, match="Anthropic API key not provided"):
            get_provider_config(settings)

    def test_anthropic_missing_key_raises(self):
        settings = _make_settings(ai_provider="anthropic", anthropic_api_key=None)
        with pytest.raises(ValueError, match="Anthropic API key not provided"):
            get_provider_config(settings)


class TestGetProviderConfigWithKey:
    """When the API key is present, config should be returned normally."""

    def test_gemini_with_key(self):
        settings = _make_settings(ai_provider="gemini", gemini_api_key="test-key")
        config = get_provider_config(settings)
        assert config["ai_provider"] == "gemini"
        assert config["api_key"] == "test-key"

    def test_openai_with_key(self):
        settings = _make_settings(ai_provider="openai", openai_api_key="test-key")
        config = get_provider_config(settings)
        assert config["ai_provider"] == "openai"
        assert config["api_key"] == "test-key"

    def test_claude_with_key(self):
        settings = _make_settings(ai_provider="claude", anthropic_api_key="test-key")
        config = get_provider_config(settings)
        assert config["ai_provider"] == "claude"
        assert config["api_key"] == "test-key"


class TestGetProviderConfigVertex:
    """Anthropic Vertex AI provider config tests."""

    def test_vertex_with_project_id(self):
        settings = _make_settings(
            ai_provider="anthropic-vertex",
            anthropic_vertex_project_id="my-gcp-project",
        )
        config = get_provider_config(settings)
        assert config["ai_provider"] == "anthropic-vertex"
        assert config["vertex_project_id"] == "my-gcp-project"
        assert config["region"] == "global"
        assert "api_key" not in config

    def test_vertex_with_custom_region(self):
        settings = _make_settings(
            ai_provider="anthropic-vertex",
            anthropic_vertex_project_id="my-gcp-project",
            google_cloud_location="europe-west4",
        )
        config = get_provider_config(settings)
        assert config["region"] == "europe-west4"

    def test_vertex_missing_project_id_raises(self):
        settings = _make_settings(
            ai_provider="anthropic-vertex",
            anthropic_vertex_project_id=None,
            google_cloud_project=None,
        )
        with pytest.raises(
            ValueError, match=r"Anthropic Vertex project ID not provided"
        ):
            get_provider_config(settings)

    def test_vertex_empty_project_id_raises(self):
        settings = _make_settings(
            ai_provider="anthropic-vertex",
            anthropic_vertex_project_id="",
            google_cloud_project=None,
        )
        with pytest.raises(
            ValueError, match=r"Anthropic Vertex project ID not provided"
        ):
            get_provider_config(settings)

    def test_vertex_whitespace_project_id_raises(self):
        settings = _make_settings(
            ai_provider="anthropic-vertex",
            anthropic_vertex_project_id="   ",
            google_cloud_project=None,
        )
        with pytest.raises(
            ValueError, match=r"Anthropic Vertex project ID not provided"
        ):
            get_provider_config(settings)

    def test_vertex_default_model(self):
        settings = _make_settings(
            ai_provider="anthropic-vertex",
            anthropic_vertex_project_id="my-gcp-project",
        )
        config = get_provider_config(settings)
        assert config["model_id"] == "claude-sonnet-4-6"

    def test_vertex_config_excludes_api_key(self):
        """Vertex config should not include api_key even if anthropic_api_key is set."""
        settings = _make_settings(
            ai_provider="anthropic-vertex",
            anthropic_vertex_project_id="my-gcp-project",
            anthropic_api_key="not-a-real-key",
        )
        config = get_provider_config(settings)
        assert config["vertex_project_id"] == "my-gcp-project"
        assert "api_key" not in config

    def test_vertex_none_region_falls_back_to_default(self):
        settings = _make_settings(
            ai_provider="anthropic-vertex",
            anthropic_vertex_project_id="my-gcp-project",
            cloud_ml_region=None,
        )
        config = get_provider_config(settings)
        assert config["region"] == "global"

    def test_vertex_whitespace_region_falls_back_to_default(self):
        settings = _make_settings(
            ai_provider="anthropic-vertex",
            anthropic_vertex_project_id="my-gcp-project",
            cloud_ml_region="   ",
        )
        config = get_provider_config(settings)
        assert config["region"] == "global"

    def test_vertex_falls_back_to_google_cloud_project(self):
        """When ANTHROPIC_VERTEX_PROJECT_ID is not set, fall back to GOOGLE_CLOUD_PROJECT."""
        settings = _make_settings(
            ai_provider="anthropic-vertex",
            anthropic_vertex_project_id=None,
            google_cloud_project="shared-gcp-project",
        )
        config = get_provider_config(settings)
        assert config["vertex_project_id"] == "shared-gcp-project"

    def test_vertex_specific_project_takes_precedence(self):
        """ANTHROPIC_VERTEX_PROJECT_ID takes precedence over GOOGLE_CLOUD_PROJECT."""
        settings = _make_settings(
            ai_provider="anthropic-vertex",
            anthropic_vertex_project_id="anthropic-project",
            google_cloud_project="shared-project",
        )
        config = get_provider_config(settings)
        assert config["vertex_project_id"] == "anthropic-project"

    def test_vertex_uses_google_cloud_location(self):
        """GOOGLE_CLOUD_LOCATION is the primary region variable."""
        settings = _make_settings(
            ai_provider="anthropic-vertex",
            anthropic_vertex_project_id="my-gcp-project",
            google_cloud_location="europe-west4",
        )
        config = get_provider_config(settings)
        assert config["region"] == "europe-west4"

    def test_vertex_deprecated_cloud_ml_region_is_ignored(self):
        """Deprecated CLOUD_ML_REGION is ignored; region uses GOOGLE_CLOUD_LOCATION."""
        settings = _make_settings(
            ai_provider="anthropic-vertex",
            anthropic_vertex_project_id="my-gcp-project",
            cloud_ml_region="us-central1",
        )
        config = get_provider_config(settings)
        assert config["region"] == "global"

    def test_vertex_google_cloud_location_overrides_default(self):
        """GOOGLE_CLOUD_LOCATION overrides the default even when CLOUD_ML_REGION is set."""
        settings = _make_settings(
            ai_provider="anthropic-vertex",
            anthropic_vertex_project_id="my-gcp-project",
            cloud_ml_region="us-central1",
            google_cloud_location="europe-west4",
        )
        config = get_provider_config(settings)
        assert config["region"] == "europe-west4"

    def test_vertex_whitespace_project_falls_back_to_gcp(self):
        """Whitespace-only ANTHROPIC_VERTEX_PROJECT_ID falls back to GOOGLE_CLOUD_PROJECT."""
        settings = _make_settings(
            ai_provider="anthropic-vertex",
            anthropic_vertex_project_id="   ",
            google_cloud_project="fallback-project",
        )
        config = get_provider_config(settings)
        assert config["vertex_project_id"] == "fallback-project"

    def test_vertex_project_fallback_logs_warning(self, capsys):
        """Falling back to GOOGLE_CLOUD_PROJECT should log a warning."""
        settings = _make_settings(
            ai_provider="anthropic-vertex",
            anthropic_vertex_project_id=None,
            google_cloud_project="shared-project",
        )
        get_provider_config(settings)
        captured = capsys.readouterr()
        assert "ANTHROPIC_VERTEX_PROJECT_ID not set" in captured.out

    def test_vertex_deprecated_region_logs_warning(self, capsys):
        """Setting deprecated CLOUD_ML_REGION should log a deprecation warning."""
        settings = _make_settings(
            ai_provider="anthropic-vertex",
            anthropic_vertex_project_id="my-project",
            cloud_ml_region="us-east5",
        )
        get_provider_config(settings)
        captured = capsys.readouterr()
        assert "CLOUD_ML_REGION is deprecated and ignored" in captured.out


class TestGetProviderConfigGeminiVertex:
    """Gemini Vertex AI provider config tests."""

    def test_gemini_vertex_with_project(self):
        settings = _make_settings(
            ai_provider="gemini-vertex",
            google_cloud_project="my-gcp-project",
        )
        config = get_provider_config(settings)
        assert config["ai_provider"] == "gemini-vertex"
        assert config["vertexai"] is True
        assert config["google_cloud_project"] == "my-gcp-project"
        assert config["google_cloud_location"] == "global"
        assert "api_key" not in config

    def test_gemini_vertex_with_custom_location(self):
        settings = _make_settings(
            ai_provider="gemini-vertex",
            google_cloud_project="my-gcp-project",
            google_cloud_location="europe-west4",
        )
        config = get_provider_config(settings)
        assert config["google_cloud_location"] == "europe-west4"

    def test_gemini_vertex_missing_project_raises(self):
        settings = _make_settings(
            ai_provider="gemini-vertex",
            google_cloud_project=None,
        )
        with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
            get_provider_config(settings)

    def test_gemini_vertex_empty_project_raises(self):
        settings = _make_settings(
            ai_provider="gemini-vertex",
            google_cloud_project="",
        )
        with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
            get_provider_config(settings)

    def test_gemini_vertex_whitespace_project_raises(self):
        settings = _make_settings(
            ai_provider="gemini-vertex",
            google_cloud_project="   ",
        )
        with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
            get_provider_config(settings)

    def test_gemini_vertex_default_model(self):
        settings = _make_settings(
            ai_provider="gemini-vertex",
            google_cloud_project="my-gcp-project",
        )
        config = get_provider_config(settings)
        assert config["model_id"] == "gemini-3-flash-preview"

    def test_gemini_vertex_none_location_falls_back_to_default(self):
        settings = _make_settings(
            ai_provider="gemini-vertex",
            google_cloud_project="my-gcp-project",
            google_cloud_location=None,
        )
        config = get_provider_config(settings)
        assert config["google_cloud_location"] == "global"

    def test_gemini_vertex_whitespace_location_falls_back_to_default(self):
        settings = _make_settings(
            ai_provider="gemini-vertex",
            google_cloud_project="my-gcp-project",
            google_cloud_location="   ",
        )
        config = get_provider_config(settings)
        assert config["google_cloud_location"] == "global"


class TestGeminiAutoFallbackToVertex:
    """When AI_PROVIDER=gemini and no API key, auto-fallback to Vertex AI."""

    def test_gemini_falls_back_to_vertex_when_project_set(self):
        settings = _make_settings(
            ai_provider="gemini",
            gemini_api_key=None,
            google_cloud_project="my-gcp-project",
        )
        config = get_provider_config(settings)
        assert config["ai_provider"] == "gemini-vertex"
        assert config["vertexai"] is True
        assert config["google_cloud_project"] == "my-gcp-project"
        assert config["google_cloud_location"] == "global"
        assert "api_key" not in config

    def test_gemini_fallback_respects_custom_location(self):
        settings = _make_settings(
            ai_provider="gemini",
            gemini_api_key=None,
            google_cloud_project="my-gcp-project",
            google_cloud_location="asia-southeast1",
        )
        config = get_provider_config(settings)
        assert config["google_cloud_location"] == "asia-southeast1"

    def test_gemini_api_key_takes_precedence_over_project(self):
        settings = _make_settings(
            ai_provider="gemini",
            gemini_api_key="my-api-key",
            google_cloud_project="my-gcp-project",
        )
        config = get_provider_config(settings)
        assert config["ai_provider"] == "gemini"
        assert config["api_key"] == "my-api-key"
        assert "vertexai" not in config
        assert "google_cloud_project" not in config

    def test_gemini_empty_key_with_project_falls_back(self):
        settings = _make_settings(
            ai_provider="gemini",
            gemini_api_key="",
            google_cloud_project="my-gcp-project",
        )
        config = get_provider_config(settings)
        assert config["ai_provider"] == "gemini-vertex"
        assert config["vertexai"] is True

    def test_gemini_whitespace_key_with_project_falls_back(self):
        settings = _make_settings(
            ai_provider="gemini",
            gemini_api_key="   ",
            google_cloud_project="my-gcp-project",
        )
        config = get_provider_config(settings)
        assert config["ai_provider"] == "gemini-vertex"
        assert config["vertexai"] is True
        assert "api_key" not in config

    def test_explicit_gemini_vertex_ignores_api_key(self):
        settings = _make_settings(
            ai_provider="gemini-vertex",
            gemini_api_key="should-be-ignored",
            google_cloud_project="my-gcp-project",
        )
        config = get_provider_config(settings)
        assert config["ai_provider"] == "gemini-vertex"
        assert config["vertexai"] is True
        assert "api_key" not in config


class TestCreateProviderRouting:
    """Verify create_provider returns the correct provider class."""

    def test_anthropic_vertex_returns_claude_provider(self):
        config = {
            "ai_provider": "anthropic-vertex",
            "model_id": "claude-sonnet-4-6",
            "vertex_project_id": "my-project",
            "region": "global",
            "temperature": 0.0,
        }
        provider = create_provider("anthropic-vertex", config)
        assert isinstance(provider, ClaudeProvider)

    def test_gemini_vertex_returns_gemini_provider(self):
        from cicaddy.ai_providers.gemini import GeminiProvider

        config = {
            "ai_provider": "gemini-vertex",
            "model_id": "gemini-3-flash-preview",
            "vertexai": True,
            "google_cloud_project": "my-project",
            "google_cloud_location": "global",
            "temperature": 0.0,
        }
        provider = create_provider("gemini-vertex", config)
        assert isinstance(provider, GeminiProvider)


class TestGetProviderConfigNoFallback:
    """Gemini with missing key must NOT fall back to OpenAI."""

    def test_no_fallback_to_openai(self):
        settings = _make_settings(
            ai_provider="gemini",
            gemini_api_key=None,
            openai_api_key="openai-key-present",
        )
        with pytest.raises(ValueError, match=r"GEMINI_API_KEY.*GOOGLE_CLOUD_PROJECT"):
            get_provider_config(settings)
