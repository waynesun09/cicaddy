"""Tests for AI provider factory configuration."""

from unittest.mock import MagicMock

import pytest

from cicaddy.ai_providers.factory import get_provider_config


def _make_settings(**overrides):
    """Create a mock settings object with sensible defaults."""
    defaults = {
        "ai_provider": "gemini",
        "ai_model": None,
        "ai_temperature": "0.0",
        "gemini_api_key": None,
        "openai_api_key": None,
        "anthropic_api_key": None,
    }
    defaults.update(overrides)
    settings = MagicMock()
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


class TestGetProviderConfigMissingKey:
    """When the configured provider's API key is missing, raise ValueError."""

    def test_gemini_missing_key_raises(self):
        settings = _make_settings(ai_provider="gemini", gemini_api_key=None)
        with pytest.raises(ValueError, match="Gemini API key not provided"):
            get_provider_config(settings)

    def test_gemini_empty_key_raises(self):
        settings = _make_settings(ai_provider="gemini", gemini_api_key="")
        with pytest.raises(ValueError, match="Gemini API key not provided"):
            get_provider_config(settings)

    def test_gemini_whitespace_key_raises(self):
        settings = _make_settings(ai_provider="gemini", gemini_api_key="   ")
        with pytest.raises(ValueError, match="Gemini API key not provided"):
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


class TestGetProviderConfigNoFallback:
    """Gemini with missing key must NOT fall back to OpenAI."""

    def test_no_fallback_to_openai(self):
        settings = _make_settings(
            ai_provider="gemini",
            gemini_api_key=None,
            openai_api_key="openai-key-present",
        )
        with pytest.raises(ValueError, match="Gemini API key not provided"):
            get_provider_config(settings)
