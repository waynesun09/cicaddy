"""Unit tests for settings configuration."""

import os
from unittest.mock import patch

from cicaddy.config.settings import (
    CoreSettings,
    Settings,
    load_core_settings,
    load_settings,
)


class TestCoreSettings:
    """Test CoreSettings instantiation and defaults."""

    def test_default_values(self):
        """Test that CoreSettings has sensible defaults."""
        # Use clean env to avoid picking up LOG_LEVEL etc from developer env
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in (
                "LOG_LEVEL",
                "AI_PROVIDER",
                "MAX_INFER_ITERS",
                "MAX_EXECUTION_TIME",
                "CONTEXT_SAFETY_FACTOR",
                "SSL_VERIFY",
            )
        }
        with patch.dict(os.environ, clean_env, clear=True):
            settings = CoreSettings()
            assert settings.ai_provider == "gemini"
            assert settings.max_infer_iters == 10
            assert settings.max_execution_time == 600
            assert settings.context_safety_factor == 0.85
            assert settings.log_level == "INFO"
            assert settings.ssl_verify is True

    def test_settings_is_core_settings(self):
        """Test that Settings alias points to CoreSettings."""
        assert Settings is CoreSettings

    def test_load_core_settings(self):
        """Test that load_core_settings returns a CoreSettings instance."""
        settings = load_core_settings()
        assert isinstance(settings, CoreSettings)

    def test_load_settings_returns_core(self):
        """Test that load_settings returns a CoreSettings instance."""
        settings = load_settings()
        assert isinstance(settings, CoreSettings)

    def test_context_safety_factor_valid_value(self):
        """Test that valid CONTEXT_SAFETY_FACTOR is accepted."""
        with patch.dict(os.environ, {"CONTEXT_SAFETY_FACTOR": "0.75"}, clear=False):
            settings = load_core_settings()
            assert settings.context_safety_factor == 0.75

    def test_context_safety_factor_not_set(self):
        """Test that missing CONTEXT_SAFETY_FACTOR uses default."""
        env_copy = os.environ.copy()
        env_copy.pop("CONTEXT_SAFETY_FACTOR", None)
        with patch.dict(os.environ, env_copy, clear=True):
            settings = load_core_settings()
            assert settings.context_safety_factor == 0.85

    def test_max_execution_time_valid_value(self):
        """Test that valid MAX_EXECUTION_TIME is accepted."""
        with patch.dict(os.environ, {"MAX_EXECUTION_TIME": "1200"}, clear=False):
            settings = load_core_settings()
            assert settings.max_execution_time == 1200

    def test_ai_provider_from_env(self):
        """Test that AI_PROVIDER can be set from environment."""
        with patch.dict(os.environ, {"AI_PROVIDER": "openai"}, clear=False):
            settings = load_core_settings()
            assert settings.ai_provider == "openai"

    def test_repr_masks_sensitive_fields(self):
        """Test that repr masks sensitive fields."""
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "test-secret-key-12345"},
            clear=False,
        ):
            settings = load_core_settings()
            repr_str = repr(settings)
            assert "test-secret-key-12345" not in repr_str
            assert "CoreSettings(" in repr_str
