"""Tests for cicaddy.plugin module."""

from unittest.mock import MagicMock, patch

import pytest

import cicaddy.plugin as plugin_mod


@pytest.fixture(autouse=True)
def reset_plugin_state():
    """Reset plugin discovery state between tests."""
    plugin_mod._plugins_discovered = False
    yield
    plugin_mod._plugins_discovered = False


class TestDiscoverPlugins:
    """Tests for discover_plugins()."""

    def test_idempotent(self):
        """Calling discover_plugins twice should not duplicate work."""
        with patch("cicaddy.plugin.entry_points") as mock_ep:
            mock_ep.return_value.select.return_value = []
            plugin_mod.discover_plugins()
            plugin_mod.discover_plugins()
            # entry_points() should only be called once
            assert mock_ep.call_count == 1

    def test_loads_agent_plugin(self):
        """Agent plugin entry point should be loaded and called."""
        mock_register_fn = MagicMock()
        mock_ep = MagicMock()
        mock_ep.load.return_value = mock_register_fn
        mock_ep.name = "test-plugin"

        with patch("cicaddy.plugin.entry_points") as mock_entry_points:
            mock_entry_points.return_value.select.return_value = [mock_ep]
            plugin_mod.discover_plugins()

        mock_ep.load.assert_called_once()
        mock_register_fn.assert_called_once()

    def test_plugin_load_failure_is_warned(self):
        """A failing plugin should log a warning, not raise."""
        mock_ep = MagicMock()
        mock_ep.load.side_effect = ImportError("bad plugin")
        mock_ep.name = "bad-plugin"

        with patch("cicaddy.plugin.entry_points") as mock_entry_points:
            mock_entry_points.return_value.select.return_value = [mock_ep]
            # Should not raise
            plugin_mod.discover_plugins()


class TestGetPluginCliArgs:
    """Tests for get_plugin_cli_args()."""

    def test_empty_without_plugins(self):
        """Should return empty list when no plugins installed."""
        with patch("cicaddy.plugin.entry_points") as mock_ep:
            mock_ep.return_value.select.return_value = []
            result = plugin_mod.get_plugin_cli_args()
        assert result == []

    def test_returns_plugin_args(self):
        """Should return args from plugin entry points."""
        from cicaddy.cli.arg_mapping import ArgMapping

        test_arg = ArgMapping(
            cli_arg="--test-arg", env_var="TEST_ARG", help_text="test"
        )
        mock_ep = MagicMock()
        mock_ep.load.return_value = lambda: [test_arg]
        mock_ep.name = "test"

        with patch("cicaddy.plugin.entry_points") as mock_entry_points:
            mock_entry_points.return_value.select.return_value = [mock_ep]
            result = plugin_mod.get_plugin_cli_args()

        assert len(result) == 1
        assert result[0].cli_arg == "--test-arg"


class TestGetPluginEnvVars:
    """Tests for get_plugin_env_vars()."""

    def test_empty_without_plugins(self):
        """Should return empty list when no plugins installed."""
        with patch("cicaddy.plugin.entry_points") as mock_ep:
            mock_ep.return_value.select.return_value = []
            result = plugin_mod.get_plugin_env_vars()
        assert result == []


class TestGetSettingsLoader:
    """Tests for get_settings_loader()."""

    def test_none_without_plugins(self):
        """Should return None when no settings loader plugin installed."""
        with patch("cicaddy.plugin.entry_points") as mock_ep:
            mock_ep.return_value.select.return_value = []
            result = plugin_mod.get_settings_loader()
        assert result is None

    def test_returns_loader_callable(self):
        """Should return the loader callable from entry point."""
        mock_loader = MagicMock()
        mock_ep = MagicMock()
        mock_ep.load.return_value = mock_loader
        mock_ep.name = "test"

        with patch("cicaddy.plugin.entry_points") as mock_entry_points:
            mock_entry_points.return_value.select.return_value = [mock_ep]
            result = plugin_mod.get_settings_loader()

        assert result is mock_loader


class TestGetPluginConfigSections:
    """Tests for get_plugin_config_sections()."""

    def test_empty_without_plugins(self):
        """Should return empty list when no plugins installed."""
        with patch("cicaddy.plugin.entry_points") as mock_ep:
            mock_ep.return_value.select.return_value = []
            result = plugin_mod.get_plugin_config_sections()
        assert result == []

    def test_returns_section_callable(self):
        """Should return the section callables from plugins."""
        mock_section_fn = MagicMock()
        mock_ep = MagicMock()
        mock_ep.load.return_value = mock_section_fn
        mock_ep.name = "test"

        with patch("cicaddy.plugin.entry_points") as mock_entry_points:
            mock_entry_points.return_value.select.return_value = [mock_ep]
            result = plugin_mod.get_plugin_config_sections()

        assert len(result) == 1
        assert result[0] is mock_section_fn


class TestGetPluginValidators:
    """Tests for get_plugin_validators()."""

    def test_empty_without_plugins(self):
        """Should return empty list when no plugins installed."""
        with patch("cicaddy.plugin.entry_points") as mock_ep:
            mock_ep.return_value.select.return_value = []
            result = plugin_mod.get_plugin_validators()
        assert result == []

    def test_returns_validator_callable(self):
        """Should return the validator callables from plugins."""
        mock_validator_fn = MagicMock()
        mock_ep = MagicMock()
        mock_ep.load.return_value = mock_validator_fn
        mock_ep.name = "test"

        with patch("cicaddy.plugin.entry_points") as mock_entry_points:
            mock_entry_points.return_value.select.return_value = [mock_ep]
            result = plugin_mod.get_plugin_validators()

        assert len(result) == 1
        assert result[0] is mock_validator_fn


class TestAgentTypeChoicesDynamic:
    """Tests for dynamic agent type choices."""

    def test_registered_agents_appear_in_choices(self):
        """Registered agents should appear in available types."""
        from cicaddy.agent.factory import AgentFactory

        # Built-in agents should be registered
        available = AgentFactory.get_available_agent_types()
        assert "cron" in available
        assert "branch_review" in available
        assert "merge_request" in available
