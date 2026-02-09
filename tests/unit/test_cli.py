"""Unit tests for CLI module."""

import os
import tempfile

import pytest

from cicaddy.cli.arg_mapping import (
    RUN_ARG_MAPPINGS,
    SENSITIVE_ENV_VARS,
    get_arg_mapping_by_cli_arg,
    get_arg_mapping_by_env_var,
)
from cicaddy.cli.commands import cmd_config_show, cmd_version, get_version
from cicaddy.cli.env_loader import (
    apply_cli_args_to_env,
    get_effective_config,
    load_env_file,
    mask_sensitive_value,
    validate_required_env_vars,
)
from cicaddy.cli.main import create_parser, main


class TestArgMapping:
    """Tests for arg_mapping module."""

    def test_run_arg_mappings_not_empty(self):
        """RUN_ARG_MAPPINGS should contain mappings."""
        assert len(RUN_ARG_MAPPINGS) > 0

    def test_sensitive_env_vars_contains_gitlab_token(self):
        """SENSITIVE_ENV_VARS should include GITLAB_TOKEN."""
        assert "GITLAB_TOKEN" in SENSITIVE_ENV_VARS

    def test_sensitive_env_vars_contains_api_keys(self):
        """SENSITIVE_ENV_VARS should include API keys."""
        assert "GEMINI_API_KEY" in SENSITIVE_ENV_VARS
        assert "OPENAI_API_KEY" in SENSITIVE_ENV_VARS
        assert "ANTHROPIC_API_KEY" in SENSITIVE_ENV_VARS

    def test_get_arg_mapping_by_env_var(self):
        """get_arg_mapping_by_env_var should return correct mapping."""
        mapping = get_arg_mapping_by_env_var("AGENT_TYPE")
        assert mapping is not None
        assert mapping.cli_arg == "--agent-type"
        assert mapping.env_var == "AGENT_TYPE"

    def test_get_arg_mapping_by_env_var_not_found(self):
        """get_arg_mapping_by_env_var should return None for unknown var."""
        mapping = get_arg_mapping_by_env_var("UNKNOWN_VAR")
        assert mapping is None

    def test_get_arg_mapping_by_cli_arg(self):
        """get_arg_mapping_by_cli_arg should return correct mapping."""
        mapping = get_arg_mapping_by_cli_arg("--agent-type")
        assert mapping is not None
        assert mapping.env_var == "AGENT_TYPE"

    def test_get_arg_mapping_by_cli_arg_normalized(self):
        """get_arg_mapping_by_cli_arg should handle different formats."""
        mapping = get_arg_mapping_by_cli_arg("agent_type")
        assert mapping is not None
        assert mapping.env_var == "AGENT_TYPE"


class TestEnvLoader:
    """Tests for env_loader module."""

    def test_load_env_file_not_found(self):
        """load_env_file should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_env_file("/nonexistent/.env.test")

    def test_load_env_file_success(self):
        """load_env_file should load variables from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("TEST_VAR=test_value\n")
            f.write("ANOTHER_VAR=another_value\n")
            f.flush()
            temp_path = f.name

        try:
            loaded = load_env_file(temp_path)
            assert "TEST_VAR" in loaded
            assert loaded["TEST_VAR"] == "test_value"
            assert os.environ.get("TEST_VAR") == "test_value"
        finally:
            os.unlink(temp_path)
            if "TEST_VAR" in os.environ:
                del os.environ["TEST_VAR"]
            if "ANOTHER_VAR" in os.environ:
                del os.environ["ANOTHER_VAR"]

    def test_apply_cli_args_to_env(self):
        """apply_cli_args_to_env should set environment variables."""
        args = {
            "agent_type": "cron",
            "ai_provider": "gemini",
            "log_level": "DEBUG",
        }

        try:
            applied = apply_cli_args_to_env(args)
            assert "AGENT_TYPE" in applied
            assert applied["AGENT_TYPE"] == "cron"
            assert os.environ.get("AGENT_TYPE") == "cron"
            assert os.environ.get("AI_PROVIDER") == "gemini"
        finally:
            for var in ["AGENT_TYPE", "AI_PROVIDER", "LOG_LEVEL"]:
                if var in os.environ:
                    del os.environ[var]

    def test_apply_cli_args_verbose_flag(self):
        """apply_cli_args_to_env should handle verbose flag."""
        args = {"verbose": True}

        try:
            apply_cli_args_to_env(args)
            assert os.environ.get("LOG_LEVEL") == "DEBUG"
        finally:
            if "LOG_LEVEL" in os.environ:
                del os.environ["LOG_LEVEL"]

    def test_mask_sensitive_value_not_set(self):
        """mask_sensitive_value should return '(not set)' for None."""
        assert mask_sensitive_value(None) == "(not set)"

    def test_mask_sensitive_value_short(self):
        """mask_sensitive_value should mask short values completely."""
        assert mask_sensitive_value("abc") == "***"

    def test_mask_sensitive_value_long(self):
        """mask_sensitive_value should show last 4 chars of long values."""
        result = mask_sensitive_value("my_secret_token_abcd")
        assert result.endswith("abcd")
        assert result.startswith("*")

    def test_validate_required_env_vars(self):
        """validate_required_env_vars should return missing vars."""
        os.environ["EXISTING_VAR"] = "value"
        try:
            missing = validate_required_env_vars(["EXISTING_VAR", "MISSING_VAR"])
            assert "MISSING_VAR" in missing
            assert "EXISTING_VAR" not in missing
        finally:
            del os.environ["EXISTING_VAR"]

    def test_get_effective_config(self):
        """get_effective_config should return current environment values."""
        os.environ["AGENT_TYPE"] = "cron"
        try:
            config = get_effective_config()
            assert config["AGENT_TYPE"] == "cron"
        finally:
            del os.environ["AGENT_TYPE"]


class TestParser:
    """Tests for argument parser."""

    def test_create_parser(self):
        """create_parser should return an ArgumentParser."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "cicaddy"

    def test_parser_run_command(self):
        """Parser should accept 'run' command."""
        parser = create_parser()
        args = parser.parse_args(["run"])
        assert args.command == "run"

    def test_parser_run_with_env_file(self):
        """Parser should accept --env-file option."""
        parser = create_parser()
        args = parser.parse_args(["run", "--env-file", ".env.test"])
        assert args.env_file == ".env.test"

    def test_parser_run_with_short_env_file(self):
        """Parser should accept -e option."""
        parser = create_parser()
        args = parser.parse_args(["run", "-e", ".env.test"])
        assert args.env_file == ".env.test"

    def test_parser_run_with_agent_type(self):
        """Parser should accept --agent-type option."""
        parser = create_parser()
        args = parser.parse_args(["run", "--agent-type", "cron"])
        assert args.agent_type == "cron"

    def test_parser_run_with_short_agent_type(self):
        """Parser should accept -t option."""
        parser = create_parser()
        args = parser.parse_args(["run", "-t", "cron"])
        assert args.agent_type == "cron"

    def test_parser_run_with_verbose(self):
        """Parser should accept --verbose flag."""
        parser = create_parser()
        args = parser.parse_args(["run", "--verbose"])
        assert args.verbose is True

    def test_parser_run_with_dry_run(self):
        """Parser should accept --dry-run flag."""
        parser = create_parser()
        args = parser.parse_args(["run", "--dry-run"])
        assert args.dry_run is True

    def test_parser_version_command(self):
        """Parser should accept 'version' command."""
        parser = create_parser()
        args = parser.parse_args(["version"])
        assert args.command == "version"

    def test_parser_config_show_command(self):
        """Parser should accept 'config show' command."""
        parser = create_parser()
        args = parser.parse_args(["config", "show"])
        assert args.command == "config"
        assert args.config_command == "show"

    def test_parser_invalid_agent_type(self):
        """Parser should reject invalid agent type."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--agent-type", "invalid"])


class TestCommands:
    """Tests for command handlers."""

    def test_get_version(self):
        """get_version should return a version string."""
        version = get_version()
        assert version is not None
        assert isinstance(version, str)

    def test_cmd_version(self, capsys):
        """cmd_version should print version."""
        from argparse import Namespace

        args = Namespace()
        result = cmd_version(args)
        assert result == 0

        captured = capsys.readouterr()
        assert "cicaddy version" in captured.out

    def test_cmd_config_show(self, capsys):
        """cmd_config_show should print configuration."""
        from argparse import Namespace

        args = Namespace(env_file=None)
        result = cmd_config_show(args)
        assert result == 0

        captured = capsys.readouterr()
        assert "Current Configuration" in captured.out
        assert "[Agent Settings]" in captured.out

    def test_cmd_config_show_missing_env_file(self, capsys):
        """cmd_config_show should error on missing env file."""
        from argparse import Namespace

        args = Namespace(env_file="/nonexistent/.env")
        result = cmd_config_show(args)
        assert result == 1

        captured = capsys.readouterr()
        assert "Error" in captured.err


class TestMain:
    """Tests for main entry point."""

    def test_main_no_command(self, capsys):
        """main with no command should show help."""
        result = main([])
        assert result == 0

        captured = capsys.readouterr()
        assert "usage:" in captured.out

    def test_main_version_command(self, capsys):
        """main with version command should show version."""
        result = main(["version"])
        assert result == 0

        captured = capsys.readouterr()
        assert "cicaddy version" in captured.out

    def test_main_config_show_command(self, capsys):
        """main with config show should display config."""
        result = main(["config", "show"])
        assert result == 0

        captured = capsys.readouterr()
        assert "Current Configuration" in captured.out
