"""CLI argument to environment variable mappings."""

from dataclasses import dataclass
from typing import Any, List, Optional

from cicaddy.config.settings import SENSITIVE_ENV_VAR_NAMES


@dataclass
class ArgMapping:
    """Mapping between CLI argument and environment variable."""

    cli_arg: str  # CLI argument name (e.g., "--agent-type")
    env_var: str  # Environment variable name (e.g., "AGENT_TYPE")
    arg_type: type = str  # Argument type
    choices: Optional[List[str]] = None  # Valid choices
    help_text: str = ""  # Help text for argparse
    short_arg: Optional[str] = None  # Short argument (e.g., "-t")
    default: Any = None  # Default value


# CLI argument mappings for the 'run' command
RUN_ARG_MAPPINGS: List[ArgMapping] = [
    ArgMapping(
        cli_arg="--agent-type",
        env_var="AGENT_TYPE",
        choices=["cron", "branch"],
        help_text="Agent type: cron (scheduled), branch (review)",
        short_arg="-t",
    ),
    ArgMapping(
        cli_arg="--ai-provider",
        env_var="AI_PROVIDER",
        choices=["gemini", "openai", "claude", "azure", "ollama"],
        help_text="AI provider to use",
    ),
    ArgMapping(
        cli_arg="--ai-model",
        env_var="AI_MODEL",
        help_text="AI model name (e.g., gemini-2.5-pro, gpt-4o)",
    ),
    ArgMapping(
        cli_arg="--mcp-config",
        env_var="MCP_SERVERS_CONFIG",
        help_text="MCP servers configuration (JSON string or file path)",
    ),
    ArgMapping(
        cli_arg="--log-level",
        env_var="LOG_LEVEL",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help_text="Logging level",
        default="INFO",
    ),
    ArgMapping(
        cli_arg="--max-iters",
        env_var="MAX_INFER_ITERS",
        arg_type=int,
        help_text="Maximum inference iterations",
    ),
    ArgMapping(
        cli_arg="--task-prompt",
        env_var="AI_TASK_PROMPT",
        help_text="Custom task prompt for the agent",
    ),
]

# Sensitive environment variables that should NOT be exposed as CLI arguments
# These must be provided via .env file or environment variables only
# Imported from settings.py to maintain DRY principle
SENSITIVE_ENV_VARS: frozenset = SENSITIVE_ENV_VAR_NAMES


def get_arg_mapping_by_env_var(env_var: str) -> Optional[ArgMapping]:
    """Get an ArgMapping by its environment variable name."""
    for mapping in RUN_ARG_MAPPINGS:
        if mapping.env_var == env_var:
            return mapping
    return None


def get_arg_mapping_by_cli_arg(cli_arg: str) -> Optional[ArgMapping]:
    """Get an ArgMapping by its CLI argument name."""
    # Normalize the CLI arg (remove leading dashes, replace dashes with underscores)
    normalized = cli_arg.lstrip("-").replace("-", "_")
    for mapping in RUN_ARG_MAPPINGS:
        mapping_normalized = mapping.cli_arg.lstrip("-").replace("-", "_")
        if mapping_normalized == normalized:
            return mapping
    return None
