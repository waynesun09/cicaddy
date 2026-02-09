"""Environment file loader using python-dotenv."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import dotenv_values, load_dotenv

from cicaddy.cli.arg_mapping import RUN_ARG_MAPPINGS


def load_env_file(env_file: str, override: bool = False) -> Dict[str, str]:
    """
    Load environment variables from a .env file.

    Args:
        env_file: Path to the .env file
        override: If True, override existing environment variables

    Returns:
        Dictionary of loaded environment variables

    Raises:
        FileNotFoundError: If the env file doesn't exist
    """
    env_path = Path(env_file)
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {env_file}")

    # Load into os.environ
    load_dotenv(env_path, override=override)

    # Return the values that were loaded (filter out None values)
    return {k: v for k, v in dotenv_values(env_path).items() if v is not None}


def apply_cli_args_to_env(args: Dict[str, Any]) -> Dict[str, str]:
    """
    Apply CLI arguments to environment variables.

    CLI arguments take precedence over existing environment variables.

    Args:
        args: Dictionary of CLI argument values (from argparse namespace)

    Returns:
        Dictionary of environment variables that were set
    """
    applied: Dict[str, str] = {}

    for mapping in RUN_ARG_MAPPINGS:
        # Convert CLI arg name to attribute name (--agent-type -> agent_type)
        attr_name = mapping.cli_arg.lstrip("-").replace("-", "_")
        value = args.get(attr_name)

        if value is not None:
            str_value = str(value)
            os.environ[mapping.env_var] = str_value
            applied[mapping.env_var] = str_value

    # Handle verbose flag specially
    if args.get("verbose"):
        os.environ["LOG_LEVEL"] = "DEBUG"
        applied["LOG_LEVEL"] = "DEBUG"

    return applied


def get_effective_config() -> Dict[str, Optional[str]]:
    """
    Get the effective configuration from current environment.

    Returns:
        Dictionary mapping environment variable names to their current values
    """
    config: Dict[str, Optional[str]] = {}

    # Get all mapped environment variables
    for mapping in RUN_ARG_MAPPINGS:
        config[mapping.env_var] = os.environ.get(mapping.env_var)

    # Add some additional commonly used variables
    additional_vars = [
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ]

    for var in additional_vars:
        if var not in config:
            config[var] = os.environ.get(var)

    return config


def mask_sensitive_value(value: Optional[str], visible_chars: int = 4) -> str:
    """
    Mask a sensitive value for display.

    Args:
        value: The value to mask
        visible_chars: Number of characters to show at the end

    Returns:
        Masked string (e.g., "****abcd") or "(not set)"
    """
    if value is None:
        return "(not set)"
    if len(value) <= visible_chars:
        return "*" * len(value)
    return "*" * (len(value) - visible_chars) + value[-visible_chars:]


def validate_required_env_vars(required: list[str]) -> list[str]:
    """
    Validate that required environment variables are set.

    Args:
        required: List of required environment variable names

    Returns:
        List of missing environment variable names
    """
    missing = []
    for var in required:
        if not os.environ.get(var):
            missing.append(var)
    return missing
