"""Centralized environment variable substitution utility.

Provides a single implementation of {{VAR_NAME}} and {{VAR_NAME:default}}
pattern substitution, used by both the legacy prompt path and the DSPy
task loader.
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

# Pattern for environment variable substitution:
# {{VAR_NAME}} or {{VAR_NAME:default_value}}
ENV_VAR_PATTERN = re.compile(r"\{\{([A-Z_][A-Z0-9_]*?)(?::([^}]*))?\}\}")


def substitute_env_variables(content: str) -> str:
    """Replace {{VAR_NAME}} or {{VAR_NAME:default}} placeholders with env values.

    Supports two syntaxes:
    - ``{{VAR_NAME}}`` – replaced by the environment variable value, or left
      as-is with a warning if the variable is not set.
    - ``{{VAR_NAME:default}}`` – replaced by the environment variable value,
      falling back to *default* when the variable is not set.

    Args:
        content: String content with potential variable placeholders.

    Returns:
        Content with environment variables substituted.
    """
    replacements: list[tuple[str, str]] = []

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        default_value = match.group(2)  # May be None

        value = os.getenv(var_name)
        if value is not None:
            replacements.append((var_name, value))
            return value
        if default_value is not None:
            replacements.append((var_name, default_value))
            return default_value

        logger.warning(f"Environment variable {var_name} not set and no default")
        return match.group(0)

    result = ENV_VAR_PATTERN.sub(_replace, content)

    if replacements:
        logger.info(
            f"Completed {len(replacements)} variable substitution(s)",
            extra={"replacements": [(n, v) for n, v in replacements]},
        )
    logger.debug(f"Content after variable substitution:\n{result}")

    return result
