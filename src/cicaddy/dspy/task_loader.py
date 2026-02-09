"""YAML task definition loader with validation.

Loads task definitions from YAML files and validates them against the schema.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml

from cicaddy.dspy.task_schema import TaskDefinition
from cicaddy.utils.env_substitution import substitute_env_variables

logger = logging.getLogger(__name__)


class TaskLoadError(Exception):
    """Raised when a task definition cannot be loaded."""

    pass


class TaskLoader:
    """Loads and validates YAML task definitions."""

    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """Initialize the task loader.

        Args:
            base_path: Base directory for resolving relative task file paths.
                       Defaults to current working directory.
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()

    def load(
        self, task_file: Union[str, Path]
    ) -> Tuple[TaskDefinition, Dict[str, Optional[str]]]:
        """Load a task definition from a YAML file.

        Args:
            task_file: Path to the YAML task file (absolute or relative to base_path)

        Returns:
            Tuple of (validated TaskDefinition, resolved input values dict).

        Raises:
            TaskLoadError: If the file cannot be loaded or validated
        """
        file_path = self._resolve_path(task_file)

        if not file_path.exists():
            raise TaskLoadError(f"Task file not found: {file_path}")

        if file_path.suffix.lower() not in (".yaml", ".yml"):
            raise TaskLoadError(f"Task file must be YAML (.yaml or .yml): {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_content = f.read()
        except IOError as e:
            raise TaskLoadError(f"Failed to read task file {file_path}: {e}")

        try:
            data = yaml.safe_load(raw_content)
        except yaml.YAMLError as e:
            raise TaskLoadError(f"Invalid YAML in task file {file_path}: {e}")

        if not isinstance(data, dict):
            raise TaskLoadError(
                f"Task file must contain a YAML mapping, got {type(data).__name__}"
            )

        # Substitute {{VAR}} placeholders in parsed data (string leaves only).
        # This is done *after* YAML parsing so that env var values containing
        # YAML-significant characters (colons, dashes, newlines) cannot break
        # the YAML structure.
        #
        # Two-tier env var strategy:
        #   1. {{VAR}} in string values  → inline embedding in context,
        #      description, defaults, etc.  Resolved here at load time.
        #   2. TaskInput.env_var          → structured input resolution with
        #      defaults and validation.  Resolved in _resolve_inputs().
        data = self._substitute_in_data(data)

        try:
            task = TaskDefinition(**data)
        except Exception as e:
            raise TaskLoadError(f"Invalid task definition in {file_path}: {e}")

        # Resolve input values from environment and validate required inputs
        resolved = self._resolve_inputs(task)

        logger.info(
            f"Loaded task definition: {task.name} ({task.type})",
            extra={
                "task_name": task.name,
                "task_type": task.type,
                "inputs": len(task.inputs),
                "outputs": len(task.outputs),
                "constraints": len(task.constraints),
            },
        )

        return task, resolved

    def _resolve_path(self, task_file: Union[str, Path]) -> Path:
        """Resolve task file path relative to base_path if not absolute."""
        path = Path(task_file)
        if path.is_absolute():
            return path
        return self.base_path / path

    @staticmethod
    def _substitute_in_data(data: Any) -> Any:
        """Recursively substitute {{VAR}} placeholders in parsed YAML data.

        Walks dicts and lists, applying substitute_env_variables() to
        string leaf values only.  Non-string leaves are returned as-is.
        """
        if isinstance(data, dict):
            return {k: TaskLoader._substitute_in_data(v) for k, v in data.items()}
        if isinstance(data, list):
            return [TaskLoader._substitute_in_data(item) for item in data]
        if isinstance(data, str):
            return substitute_env_variables(data)
        return data

    def _resolve_inputs(self, task: TaskDefinition) -> Dict[str, Optional[str]]:
        """Resolve input values from environment variables.

        Returns a dictionary of resolved values without mutating the
        TaskDefinition object, so the same definition can be safely
        reused across multiple invocations.

        Args:
            task: TaskDefinition to resolve inputs for.

        Returns:
            Dictionary mapping input names to their resolved values.

        Raises:
            TaskLoadError: If a required input has no value after resolution.
        """
        resolved: Dict[str, Optional[str]] = {}

        for input_spec in task.inputs:
            value = input_spec.default

            if input_spec.env_var:
                env_value = os.getenv(input_spec.env_var)
                if env_value is not None:
                    value = env_value
                    logger.debug(
                        f"Resolved input {input_spec.name} from {input_spec.env_var}"
                    )

            resolved[input_spec.name] = value

            # Enforce required inputs that should be resolvable at load time
            # (i.e. those with an env_var). Inputs without env_var are expected
            # to be supplied later via runtime context in PromptBuilder.build().
            if input_spec.required and input_spec.env_var and value is None:
                raise TaskLoadError(
                    f"Required input '{input_spec.name}' has no value "
                    f"(env_var: {input_spec.env_var}) and no default provided."
                )

        return resolved

    def get_resolved_inputs(self, task: TaskDefinition) -> Dict[str, str]:
        """Get a dictionary of resolved input values.

        Args:
            task: TaskDefinition to resolve inputs for.

        Returns:
            Dictionary mapping input names to their resolved string values.
        """
        resolved = self._resolve_inputs(task)
        return {name: value or "" for name, value in resolved.items()}
