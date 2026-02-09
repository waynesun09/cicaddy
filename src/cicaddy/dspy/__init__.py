"""DSPy integration for declarative task definitions.

This module provides YAML-based task configuration as an alternative to
monolithic AI_TASK_PROMPT environment variables.
"""

from cicaddy.dspy.prompt_builder import PromptBuilder
from cicaddy.dspy.task_loader import TaskLoader, TaskLoadError
from cicaddy.dspy.task_schema import (
    TaskDefinition,
    TaskInput,
    TaskOutput,
    ToolConfig,
)

__all__ = [
    "TaskDefinition",
    "TaskInput",
    "TaskOutput",
    "ToolConfig",
    "TaskLoader",
    "TaskLoadError",
    "PromptBuilder",
]
