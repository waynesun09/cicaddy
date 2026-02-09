"""Step classes for multi-step execution following Llama Stack patterns."""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class StepType(Enum):
    """Step types for execution tracking."""

    inference = "inference"
    tool_execution = "tool_execution"


class Step(ABC):
    """Base class for execution steps."""

    def __init__(
        self,
        turn_id: str,
        step_id: str,
        step_type: StepType,
        started_at: datetime,
        completed_at: Optional[datetime] = None,
    ):
        self.turn_id = turn_id
        self.step_id = step_id
        self.step_type = step_type
        self.started_at = started_at
        self.completed_at = completed_at

    def mark_completed(self):
        """Mark step as completed."""
        self.completed_at = datetime.now()

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization."""
        pass


class InferenceStep(Step):
    """Step for AI model inference calls."""

    def __init__(
        self,
        turn_id: str,
        step_id: str,
        step_type: StepType,
        started_at: datetime,
        model_response: Optional[str] = None,
        model_used: Optional[str] = None,
        usage_stats: Optional[Dict[str, Any]] = None,
        completed_at: Optional[datetime] = None,
    ):
        super().__init__(turn_id, step_id, step_type, started_at, completed_at)
        self.model_response = model_response
        self.model_used = model_used
        self.usage_stats = usage_stats

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "model_response": self.model_response,
            "model_used": self.model_used,
            "usage_stats": self.usage_stats,
        }


class ToolExecutionStep(Step):
    """Step for MCP tool execution."""

    def __init__(
        self,
        turn_id: str,
        step_id: str,
        step_type: StepType,
        started_at: datetime,
        tool_name: str,
        tool_server: str,
        tool_arguments: Dict[str, Any],
        tool_response: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        completed_at: Optional[datetime] = None,
    ):
        super().__init__(turn_id, step_id, step_type, started_at, completed_at)
        self.tool_name = tool_name
        self.tool_server = tool_server
        self.tool_arguments = tool_arguments
        self.tool_response = tool_response
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "tool_name": self.tool_name,
            "tool_server": self.tool_server,
            "tool_arguments": self.tool_arguments,
            "tool_response": self.tool_response,
            "error": self.error,
        }
