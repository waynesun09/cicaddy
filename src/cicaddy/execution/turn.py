"""Turn class for managing multi-step execution sessions."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from cicaddy.ai_providers.base import ProviderMessage

from .knowledge_store import AccumulatedKnowledge
from .steps import Step


@dataclass
class Turn:
    """Represents a single interaction turn in an agent session."""

    turn_id: str
    session_id: str
    input_messages: List[ProviderMessage]
    started_at: datetime = field(default_factory=datetime.now)
    steps: List[Step] = field(default_factory=list)
    output_message: Optional[str] = None
    completed_at: Optional[datetime] = None
    execution_summary: Optional[Dict[str, Any]] = None
    status: str = "in_progress"
    # Data preservation: accumulated knowledge from MCP tool executions
    accumulated_knowledge: Optional[AccumulatedKnowledge] = None

    def add_step(self, step: Step):
        """Add a step to this turn."""
        self.steps.append(step)

    def mark_completed(self, output_message: str):
        """Mark turn as completed with final output."""
        self.output_message = output_message
        self.completed_at = datetime.now()
        self.status = "completed"

    def get_latest_step(self) -> Optional[Step]:
        """Get the most recent step."""
        return self.steps[-1] if self.steps else None

    def get_steps_by_type(self, step_type) -> List[Step]:
        """Get all steps of a specific type."""
        return [step for step in self.steps if step.step_type == step_type]

    def to_dict(self) -> Dict[str, Any]:
        """Convert turn to dictionary for serialization."""
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "input_messages": [
                {"role": msg.role, "content": msg.content}
                for msg in self.input_messages
            ],
            "started_at": self.started_at.isoformat(),
            "steps": [step.to_dict() for step in self.steps],
            "output_message": self.output_message,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "execution_summary": self.execution_summary,
            "status": self.status,
            "accumulated_knowledge": self.accumulated_knowledge.to_dict()
            if self.accumulated_knowledge
            else None,
        }
