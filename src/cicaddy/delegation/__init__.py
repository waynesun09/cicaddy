"""General-purpose sub-agent delegation framework for cicaddy.

Provides AI-powered triage, sub-agent spawning, and result aggregation
that any agent type (review, task, cron) can use via delegation hooks
in BaseAIAgent.
"""

from cicaddy.delegation.orchestrator import DelegationOrchestrator, DelegationResult
from cicaddy.delegation.registry import SubAgentRegistry, SubAgentSpec
from cicaddy.delegation.sub_agent import DelegationSubAgent
from cicaddy.delegation.triage import (
    DelegationEntry,
    DelegationPlan,
    SiblingInfo,
    TriageAgent,
)

__all__ = [
    "DelegationEntry",
    "DelegationOrchestrator",
    "DelegationPlan",
    "DelegationResult",
    "DelegationSubAgent",
    "SiblingInfo",
    "SubAgentRegistry",
    "SubAgentSpec",
    "TriageAgent",
]
