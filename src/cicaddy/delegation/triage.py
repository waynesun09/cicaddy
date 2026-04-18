"""AI-powered triage agent for delegation planning.

Uses the parent agent's AI provider to analyze context and determine
which sub-agents should be activated, what they should focus on, and
in what priority order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

from cicaddy.utils.logger import get_logger

if TYPE_CHECKING:
    from cicaddy.ai_providers.base import BaseProvider
    from cicaddy.delegation.registry import SubAgentSpec

logger = get_logger(__name__)

# Data boundary markers for prompt injection protection (per hermes-agent pattern)
DATA_BOUNDARY_START = "<<<BEGIN_CONTEXT_DATA>>>"
DATA_BOUNDARY_END = "<<<END_CONTEXT_DATA>>>"


@dataclass
class DelegationEntry:
    """A single sub-agent activation entry in a delegation plan."""

    agent_name: str
    categories: List[str] = field(default_factory=list)
    rationale: str = ""
    relevant_context_keys: List[str] = field(default_factory=list)
    relevant_files: List[str] = field(default_factory=list)
    priority: int = 0


@dataclass
class DelegationPlan:
    """Result of AI-powered triage: which sub-agents to activate."""

    entries: List[DelegationEntry] = field(default_factory=list)
    context_summary: str = ""
    estimated_complexity: str = "medium"


class TriageAgent:
    """AI-powered triage that analyzes context and produces a DelegationPlan.

    Uses the parent agent's AI provider (single lightweight call, no new
    provider needed) to decide which sub-agents should handle the task.
    """

    def __init__(self, ai_provider: "BaseProvider"):
        self.ai_provider = ai_provider

    async def triage(
        self,
        context: Dict[str, Any],
        available_agents: Dict[str, "SubAgentSpec"],
        triage_prompt: str = "",
    ) -> DelegationPlan:
        """Analyze context and produce a delegation plan.

        Args:
            context: Agent-type-specific context (diff for review, task
                description for task agents, etc.)
            available_agents: Registry of available sub-agent specs.
            triage_prompt: Optional user-provided triage instructions.

        Returns:
            DelegationPlan with entries for each sub-agent to activate.
        """
        prompt = self._build_triage_prompt(context, available_agents, triage_prompt)

        try:
            from cicaddy.ai_providers.base import ProviderMessage

            messages = [ProviderMessage(content=prompt, role="user")]
            response = await self.ai_provider.chat_completion(messages)

            plan = self._parse_response(response.content, available_agents)
            logger.info(
                f"Triage complete: {len(plan.entries)} sub-agents selected, "
                f"complexity={plan.estimated_complexity}"
            )
            return plan

        except Exception as e:
            logger.error(f"Triage failed, falling back to general agent: {e}")
            return self._fallback_plan(available_agents)

    def _build_triage_prompt(
        self,
        context: Dict[str, Any],
        available_agents: Dict[str, "SubAgentSpec"],
        triage_prompt: str,
    ) -> str:
        """Build the triage prompt for the AI."""
        # Format available agents
        agents_desc = []
        for name, spec in sorted(available_agents.items()):
            agents_desc.append(
                f"- **{name}**: {spec.description} "
                f"(categories: {', '.join(spec.categories)})"
            )
        agents_section = "\n".join(agents_desc)

        # Format context summary (truncate large values)
        context_keys = list(context.keys())
        context_preview = {}
        for key in context_keys:
            value = context[key]
            if isinstance(value, str) and len(value) > 500:
                context_preview[key] = f"[{len(value)} chars]"
            elif isinstance(value, (list, dict)):
                context_preview[key] = f"[{type(value).__name__}, {len(value)} items]"
            else:
                context_preview[key] = str(value)[:200]

        # Build context data section with boundary markers
        context_data = ""
        if "diff" in context:
            diff_preview = (
                context["diff"][:3000]
                if len(context.get("diff", "")) > 3000
                else context.get("diff", "")
            )
            context_data = (
                f"\n{DATA_BOUNDARY_START}\n{diff_preview}\n{DATA_BOUNDARY_END}\n"
            )
        elif context_preview:
            context_data = f"\n{DATA_BOUNDARY_START}\n{json.dumps(context_preview, indent=2)}\n{DATA_BOUNDARY_END}\n"

        user_instructions = ""
        if triage_prompt:
            user_instructions = f"\n## Additional Instructions\n{triage_prompt}\n"

        return f"""You are a triage agent. Analyze the provided context and determine which specialized sub-agents should review it.

## Available Sub-Agents
{agents_section}

## Context Keys
{json.dumps(context_keys)}

## Context Data
{context_data}
{user_instructions}
## Task

Analyze the context and produce a JSON delegation plan. Select ONLY the agents that are relevant to the content. Do NOT select agents for aspects not present in the context.

Respond with ONLY a JSON object in this exact format (no markdown, no explanation):
{{
  "context_summary": "Brief description of what the context contains",
  "estimated_complexity": "low|medium|high",
  "entries": [
    {{
      "agent_name": "name from available agents list",
      "categories": ["relevant", "categories"],
      "rationale": "Why this agent is needed for this specific context",
      "relevant_context_keys": ["keys from context this agent needs"],
      "relevant_files": ["file paths this agent should focus on, if applicable"],
      "priority": 1
    }}
  ]
}}"""

    def _parse_response(
        self,
        response_content: str,
        available_agents: Dict[str, "SubAgentSpec"],
    ) -> DelegationPlan:
        """Parse AI response into a DelegationPlan."""
        # Extract JSON from response (handle markdown code blocks)
        content = response_content.strip()
        if content.startswith("```"):
            # Remove markdown code block wrapper
            lines = content.splitlines()
            # Skip first line (```json) and last line (```)
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```") and not in_block:
                    in_block = True
                    continue
                if line.strip() == "```" and in_block:
                    break
                if in_block:
                    json_lines.append(line)
            content = "\n".join(json_lines)

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse triage JSON response: {e}")
            raise

        entries = []
        for entry_data in data.get("entries", []):
            agent_name = entry_data.get("agent_name", "")
            # Validate agent exists in registry
            if agent_name not in available_agents:
                logger.warning(
                    f"Triage selected unknown agent '{agent_name}', skipping"
                )
                continue

            entries.append(
                DelegationEntry(
                    agent_name=agent_name,
                    categories=entry_data.get("categories", []),
                    rationale=entry_data.get("rationale", ""),
                    relevant_context_keys=entry_data.get("relevant_context_keys", []),
                    relevant_files=entry_data.get("relevant_files", []),
                    priority=entry_data.get("priority", 0),
                )
            )

        if not entries:
            logger.warning("Triage produced no valid entries, using fallback")
            return self._fallback_plan(available_agents)

        # Sort by priority (lower = higher priority)
        entries.sort(key=lambda e: e.priority)

        return DelegationPlan(
            entries=entries,
            context_summary=data.get("context_summary", ""),
            estimated_complexity=data.get("estimated_complexity", "medium"),
        )

    def _fallback_plan(
        self, available_agents: Dict[str, "SubAgentSpec"]
    ) -> DelegationPlan:
        """Create fallback plan using the general agent."""
        # Find a general/catch-all agent
        general_name = None
        for name in available_agents:
            if "general" in name:
                general_name = name
                break

        if not general_name:
            # Use first available agent
            general_name = next(iter(available_agents))

        return DelegationPlan(
            entries=[
                DelegationEntry(
                    agent_name=general_name,
                    categories=available_agents[general_name].categories,
                    rationale="Fallback: triage could not determine specific agents",
                    priority=0,
                )
            ],
            context_summary="Fallback plan",
            estimated_complexity="medium",
        )
