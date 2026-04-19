"""Lightweight sub-agent executor for delegation.

Each DelegationSubAgent receives a focused context subset, filtered tools,
and a reduced token budget. It shares the parent's MCP connections and
local tool registry for actual tool execution.
"""

from __future__ import annotations

import importlib.metadata
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from cicaddy.ai_providers.base import ProviderMessage
from cicaddy.ai_providers.factory import create_provider, get_provider_config
from cicaddy.delegation.triage import (
    DelegationEntry,
    SiblingInfo,
    _make_boundary_pair,
    _sanitize_for_boundary,
)
from cicaddy.utils.logger import get_logger

if TYPE_CHECKING:
    from cicaddy.config.settings import Settings
    from cicaddy.delegation.registry import SubAgentSpec
    from cicaddy.execution.engine import ExecutionEngine
    from cicaddy.mcp_client.client import OfficialMCPClientManager
    from cicaddy.skills import SkillMetadata
    from cicaddy.tools import ToolRegistry

logger = get_logger(__name__)

# Base-level blocked tools (only recursion prevention — no plugin-specific names)
BASE_BLOCKED_TOOLS: Set[str] = {"delegate_task"}

# Cache for plugin-contributed blocked tools
_plugin_blocked_tools: Optional[Set[str]] = None


def collect_blocked_tools() -> Set[str]:
    """Collect blocked tools from base + all installed plugins.

    Plugins register their blocked tools via the
    ``cicaddy.delegation_blocked_tools`` entry point group.
    Each entry point should be a callable returning a Set[str].
    """
    global _plugin_blocked_tools
    if _plugin_blocked_tools is not None:
        return _plugin_blocked_tools

    blocked = set(BASE_BLOCKED_TOOLS)
    eps = importlib.metadata.entry_points()
    delegation_eps = (
        eps.select(group="cicaddy.delegation_blocked_tools")
        if hasattr(eps, "select")
        else eps.get("cicaddy.delegation_blocked_tools", [])
    )
    for ep in delegation_eps:
        try:
            plugin_blocked = ep.load()()
            blocked.update(plugin_blocked)
            logger.debug(
                f"Loaded delegation blocked tools from plugin '{ep.name}': "
                f"{len(plugin_blocked)} tools"
            )
        except Exception as e:
            logger.warning(
                f"Failed to load delegation blocked tools from '{ep.name}': {e}"
            )

    _plugin_blocked_tools = blocked
    logger.info(f"Total blocked tools for sub-agents: {len(blocked)}")
    return blocked


class DelegationSubAgent:
    """Lightweight sub-agent that executes a focused analysis task.

    Does NOT create new MCP connections — shares the parent's backends.
    Creates its own AI provider instance with a reduced token budget.
    """

    def __init__(
        self,
        spec: "SubAgentSpec",
        delegation_entry: DelegationEntry,
        settings: "Settings",
        context: Dict[str, Any],
        parent_tools: List[Dict[str, Any]],
        parent_mcp_manager: Optional["OfficialMCPClientManager"],
        parent_local_registry: Optional["ToolRegistry"],
        sibling_agents: Optional[List[SiblingInfo]] = None,
        bundled_context: str = "",
        agent_rules: str = "",
        skills: Optional[List["SkillMetadata"]] = None,
    ):
        # bundled_context, agent_rules, and skills are pre-scanned for prompt
        # injection by the parent agent's initialize(). Do not re-load these
        # from disk in sub-agents — always receive them from the parent.
        self.spec = spec
        self.delegation_entry = delegation_entry
        self.settings = settings
        self.context = context
        self.parent_tools = parent_tools
        self.parent_mcp_manager = parent_mcp_manager
        self.parent_local_registry = parent_local_registry
        self.sibling_agents = sibling_agents or []
        self.bundled_context = bundled_context
        self.agent_rules = agent_rules
        self.skills = skills or []

        self.ai_provider = None
        self.execution_engine: Optional["ExecutionEngine"] = None

    async def initialize(self, num_agents: int = 1) -> None:
        """Initialize AI provider and execution engine with reduced budget."""
        from cicaddy.execution.engine import ExecutionEngine
        from cicaddy.execution.token_aware_executor import ExecutionLimits
        from cicaddy.utils.token_utils import TokenLimitManager

        provider_config = get_provider_config(self.settings)
        provider_name = self.settings.ai_provider or "gemini"
        self.ai_provider = create_provider(provider_name, provider_config)
        await self.ai_provider.initialize()

        # Reduced budget: divide parent's budget by number of agents
        from cicaddy.ai_providers.factory import DEFAULT_AI_PROVIDER, get_default_model

        provider = self.settings.ai_provider or DEFAULT_AI_PROVIDER
        model = self.settings.ai_model or get_default_model(provider)
        token_limits = TokenLimitManager.get_limits(provider, model)

        max_iters = getattr(self.settings, "sub_agent_max_iters", 5)
        budget_fraction = max(1, num_agents)
        max_tokens_total = token_limits["input"] // budget_fraction

        per_iter = max(4096, int(max_tokens_total * 0.0625))
        # Ensure per-iteration limit never exceeds total budget
        per_iter = min(per_iter, max_tokens_total)

        execution_limits = ExecutionLimits(
            max_infer_iters=max_iters,
            max_tokens_total=max_tokens_total,
            max_tokens_per_iteration=per_iter,
            max_tokens_per_tool_result=max(1024, int(token_limits["output"] * 0.25)),
            max_execution_time=min(
                300,
                getattr(self.settings, "max_execution_time", 600) // budget_fraction,
            ),
        )

        self.execution_engine = ExecutionEngine(
            ai_provider=self.ai_provider,
            mcp_manager=self.parent_mcp_manager,
            local_tool_registry=self.parent_local_registry,
            session_id=f"delegation-{self.spec.name}",
            execution_limits=execution_limits,
            context_safety_factor=self.settings.context_safety_factor,
        )

        logger.info(
            f"Sub-agent '{self.spec.name}' initialized: "
            f"max_iters={max_iters}, "
            f"max_tokens={max_tokens_total:,}, "
            f"tools={len(self._filter_tools(self.parent_tools))}"
        )

    def _filter_tools(self, parent_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter parent tools to a safe subset for this sub-agent."""
        blocked = collect_blocked_tools()

        tools = [t for t in parent_tools if t["name"] not in blocked]

        if self.spec.blocked_tools:
            spec_blocked = set(self.spec.blocked_tools)
            tools = [t for t in tools if t["name"] not in spec_blocked]

        if self.spec.allowed_tools:
            spec_allowed = set(self.spec.allowed_tools)
            tools = [t for t in tools if t["name"] in spec_allowed]

        return sorted(tools, key=lambda t: t.get("name", ""))

    def _build_prompt(self) -> str:
        """Build the focused prompt for this sub-agent."""
        # Get relevant context subset
        relevant_context = self._get_relevant_context()

        # Generate nonce-based boundary markers for prompt injection protection
        boundary_start, boundary_end = _make_boundary_pair()

        # Build context data section with boundary markers
        context_parts = []
        for key, value in relevant_context.items():
            if isinstance(value, str):
                sanitized = _sanitize_for_boundary(value, boundary_start, boundary_end)
                context_parts.append(f"### {key}\n{sanitized}")
            else:
                context_parts.append(f"### {key}\n{value!s}")

        context_section = "\n\n".join(context_parts)

        # Build constraints section
        constraints_text = ""
        if self.spec.constraints:
            constraints_list = "\n".join(f"- {c}" for c in self.spec.constraints)
            constraints_text = f"\n## Constraints\n{constraints_list}\n"

        # Build output sections
        output_text = ""
        if self.spec.output_sections:
            sections_list = "\n".join(f"- {s}" for s in self.spec.output_sections)
            output_text = f"\n## Expected Output Sections\n{sections_list}\n"

        # Include user custom prompt if set (sanitize against boundary injection)
        user_prompt = ""
        review_prompt = getattr(self.settings, "review_prompt", None)
        if review_prompt:
            sanitized_prompt = _sanitize_for_boundary(
                review_prompt, boundary_start, boundary_end
            )
            user_prompt = f"\n## Additional Instructions\n{sanitized_prompt}\n"

        # Build delegation context so the agent knows if it's solo or combined
        my_name = self.delegation_entry.agent_name
        siblings = [s for s in self.sibling_agents if s.name != my_name]
        # Deduplicate by name, preserving order
        seen: set[str] = set()
        unique_siblings: list[SiblingInfo] = []
        for s in siblings:
            if s.name not in seen:
                seen.add(s.name)
                unique_siblings.append(s)

        if unique_siblings:
            parts = []
            for s in unique_siblings:
                if s.categories:
                    parts.append(f"{s.name} ({', '.join(s.categories)})")
                else:
                    parts.append(s.name)
            sibling_list = "; ".join(parts)
            delegation_text = (
                f"\n## Delegation Context\n"
                f"You are running alongside these other agents: {sibling_list}.\n"
                f"They will cover their listed categories. "
                f"Focus on aspects they do not cover and avoid duplicating their work.\n"
            )
        else:
            delegation_text = (
                "\n## Delegation Context\n"
                "You are the sole reviewer for this change. "
                "Provide comprehensive coverage across all relevant aspects.\n"
            )

        core_prompt = f"""You are a {self.spec.persona}.

## Your Role
{self.spec.description}

## Focus Areas
Categories: {", ".join(self.delegation_entry.categories)}
Rationale: {self.delegation_entry.rationale}
{constraints_text}{output_text}{delegation_text}{user_prompt}
## Code References

When identifying issues in code diffs, quote the relevant code snippet (1-3 \
lines) so findings can be precisely located. For example:

```
problematic_function(arg, None)
```
Missing null check before calling `problematic_function`.

This is more useful than citing line numbers.

## Context

{boundary_start}
{context_section}
{boundary_end}

Analyze the context above according to your role and focus areas. Provide structured, actionable findings."""

        # Layer workspace context following the same pattern as parent agents:
        # bundled_context → agent_rules → core prompt → skills
        sections: list[str] = []
        if self.bundled_context:
            sections.append(self.bundled_context)
        if self.agent_rules:
            sections.append(self.agent_rules)
        sections.append(core_prompt)
        if self.skills:
            from cicaddy.skills import render_skills_prompt

            skills_section = render_skills_prompt(self.skills)
            if skills_section:
                sections.append(skills_section)

        return "\n\n".join(sections)

    def _get_relevant_context(self) -> Dict[str, Any]:
        """Filter context to what this sub-agent needs."""
        if not self.delegation_entry.relevant_context_keys:
            return self.context

        relevant = {}
        for key in self.delegation_entry.relevant_context_keys:
            if key in self.context:
                relevant[key] = self.context[key]

        # Always include project info if available
        if "project" in self.context and "project" not in relevant:
            relevant["project"] = self.context["project"]

        return relevant if relevant else self.context

    async def execute(self) -> Dict[str, Any]:
        """Execute the sub-agent analysis.

        Returns:
            Dict with agent_name, categories, rationale, analysis,
            status, execution_time, and tokens.
        """
        start = time.monotonic()

        try:
            prompt = self._build_prompt()
            filtered_tools = self._filter_tools(self.parent_tools)

            messages = [ProviderMessage(content=prompt, role="user")]
            turn = await self.execution_engine.execute_turn(
                messages=messages,
                available_tools=filtered_tools,
                max_infer_iters=getattr(self.settings, "sub_agent_max_iters", 5),
            )

            elapsed = time.monotonic() - start

            return {
                "agent_name": self.spec.name,
                "categories": self.delegation_entry.categories,
                "rationale": self.delegation_entry.rationale,
                "analysis": turn.output_message,
                "status": "success",
                "execution_time": round(elapsed, 2),
                "tokens": turn.execution_summary.get("total_tokens", 0)
                if turn.execution_summary
                else 0,
            }

        except Exception as e:
            elapsed = time.monotonic() - start
            logger.error(f"Sub-agent '{self.spec.name}' failed: {e}", exc_info=True)
            return {
                "agent_name": self.spec.name,
                "categories": self.delegation_entry.categories,
                "rationale": self.delegation_entry.rationale,
                "analysis": f"Sub-agent execution failed: {e}",
                "status": "failed",
                "execution_time": round(elapsed, 2),
                "tokens": 0,
            }

    async def cleanup(self) -> None:
        """Shut down AI provider. Does NOT clean up parent's MCP/registry."""
        if self.ai_provider:
            try:
                await self.ai_provider.shutdown()
            except Exception as e:
                logger.debug(f"Sub-agent cleanup error (expected): {e}")
