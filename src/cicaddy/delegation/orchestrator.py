"""Parallel delegation orchestrator and result aggregation.

Spawns DelegationSubAgent instances in parallel (with semaphore for
rate limit safety) and aggregates results into unified output.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from cicaddy.delegation.sub_agent import DelegationSubAgent
from cicaddy.delegation.triage import DelegationPlan, SiblingInfo
from cicaddy.utils.logger import get_logger

if TYPE_CHECKING:
    from cicaddy.ai_providers.base import BaseProvider
    from cicaddy.config.settings import Settings
    from cicaddy.delegation.registry import SubAgentSpec
    from cicaddy.delegation.summarizer import Finding
    from cicaddy.mcp_client.client import OfficialMCPClientManager
    from cicaddy.skills import SkillMetadata
    from cicaddy.tools import ToolRegistry

logger = get_logger(__name__)


@dataclass
class DelegationResult:
    """Result of delegated sub-agent execution."""

    agent_results: List[Dict[str, Any]] = field(default_factory=list)
    aggregated_analysis: str = ""
    delegation_plan: Optional[DelegationPlan] = None
    total_execution_time: float = 0.0
    agents_succeeded: int = 0
    agents_failed: int = 0
    categories_covered: List[str] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)
    summarized: bool = False


class DelegationOrchestrator:
    """Orchestrates parallel sub-agent execution and result aggregation.

    Uses asyncio.Semaphore to limit concurrency for API rate safety.
    Handles partial failures gracefully — failed agents show errors
    while successful agents' results are preserved.
    """

    def __init__(
        self,
        settings: "Settings",
        max_concurrent: int = 3,
        ai_provider: Optional["BaseProvider"] = None,
    ):
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        self.settings = settings
        self.max_concurrent = max_concurrent
        self.ai_provider = ai_provider

    async def execute(
        self,
        plan: DelegationPlan,
        registry: Dict[str, "SubAgentSpec"],
        context: Dict[str, Any],
        parent_tools: List[Dict[str, Any]],
        mcp_manager: Optional["OfficialMCPClientManager"],
        local_registry: Optional["ToolRegistry"],
        bundled_context: str = "",
        agent_rules: str = "",
        skills: Optional[List["SkillMetadata"]] = None,
        summarize_results: bool = False,
        summarization_prompt: str = "",
    ) -> DelegationResult:
        """Execute delegation plan by spawning sub-agents in parallel.

        Args:
            plan: The triage-generated delegation plan.
            registry: Available sub-agent specs.
            context: Full context dict from parent agent.
            parent_tools: Parent's collected tool list.
            mcp_manager: Parent's MCP client manager (shared).
            local_registry: Parent's local tool registry (shared).
            bundled_context: Pre-rendered bundled skills text from parent.
            agent_rules: Per-repo rules (AGENT.md/CLAUDE.md/GEMINI.md) from parent.
            skills: Per-repo skill metadata from parent.
            summarize_results: Whether to use AI summarization for 2+ agents.
            summarization_prompt: Optional custom instructions for the summarizer.

        Returns:
            DelegationResult with aggregated analysis and per-agent results.
        """
        start = time.monotonic()
        num_agents = len(plan.entries)

        logger.info(
            f"Starting delegation: {num_agents} sub-agents, "
            f"max_concurrent={self.max_concurrent}"
        )

        semaphore = asyncio.Semaphore(self.max_concurrent)
        all_agent_info = []
        for entry in plan.entries:
            spec = registry.get(entry.agent_name)
            if spec:
                all_agent_info.append(
                    SiblingInfo(
                        name=entry.agent_name,
                        categories=list(spec.categories),
                    )
                )
            else:
                logger.warning(
                    f"Triage selected '{entry.agent_name}' but spec not in registry, "
                    f"excluding from sibling list"
                )

        async def _run_agent(entry):
            async with semaphore:
                spec = registry.get(entry.agent_name)
                if not spec:
                    logger.warning(
                        f"Agent '{entry.agent_name}' not in registry, skipping"
                    )
                    return {
                        "agent_name": entry.agent_name,
                        "status": "skipped",
                        "analysis": f"Agent '{entry.agent_name}' not found in registry",
                        "categories": entry.categories,
                        "rationale": entry.rationale,
                        "execution_time": 0,
                        "tokens": 0,
                    }

                agent = DelegationSubAgent(
                    spec=spec,
                    delegation_entry=entry,
                    settings=self.settings,
                    context=context,
                    parent_tools=parent_tools,
                    parent_mcp_manager=mcp_manager,
                    parent_local_registry=local_registry,
                    sibling_agents=all_agent_info,
                    bundled_context=bundled_context,
                    agent_rules=agent_rules,
                    skills=skills,
                )

                try:
                    await agent.initialize(num_agents=num_agents)
                    result = await agent.execute()
                    return result
                finally:
                    await agent.cleanup()

        # Run all agents in parallel with semaphore
        tasks = [_run_agent(entry) for entry in plan.entries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        agent_results = []
        succeeded = 0
        failed = 0
        all_categories = set()

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                entry = plan.entries[i]
                logger.error(
                    f"Sub-agent '{entry.agent_name}' raised exception: {result}"
                )
                agent_results.append(
                    {
                        "agent_name": entry.agent_name,
                        "status": "error",
                        "analysis": f"Unexpected error: {result}",
                        "categories": entry.categories,
                        "rationale": entry.rationale,
                        "execution_time": 0,
                        "tokens": 0,
                    }
                )
                failed += 1
            else:
                agent_results.append(result)
                if result.get("status") == "success":
                    succeeded += 1
                    all_categories.update(result.get("categories", []))
                elif result.get("status") == "skipped":
                    pass  # Don't count skipped as failed
                else:
                    failed += 1

        elapsed = time.monotonic() - start

        aggregated, findings, summarized = await self._aggregate_results(
            agent_results,
            summarize=summarize_results,
            summarization_prompt=summarization_prompt,
        )

        logger.info(
            f"Delegation complete: {succeeded} succeeded, {failed} failed, "
            f"{elapsed:.2f}s total"
        )

        return DelegationResult(
            agent_results=agent_results,
            aggregated_analysis=aggregated,
            delegation_plan=plan,
            total_execution_time=round(elapsed, 2),
            agents_succeeded=succeeded,
            agents_failed=failed,
            categories_covered=sorted(all_categories),
            findings=findings,
            summarized=summarized,
        )

    async def _aggregate_results(
        self,
        results: List[Dict[str, Any]],
        summarize: bool = False,
        summarization_prompt: str = "",
    ) -> tuple[str, list, bool]:
        """Aggregate sub-agent results into unified markdown output.

        When summarize=True and an AI provider is available with 2+
        successful results, uses AI-powered summarization. Otherwise
        falls back to deterministic structured concatenation.

        Returns:
            Tuple of (aggregated_analysis, findings, summarized).
        """
        num_successful = sum(1 for r in results if r.get("status") == "success")

        if summarize and self.ai_provider and num_successful >= 2:
            from cicaddy.delegation.summarizer import SummarizationAgent

            summarizer = SummarizationAgent(self.ai_provider)
            result = await summarizer.summarize(results, summarization_prompt)

            # Assemble: summary + individual sections + footer
            parts = [result.summary]
            if result.individual_sections:
                parts.append(result.individual_sections)
            if result.footer:
                parts.append(f"\n---\n\n{result.footer}")

            return (
                "\n\n".join(parts),
                result.findings,
                bool(result.findings or result.summary),
            )

        # Deterministic concatenation (original behavior)
        return self._concat_results(results), [], False

    @staticmethod
    def _concat_results(results: List[Dict[str, Any]]) -> str:
        """Deterministic structured concatenation of sub-agent results."""
        sections = []

        for result in results:
            agent_name = result.get("agent_name", "Unknown")
            status = result.get("status", "unknown")
            analysis = result.get("analysis", "")

            if status == "skipped":
                continue

            header = f"## {agent_name}"
            if status != "success":
                header += f" ({status})"

            sections.append(f"{header}\n\n{analysis}")

        if not sections:
            return "No sub-agent results available."

        body = "\n\n---\n\n".join(sections)

        # Add delegation summary footer
        succeeded = sum(1 for r in results if r.get("status") == "success")
        failed = sum(
            1 for r in results if r.get("status") not in ("success", "skipped")
        )
        total_time = sum(r.get("execution_time", 0) for r in results)
        agent_names = [r["agent_name"] for r in results if r.get("status") != "skipped"]

        footer = f"\n\n---\n\n*Delegation summary: {succeeded} agent(s) succeeded"
        if failed:
            footer += f", {failed} failed"
        footer += (
            f" | Agents: {', '.join(agent_names)}"
            f" | Total sub-agent time: {total_time:.1f}s*"
        )

        return body + footer
