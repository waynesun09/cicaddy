"""Tests for cicaddy.delegation.orchestrator module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cicaddy.delegation.orchestrator import DelegationOrchestrator, DelegationResult
from cicaddy.delegation.registry import SubAgentSpec
from cicaddy.delegation.triage import DelegationEntry, DelegationPlan


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.ai_provider = "gemini"
    settings.ai_model = "gemini-2.0-flash"
    settings.sub_agent_max_iters = 5
    settings.max_execution_time = 600
    settings.context_safety_factor = 0.85
    settings.review_prompt = None
    return settings


@pytest.fixture
def sample_registry():
    return {
        "security-reviewer": SubAgentSpec(
            name="security-reviewer",
            persona="security expert",
            description="Reviews security",
            categories=["security"],
            agent_type="review",
        ),
        "general-reviewer": SubAgentSpec(
            name="general-reviewer",
            persona="senior engineer",
            description="General review",
            categories=["architecture"],
            agent_type="review",
        ),
    }


@pytest.fixture
def sample_plan():
    return DelegationPlan(
        entries=[
            DelegationEntry(
                agent_name="security-reviewer",
                categories=["security"],
                rationale="Security changes detected",
                priority=10,
            ),
            DelegationEntry(
                agent_name="general-reviewer",
                categories=["architecture"],
                rationale="General review needed",
                priority=100,
            ),
        ],
        context_summary="Test changes",
        estimated_complexity="medium",
    )


@pytest.fixture
def sample_context():
    return {"project": {"name": "test"}, "diff": "some diff"}


class TestDelegationResult:
    """Tests for DelegationResult dataclass."""

    def test_defaults(self):
        result = DelegationResult()
        assert result.agent_results == []
        assert result.aggregated_analysis == ""
        assert result.delegation_plan is None
        assert result.total_execution_time == 0.0
        assert result.agents_succeeded == 0
        assert result.agents_failed == 0
        assert result.categories_covered == []


class TestDelegationOrchestrator:
    """Tests for DelegationOrchestrator."""

    def test_init(self, mock_settings):
        orch = DelegationOrchestrator(mock_settings, max_concurrent=5)
        assert orch.max_concurrent == 5

    @pytest.mark.asyncio
    async def test_execute_success(
        self, mock_settings, sample_plan, sample_registry, sample_context
    ):
        """All agents succeed."""
        orch = DelegationOrchestrator(mock_settings, max_concurrent=3)

        mock_sub_agent = MagicMock()
        mock_sub_agent.initialize = AsyncMock()
        mock_sub_agent.execute = AsyncMock(
            return_value={
                "agent_name": "test",
                "status": "success",
                "analysis": "All good",
                "categories": ["security"],
                "rationale": "test",
                "execution_time": 1.5,
                "tokens": 100,
            }
        )
        mock_sub_agent.cleanup = AsyncMock()

        with patch(
            "cicaddy.delegation.orchestrator.DelegationSubAgent",
            return_value=mock_sub_agent,
        ):
            result = await orch.execute(
                plan=sample_plan,
                registry=sample_registry,
                context=sample_context,
                parent_tools=[],
                mcp_manager=None,
                local_registry=None,
            )

        assert result.agents_succeeded == 2
        assert result.agents_failed == 0
        assert len(result.agent_results) == 2
        assert result.total_execution_time >= 0  # mocks run instantly

    @pytest.mark.asyncio
    async def test_execute_partial_failure(
        self, mock_settings, sample_registry, sample_context
    ):
        """One agent fails, one succeeds."""
        plan = DelegationPlan(
            entries=[
                DelegationEntry(
                    agent_name="security-reviewer",
                    categories=["security"],
                    rationale="sec",
                ),
                DelegationEntry(
                    agent_name="general-reviewer", categories=["arch"], rationale="gen"
                ),
            ],
        )

        call_count = 0

        async def mock_execute():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "agent_name": "security-reviewer",
                    "status": "success",
                    "analysis": "No issues",
                    "categories": ["security"],
                    "rationale": "sec",
                    "execution_time": 1.0,
                    "tokens": 50,
                }
            return {
                "agent_name": "general-reviewer",
                "status": "failed",
                "analysis": "Error occurred",
                "categories": ["arch"],
                "rationale": "gen",
                "execution_time": 0.5,
                "tokens": 0,
            }

        mock_sub_agent = MagicMock()
        mock_sub_agent.initialize = AsyncMock()
        mock_sub_agent.execute = AsyncMock(side_effect=mock_execute)
        mock_sub_agent.cleanup = AsyncMock()

        orch = DelegationOrchestrator(mock_settings, max_concurrent=3)
        with patch(
            "cicaddy.delegation.orchestrator.DelegationSubAgent",
            return_value=mock_sub_agent,
        ):
            result = await orch.execute(
                plan=plan,
                registry=sample_registry,
                context=sample_context,
                parent_tools=[],
                mcp_manager=None,
                local_registry=None,
            )

        assert result.agents_succeeded == 1
        assert result.agents_failed == 1

    @pytest.mark.asyncio
    async def test_execute_exception_handled(
        self, mock_settings, sample_registry, sample_context
    ):
        """Agent raising exception should be caught and reported."""
        plan = DelegationPlan(
            entries=[
                DelegationEntry(
                    agent_name="security-reviewer",
                    categories=["security"],
                    rationale="test",
                ),
            ],
        )

        mock_sub_agent = MagicMock()
        mock_sub_agent.initialize = AsyncMock()
        mock_sub_agent.execute = AsyncMock(side_effect=RuntimeError("crash"))
        mock_sub_agent.cleanup = AsyncMock()

        orch = DelegationOrchestrator(mock_settings, max_concurrent=3)
        with patch(
            "cicaddy.delegation.orchestrator.DelegationSubAgent",
            return_value=mock_sub_agent,
        ):
            result = await orch.execute(
                plan=plan,
                registry=sample_registry,
                context=sample_context,
                parent_tools=[],
                mcp_manager=None,
                local_registry=None,
            )

        # Exception should be caught, not propagated
        assert result.agents_failed == 1
        assert result.agents_succeeded == 0

    @pytest.mark.asyncio
    async def test_execute_skips_unknown_agent(self, mock_settings, sample_context):
        """Agent not in registry should be skipped."""
        plan = DelegationPlan(
            entries=[
                DelegationEntry(
                    agent_name="nonexistent", categories=["x"], rationale="test"
                ),
            ],
        )

        orch = DelegationOrchestrator(mock_settings, max_concurrent=3)
        result = await orch.execute(
            plan=plan,
            registry={},  # empty registry
            context=sample_context,
            parent_tools=[],
            mcp_manager=None,
            local_registry=None,
        )

        assert len(result.agent_results) == 1
        assert result.agent_results[0]["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_delegation_plan_passthrough(
        self, mock_settings, sample_plan, sample_registry, sample_context
    ):
        """DelegationResult should carry the original plan."""
        mock_sub_agent = MagicMock()
        mock_sub_agent.initialize = AsyncMock()
        mock_sub_agent.execute = AsyncMock(
            return_value={
                "agent_name": "test",
                "status": "success",
                "analysis": "ok",
                "categories": [],
                "rationale": "",
                "execution_time": 0.1,
                "tokens": 10,
            }
        )
        mock_sub_agent.cleanup = AsyncMock()

        orch = DelegationOrchestrator(mock_settings)
        with patch(
            "cicaddy.delegation.orchestrator.DelegationSubAgent",
            return_value=mock_sub_agent,
        ):
            result = await orch.execute(
                plan=sample_plan,
                registry=sample_registry,
                context=sample_context,
                parent_tools=[],
                mcp_manager=None,
                local_registry=None,
            )

        assert result.delegation_plan is sample_plan


class TestAggregateResults:
    """Tests for DelegationOrchestrator._aggregate_results()."""

    def test_empty_results(self, mock_settings):
        orch = DelegationOrchestrator(mock_settings)
        output = orch._aggregate_results([])
        assert output == "No sub-agent results available."

    def test_single_success(self, mock_settings):
        orch = DelegationOrchestrator(mock_settings)
        results = [
            {
                "agent_name": "security-reviewer",
                "status": "success",
                "analysis": "No vulnerabilities found.",
                "execution_time": 2.5,
            }
        ]
        output = orch._aggregate_results(results)
        assert "## security-reviewer" in output
        assert "No vulnerabilities found." in output
        assert "1 agent(s) succeeded" in output

    def test_skipped_not_included(self, mock_settings):
        orch = DelegationOrchestrator(mock_settings)
        results = [
            {
                "agent_name": "skipped-agent",
                "status": "skipped",
                "analysis": "",
                "execution_time": 0,
            },
        ]
        output = orch._aggregate_results(results)
        assert output == "No sub-agent results available."

    def test_failed_shown_with_status(self, mock_settings):
        orch = DelegationOrchestrator(mock_settings)
        results = [
            {
                "agent_name": "broken-agent",
                "status": "error",
                "analysis": "Something went wrong",
                "execution_time": 0.1,
            }
        ]
        output = orch._aggregate_results(results)
        assert "## broken-agent (error)" in output

    def test_multiple_agents_separated(self, mock_settings):
        orch = DelegationOrchestrator(mock_settings)
        results = [
            {
                "agent_name": "agent-a",
                "status": "success",
                "analysis": "A output",
                "execution_time": 1,
            },
            {
                "agent_name": "agent-b",
                "status": "success",
                "analysis": "B output",
                "execution_time": 2,
            },
        ]
        output = orch._aggregate_results(results)
        assert "## agent-a" in output
        assert "## agent-b" in output
        assert "---" in output
        assert "2 agent(s) succeeded" in output

    def test_summary_footer_includes_agent_names(self, mock_settings):
        orch = DelegationOrchestrator(mock_settings)
        results = [
            {
                "agent_name": "sec",
                "status": "success",
                "analysis": "ok",
                "execution_time": 1,
            },
            {
                "agent_name": "gen",
                "status": "failed",
                "analysis": "err",
                "execution_time": 0.5,
            },
        ]
        output = orch._aggregate_results(results)
        assert "Agents: sec, gen" in output
        assert "1 failed" in output
