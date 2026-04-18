"""Tests for cicaddy.delegation.triage module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from cicaddy.delegation.registry import SubAgentSpec
from cicaddy.delegation.triage import (
    DelegationEntry,
    DelegationPlan,
    TriageAgent,
)


@pytest.fixture
def sample_registry():
    """Provide a small registry for triage tests."""
    return {
        "security-reviewer": SubAgentSpec(
            name="security-reviewer",
            persona="security expert",
            description="Reviews security-sensitive changes",
            categories=["security", "api"],
            priority=10,
            agent_type="review",
        ),
        "general-reviewer": SubAgentSpec(
            name="general-reviewer",
            persona="senior engineer",
            description="General code review",
            categories=["architecture", "tests"],
            priority=100,
            agent_type="review",
        ),
    }


@pytest.fixture
def sample_context():
    """Provide a minimal context dict."""
    return {
        "project": {"name": "test-project"},
        "diff": "diff --git a/app.py b/app.py\n+import os\n+secret = os.getenv('SECRET')",
        "changed_files": ["app.py"],
    }


class TestDelegationEntry:
    """Tests for the DelegationEntry dataclass."""

    def test_defaults(self):
        entry = DelegationEntry(agent_name="test", categories=["security"])
        assert entry.rationale == ""
        assert entry.relevant_context_keys == []
        assert entry.relevant_files == []
        assert entry.priority == 0

    def test_fields(self):
        entry = DelegationEntry(
            agent_name="sec",
            categories=["security"],
            rationale="has secrets",
            relevant_context_keys=["diff"],
            relevant_files=["app.py"],
            priority=10,
        )
        assert entry.agent_name == "sec"
        assert entry.categories == ["security"]
        assert entry.priority == 10


class TestDelegationPlan:
    """Tests for the DelegationPlan dataclass."""

    def test_defaults(self):
        plan = DelegationPlan(entries=[])
        assert plan.context_summary == ""
        assert plan.estimated_complexity == "medium"

    def test_with_entries(self):
        entry = DelegationEntry(agent_name="test", categories=["api"])
        plan = DelegationPlan(
            entries=[entry],
            context_summary="API changes detected",
            estimated_complexity="medium",
        )
        assert len(plan.entries) == 1
        assert plan.estimated_complexity == "medium"


class TestTriageAgent:
    """Tests for the TriageAgent class."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock AI provider."""
        provider = MagicMock()
        provider.chat_completion = AsyncMock()
        return provider

    @pytest.fixture
    def triage_agent(self, mock_provider):
        return TriageAgent(mock_provider)

    def test_build_triage_prompt_includes_agents(
        self, triage_agent, sample_registry, sample_context
    ):
        """Triage prompt should list available agents."""
        prompt = triage_agent._build_triage_prompt(sample_context, sample_registry, "")
        assert "security-reviewer" in prompt
        assert "general-reviewer" in prompt
        assert "Reviews security-sensitive changes" in prompt

    def test_build_triage_prompt_includes_context(
        self, triage_agent, sample_registry, sample_context
    ):
        """Triage prompt should include context data with nonce-based boundary markers."""
        prompt = triage_agent._build_triage_prompt(sample_context, sample_registry, "")
        assert "<<<BEGIN_CONTEXT_DATA_" in prompt
        assert "<<<END_CONTEXT_DATA_" in prompt

    def test_build_triage_prompt_includes_custom_prompt(
        self, triage_agent, sample_registry, sample_context
    ):
        """Custom triage prompt should be appended."""
        prompt = triage_agent._build_triage_prompt(
            sample_context, sample_registry, triage_prompt="Focus on auth"
        )
        assert "Focus on auth" in prompt

    def test_parse_response_valid_json(self, triage_agent, sample_registry):
        """Should parse valid JSON response into DelegationPlan."""
        response_json = json.dumps(
            {
                "context_summary": "Security-related changes",
                "estimated_complexity": "medium",
                "entries": [
                    {
                        "agent_name": "security-reviewer",
                        "categories": ["security"],
                        "rationale": "Contains secret handling",
                        "relevant_context_keys": ["diff"],
                        "relevant_files": ["app.py"],
                        "priority": 10,
                    }
                ],
            }
        )
        plan = triage_agent._parse_response(response_json, sample_registry)
        assert len(plan.entries) == 1
        assert plan.entries[0].agent_name == "security-reviewer"
        assert plan.estimated_complexity == "medium"

    def test_parse_response_markdown_code_block(self, triage_agent, sample_registry):
        """Should extract JSON from markdown code blocks."""
        response = '```json\n{"entries": [{"agent_name": "general-reviewer", "categories": ["tests"], "rationale": "test changes"}]}\n```'
        plan = triage_agent._parse_response(response, sample_registry)
        assert len(plan.entries) == 1
        assert plan.entries[0].agent_name == "general-reviewer"

    def test_parse_response_filters_unknown_agents(self, triage_agent, sample_registry):
        """Agents not in registry should be filtered out."""
        response_json = json.dumps(
            {
                "entries": [
                    {
                        "agent_name": "security-reviewer",
                        "categories": ["security"],
                        "rationale": "ok",
                    },
                    {
                        "agent_name": "nonexistent-agent",
                        "categories": ["x"],
                        "rationale": "bad",
                    },
                ]
            }
        )
        plan = triage_agent._parse_response(response_json, sample_registry)
        assert len(plan.entries) == 1
        assert plan.entries[0].agent_name == "security-reviewer"

    def test_parse_response_sorts_by_priority(self, triage_agent, sample_registry):
        """Entries should be sorted by priority (lower first)."""
        response_json = json.dumps(
            {
                "entries": [
                    {
                        "agent_name": "general-reviewer",
                        "categories": ["tests"],
                        "rationale": "a",
                        "priority": 100,
                    },
                    {
                        "agent_name": "security-reviewer",
                        "categories": ["security"],
                        "rationale": "b",
                        "priority": 10,
                    },
                ]
            }
        )
        plan = triage_agent._parse_response(response_json, sample_registry)
        assert plan.entries[0].agent_name == "security-reviewer"
        assert plan.entries[1].agent_name == "general-reviewer"

    def test_fallback_plan_uses_general_agent(self, triage_agent, sample_registry):
        """Fallback should use the 'general-*' agent if available."""
        plan = triage_agent._fallback_plan(sample_registry)
        assert len(plan.entries) == 1
        assert plan.entries[0].agent_name == "general-reviewer"

    def test_fallback_plan_uses_first_agent_when_no_general(self, triage_agent):
        """Fallback should use the first agent if no 'general' agent exists."""
        registry = {
            "custom-agent": SubAgentSpec(
                name="custom-agent",
                persona="custom",
                description="custom",
                categories=["misc"],
            )
        }
        plan = triage_agent._fallback_plan(registry)
        assert len(plan.entries) == 1
        assert plan.entries[0].agent_name == "custom-agent"

    def test_parse_response_invalid_json_raises(self, triage_agent, sample_registry):
        """Invalid JSON should raise JSONDecodeError (caller handles fallback)."""
        with pytest.raises(json.JSONDecodeError):
            triage_agent._parse_response("not valid json at all", sample_registry)

    @pytest.mark.asyncio
    async def test_triage_calls_ai_provider(
        self, triage_agent, mock_provider, sample_registry, sample_context
    ):
        """triage() should call AI provider and parse response."""
        mock_response = MagicMock()
        mock_response.content = json.dumps(
            {
                "entries": [
                    {
                        "agent_name": "security-reviewer",
                        "categories": ["security"],
                        "rationale": "test",
                    }
                ]
            }
        )
        mock_provider.chat_completion.return_value = mock_response

        plan = await triage_agent.triage(sample_context, sample_registry)
        mock_provider.chat_completion.assert_called_once()
        assert len(plan.entries) == 1

    @pytest.mark.asyncio
    async def test_triage_fallback_on_provider_error(
        self, triage_agent, mock_provider, sample_registry, sample_context
    ):
        """triage() should return fallback plan on AI provider failure."""
        mock_provider.chat_completion.side_effect = Exception("API error")

        plan = await triage_agent.triage(sample_context, sample_registry)
        assert len(plan.entries) == 1
        assert "general" in plan.entries[0].agent_name
