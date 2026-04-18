"""Tests for cicaddy.delegation.sub_agent module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cicaddy.delegation.registry import SubAgentSpec
from cicaddy.delegation.sub_agent import (
    BASE_BLOCKED_TOOLS,
    DelegationSubAgent,
    collect_blocked_tools,
)
from cicaddy.delegation.triage import DelegationEntry


@pytest.fixture(autouse=True)
def reset_plugin_cache():
    """Reset the plugin blocked tools cache between tests."""
    import cicaddy.delegation.sub_agent as mod

    mod._plugin_blocked_tools = None
    yield
    mod._plugin_blocked_tools = None


@pytest.fixture
def sample_spec():
    return SubAgentSpec(
        name="security-reviewer",
        persona="security expert",
        description="Reviews security changes",
        categories=["security"],
        constraints=["Focus on auth"],
        output_sections=["Vulnerabilities"],
        priority=10,
        agent_type="review",
    )


@pytest.fixture
def sample_entry():
    return DelegationEntry(
        agent_name="security-reviewer",
        categories=["security"],
        rationale="Contains secret handling",
        relevant_context_keys=["diff", "project"],
        relevant_files=["app.py"],
        priority=10,
    )


@pytest.fixture
def sample_context():
    return {
        "project": {"name": "test"},
        "diff": "diff content here",
        "mr_description": "Fix auth bug",
        "extra_key": "should be filtered",
    }


@pytest.fixture
def sample_tools():
    return [
        {"name": "read_file", "description": "Read a file"},
        {"name": "list_directory", "description": "List dir"},
        {"name": "delegate_task", "description": "Delegate (should be blocked)"},
        {"name": "create_mr_comment", "description": "Comment (plugin blocked)"},
    ]


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.ai_provider = "gemini"
    settings.ai_model = "gemini-2.0-flash"
    settings.review_prompt = None
    settings.sub_agent_max_iters = 5
    settings.max_execution_time = 600
    settings.context_safety_factor = 0.85
    return settings


class TestBaseBlockedTools:
    """Tests for BASE_BLOCKED_TOOLS constant."""

    def test_delegate_task_blocked(self):
        assert "delegate_task" in BASE_BLOCKED_TOOLS

    def test_minimal_set(self):
        """Base should only block recursion prevention tools."""
        assert len(BASE_BLOCKED_TOOLS) == 1


class TestCollectBlockedTools:
    """Tests for collect_blocked_tools()."""

    def test_includes_base_blocked(self):
        with patch(
            "cicaddy.delegation.sub_agent.importlib.metadata.entry_points"
        ) as mock_ep:
            mock_ep.return_value.select.return_value = []
            blocked = collect_blocked_tools()
        assert "delegate_task" in blocked

    def test_includes_plugin_blocked_tools(self):
        mock_ep = MagicMock()
        mock_ep.load.return_value = lambda: {"create_mr_comment", "merge_mr"}
        mock_ep.name = "gitlab"

        with patch(
            "cicaddy.delegation.sub_agent.importlib.metadata.entry_points"
        ) as mock_eps:
            mock_eps.return_value.select.return_value = [mock_ep]
            blocked = collect_blocked_tools()

        assert "create_mr_comment" in blocked
        assert "merge_mr" in blocked
        assert "delegate_task" in blocked

    def test_caches_result(self):
        with patch(
            "cicaddy.delegation.sub_agent.importlib.metadata.entry_points"
        ) as mock_ep:
            mock_ep.return_value.select.return_value = []
            first = collect_blocked_tools()
            second = collect_blocked_tools()
        # Should call entry_points only once
        assert mock_ep.call_count == 1
        assert first is second

    def test_plugin_load_failure_skipped(self):
        mock_ep = MagicMock()
        mock_ep.load.side_effect = ImportError("bad plugin")
        mock_ep.name = "broken"

        with patch(
            "cicaddy.delegation.sub_agent.importlib.metadata.entry_points"
        ) as mock_eps:
            mock_eps.return_value.select.return_value = [mock_ep]
            blocked = collect_blocked_tools()

        # Should still have base blocked tools
        assert "delegate_task" in blocked


class TestDelegationSubAgentFilterTools:
    """Tests for DelegationSubAgent._filter_tools()."""

    def _make_agent(self, spec, entry, settings, context):
        return DelegationSubAgent(
            spec=spec,
            delegation_entry=entry,
            settings=settings,
            context=context,
            parent_tools=[],
            parent_mcp_manager=None,
            parent_local_registry=None,
        )

    def test_blocks_delegate_task(
        self, sample_spec, sample_entry, mock_settings, sample_context, sample_tools
    ):
        agent = self._make_agent(
            sample_spec, sample_entry, mock_settings, sample_context
        )
        with patch(
            "cicaddy.delegation.sub_agent.collect_blocked_tools",
            return_value={"delegate_task"},
        ):
            filtered = agent._filter_tools(sample_tools)
        names = [t["name"] for t in filtered]
        assert "delegate_task" not in names
        assert "read_file" in names

    def test_spec_blocked_tools(
        self, sample_entry, mock_settings, sample_context, sample_tools
    ):
        spec = SubAgentSpec(
            name="test",
            persona="test",
            description="test",
            blocked_tools=["list_directory"],
        )
        agent = self._make_agent(spec, sample_entry, mock_settings, sample_context)
        with patch(
            "cicaddy.delegation.sub_agent.collect_blocked_tools",
            return_value={"delegate_task"},
        ):
            filtered = agent._filter_tools(sample_tools)
        names = [t["name"] for t in filtered]
        assert "list_directory" not in names
        assert "read_file" in names

    def test_spec_allowed_tools_whitelist(
        self, sample_entry, mock_settings, sample_context, sample_tools
    ):
        spec = SubAgentSpec(
            name="test",
            persona="test",
            description="test",
            allowed_tools=["read_file"],
        )
        agent = self._make_agent(spec, sample_entry, mock_settings, sample_context)
        with patch(
            "cicaddy.delegation.sub_agent.collect_blocked_tools",
            return_value={"delegate_task"},
        ):
            filtered = agent._filter_tools(sample_tools)
        names = [t["name"] for t in filtered]
        assert names == ["read_file"]

    def test_tools_sorted_alphabetically(
        self, sample_spec, sample_entry, mock_settings, sample_context, sample_tools
    ):
        agent = self._make_agent(
            sample_spec, sample_entry, mock_settings, sample_context
        )
        with patch(
            "cicaddy.delegation.sub_agent.collect_blocked_tools",
            return_value={"delegate_task"},
        ):
            filtered = agent._filter_tools(sample_tools)
        names = [t["name"] for t in filtered]
        assert names == sorted(names)


class TestDelegationSubAgentPrompt:
    """Tests for DelegationSubAgent._build_prompt()."""

    def _make_agent(self, spec, entry, settings, context):
        return DelegationSubAgent(
            spec=spec,
            delegation_entry=entry,
            settings=settings,
            context=context,
            parent_tools=[],
            parent_mcp_manager=None,
            parent_local_registry=None,
        )

    def test_prompt_includes_persona(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        agent = self._make_agent(
            sample_spec, sample_entry, mock_settings, sample_context
        )
        prompt = agent._build_prompt()
        assert "security expert" in prompt

    def test_prompt_includes_categories(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        agent = self._make_agent(
            sample_spec, sample_entry, mock_settings, sample_context
        )
        prompt = agent._build_prompt()
        assert "security" in prompt

    def test_prompt_includes_constraints(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        agent = self._make_agent(
            sample_spec, sample_entry, mock_settings, sample_context
        )
        prompt = agent._build_prompt()
        assert "Focus on auth" in prompt

    def test_prompt_includes_data_boundaries(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        agent = self._make_agent(
            sample_spec, sample_entry, mock_settings, sample_context
        )
        prompt = agent._build_prompt()
        assert "<<<BEGIN_CONTEXT_DATA_" in prompt
        assert "<<<END_CONTEXT_DATA_" in prompt

    def test_prompt_includes_user_review_prompt(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        mock_settings.review_prompt = "Also check for SQL injection"
        agent = self._make_agent(
            sample_spec, sample_entry, mock_settings, sample_context
        )
        prompt = agent._build_prompt()
        assert "SQL injection" in prompt


class TestDelegationSubAgentContext:
    """Tests for DelegationSubAgent._get_relevant_context()."""

    def _make_agent(self, spec, entry, settings, context):
        return DelegationSubAgent(
            spec=spec,
            delegation_entry=entry,
            settings=settings,
            context=context,
            parent_tools=[],
            parent_mcp_manager=None,
            parent_local_registry=None,
        )

    def test_filters_by_relevant_keys(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        agent = self._make_agent(
            sample_spec, sample_entry, mock_settings, sample_context
        )
        relevant = agent._get_relevant_context()
        assert "diff" in relevant
        assert "project" in relevant
        assert "extra_key" not in relevant

    def test_always_includes_project(self, sample_spec, mock_settings, sample_context):
        entry = DelegationEntry(
            agent_name="test",
            categories=["security"],
            relevant_context_keys=["diff"],  # no "project"
        )
        agent = self._make_agent(sample_spec, entry, mock_settings, sample_context)
        relevant = agent._get_relevant_context()
        assert "project" in relevant

    def test_no_keys_returns_full_context(
        self, sample_spec, mock_settings, sample_context
    ):
        entry = DelegationEntry(
            agent_name="test",
            categories=["security"],
            relevant_context_keys=[],
        )
        agent = self._make_agent(sample_spec, entry, mock_settings, sample_context)
        relevant = agent._get_relevant_context()
        assert relevant == sample_context


class TestDelegationSubAgentExecute:
    """Tests for DelegationSubAgent.execute()."""

    def _make_agent(self, spec, entry, settings, context):
        return DelegationSubAgent(
            spec=spec,
            delegation_entry=entry,
            settings=settings,
            context=context,
            parent_tools=[],
            parent_mcp_manager=None,
            parent_local_registry=None,
        )

    @pytest.mark.asyncio
    async def test_execute_success(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        agent = self._make_agent(
            sample_spec, sample_entry, mock_settings, sample_context
        )

        # Mock execution engine
        mock_turn = MagicMock()
        mock_turn.output_message = "Found 2 vulnerabilities"
        mock_turn.execution_summary = {"total_tokens": 500}

        mock_engine = MagicMock()
        mock_engine.execute_turn = AsyncMock(return_value=mock_turn)
        agent.execution_engine = mock_engine

        result = await agent.execute()
        assert result["status"] == "success"
        assert result["agent_name"] == "security-reviewer"
        assert "vulnerabilities" in result["analysis"]
        assert result["tokens"] == 500

    @pytest.mark.asyncio
    async def test_execute_failure(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        agent = self._make_agent(
            sample_spec, sample_entry, mock_settings, sample_context
        )

        mock_engine = MagicMock()
        mock_engine.execute_turn = AsyncMock(side_effect=Exception("API timeout"))
        agent.execution_engine = mock_engine

        result = await agent.execute()
        assert result["status"] == "failed"
        assert "API timeout" in result["analysis"]
        assert result["tokens"] == 0

    @pytest.mark.asyncio
    async def test_cleanup_shuts_down_provider(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        agent = self._make_agent(
            sample_spec, sample_entry, mock_settings, sample_context
        )
        agent.ai_provider = MagicMock()
        agent.ai_provider.shutdown = AsyncMock()

        await agent.cleanup()
        agent.ai_provider.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_handles_no_provider(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        agent = self._make_agent(
            sample_spec, sample_entry, mock_settings, sample_context
        )
        agent.ai_provider = None
        # Should not raise
        await agent.cleanup()
