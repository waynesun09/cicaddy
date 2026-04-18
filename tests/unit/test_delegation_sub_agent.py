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
from cicaddy.delegation.triage import DelegationEntry, SiblingInfo


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


class TestDelegationSubAgentSiblingAwareness:
    """Tests for sibling-agent awareness in prompts."""

    def _make_agent(self, spec, entry, settings, context, sibling_agents=None):
        return DelegationSubAgent(
            spec=spec,
            delegation_entry=entry,
            settings=settings,
            context=context,
            parent_tools=[],
            parent_mcp_manager=None,
            parent_local_registry=None,
            sibling_agents=sibling_agents,
        )

    def test_sole_reviewer_when_no_siblings(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        agent = self._make_agent(
            sample_spec, sample_entry, mock_settings, sample_context
        )
        prompt = agent._build_prompt()
        assert "sole reviewer" in prompt
        assert "running alongside" not in prompt

    def test_sole_reviewer_when_only_self_in_list(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        agent = self._make_agent(
            sample_spec,
            sample_entry,
            mock_settings,
            sample_context,
            sibling_agents=[
                SiblingInfo(name="security-reviewer", categories=["security"])
            ],
        )
        prompt = agent._build_prompt()
        assert "sole reviewer" in prompt
        assert "running alongside" not in prompt

    def test_siblings_shown_with_categories(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        agent = self._make_agent(
            sample_spec,
            sample_entry,
            mock_settings,
            sample_context,
            sibling_agents=[
                SiblingInfo(name="security-reviewer", categories=["security"]),
                SiblingInfo(name="performance-reviewer", categories=["performance"]),
                SiblingInfo(
                    name="general-reviewer", categories=["code_quality", "tests"]
                ),
            ],
        )
        prompt = agent._build_prompt()
        assert "running alongside" in prompt
        assert "performance-reviewer (performance)" in prompt
        assert "general-reviewer (code_quality, tests)" in prompt
        # Self should be excluded
        assert (
            "security-reviewer"
            not in prompt.split("running alongside")[1].split("\n")[0]
        )

    def test_siblings_without_categories(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        agent = self._make_agent(
            sample_spec,
            sample_entry,
            mock_settings,
            sample_context,
            sibling_agents=[
                SiblingInfo(name="security-reviewer", categories=["security"]),
                SiblingInfo(name="custom-agent"),
            ],
        )
        prompt = agent._build_prompt()
        # Agent without categories shows just the name
        assert (
            "custom-agent;" in prompt
            or "custom-agent\n" in prompt
            or "custom-agent." in prompt
        )

    def test_sibling_agents_defaults_to_empty_list(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        agent = self._make_agent(
            sample_spec,
            sample_entry,
            mock_settings,
            sample_context,
            sibling_agents=None,
        )
        assert agent.sibling_agents == []

    def test_duplicate_siblings_deduplicated(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        agent = self._make_agent(
            sample_spec,
            sample_entry,
            mock_settings,
            sample_context,
            sibling_agents=[
                SiblingInfo(name="security-reviewer", categories=["security"]),
                SiblingInfo(name="general-reviewer", categories=["code_quality"]),
                SiblingInfo(name="general-reviewer", categories=["code_quality"]),
            ],
        )
        prompt = agent._build_prompt()
        # general-reviewer should appear only once in the delegation context
        delegation_section = prompt.split("## Delegation Context")[1].split("##")[0]
        assert delegation_section.count("general-reviewer") == 1

    def test_self_exclusion_uses_delegation_entry_name(
        self, mock_settings, sample_context
    ):
        """Self-exclusion matches on delegation_entry.agent_name, not spec.name."""
        spec = SubAgentSpec(
            name="my-agent",
            persona="test",
            description="test",
            categories=["test"],
            constraints=["test"],
            output_sections=["test"],
            agent_type="review",
        )
        entry = DelegationEntry(
            agent_name="my-agent",
            categories=["test"],
            rationale="test",
        )
        agent = self._make_agent(
            spec,
            entry,
            mock_settings,
            sample_context,
            sibling_agents=[
                SiblingInfo(name="my-agent", categories=["test"]),
                SiblingInfo(name="other-agent", categories=["other"]),
            ],
        )
        prompt = agent._build_prompt()
        delegation_section = prompt.split("## Delegation Context")[1].split("##")[0]
        assert "my-agent" not in delegation_section
        assert "other-agent" in delegation_section


class TestDelegationSubAgentWorkspaceContext:
    """Tests for workspace context (bundled_context, agent_rules, skills) in sub-agents."""

    def _make_agent(
        self,
        spec,
        entry,
        settings,
        context,
        bundled_context="",
        agent_rules="",
        skills=None,
    ):
        return DelegationSubAgent(
            spec=spec,
            delegation_entry=entry,
            settings=settings,
            context=context,
            parent_tools=[],
            parent_mcp_manager=None,
            parent_local_registry=None,
            bundled_context=bundled_context,
            agent_rules=agent_rules,
            skills=skills,
        )

    def test_init_stores_bundled_context(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        agent = self._make_agent(
            sample_spec,
            sample_entry,
            mock_settings,
            sample_context,
            bundled_context="bundled knowledge here",
        )
        assert agent.bundled_context == "bundled knowledge here"

    def test_init_stores_agent_rules(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        agent = self._make_agent(
            sample_spec,
            sample_entry,
            mock_settings,
            sample_context,
            agent_rules="# Project Rules\nUse type hints.",
        )
        assert agent.agent_rules == "# Project Rules\nUse type hints."

    def test_init_stores_skills(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        mock_skill = MagicMock()
        mock_skill.name = "test-skill"
        agent = self._make_agent(
            sample_spec,
            sample_entry,
            mock_settings,
            sample_context,
            skills=[mock_skill],
        )
        assert len(agent.skills) == 1
        assert agent.skills[0].name == "test-skill"

    def test_init_defaults_empty(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        agent = DelegationSubAgent(
            spec=sample_spec,
            delegation_entry=sample_entry,
            settings=mock_settings,
            context=sample_context,
            parent_tools=[],
            parent_mcp_manager=None,
            parent_local_registry=None,
        )
        assert agent.bundled_context == ""
        assert agent.agent_rules == ""
        assert agent.skills == []

    def test_prompt_includes_bundled_context(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        agent = self._make_agent(
            sample_spec,
            sample_entry,
            mock_settings,
            sample_context,
            bundled_context="## Model Reference\nUse gemini-3-flash-preview.",
        )
        prompt = agent._build_prompt()
        assert "## Model Reference" in prompt
        assert "gemini-3-flash-preview" in prompt

    def test_prompt_includes_agent_rules(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        agent = self._make_agent(
            sample_spec,
            sample_entry,
            mock_settings,
            sample_context,
            agent_rules="# AGENT.md\nAlways use type hints.\nPrefer async.",
        )
        prompt = agent._build_prompt()
        assert "# AGENT.md" in prompt
        assert "Always use type hints." in prompt

    def test_prompt_includes_rendered_skills(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        mock_skill = MagicMock()
        mock_skill.name = "test-skill"

        agent = self._make_agent(
            sample_spec,
            sample_entry,
            mock_settings,
            sample_context,
            skills=[mock_skill],
        )

        with patch(
            "cicaddy.skills.render_skills_prompt",
            return_value="## Skills\n- test-skill: does things",
        ) as mock_render:
            prompt = agent._build_prompt()

        mock_render.assert_called_once_with([mock_skill])
        assert "## Skills" in prompt
        assert "test-skill: does things" in prompt

    def test_prompt_ordering_bundled_rules_core_skills(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        """Sections must be ordered: bundled → rules → core → skills."""
        mock_skill = MagicMock()

        agent = self._make_agent(
            sample_spec,
            sample_entry,
            mock_settings,
            sample_context,
            bundled_context="BUNDLED_SECTION",
            agent_rules="RULES_SECTION",
            skills=[mock_skill],
        )

        with patch(
            "cicaddy.skills.render_skills_prompt",
            return_value="SKILLS_SECTION",
        ):
            prompt = agent._build_prompt()

        bundled_pos = prompt.index("BUNDLED_SECTION")
        rules_pos = prompt.index("RULES_SECTION")
        core_pos = prompt.index("You are a security expert")
        skills_pos = prompt.index("SKILLS_SECTION")

        assert bundled_pos < rules_pos < core_pos < skills_pos

    def test_empty_values_produce_same_structure(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        """Empty bundled_context/agent_rules and no skills should not add extra sections."""
        agent = self._make_agent(
            sample_spec,
            sample_entry,
            mock_settings,
            sample_context,
        )
        prompt = agent._build_prompt()
        # Should start directly with core prompt (no bundled/rules prefix)
        assert prompt.startswith("You are a security expert")
        # Should not contain skills section
        assert "## Skills" not in prompt

    def test_skills_render_returns_empty_skipped(
        self, sample_spec, sample_entry, mock_settings, sample_context
    ):
        """If render_skills_prompt returns empty string, skills section is omitted."""
        mock_skill = MagicMock()

        agent = self._make_agent(
            sample_spec,
            sample_entry,
            mock_settings,
            sample_context,
            skills=[mock_skill],
        )

        with patch(
            "cicaddy.skills.render_skills_prompt",
            return_value="",
        ):
            prompt = agent._build_prompt()

        # The core prompt should end the output (no trailing skills section)
        assert prompt.rstrip().endswith("Provide structured, actionable findings.")
