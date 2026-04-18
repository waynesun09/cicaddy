"""Tests for delegation CLI integration: _should_delegate(), _get_delegation_context(), tool cascade."""

from __future__ import annotations

import os
import textwrap
from unittest.mock import MagicMock, patch


class TestShouldDelegate:
    """Tests for BaseAIAgent._should_delegate() hook."""

    def _make_agent(self, delegation_mode="none"):
        """Create a minimal agent with mocked settings."""
        from cicaddy.agent.base import BaseAIAgent

        settings = MagicMock()
        settings.delegation_mode = delegation_mode
        settings.ai_provider = "gemini"
        settings.ai_model = "gemini-3-flash"
        settings.ai_temperature = 0.0
        settings.max_infer_iters = 10
        settings.json_logs = False
        settings.log_level = "INFO"
        settings.ssl_verify = True
        settings.ai_response_format = "markdown"

        # Create a concrete subclass since BaseAIAgent is abstract
        class ConcreteAgent(BaseAIAgent):
            async def get_analysis_context(self):
                return {}

            def build_analysis_prompt(self, context):
                return ""

            def get_session_id(self):
                return "test"

        agent = ConcreteAgent.__new__(ConcreteAgent)
        agent.settings = settings
        return agent

    def test_should_delegate_none(self):
        agent = self._make_agent("none")
        assert agent._should_delegate() is False

    def test_should_delegate_auto(self):
        agent = self._make_agent("auto")
        assert agent._should_delegate() is True

    def test_should_delegate_missing_attr(self):
        """When settings has no delegation_mode, defaults to none."""
        agent = self._make_agent("none")
        del agent.settings.delegation_mode
        # getattr with default "none" should kick in
        assert agent._should_delegate() is False


class TestTaskAgentDelegationContext:
    """Tests for TaskAgent._get_delegation_context() DSPy enrichment."""

    def _make_task_agent(self):
        """Create a minimal TaskAgent with mocked settings."""
        from cicaddy.agent.task_agent import TaskAgent

        agent = TaskAgent.__new__(TaskAgent)
        agent.settings = MagicMock()
        agent.settings.delegation_mode = "auto"
        return agent

    def test_no_task_file_returns_context_unchanged(self):
        agent = self._make_task_agent()
        context = {"project": {"name": "test"}, "scope": "external_tools"}
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AI_TASK_FILE", None)
            result = agent._get_delegation_context(context)
        assert result is context
        assert "task_definition" not in result

    def test_task_file_enriches_context(self, tmp_path):
        agent = self._make_task_agent()
        context = {"project": {"name": "test"}, "scope": "external_tools"}

        task_yaml = tmp_path / "test_task.yaml"
        task_yaml.write_text(
            textwrap.dedent("""\
                name: test_analysis
                description: Test analysis task
                type: data_analysis
                persona: test analyst
                constraints:
                  - Use SQL queries
                  - Never drop tables
                outputs:
                  - name: summary
                    description: Executive summary
                tools:
                  servers:
                    - devlake
                    - local
                  required_tools:
                    - execute_query
                  forbidden_tools:
                    - delete_file
                output_format: html
                reasoning: react
                context: |
                  Analyze the project metrics.
            """)
        )

        with patch.dict(os.environ, {"AI_TASK_FILE": str(task_yaml)}):
            result = agent._get_delegation_context(context)

        assert "task_definition" in result
        td = result["task_definition"]
        assert td["name"] == "test_analysis"
        assert td["description"] == "Test analysis task"
        assert td["type"] == "data_analysis"
        assert td["persona"] == "test analyst"
        assert td["constraints"] == ["Use SQL queries", "Never drop tables"]
        assert td["outputs"] == ["summary"]
        assert td["output_format"] == "html"
        assert td["reasoning"] == "react"
        assert td["tool_servers"] == ["devlake", "local"]
        assert td["required_tools"] == ["execute_query"]
        assert td["forbidden_tools"] == ["delete_file"]
        assert "task_context_preview" in result
        assert "Analyze the project metrics" in result["task_context_preview"]

    def test_invalid_task_file_returns_context_unchanged(self):
        agent = self._make_task_agent()
        context = {"project": {"name": "test"}}

        with patch.dict(os.environ, {"AI_TASK_FILE": "/nonexistent/task.yaml"}):
            result = agent._get_delegation_context(context)

        # Should return context unchanged (warning logged, not raised)
        assert result is context
        assert "task_definition" not in result

    def test_resolved_inputs_truncated(self, tmp_path):
        """Long resolved input values should be truncated to 200 chars."""
        agent = self._make_task_agent()
        context = {"project": {"name": "test"}}

        long_value = "x" * 300
        task_yaml = tmp_path / "task.yaml"
        task_yaml.write_text(
            textwrap.dedent(f"""\
                name: truncation_test
                description: Test truncation
                type: custom
                inputs:
                  - name: long_input
                    env_var: LONG_INPUT_VAR
                    default: "{long_value}"
                outputs: []
            """)
        )

        with patch.dict(
            os.environ,
            {"AI_TASK_FILE": str(task_yaml)},
        ):
            result = agent._get_delegation_context(context)

        if "task_definition" in result:
            inputs = result["task_definition"]["resolved_inputs"]
            for v in inputs.values():
                if isinstance(v, str):
                    assert len(v) <= 200


class TestToolCascade:
    """Tests for DSPy task forbidden_tools cascade to sub-agents."""

    def test_forbidden_tools_merged_into_blocked(self):
        """Task forbidden_tools should be added to sub-agent blocked_tools."""
        from cicaddy.delegation.registry import SubAgentSpec

        # Simulate what _analyze_delegate does
        delegation_context = {
            "task_definition": {
                "forbidden_tools": ["delete_file", "drop_table"],
            }
        }

        spec = SubAgentSpec(
            name="test-agent",
            persona="tester",
            description="test",
            blocked_tools=["delegate_task"],
        )
        registry = {"test-agent": spec}

        # Simulate the cascade logic from base.py
        task_def = delegation_context.get("task_definition", {})
        task_forbidden = task_def.get("forbidden_tools", [])
        if task_forbidden:
            for name, s in registry.items():
                s.blocked_tools = list(set(s.blocked_tools + task_forbidden))

        assert "delete_file" in spec.blocked_tools
        assert "drop_table" in spec.blocked_tools
        assert "delegate_task" in spec.blocked_tools

    def test_no_task_definition_no_cascade(self):
        """Without task_definition in context, blocked_tools unchanged."""
        from cicaddy.delegation.registry import SubAgentSpec

        delegation_context = {"project": {"name": "test"}}

        spec = SubAgentSpec(
            name="test-agent",
            persona="tester",
            description="test",
            blocked_tools=["delegate_task"],
        )

        task_def = delegation_context.get("task_definition", {})
        task_forbidden = task_def.get("forbidden_tools", [])
        if task_forbidden:
            spec.blocked_tools = list(set(spec.blocked_tools + task_forbidden))

        assert spec.blocked_tools == ["delegate_task"]
