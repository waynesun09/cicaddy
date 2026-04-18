"""Tests for cicaddy.delegation.registry module."""

from __future__ import annotations

import json
import textwrap

from cicaddy.delegation.registry import (
    _BUILT_IN_REVIEW_AGENTS,
    _BUILT_IN_TASK_AGENTS,
    SubAgentRegistry,
    SubAgentSpec,
)


class TestSubAgentSpec:
    """Tests for SubAgentSpec dataclass."""

    def test_defaults(self):
        spec = SubAgentSpec(name="test", persona="tester", description="a test agent")
        assert spec.categories == []
        assert spec.constraints == []
        assert spec.output_sections == []
        assert spec.priority == 0
        assert spec.allowed_tools is None
        assert spec.blocked_tools == []
        assert spec.agent_type == "*"
        assert spec.source_file is None

    def test_all_fields(self):
        spec = SubAgentSpec(
            name="sec",
            persona="security expert",
            description="reviews security",
            categories=["security"],
            constraints=["focus on auth"],
            output_sections=["Findings"],
            priority=10,
            allowed_tools=["read_file"],
            blocked_tools=["delete_file"],
            agent_type="review",
            source_file="/tmp/sec.yaml",
        )
        assert spec.name == "sec"
        assert spec.agent_type == "review"
        assert spec.allowed_tools == ["read_file"]


class TestBuiltInAgents:
    """Tests for built-in agent definitions."""

    def test_review_agents_present(self):
        """All 8 built-in review agents should exist."""
        expected = {
            "security-reviewer",
            "architecture-reviewer",
            "api-reviewer",
            "database-reviewer",
            "ui-reviewer",
            "devops-reviewer",
            "performance-reviewer",
            "general-reviewer",
        }
        assert set(_BUILT_IN_REVIEW_AGENTS.keys()) == expected

    def test_task_agents_present(self):
        """All 3 built-in task agents should exist."""
        expected = {"data-analyst", "report-writer", "general-task"}
        assert set(_BUILT_IN_TASK_AGENTS.keys()) == expected

    def test_review_agents_have_review_type(self):
        for spec in _BUILT_IN_REVIEW_AGENTS.values():
            assert spec.agent_type == "review"

    def test_task_agents_have_task_type(self):
        for spec in _BUILT_IN_TASK_AGENTS.values():
            assert spec.agent_type == "task"

    def test_all_agents_have_persona(self):
        for spec in _BUILT_IN_REVIEW_AGENTS.values():
            assert spec.persona, f"{spec.name} missing persona"
        for spec in _BUILT_IN_TASK_AGENTS.values():
            assert spec.persona, f"{spec.name} missing persona"

    def test_general_reviewer_is_catch_all(self):
        gen = _BUILT_IN_REVIEW_AGENTS["general-reviewer"]
        assert gen.priority == 100  # Lowest priority = catch-all


class TestSubAgentRegistry:
    """Tests for SubAgentRegistry.load_registry()."""

    def test_load_review_agents(self):
        registry = SubAgentRegistry().load_registry("review")
        assert "security-reviewer" in registry
        assert "general-reviewer" in registry
        assert len(registry) >= 8

    def test_load_task_agents(self):
        registry = SubAgentRegistry().load_registry("task")
        assert "data-analyst" in registry
        assert "general-task" in registry
        assert len(registry) >= 3

    def test_unknown_type_returns_empty_builtin(self):
        """Unknown agent type should return no built-in agents."""
        registry = SubAgentRegistry().load_registry("cron")
        assert len(registry) == 0

    def test_user_yaml_override(self, tmp_path):
        """User YAML should override built-in agent by name."""
        review_dir = tmp_path / "review"
        review_dir.mkdir()
        (review_dir / "security-reviewer.yaml").write_text(
            textwrap.dedent("""\
                name: security-reviewer
                persona: custom security persona
                description: custom security description
                categories: [security, crypto]
                priority: 5
                agent_type: review
            """)
        )
        registry = SubAgentRegistry().load_registry("review", agents_dir=str(tmp_path))
        spec = registry["security-reviewer"]
        assert spec.persona == "custom security persona"
        assert spec.priority == 5
        assert "crypto" in spec.categories

    def test_user_yaml_wildcard_agent(self, tmp_path):
        """YAML with agent_type=* should be loaded for any agent type."""
        (tmp_path / "shared.yaml").write_text(
            textwrap.dedent("""\
                name: shared-agent
                persona: shared
                description: available to all
                categories: [misc]
                agent_type: "*"
            """)
        )
        registry = SubAgentRegistry().load_registry("review", agents_dir=str(tmp_path))
        assert "shared-agent" in registry

    def test_user_yaml_wrong_type_excluded(self, tmp_path):
        """YAML with agent_type=task should NOT load for review."""
        (tmp_path / "task-only.yaml").write_text(
            textwrap.dedent("""\
                name: task-only-agent
                persona: task
                description: only for tasks
                categories: [data]
                agent_type: task
            """)
        )
        registry = SubAgentRegistry().load_registry("review", agents_dir=str(tmp_path))
        assert "task-only-agent" not in registry

    def test_invalid_yaml_skipped(self, tmp_path):
        """Invalid YAML should be skipped with a warning, not raise."""
        review_dir = tmp_path / "review"
        review_dir.mkdir()
        (review_dir / "bad.yaml").write_text("this is not: valid: yaml: [")
        registry = SubAgentRegistry().load_registry("review", agents_dir=str(tmp_path))
        # Should still have built-in agents
        assert "security-reviewer" in registry

    def test_yaml_missing_name_skipped(self, tmp_path):
        """YAML without 'name' field should be skipped."""
        review_dir = tmp_path / "review"
        review_dir.mkdir()
        (review_dir / "noname.yaml").write_text("persona: test\ndescription: no name")
        registry = SubAgentRegistry().load_registry("review", agents_dir=str(tmp_path))
        assert "security-reviewer" in registry  # built-ins still loaded

    def test_json_config_override(self):
        """DELEGATION_AGENTS JSON should override agents."""
        config = json.dumps(
            [
                {
                    "name": "custom-json-agent",
                    "persona": "JSON agent",
                    "description": "from JSON config",
                    "categories": ["custom"],
                    "agent_type": "review",
                }
            ]
        )
        registry = SubAgentRegistry().load_registry("review", user_config=config)
        assert "custom-json-agent" in registry
        assert registry["custom-json-agent"].persona == "JSON agent"

    def test_json_config_type_filtering(self):
        """JSON agent with wrong type should be excluded."""
        config = json.dumps(
            [
                {
                    "name": "task-json",
                    "persona": "task",
                    "description": "task only",
                    "agent_type": "task",
                }
            ]
        )
        registry = SubAgentRegistry().load_registry("review", user_config=config)
        assert "task-json" not in registry

    def test_json_config_invalid_json(self):
        """Invalid JSON should be handled gracefully."""
        registry = SubAgentRegistry().load_registry("review", user_config="not json")
        # Should still have built-in agents
        assert "security-reviewer" in registry

    def test_json_config_empty_string(self):
        """Empty config string should be fine."""
        registry = SubAgentRegistry().load_registry("review", user_config="")
        assert len(registry) >= 8

    def test_merge_precedence(self, tmp_path):
        """JSON config should override YAML which overrides built-in."""
        review_dir = tmp_path / "review"
        review_dir.mkdir()
        (review_dir / "security-reviewer.yaml").write_text(
            textwrap.dedent("""\
                name: security-reviewer
                persona: yaml persona
                description: yaml desc
                agent_type: review
            """)
        )
        json_config = json.dumps(
            [
                {
                    "name": "security-reviewer",
                    "persona": "json persona",
                    "description": "json desc",
                    "agent_type": "review",
                }
            ]
        )
        registry = SubAgentRegistry().load_registry(
            "review", user_config=json_config, agents_dir=str(tmp_path)
        )
        # JSON overrides YAML overrides built-in
        assert registry["security-reviewer"].persona == "json persona"
