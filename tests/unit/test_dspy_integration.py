"""Unit tests for DSPy integration."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from cicaddy.dspy import (
    PromptBuilder,
    TaskDefinition,
    TaskInput,
    TaskLoader,
    TaskLoadError,
    TaskOutput,
    ToolConfig,
)
from cicaddy.utils.env_substitution import substitute_env_variables


class TestTaskSchema:
    """Test Pydantic models for task configuration."""

    def test_task_input_basic(self):
        """Test basic TaskInput creation."""
        input_spec = TaskInput(name="repo_name", env_var="REPO_NAME")
        assert input_spec.name == "repo_name"
        assert input_spec.env_var == "REPO_NAME"
        assert input_spec.required is True
        assert input_spec.default is None

    def test_task_input_with_default(self):
        """Test TaskInput with default value."""
        input_spec = TaskInput(
            name="days", env_var="ANALYSIS_DAYS", default="30", required=False
        )
        assert input_spec.default == "30"
        assert input_spec.required is False

    def test_task_output_basic(self):
        """Test basic TaskOutput creation."""
        output_spec = TaskOutput(name="summary", description="Executive summary")
        assert output_spec.name == "summary"
        assert output_spec.description == "Executive summary"
        assert output_spec.required is True

    def test_task_output_with_format(self):
        """Test TaskOutput with format hint."""
        output_spec = TaskOutput(name="data_table", format="table", required=False)
        assert output_spec.format == "table"
        assert output_spec.required is False

    def test_tool_config_default(self):
        """Test ToolConfig with defaults."""
        config = ToolConfig()
        assert config.servers == []
        assert config.required_tools == []
        assert config.forbidden_tools == []

    def test_tool_config_with_servers(self):
        """Test ToolConfig with server list."""
        config = ToolConfig(servers=["devlake-mcp-instance", "monitoring"])
        assert "devlake-mcp-instance" in config.servers
        assert "monitoring" in config.servers

    def test_task_definition_minimal(self):
        """Test minimal TaskDefinition."""
        task = TaskDefinition(
            name="test_task",
            description="A test task",
        )
        assert task.name == "test_task"
        assert task.type == "custom"
        assert task.reasoning == "chain_of_thought"
        assert task.output_format == "markdown"

    def test_task_definition_full(self):
        """Test TaskDefinition with all fields."""
        task = TaskDefinition(
            name="analysis_task",
            description="Analyze data",
            type="data_analysis",
            inputs=[
                TaskInput(name="repo", env_var="REPO_NAME"),
            ],
            outputs=[
                TaskOutput(name="summary", required=True),
            ],
            tools=ToolConfig(servers=["prod-server"]),
            constraints=["Use SQL aggregation", "Avoid CTE"],
            reasoning="react",
            output_format="html",
            version="2.0",
            author="Test Author",
            tags=["analysis", "sql"],
        )
        assert task.type == "data_analysis"
        assert len(task.inputs) == 1
        assert len(task.outputs) == 1
        assert len(task.constraints) == 2
        assert task.reasoning == "react"
        assert task.output_format == "html"

    def test_task_definition_invalid_type(self):
        """Test TaskDefinition with invalid type."""
        with pytest.raises(ValueError):
            TaskDefinition(
                name="test",
                description="test",
                type="invalid_type",  # type: ignore
            )


class TestEnvSubstitution:
    """Test centralized environment variable substitution utility."""

    def test_substitute_with_env_value(self):
        """Test substitution when env var is set."""
        with patch.dict(os.environ, {"MY_VAR": "hello"}, clear=False):
            result = substitute_env_variables("Value is {{MY_VAR}}")
            assert result == "Value is hello"

    def test_substitute_with_default(self):
        """Test substitution falls back to default."""
        result = substitute_env_variables("Value is {{UNSET_VAR_XYZ:fallback}}")
        assert result == "Value is fallback"

    def test_substitute_env_overrides_default(self):
        """Test env value takes precedence over default."""
        with patch.dict(os.environ, {"MY_VAR": "from_env"}, clear=False):
            result = substitute_env_variables("{{MY_VAR:default}}")
            assert result == "from_env"

    def test_substitute_keeps_placeholder_when_unset(self):
        """Test unset var without default keeps original placeholder."""
        result = substitute_env_variables("{{TOTALLY_UNSET_VAR_ABC}}")
        assert result == "{{TOTALLY_UNSET_VAR_ABC}}"

    def test_substitute_multiple_vars(self):
        """Test multiple substitutions in one string."""
        with patch.dict(os.environ, {"A_VAR": "x", "B_VAR": "y"}, clear=False):
            result = substitute_env_variables("{{A_VAR}} and {{B_VAR}}")
            assert result == "x and y"


class TestTaskLoader:
    """Test YAML task loading functionality."""

    def test_load_valid_yaml(self):
        """Test loading a valid YAML task file."""
        yaml_content = """
name: test_task
description: A test task for unit testing
type: data_analysis

inputs:
  - name: repo_name
    env_var: TEST_REPO
    default: test-repo

outputs:
  - name: summary
    required: true

constraints:
  - Use SQL aggregation
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            loader = TaskLoader()
            task, resolved = loader.load(f.name)

            assert task.name == "test_task"
            assert task.type == "data_analysis"
            assert len(task.inputs) == 1
            # Original default is preserved on the model
            assert task.inputs[0].default == "test-repo"
            # Resolved dict also has the value
            assert resolved["repo_name"] == "test-repo"

        os.unlink(f.name)

    def test_load_file_not_found(self):
        """Test loading non-existent file raises error."""
        loader = TaskLoader()
        with pytest.raises(TaskLoadError) as exc_info:
            loader.load("/nonexistent/path/task.yaml")
        assert "not found" in str(exc_info.value)

    def test_load_invalid_extension(self):
        """Test loading non-YAML file raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("name: test")
            f.flush()

            loader = TaskLoader()
            with pytest.raises(TaskLoadError) as exc_info:
                loader.load(f.name)
            assert "must be YAML" in str(exc_info.value)

        os.unlink(f.name)

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()

            loader = TaskLoader()
            with pytest.raises(TaskLoadError) as exc_info:
                loader.load(f.name)
            assert "Invalid YAML" in str(exc_info.value)

        os.unlink(f.name)

    def test_env_variable_substitution(self):
        """Test environment variable substitution in YAML."""
        yaml_content = """
name: env_test
description: Test with {{TEST_VAR}} value
type: custom

inputs:
  - name: days
    default: "{{DAYS_VAR:7}}"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            with patch.dict(os.environ, {"TEST_VAR": "substituted"}, clear=False):
                loader = TaskLoader()
                task, resolved = loader.load(f.name)

                assert "substituted" in task.description
                # Default value should be used when env var not set
                assert task.inputs[0].default == "7"
                assert resolved["days"] == "7"

        os.unlink(f.name)

    def test_env_variable_with_env_value(self):
        """Test environment variable substitution uses env value over default."""
        yaml_content = """
name: env_test
description: Test task
type: custom

inputs:
  - name: days
    default: "{{DAYS_VAR:7}}"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            with patch.dict(os.environ, {"DAYS_VAR": "14"}, clear=False):
                loader = TaskLoader()
                task, resolved = loader.load(f.name)

                # Env value should override default in the YAML substitution
                assert task.inputs[0].default == "14"
                assert resolved["days"] == "14"

        os.unlink(f.name)

    def test_input_resolution_from_env(self):
        """Test that inputs are resolved from environment variables."""
        yaml_content = """
name: input_test
description: Test input resolution
type: custom

inputs:
  - name: repo
    env_var: TEST_REPO_NAME
    required: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            with patch.dict(os.environ, {"TEST_REPO_NAME": "my-repo"}, clear=False):
                loader = TaskLoader()
                task, resolved = loader.load(f.name)

                # Resolved dict should have the env value
                assert resolved["repo"] == "my-repo"
                # Original model default should NOT be mutated
                assert task.inputs[0].default is None

        os.unlink(f.name)

    def test_required_input_missing_raises_error(self):
        """Test that a missing required input raises TaskLoadError."""
        yaml_content = """
name: strict_test
description: Test strict validation
type: custom

inputs:
  - name: critical_param
    env_var: NONEXISTENT_CRITICAL_VAR
    required: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            loader = TaskLoader()
            with pytest.raises(TaskLoadError) as exc_info:
                loader.load(f.name)
            assert "critical_param" in str(exc_info.value)
            assert "Required input" in str(exc_info.value)

        os.unlink(f.name)

    def test_required_context_input_without_env_var_loads_ok(self):
        """Test that a required input without env_var loads without error.

        Inputs without env_var are expected to be provided later via runtime
        context in PromptBuilder.build().
        """
        yaml_content = """
name: context_input_test
description: Test context-provided inputs
type: code_review

inputs:
  - name: mr_title
    description: MR title
    required: true
  - name: diff_content
    description: Code diff
    required: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            loader = TaskLoader()
            task, resolved = loader.load(f.name)
            # Values are None at load time — will be supplied via context
            assert resolved["mr_title"] is None
            assert resolved["diff_content"] is None

        os.unlink(f.name)

    def test_optional_input_missing_is_ok(self):
        """Test that a missing optional input does not raise."""
        yaml_content = """
name: optional_test
description: Test optional inputs
type: custom

inputs:
  - name: optional_param
    env_var: NONEXISTENT_OPT_VAR
    required: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            loader = TaskLoader()
            task, resolved = loader.load(f.name)
            assert resolved["optional_param"] is None

        os.unlink(f.name)

    def test_no_mutation_on_task_definition(self):
        """Test that load() does not mutate the TaskDefinition inputs."""
        yaml_content = """
name: mutation_test
description: Test no mutation
type: custom

inputs:
  - name: param
    env_var: MUTATION_TEST_VAR
    default: original
    required: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            with patch.dict(os.environ, {"MUTATION_TEST_VAR": "from_env"}, clear=False):
                loader = TaskLoader()
                task, resolved = loader.load(f.name)

                # Resolved dict has the env value
                assert resolved["param"] == "from_env"
                # The model's default is NOT mutated
                assert task.inputs[0].default == "original"

        os.unlink(f.name)

    def test_base_path_resolution(self):
        """Test relative path resolution with base_path."""
        yaml_content = """
name: relative_test
description: Test relative path
type: custom
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir) / "tasks" / "test.yaml"
            task_path.parent.mkdir(parents=True)
            task_path.write_text(yaml_content)

            loader = TaskLoader(base_path=tmpdir)
            task, _ = loader.load("tasks/test.yaml")

            assert task.name == "relative_test"


class TestPromptBuilder:
    """Test prompt building from task definitions."""

    def test_build_basic_prompt(self):
        """Test building a basic prompt."""
        task = TaskDefinition(
            name="basic_task",
            description="A basic test task",
            type="data_analysis",
        )
        builder = PromptBuilder(task)
        prompt = builder.build()

        assert "basic_task" in prompt
        assert "A basic test task" in prompt
        assert "data analyst" in prompt.lower()

    def test_build_with_inputs(self):
        """Test prompt includes input values."""
        task = TaskDefinition(
            name="input_task",
            description="Task with inputs",
            inputs=[
                TaskInput(name="repo", default="my-repo", description="Repository"),
                TaskInput(name="days", default="30", required=False),
            ],
        )
        builder = PromptBuilder(task)
        prompt = builder.build()

        assert "my-repo" in prompt
        assert "30" in prompt
        assert "(required)" in prompt
        assert "(optional)" in prompt

    def test_build_with_resolved_inputs(self):
        """Test prompt uses resolved_inputs over defaults."""
        task = TaskDefinition(
            name="resolved_task",
            description="Task with resolved inputs",
            inputs=[
                TaskInput(name="repo", default="default-repo"),
            ],
        )
        resolved = {"repo": "resolved-repo"}
        builder = PromptBuilder(task, resolved_inputs=resolved)
        prompt = builder.build()

        assert "resolved-repo" in prompt
        assert "default-repo" not in prompt

    def test_build_does_not_mutate_task(self):
        """Test that build() does not modify the TaskDefinition."""
        task = TaskDefinition(
            name="immutable_task",
            description="Test immutability",
            inputs=[
                TaskInput(name="param", default="original"),
            ],
        )
        builder = PromptBuilder(task)
        builder.build(context={"param": "from_context"})

        # The task's input default should be unchanged
        assert task.inputs[0].default == "original"

    def test_build_with_outputs(self):
        """Test prompt includes output specifications."""
        task = TaskDefinition(
            name="output_task",
            description="Task with outputs",
            outputs=[
                TaskOutput(name="summary", description="Executive summary"),
                TaskOutput(name="table", format="table", required=False),
            ],
        )
        builder = PromptBuilder(task)
        prompt = builder.build()

        assert "summary" in prompt
        assert "Executive summary" in prompt
        assert "[table]" in prompt

    def test_build_with_constraints(self):
        """Test prompt includes constraints."""
        task = TaskDefinition(
            name="constrained_task",
            description="Task with constraints",
            constraints=[
                "Use SQL aggregation",
                "Avoid CTE clauses",
                "Focus on critical issues",
            ],
        )
        builder = PromptBuilder(task)
        prompt = builder.build()

        assert "Constraints" in prompt
        assert "Use SQL aggregation" in prompt
        assert "Avoid CTE clauses" in prompt
        assert "1." in prompt  # Numbered list
        assert "2." in prompt
        assert "3." in prompt

    def test_build_with_tools(self):
        """Test prompt includes tool information."""
        task = TaskDefinition(
            name="tool_task",
            description="Task with tools",
            tools=ToolConfig(servers=["prod-server"]),
        )
        mcp_tools = [
            {
                "name": "query_database",
                "description": "Execute SQL queries",
                "server": "prod-server",
                "inputSchema": {
                    "properties": {
                        "query": {"type": "string", "description": "SQL query"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "other_tool",
                "description": "Tool from other server",
                "server": "other-server",
            },
        ]
        builder = PromptBuilder(task)
        prompt = builder.build(mcp_tools=mcp_tools)

        assert "query_database" in prompt
        assert "Execute SQL queries" in prompt
        # Tool from other server should be filtered out
        assert "other_tool" not in prompt

    def test_build_with_forbidden_tools(self):
        """Test that forbidden tools are excluded."""
        task = TaskDefinition(
            name="filtered_task",
            description="Task with forbidden tools",
            tools=ToolConfig(forbidden_tools=["dangerous_tool"]),
        )
        mcp_tools = [
            {"name": "safe_tool", "description": "Safe tool"},
            {"name": "dangerous_tool", "description": "Dangerous tool"},
        ]
        builder = PromptBuilder(task)
        prompt = builder.build(mcp_tools=mcp_tools)

        assert "safe_tool" in prompt
        assert "dangerous_tool" not in prompt

    def test_build_with_context(self):
        """Test prompt includes runtime context."""
        task = TaskDefinition(
            name="context_task",
            description="Task with context",
        )
        context = {
            "project": {"name": "my-project", "id": "123"},
            "timestamp": "2024-01-15T10:00:00",
            "scope": "full_project",
        }
        builder = PromptBuilder(task)
        prompt = builder.build(context=context)

        assert "my-project" in prompt
        assert "123" in prompt
        assert "2024-01-15" in prompt
        assert "full_project" in prompt

    def test_build_with_examples(self):
        """Test prompt includes examples."""
        task = TaskDefinition(
            name="example_task",
            description="Task with examples",
            examples=[
                {"input": "Query the database", "output": "SELECT * FROM users"},
            ],
        )
        builder = PromptBuilder(task)
        prompt = builder.build()

        assert "Examples" in prompt
        assert "Query the database" in prompt
        assert "SELECT * FROM users" in prompt

    def test_reasoning_strategies(self):
        """Test different reasoning strategy templates."""
        for strategy in ["chain_of_thought", "react", "simple"]:
            task = TaskDefinition(
                name=f"{strategy}_task",
                description="Test task",
                reasoning=strategy,  # type: ignore
            )
            builder = PromptBuilder(task)
            prompt = builder.build()

            assert "Approach" in prompt
            if strategy == "chain_of_thought":
                assert "step-by-step" in prompt.lower()
            elif strategy == "react":
                assert "Thought" in prompt
                assert "Action" in prompt
            elif strategy == "simple":
                assert "directly" in prompt.lower()

    def test_output_formats(self):
        """Test different output format instructions."""
        for fmt in ["markdown", "html", "json"]:
            task = TaskDefinition(
                name=f"{fmt}_task",
                description="Test task",
                output_format=fmt,  # type: ignore
            )
            builder = PromptBuilder(task)
            prompt = builder.build()

            assert "Output Format" in prompt
            if fmt == "markdown":
                assert "Markdown" in prompt
            elif fmt == "html":
                assert "HTML" in prompt
            elif fmt == "json":
                assert "JSON" in prompt

    def test_get_input_values(self):
        """Test extracting input values as dictionary."""
        task = TaskDefinition(
            name="values_task",
            description="Test task",
            inputs=[
                TaskInput(name="repo", default="my-repo"),
                TaskInput(name="days", default="30"),
                TaskInput(name="empty"),  # No default
            ],
        )
        resolved = {"repo": "my-repo", "days": "30", "empty": None}
        builder = PromptBuilder(task, resolved_inputs=resolved)
        values = builder.get_input_values()

        assert values["repo"] == "my-repo"
        assert values["days"] == "30"
        assert values["empty"] == ""

    def test_build_with_context_populating_inputs(self):
        """Test that context values populate task inputs."""
        task = TaskDefinition(
            name="context_input_task",
            description="Task with context-populated inputs",
            type="code_review",
            inputs=[
                TaskInput(name="mr_title", description="MR title"),
                TaskInput(name="diff_content", description="Code diff"),
            ],
        )
        context = {
            "mr_title": "Add new feature",
            "diff_content": "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
        }
        builder = PromptBuilder(task)
        prompt = builder.build(context=context)

        # Context values should populate inputs
        assert "Add new feature" in prompt
        assert "Code Changes" in prompt
        assert "+new" in prompt
        assert "-old" in prompt

    def test_build_with_large_diff_content(self):
        """Test that large diff content is rendered in code block."""
        task = TaskDefinition(
            name="diff_task",
            description="Task with diff",
            type="code_review",
            inputs=[
                TaskInput(name="diff_content", description="Code diff"),
            ],
        )
        diff = "--- a/file.py\n+++ b/file.py\n" + "\n".join(
            [f"+line {i}" for i in range(50)]
        )
        builder = PromptBuilder(task)
        prompt = builder.build(context={"diff_content": diff})

        assert "~~~diff" in prompt
        assert "+line 49" in prompt


class TestIntegration:
    """Integration tests for the full DSPy workflow."""

    def test_load_and_build_prompt(self):
        """Test loading a task and building a prompt."""
        yaml_content = """
name: integration_test
description: Full integration test task
type: monitoring

inputs:
  - name: target
    default: production
    description: Target environment

outputs:
  - name: status
    description: System status report
    required: true

tools:
  servers:
    - monitoring-server

constraints:
  - Check all critical services
  - Report issues immediately

reasoning: react
output_format: markdown
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            # Load task
            loader = TaskLoader()
            task, resolved = loader.load(f.name)

            # Build prompt
            builder = PromptBuilder(task, resolved_inputs=resolved)
            prompt = builder.build(
                context={"timestamp": "2024-01-15T10:00:00"},
                mcp_tools=[
                    {
                        "name": "check_health",
                        "description": "Check service health",
                        "server": "monitoring-server",
                    }
                ],
            )

            # Verify prompt content
            assert "integration_test" in prompt
            assert "monitoring specialist" in prompt.lower()
            assert "production" in prompt
            assert "check_health" in prompt
            assert "ReAct" in prompt

        os.unlink(f.name)

    def test_task_reuse_across_builds(self):
        """Test that the same task can be reused without state leaking."""
        task = TaskDefinition(
            name="reuse_test",
            description="Test reuse",
            inputs=[
                TaskInput(name="param", default="base_value", required=False),
            ],
        )
        resolved = {"param": "base_value"}

        # First build with context override
        builder1 = PromptBuilder(task, resolved_inputs=resolved)
        prompt1 = builder1.build(context={"param": "override_1"})
        assert "override_1" in prompt1

        # Second build with different context — should NOT see override_1
        builder2 = PromptBuilder(task, resolved_inputs=resolved)
        prompt2 = builder2.build(context={"param": "override_2"})
        assert "override_2" in prompt2
        assert "override_1" not in prompt2

        # Third build without context — should use base value
        builder3 = PromptBuilder(task, resolved_inputs=resolved)
        prompt3 = builder3.build()
        assert "base_value" in prompt3
        assert "override" not in prompt3

        # Verify task itself was never mutated
        assert task.inputs[0].default == "base_value"


class TestYamlSubstitutionSafety:
    """Tests for parse-then-substitute YAML handling."""

    def test_env_var_with_yaml_special_characters(self):
        """Env var values with YAML-significant chars don't break parsing."""
        yaml_content = """
name: special_chars_test
description: "Desc with {{SPECIAL_VAR}} here"
type: custom
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            # Value contains colons, dashes, and newlines — would break
            # yaml.safe_load() if substituted before parsing.
            special_value = "key: value\n- item1\n- item2"
            with patch.dict(os.environ, {"SPECIAL_VAR": special_value}, clear=False):
                loader = TaskLoader()
                task, _ = loader.load(f.name)

                assert special_value in task.description

        os.unlink(f.name)

    def test_substitute_in_data_recursive(self):
        """Recursive substitution works on nested dicts and lists."""
        with patch.dict(os.environ, {"A": "alpha", "B": "beta"}, clear=False):
            data = {
                "top": "{{A}}",
                "nested": {"inner": "{{B}}", "number": 42},
                "items": ["{{A}}", "literal", "{{B}}"],
            }
            result = TaskLoader._substitute_in_data(data)

            assert result["top"] == "alpha"
            assert result["nested"]["inner"] == "beta"
            assert result["nested"]["number"] == 42
            assert result["items"] == ["alpha", "literal", "beta"]


class TestTaskInputFormat:
    """Tests for TaskInput.format field."""

    def test_task_input_with_format(self):
        """TaskInput accepts a format field."""
        inp = TaskInput(name="diff_content", format="diff")
        assert inp.format == "diff"

    def test_task_input_format_default_none(self):
        """TaskInput.format defaults to None."""
        inp = TaskInput(name="some_input")
        assert inp.format is None

    def test_task_input_code_format(self):
        """TaskInput accepts format='code'."""
        inp = TaskInput(name="source_code", format="code")
        assert inp.format == "code"

    def test_task_input_invalid_format_rejected(self):
        """TaskInput rejects format values outside the Literal."""
        with pytest.raises(ValueError):
            TaskInput(name="bad", format="xml")  # type: ignore


class TestPromptBuilderFormat:
    """Tests for format-aware prompt rendering."""

    def test_build_with_explicit_diff_format(self):
        """format='diff' renders ~~~diff fencing."""
        task = TaskDefinition(
            name="diff_format_test",
            description="Test diff format",
            inputs=[
                TaskInput(
                    name="my_changes",
                    description="Changes",
                    format="diff",
                ),
            ],
        )
        diff = "--- a/file.py\n+++ b/file.py\n-old\n+new"
        builder = PromptBuilder(task)
        prompt = builder.build(context={"my_changes": diff})

        assert "~~~diff" in prompt
        assert "+new" in prompt
        assert "Code Changes" in prompt

    def test_build_with_code_format(self):
        """format='code' renders plain ~~~ fencing."""
        task = TaskDefinition(
            name="code_format_test",
            description="Test code format",
            inputs=[
                TaskInput(
                    name="source_snippet",
                    description="Source code",
                    format="code",
                ),
            ],
        )
        code = "def hello():\n    print('hello')"
        builder = PromptBuilder(task)
        prompt = builder.build(context={"source_snippet": code})

        assert "~~~\n" in prompt
        assert "~~~diff" not in prompt
        assert "def hello():" in prompt

    def test_build_legacy_diff_name_still_works(self):
        """Backward compat: diff_content without format still gets ~~~diff."""
        task = TaskDefinition(
            name="legacy_test",
            description="Test legacy names",
            inputs=[
                TaskInput(name="diff_content", description="Code diff"),
            ],
        )
        diff = "--- a/file.py\n+++ b/file.py\n-old\n+new"
        builder = PromptBuilder(task)
        prompt = builder.build(context={"diff_content": diff})

        assert "~~~diff" in prompt
        assert "+new" in prompt

    def test_build_explicit_format_overrides_legacy_name(self):
        """Explicit format='code' on 'diff_content' overrides legacy diff."""
        task = TaskDefinition(
            name="override_test",
            description="Test format override",
            inputs=[
                TaskInput(
                    name="diff_content",
                    description="Actually code, not diff",
                    format="code",
                ),
            ],
        )
        code = "def foo(): pass"
        builder = PromptBuilder(task)
        prompt = builder.build(context={"diff_content": code})

        # Should use plain ~~~ fencing, NOT ~~~diff
        assert "~~~\n" in prompt
        assert "~~~diff" not in prompt


class TestPersonaField:
    """Tests for TaskDefinition.persona and PromptBuilder role rendering."""

    def test_persona_default_none(self):
        """TaskDefinition.persona defaults to None."""
        task = TaskDefinition(name="t", description="d")
        assert task.persona is None

    def test_persona_used_in_role_section(self):
        """Explicit persona overrides ROLE_MAPPING in the prompt."""
        task = TaskDefinition(
            name="custom_task",
            description="A task",
            type="custom",
            persona="world-class security auditor with OSCP certification",
        )
        builder = PromptBuilder(task)
        prompt = builder.build()

        assert "world-class security auditor with OSCP certification" in prompt
        # The default ROLE_MAPPING value should NOT appear
        assert "AI assistant specialized" not in prompt

    def test_role_mapping_fallback_without_persona(self):
        """Without persona, ROLE_MAPPING is used based on type."""
        task = TaskDefinition(
            name="review_task",
            description="Review code",
            type="code_review",
        )
        builder = PromptBuilder(task)
        prompt = builder.build()

        assert "senior software engineer and code reviewer" in prompt
