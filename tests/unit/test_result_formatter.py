"""Unit tests for ResultFormatter and related components."""

from unittest.mock import patch

import pytest

from cicaddy.execution.result_formatter import (
    FormattedResult,
    GenericResultFormatter,
    ResultPrioritizer,
    ResultPriority,
)


class TestResultPriority:
    """Test ResultPriority constants."""

    def test_priority_values(self):
        """Test ResultPriority constant values."""
        assert ResultPriority.CRITICAL == 1
        assert ResultPriority.HIGH == 2
        assert ResultPriority.MEDIUM == 3
        assert ResultPriority.LOW == 4


class TestFormattedResult:
    """Test FormattedResult dataclass."""

    def test_formatted_result_creation(self):
        """Test FormattedResult creation with all fields."""
        result = FormattedResult(
            content="Test content",
            priority=ResultPriority.HIGH,
            tool_name="test_tool",
            server_name="test_server",
            execution_time=1.5,
            truncated=False,
            metadata={"test": "data"},
        )

        assert result.content == "Test content"
        assert result.priority == ResultPriority.HIGH
        assert result.tool_name == "test_tool"
        assert result.server_name == "test_server"
        assert result.execution_time == 1.5
        assert result.truncated is False
        assert result.metadata == {"test": "data"}

    def test_formatted_result_post_init(self):
        """Test FormattedResult post-initialization processing."""
        result = FormattedResult(content="Test content")

        # Should calculate size_bytes automatically
        assert result.size_bytes == len("Test content".encode("utf-8"))

        # Should initialize empty metadata
        assert result.metadata == {}

    def test_formatted_result_metadata_initialization(self):
        """Test FormattedResult metadata field initialization."""
        # With provided metadata
        result = FormattedResult(content="Test", metadata={"key": "value"})
        assert result.metadata == {"key": "value"}

        # Without provided metadata (None)
        result = FormattedResult(content="Test", metadata=None)
        assert result.metadata == {}


class TestGenericResultFormatter:
    """Test GenericResultFormatter functionality."""

    @pytest.fixture
    def formatter(self):
        """Create GenericResultFormatter for testing."""
        return GenericResultFormatter(max_result_size=1000)

    def test_initialization(self, formatter):
        """Test GenericResultFormatter initialization."""
        assert formatter.max_result_size == 1000

    def test_format_tool_result_success_string(self, formatter):
        """Test formatting successful string result."""
        result = formatter.format_tool_result(
            tool_name="test_tool",
            server_name="test_server",
            result="This is a test result",
            execution_time=2.5,
            status="success",
            arguments={"param": "value"},
        )

        assert isinstance(result, FormattedResult)
        assert result.tool_name == "test_tool"
        assert result.server_name == "test_server"
        assert result.execution_time == 2.5
        assert result.metadata["status"] == "success"
        assert result.metadata["arguments"] == {"param": "value"}

        # Should have structured boundaries
        assert "=== TEST_TOOL RESULT ===" in result.content
        assert "BEGIN_CONTENT" in result.content
        assert "END_CONTENT" in result.content
        assert "=== END TEST_TOOL ===" in result.content

    def test_format_tool_result_error(self, formatter):
        """Test formatting error result."""
        result = formatter.format_tool_result(
            tool_name="failing_tool",
            server_name="test_server",
            result="Connection timeout",
            status="error",
        )

        assert result.priority == ResultPriority.CRITICAL
        assert "❌ ERROR in failing_tool" in result.content
        assert "Connection timeout" in result.content

    def test_format_tool_result_with_arguments(self, formatter):
        """Test formatting result with arguments in header."""
        arguments = {
            "param1": "value1",
            "param2": "value2",
            "param3": "value3",
            "param4": "value4",
        }

        result = formatter.format_tool_result(
            tool_name="test_tool",
            server_name="test_server",
            result="Result",
            arguments=arguments,
        )

        # Should summarize arguments (first 3 + count)
        assert (
            "Arguments: param1=value1, param2=value2, param3=value3 (+1 more)"
            in result.content
        )

    def test_format_string_result_empty(self, formatter):
        """Test formatting empty string result."""
        content, priority = formatter._format_string_result("")
        assert content == ""
        assert priority == ResultPriority.LOW

    def test_format_string_result_error_patterns(self, formatter):
        """Test formatting string with error patterns."""
        error_texts = [
            "Error occurred during processing",
            "Failed to connect to database",
            "Exception thrown in method",
            "Warning: memory exhausted",
            "Alert: security breach",
            "Issue with configuration",
        ]

        for text in error_texts:
            content, priority = formatter._format_string_result(text)
            assert priority == ResultPriority.CRITICAL, f"Should be critical: {text}"

    def test_format_string_result_json_parsing(self, formatter):
        """Test formatting string that contains JSON."""
        json_string = '{"key": "value", "number": 42}'

        content, priority = formatter._format_string_result(json_string)

        # Should parse and format as dict
        assert content != json_string  # Should be reformatted
        assert priority in [ResultPriority.LOW, ResultPriority.MEDIUM]

    def test_format_string_result_content_length_priority(self, formatter):
        """Test string result priority based on content length."""
        # High priority for many lines
        long_content = "\n".join([f"Line {i}" for i in range(25)])
        content, priority = formatter._format_string_result(long_content)
        assert priority == ResultPriority.HIGH

        # Medium priority for moderate lines
        medium_content = "\n".join([f"Line {i}" for i in range(10)])
        content, priority = formatter._format_string_result(medium_content)
        assert priority == ResultPriority.MEDIUM

        # Medium priority for long single line
        long_single_line = "x" * 1000
        content, priority = formatter._format_string_result(long_single_line)
        assert priority == ResultPriority.MEDIUM

        # Low priority for short content
        short_content = "Brief result"
        content, priority = formatter._format_string_result(short_content)
        assert priority == ResultPriority.LOW

    def test_format_dict_result_error_field(self, formatter):
        """Test formatting dict with error field."""
        result_dict = {"error": "Database connection failed", "code": 500}

        content, priority = formatter._format_dict_result(result_dict)

        assert priority == ResultPriority.CRITICAL
        assert "ERROR: Database connection failed" in content

    def test_format_dict_result_data_field(self, formatter):
        """Test formatting dict with data field."""
        # High priority for non-empty data
        result_dict = {"data": [{"id": 1}, {"id": 2}], "status": "success"}
        content, priority = formatter._format_dict_result(result_dict)
        assert priority == ResultPriority.HIGH

        # Medium priority for empty data
        result_dict = {"data": [], "status": "success"}
        content, priority = formatter._format_dict_result(result_dict)
        assert priority == ResultPriority.MEDIUM

    def test_format_dict_result_content_field(self, formatter):
        """Test formatting dict with content field."""
        # High priority for long content
        result_dict = {"content": "x" * 200, "metadata": "info"}
        content, priority = formatter._format_dict_result(result_dict)
        assert priority == ResultPriority.HIGH

        # Medium priority for short content
        result_dict = {"content": "short", "metadata": "info"}
        content, priority = formatter._format_dict_result(result_dict)
        assert priority == ResultPriority.MEDIUM

    def test_format_dict_result_size_based_priority(self, formatter):
        """Test dict result priority based on size."""
        # High priority for large dict
        large_dict = {f"key_{i}": f"value_{i}" for i in range(50)}
        content, priority = formatter._format_dict_result(large_dict)
        assert priority == ResultPriority.HIGH

        # Medium priority for medium dict
        medium_dict = {f"key_{i}": f"value_{i}" for i in range(5)}
        content, priority = formatter._format_dict_result(medium_dict)
        assert priority == ResultPriority.MEDIUM

        # Low priority for small dict
        small_dict = {"key": "value"}
        content, priority = formatter._format_dict_result(small_dict)
        assert priority == ResultPriority.LOW

    def test_format_list_result_empty(self, formatter):
        """Test formatting empty list."""
        content, priority = formatter._format_list_result([])
        assert content == "[]"
        assert priority == ResultPriority.LOW

    def test_format_list_result_string_list(self, formatter):
        """Test formatting list of strings."""
        string_list = ["item1", "item2", "item3"]

        content, priority = formatter._format_list_result(string_list)

        assert "1. item1" in content
        assert "2. item2" in content
        assert "3. item3" in content
        assert priority == ResultPriority.MEDIUM

    def test_format_list_result_complex_list(self, formatter):
        """Test formatting list of complex objects."""
        complex_list = [{"id": 1, "name": "test"}, {"id": 2, "name": "test2"}]

        content, priority = formatter._format_list_result(complex_list)

        # Should format as JSON
        assert content != str(complex_list)  # Should be formatted JSON
        assert priority == ResultPriority.MEDIUM

    def test_format_list_result_size_based_priority(self, formatter):
        """Test list result priority based on size."""
        # High priority for large list
        large_list = [f"item{i}" for i in range(15)]
        content, priority = formatter._format_list_result(large_list)
        assert priority == ResultPriority.HIGH

        # Medium priority for medium list
        medium_list = [f"item{i}" for i in range(5)]
        content, priority = formatter._format_list_result(medium_list)
        assert priority == ResultPriority.MEDIUM

        # Low priority for small list
        small_list = ["item1", "item2"]
        content, priority = formatter._format_list_result(small_list)
        assert priority == ResultPriority.LOW

    def test_format_error_result_cleanup(self, formatter):
        """Test error result formatting and cleanup."""
        # Should remove redundant prefix
        error_with_prefix = "Tool execution failed: Connection timeout"
        formatted = formatter._format_error_result("test_tool", error_with_prefix)
        assert formatted == "❌ ERROR in test_tool: Connection timeout"

        # Should keep other errors as-is
        other_error = "Authentication failed"
        formatted = formatter._format_error_result("test_tool", other_error)
        assert formatted == "❌ ERROR in test_tool: Authentication failed"

    def test_apply_structured_boundaries(self, formatter):
        """Test application of LlamaStack-style structured boundaries."""
        content = "Test result content"

        structured = formatter._apply_structured_boundaries(
            tool_name="example_tool",
            content=content,
            server_name="example_server",
            execution_time=1.5,
            arguments={"param": "value"},
        )

        # Should contain all expected elements
        assert "=== EXAMPLE_TOOL RESULT ===" in structured
        assert "Server: example_server" in structured
        assert "Execution Time: 1.50s" in structured
        assert "Arguments: param=value" in structured
        assert "BEGIN_CONTENT" in structured
        assert content in structured
        assert "END_CONTENT" in structured
        assert "=== END EXAMPLE_TOOL ===" in structured

    def test_apply_structured_boundaries_minimal(self, formatter):
        """Test structured boundaries with minimal information."""
        structured = formatter._apply_structured_boundaries(
            tool_name="simple_tool",
            content="Simple content",
            server_name="",
            execution_time=0,
            arguments={},
        )

        # Should still have basic structure
        assert "=== SIMPLE_TOOL RESULT ===" in structured
        assert "BEGIN_CONTENT" in structured
        assert "Simple content" in structured
        assert "END_CONTENT" in structured

    def test_apply_intelligent_truncation_no_truncation(self, formatter):
        """Test intelligent truncation when no truncation is needed."""
        result = FormattedResult(content="Short content")

        truncated = formatter._apply_intelligent_truncation(result)

        assert truncated.content == "Short content"
        assert truncated.truncated is False

    def test_apply_intelligent_truncation_with_structure(self, formatter):
        """Test intelligent truncation preserving structure."""
        # Create content with structure
        large_content = "x" * 2000
        structured_content = f"""=== TEST_TOOL RESULT ===

BEGIN_CONTENT
{large_content}
END_CONTENT

=== END TEST_TOOL ==="""

        result = FormattedResult(content=structured_content)
        result.tool_name = "test_tool"

        truncated = formatter._apply_intelligent_truncation(result)

        assert truncated.truncated is True
        assert len(truncated.content) <= formatter.max_result_size
        assert "BEGIN_CONTENT" in truncated.content
        assert "END_CONTENT" in truncated.content

    def test_smart_truncate_content_no_truncation(self, formatter):
        """Test smart content truncation when no truncation needed."""
        content = "Short content"

        truncated = formatter._smart_truncate_content(content, 1000)

        assert truncated == content

    def test_smart_truncate_content_line_preservation(self, formatter):
        """Test smart content truncation preserving complete lines."""
        lines = [f"Line {i} with some content" for i in range(20)]
        content = "\n".join(lines)

        truncated = formatter._smart_truncate_content(content, 200)

        assert "[TRUNCATED:" in truncated
        assert "more lines" in truncated
        # Should preserve complete lines
        truncated_lines = truncated.split("\n")
        for line in truncated_lines[:-1]:  # Exclude truncation notice
            if line and not line.startswith("..."):
                assert line.startswith("Line")

    def test_format_tool_result_formatting_error(self, formatter):
        """Test formatting error handling."""
        # Mock an error during formatting
        with patch.object(
            formatter, "_format_success_result", side_effect=Exception("Format error")
        ):
            result = formatter.format_tool_result(
                tool_name="error_tool", server_name="test_server", result="test content"
            )

            assert result.priority == ResultPriority.CRITICAL
            assert "[FORMATTING ERROR:" in result.content
            assert result.tool_name == "error_tool"

    def test_format_tool_result_non_json_serializable(self, formatter):
        """Test formatting non-JSON serializable objects."""

        # Create a non-serializable object
        class NonSerializable:
            def __str__(self):
                return "NonSerializable object"

        obj = NonSerializable()

        result = formatter.format_tool_result(
            tool_name="test_tool", server_name="test_server", result=obj
        )

        assert "NonSerializable object" in result.content


class TestResultPrioritizer:
    """Test ResultPrioritizer functionality."""

    @pytest.fixture
    def prioritizer(self):
        """Create ResultPrioritizer for testing."""
        return ResultPrioritizer(token_budget=1000)

    def test_initialization(self, prioritizer):
        """Test ResultPrioritizer initialization."""
        assert prioritizer.token_budget == 1000
        assert isinstance(prioritizer.formatter, GenericResultFormatter)

    def test_prioritize_results_sorting(self, prioritizer):
        """Test result prioritization and sorting."""
        tool_results = [
            {"tool_name": "low_priority", "result": "short", "status": "success"},
            {
                "tool_name": "high_priority",
                "result": "error occurred",
                "status": "error",
            },
            {
                "tool_name": "medium_priority",
                "result": "found 50 items",
                "status": "success",
            },
        ]

        prioritized = prioritizer.prioritize_results(tool_results)

        # Should be sorted by priority (lower number = higher priority)
        priorities = [result.priority for result in prioritized]
        assert priorities == sorted(priorities)

    def test_prioritize_results_token_budget_enforcement(self, prioritizer):
        """Test token budget enforcement."""
        # Create results that exceed token budget
        large_result = "x" * 3000  # ~750 tokens
        tool_results = [
            {"tool_name": "tool1", "result": large_result, "status": "success"},
            {"tool_name": "tool2", "result": large_result, "status": "success"},
            {"tool_name": "tool3", "result": large_result, "status": "success"},
        ]

        prioritized = prioritizer.prioritize_results(tool_results)

        # Should limit results based on token budget
        total_tokens = sum(result.size_bytes // 4 for result in prioritized)
        assert total_tokens <= prioritizer.token_budget

    def test_prioritize_results_high_priority_summarization(self, prioritizer):
        """Test summarization of high priority results when they exceed budget."""
        # Create a high priority result that's too large for budget
        prioritizer.token_budget = 100  # Very small budget

        tool_results = [
            {
                "tool_name": "critical_tool",
                "result": "error: " + "x" * 1000,
                "status": "error",
            }
        ]

        prioritized = prioritizer.prioritize_results(tool_results)

        assert len(prioritized) == 1
        assert prioritized[0].truncated is True
        assert "SUMMARIZED" in prioritized[0].content

    def test_apply_token_budget_drop_low_priority(self, prioritizer):
        """Test dropping low priority results to fit budget."""
        results = [
            FormattedResult(
                content="x" * 2000,
                priority=ResultPriority.CRITICAL,
                tool_name="critical",
            ),
            FormattedResult(
                content="x" * 2400, priority=ResultPriority.LOW, tool_name="low"
            ),  # 600 tokens, exceeds budget
        ]

        final_results = prioritizer._apply_token_budget(results)

        # Should keep critical, drop low priority
        assert len(final_results) == 1
        assert final_results[0].tool_name == "critical"

    def test_create_summary_for_budget_truncation(self, prioritizer):
        """Test summary creation for budget-truncated results."""
        original_result = FormattedResult(
            content="x" * 1000,
            priority=ResultPriority.HIGH,
            tool_name="large_tool",
            server_name="test_server",
            execution_time=2.5,
            metadata={"status": "success"},
        )

        summary = prioritizer._create_summary(original_result, 200)

        assert summary.truncated is True
        assert "SUMMARIZED" in summary.content
        assert "LARGE_TOOL" in summary.content  # Tool name is uppercased in summary
        assert "test_server" in summary.content
        assert "2.50s" in summary.content
        assert "BUDGET TRUNCATED" in summary.content
