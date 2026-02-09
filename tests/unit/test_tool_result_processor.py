"""Tests for ToolResultProcessor."""

import pytest

from cicaddy.execution.tool_result_processor import ToolResultProcessor


class TestToolResultProcessor:
    """Test suite for ToolResultProcessor."""

    @pytest.fixture
    def processor(self):
        """Create a processor instance."""
        return ToolResultProcessor(max_entities_per_type=50)

    @pytest.fixture
    def sample_code_search_result(self):
        """Sample code search result from sourcebot."""
        return """file: https://github.com/neuralmagic/langchain/blob/HEAD/docs/docs/guides/debugging.md
num_matches: 33
repository: github.com/neuralmagic/langchain
language: Markdown
file: https://github.com/neuralmagic/langchain/blob/HEAD/cookbook/wikibase_agent.ipynb
num_matches: 4
repository: github.com/neuralmagic/langchain
language: Jupyter Notebook
file: https://github.com/openshift/sippy/blob/HEAD/chat/sippy_agent/graph.py
num_matches: 1
repository: github.com/openshift/sippy
language: Python"""

    def test_process_basic_result(self, processor, sample_code_search_result):
        """Test basic result processing."""
        result = processor.process(
            tool_name="search_code",
            tool_result=sample_code_search_result,
            tool_server="sourcebot",
        )

        # Check structure
        assert "raw_result" in result
        assert "metadata" in result
        assert "metrics" in result
        assert "entities" in result
        assert "summary_stats" in result

        # Check tool info
        assert result["tool_name"] == "search_code"
        assert result["tool_server"] == "sourcebot"

    def test_extract_metadata(self, processor, sample_code_search_result):
        """Test metadata extraction."""
        result = processor.process("search_code", sample_code_search_result)
        metadata = result["metadata"]

        assert metadata["line_count"] > 0
        assert metadata["char_count"] > 0
        assert metadata["has_errors"] is False
        assert metadata["result_type"] == "code_search"

    def test_extract_metrics(self, processor, sample_code_search_result):
        """Test metrics extraction."""
        result = processor.process("search_code", sample_code_search_result)
        counts = result["metrics"]["counts"]

        # Should find 2 unique repositories
        assert counts["repository"] == 3  # Total mentions (neuralmagic appears twice)

        # Should find 3 files
        assert counts["file"] == 3

        # Should sum all num_matches: 33 + 4 + 1 = 38
        assert counts["match"] == 38

        # No errors
        assert counts["error"] == 0

    def test_extract_entities(self, processor, sample_code_search_result):
        """Test entity extraction."""
        result = processor.process("search_code", sample_code_search_result)
        entities = result["entities"]

        # Should find 2 unique repositories (neuralmagic/langchain appears twice but deduplicated)
        assert len(entities["repositories"]) == 2  # Unique repositories
        assert "github.com/neuralmagic/langchain" in entities["repositories"]
        assert "github.com/openshift/sippy" in entities["repositories"]

        # Should find 3 unique files
        assert len(entities["files"]) == 3

        # Should find languages
        assert len(entities["languages"]) == 3
        assert "Markdown" in entities["languages"]
        assert "Python" in entities["languages"]

    def test_summary_stats(self, processor, sample_code_search_result):
        """Test summary statistics calculation."""
        result = processor.process("search_code", sample_code_search_result)
        stats = result["summary_stats"]

        assert stats["has_significant_data"] is True
        assert stats["unique_repositories"] == 2  # 2 unique repos
        assert stats["unique_files"] == 3
        assert stats["unique_languages"] == 3
        assert stats["total_entities"] > 0

    def test_aggregate_metrics_empty(self, processor):
        """Test aggregation with no results."""
        aggregated = processor.aggregate_metrics([])

        assert aggregated["total_tools"] == 0
        assert aggregated["total_repositories"] == 0
        assert aggregated["total_files"] == 0
        assert aggregated["total_matches"] == 0

    def test_aggregate_metrics_multiple_results(
        self, processor, sample_code_search_result
    ):
        """Test aggregation across multiple results."""
        # Process same result twice
        result1 = processor.process("search_code_1", sample_code_search_result)
        result2 = processor.process("search_code_2", sample_code_search_result)

        aggregated = processor.aggregate_metrics([result1, result2])

        assert aggregated["total_tools"] == 2
        assert aggregated["total_matches"] == 76  # 38 * 2
        assert (
            aggregated["unique_repository_count"] == 2
        )  # 2 unique (duplicates removed)
        assert aggregated["tools_with_data"] == 2

    def test_error_result_detection(self, processor):
        """Test detection of error results."""
        error_result = "Error: Connection failed\nException occurred during search"

        result = processor.process("search_code", error_result)

        assert result["metadata"]["has_errors"] is True
        assert result["metadata"]["result_type"] == "error"
        assert (
            result["metrics"]["counts"]["error"] >= 2
        )  # At least 2 error keywords found

    def test_json_result_detection(self, processor):
        """Test detection of JSON results."""
        json_result = '{"repositories": ["repo1", "repo2"], "count": 5}'

        result = processor.process("get_data", json_result)

        assert result["metadata"]["result_type"] == "json"

    def test_max_entities_limit(self):
        """Test that max entities per type is respected."""
        processor = ToolResultProcessor(max_entities_per_type=2)

        # Create result with many repositories
        many_repos = "\n".join(
            [
                f"repository: github.com/repo{i}\nfile: https://github.com/repo{i}/file.py"
                for i in range(10)
            ]
        )

        result = processor.process("search_code", many_repos)

        # Should be limited to 2
        assert len(result["entities"]["repositories"]) == 2
        assert len(result["entities"]["files"]) == 2

    def test_empty_result(self, processor):
        """Test processing empty result."""
        result = processor.process("search_code", "")

        assert result["metadata"]["char_count"] == 0
        assert result["metadata"]["line_count"] == 1  # Empty string has 1 line
        assert result["metrics"]["counts"]["repository"] == 0
        assert result["summary_stats"]["has_significant_data"] is False

    def test_llama_stack_result(self, processor):
        """Test with Llama Stack agent patterns."""
        llama_result = """file: https://github.com/opendatahub-io/llama-stack-demos/blob/HEAD/demos/a2a_llama_stack/notebooks/A2A_Advanced_Multi_Agent.ipynb  # noqa: E501
num_matches: 19
repository: github.com/opendatahub-io/llama-stack-demos
language: Jupyter Notebook
file: https://github.com/opendatahub-io/opendatahub-tests/blob/HEAD/tests/llama_stack/agents/test_agents.py
num_matches: 11
repository: github.com/opendatahub-io/opendatahub-tests
language: Python"""

        result = processor.process("search_code", llama_result, "sourcebot")

        assert result["metrics"]["counts"]["match"] == 30  # 19 + 11
        assert result["metrics"]["counts"]["repository"] == 2
        assert result["summary_stats"]["has_significant_data"] is True
        assert (
            "github.com/opendatahub-io/llama-stack-demos"
            in result["entities"]["repositories"]
        )
