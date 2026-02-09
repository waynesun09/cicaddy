"""Tests for ProgressiveMemory."""

import pytest

from cicaddy.execution.progressive_memory import ProgressiveMemory


class TestProgressiveMemory:
    """Test suite for ProgressiveMemory."""

    @pytest.fixture
    def memory(self):
        """Create a memory instance."""
        return ProgressiveMemory(
            tier1_max_tokens=1000,
            tier2_max_tokens=500,
            keep_raw_storage=True,
            max_entities_per_type=50,
        )

    @pytest.fixture
    def sample_tool_result(self):
        """Sample tool result."""
        return """repository: github.com/test/repo1
file: https://github.com/test/repo1/file1.py
num_matches: 10
language: Python"""

    def test_initialization(self, memory):
        """Test memory initialization."""
        assert memory.tier1_max_tokens == 1000
        assert memory.tier2_max_tokens == 500
        assert memory.keep_raw_storage is True
        assert len(memory.raw_storage) == 0
        assert len(memory.tier1_summaries) == 0
        assert len(memory.metrics_cache) == 0

    def test_add_tool_result(self, memory, sample_tool_result):
        """Test adding a tool result."""
        processed = memory.add_tool_result(
            tool_id="tool_1",
            result=sample_tool_result,
            tool_name="search_code",
            tool_server="sourcebot",
        )

        # Check raw storage
        assert "tool_1" in memory.raw_storage
        assert memory.raw_storage["tool_1"] == sample_tool_result

        # Check metrics cache
        assert "tool_1" in memory.metrics_cache
        assert memory.metrics_cache["tool_1"]["tool_name"] == "search_code"
        assert memory.metrics_cache["tool_1"]["tool_server"] == "sourcebot"

        # Check tier1 summary
        assert "tool_1" in memory.tier1_summaries
        assert memory.tier1_summaries["tool_1"]["tool_name"] == "search_code"

        # Check processed result structure
        assert "metadata" in processed
        assert "metrics" in processed
        assert "entities" in processed

    def test_add_multiple_results(self, memory, sample_tool_result):
        """Test adding multiple tool results."""
        memory.add_tool_result("tool_1", sample_tool_result, "search_1")
        memory.add_tool_result("tool_2", sample_tool_result, "search_2")
        memory.add_tool_result("tool_3", sample_tool_result, "search_3")

        assert len(memory.raw_storage) == 3
        assert len(memory.metrics_cache) == 3
        assert len(memory.tier1_summaries) == 3

    def test_get_aggregated_metrics(self, memory, sample_tool_result):
        """Test aggregated metrics calculation."""
        memory.add_tool_result("tool_1", sample_tool_result, "search_1")
        memory.add_tool_result("tool_2", sample_tool_result, "search_2")

        aggregated = memory.get_aggregated_metrics()

        assert aggregated["total_tools"] == 2
        assert aggregated["total_matches"] == 20  # 10 * 2
        assert aggregated["unique_repository_count"] == 1  # Same repo
        assert aggregated["tools_with_data"] == 2

    def test_get_final_summary(self, memory, sample_tool_result):
        """Test final summary generation."""
        memory.add_tool_result("tool_1", sample_tool_result, "search_1")
        memory.add_tool_result("tool_2", sample_tool_result, "search_2")

        final = memory.get_final_summary()

        assert "summary" in final
        assert "metrics" in final
        assert "tier1_summaries" in final
        assert "raw_available" in final
        assert "validation_ready" in final

        assert final["summary"]["total_tools"] == 2
        assert final["validation_ready"] is True
        assert len(final["raw_available"]) == 2

    def test_get_raw_result(self, memory, sample_tool_result):
        """Test retrieving raw results."""
        memory.add_tool_result("tool_1", sample_tool_result, "search_1")

        raw = memory.get_raw_result("tool_1")
        assert raw == sample_tool_result

        # Non-existent tool
        assert memory.get_raw_result("tool_999") is None

    def test_get_tool_metrics(self, memory, sample_tool_result):
        """Test retrieving tool metrics."""
        memory.add_tool_result("tool_1", sample_tool_result, "search_1", "sourcebot")

        metrics = memory.get_tool_metrics("tool_1")
        assert metrics is not None
        assert metrics["tool_name"] == "search_1"
        assert metrics["tool_server"] == "sourcebot"
        assert "metrics" in metrics
        assert "entities" in metrics

    def test_get_tools_by_criteria_matches(self, memory):
        """Test filtering tools by minimum matches."""
        memory.add_tool_result("tool_1", "num_matches: 5", "search_1")
        memory.add_tool_result("tool_2", "num_matches: 15", "search_2")
        memory.add_tool_result("tool_3", "num_matches: 25", "search_3")

        # Find tools with at least 10 matches
        matching = memory.get_tools_by_criteria(min_matches=10)

        assert len(matching) == 2
        assert "tool_1" not in matching
        assert "tool_2" in matching
        assert "tool_3" in matching

    def test_get_tools_by_criteria_errors(self, memory):
        """Test filtering tools by error status."""
        memory.add_tool_result("tool_1", "Success: found data", "search_1")
        memory.add_tool_result("tool_2", "Error: connection failed", "search_2")

        # Find tools with errors
        with_errors = memory.get_tools_by_criteria(has_errors=True)
        assert len(with_errors) == 1
        assert "tool_2" in with_errors

        # Find tools without errors
        without_errors = memory.get_tools_by_criteria(has_errors=False)
        assert len(without_errors) == 1
        assert "tool_1" in without_errors

    def test_get_tools_by_criteria_server(self, memory, sample_tool_result):
        """Test filtering tools by server name."""
        memory.add_tool_result("tool_1", sample_tool_result, "search_1", "sourcebot")
        memory.add_tool_result("tool_2", sample_tool_result, "search_2", "github")

        sourcebot_tools = memory.get_tools_by_criteria(tool_server="sourcebot")
        assert len(sourcebot_tools) == 1
        assert "tool_1" in sourcebot_tools

    def test_get_top_results_by_matches(self, memory):
        """Test getting top N results sorted by matches."""
        memory.add_tool_result("tool_1", "num_matches: 5", "search_1")
        memory.add_tool_result("tool_2", "num_matches: 25", "search_2")
        memory.add_tool_result("tool_3", "num_matches: 15", "search_3")

        top_2 = memory.get_top_results(n=2, sort_by="matches")

        assert len(top_2) == 2
        assert top_2[0]["tool_name"] == "search_2"  # Highest
        assert top_2[1]["tool_name"] == "search_3"  # Second

    def test_clear(self, memory, sample_tool_result):
        """Test clearing all memory."""
        memory.add_tool_result("tool_1", sample_tool_result, "search_1")
        memory.add_tool_result("tool_2", sample_tool_result, "search_2")

        assert len(memory.raw_storage) == 2

        memory.clear()

        assert len(memory.raw_storage) == 0
        assert len(memory.tier1_summaries) == 0
        assert len(memory.metrics_cache) == 0
        assert memory.tier2_summary is None

    def test_get_memory_stats(self, memory, sample_tool_result):
        """Test memory statistics."""
        memory.add_tool_result("tool_1", sample_tool_result, "search_1")
        memory.add_tool_result("tool_2", sample_tool_result, "search_2")

        stats = memory.get_memory_stats()

        assert stats["total_tools"] == 2
        assert stats["raw_storage_enabled"] is True
        assert stats["raw_storage_size_bytes"] > 0
        assert stats["tier1_summaries_count"] == 2
        assert stats["tier2_summary_exists"] is False  # Not generated yet

        # Generate final summary
        memory.get_final_summary()

        stats = memory.get_memory_stats()
        assert stats["tier2_summary_exists"] is True

    def test_no_raw_storage(self):
        """Test with raw storage disabled."""
        memory = ProgressiveMemory(keep_raw_storage=False)

        result = "test result"
        memory.add_tool_result("tool_1", result, "test_tool")

        assert len(memory.raw_storage) == 0
        assert memory.get_raw_result("tool_1") is None
        assert len(memory.metrics_cache) == 1  # Metrics still cached

    def test_preview_creation(self, memory):
        """Test result preview creation."""
        long_result = "A" * 1000  # 1000 char result

        memory.add_tool_result("tool_1", long_result, "test_tool")

        summary = memory.tier1_summaries["tool_1"]
        preview = summary["result_preview"]

        # Preview should be truncated to 500 chars + "..."
        assert len(preview) <= 503
        assert preview.endswith("...")

    def test_complex_aggregation(self, memory):
        """Test complex aggregation with varied data."""
        # Different results with different characteristics
        results = [
            (
                "tool_1",
                "repository: github.com/repo1\nnum_matches: 10",
                "search_1",
                "sourcebot",
            ),
            (
                "tool_2",
                "repository: github.com/repo2\nnum_matches: 20\nrepository: github.com/repo3",
                "search_2",
                "sourcebot",
            ),
            ("tool_3", "Error: failed", "search_3", "github"),
            (
                "tool_4",
                "repository: github.com/repo1\nnum_matches: 5",
                "search_4",
                "sourcebot",
            ),
        ]

        for tool_id, result, tool_name, tool_server in results:
            memory.add_tool_result(tool_id, result, tool_name, tool_server)

        aggregated = memory.get_aggregated_metrics()

        assert aggregated["total_tools"] == 4
        assert aggregated["total_matches"] == 35  # 10 + 20 + 0 + 5
        assert aggregated["tools_with_errors"] == 1
        assert aggregated["tools_with_data"] == 3
        assert aggregated["unique_repository_count"] == 3  # repo1, repo2, repo3

    def test_final_summary_structure(self, memory, sample_tool_result):
        """Test final summary has complete structure."""
        memory.add_tool_result("tool_1", sample_tool_result, "search_1")

        final = memory.get_final_summary()

        # Check tier 2 summary structure
        assert "total_tools" in final["summary"]
        assert "tools_executed" in final["summary"]
        assert "aggregated_metrics" in final["summary"]
        assert "created_at" in final["summary"]
        assert "updated_at" in final["summary"]

        # Check metrics completeness
        assert "total_repositories" in final["metrics"]
        assert "unique_repository_count" in final["metrics"]
        assert "unique_repositories_list" in final["metrics"]
