"""Unit tests for knowledge store data preservation."""

import pytest

from cicaddy.execution.knowledge_store import AccumulatedKnowledge


class TestAccumulatedKnowledge:
    """Test the knowledge accumulation store for MCP tool results."""

    def test_initialize_empty_store(self):
        """Test that knowledge store initializes empty."""
        store = AccumulatedKnowledge()

        assert len(store.tool_results) == 0
        assert store.total_tools_executed == 0
        assert len(store.servers_used) == 0
        assert len(store.tools_used) == 0

    def test_add_tool_result(self):
        """Test adding a single tool result."""
        store = AccumulatedKnowledge()

        store.add_tool_result(
            iteration=1,
            server="datarouter",
            tool="getTotalRequestNumberFromPastDays",
            arguments={"numOfDays": "1"},
            result={"total": 15234},
            execution_time=0.5,
            result_size_bytes=100,
        )

        assert len(store.tool_results) == 1
        assert store.total_tools_executed == 1
        assert "datarouter" in store.servers_used
        assert "getTotalRequestNumberFromPastDays" in store.tools_used

        # Check result structure
        result = store.tool_results[0]
        assert result["iteration"] == 1
        assert result["server"] == "datarouter"
        assert result["tool"] == "getTotalRequestNumberFromPastDays"
        assert result["arguments"] == {"numOfDays": "1"}
        assert result["result"] == {"total": 15234}
        assert result["execution_time"] == 0.5
        assert result["result_size_bytes"] == 100

    def test_add_multiple_tool_results(self):
        """Test adding multiple tool results from different servers."""
        store = AccumulatedKnowledge()

        # Add DataRouter tool
        store.add_tool_result(
            iteration=1,
            server="datarouter",
            tool="getTotalRequests",
            arguments={"numOfDays": "1"},
            result={"total": 15234},
        )

        # Add GitHub tool
        store.add_tool_result(
            iteration=2,
            server="github",
            tool="search_code",
            arguments={"query": "function.*main"},
            result=[{"file": "main.py", "line": 10}],
        )

        # Add another DataRouter tool
        store.add_tool_result(
            iteration=3,
            server="datarouter",
            tool="getComponentStatus",
            arguments={"numOfDays": "1"},
            result=[{"component": "api-gateway"}],
        )

        assert len(store.tool_results) == 3
        assert store.total_tools_executed == 3
        assert "datarouter" in store.servers_used
        assert "github" in store.servers_used
        assert len(store.servers_used) == 2

    def test_get_results_for_server(self):
        """Test retrieving results for a specific server."""
        store = AccumulatedKnowledge()

        store.add_tool_result(
            iteration=1,
            server="datarouter",
            tool="tool1",
            arguments={},
            result={"data": 1},
        )
        store.add_tool_result(
            iteration=2, server="github", tool="tool2", arguments={}, result={"data": 2}
        )
        store.add_tool_result(
            iteration=3,
            server="datarouter",
            tool="tool3",
            arguments={},
            result={"data": 3},
        )

        datarouter_results = store.get_results_for_server("datarouter")
        assert len(datarouter_results) == 2
        assert all(r["server"] == "datarouter" for r in datarouter_results)

        github_results = store.get_results_for_server("github")
        assert len(github_results) == 1
        assert github_results[0]["server"] == "github"

    def test_get_results_for_tool(self):
        """Test retrieving results for a specific tool."""
        store = AccumulatedKnowledge()

        store.add_tool_result(
            iteration=1,
            server="datarouter",
            tool="getTotalRequests",
            arguments={"days": 1},
            result={"total": 100},
        )
        store.add_tool_result(
            iteration=2,
            server="datarouter",
            tool="getTotalRequests",
            arguments={"days": 7},
            result={"total": 700},
        )
        store.add_tool_result(
            iteration=3,
            server="datarouter",
            tool="getComponentStatus",
            arguments={},
            result=[],
        )

        total_requests_results = store.get_results_for_tool("getTotalRequests")
        assert len(total_requests_results) == 2
        assert all(r["tool"] == "getTotalRequests" for r in total_requests_results)

    def test_get_latest_result(self):
        """Test getting the most recent result for a tool."""
        store = AccumulatedKnowledge()

        store.add_tool_result(
            iteration=1,
            server="datarouter",
            tool="getTotalRequests",
            arguments={"days": 1},
            result={"total": 100},
        )
        store.add_tool_result(
            iteration=2,
            server="datarouter",
            tool="getTotalRequests",
            arguments={"days": 7},
            result={"total": 700},
        )

        latest = store.get_latest_result("getTotalRequests")
        assert latest is not None
        assert latest["iteration"] == 2
        assert latest["result"]["total"] == 700

    def test_get_results_by_iteration(self):
        """Test retrieving all results from a specific iteration."""
        store = AccumulatedKnowledge()

        store.add_tool_result(
            iteration=1,
            server="datarouter",
            tool="tool1",
            arguments={},
            result={"data": 1},
        )
        store.add_tool_result(
            iteration=2,
            server="datarouter",
            tool="tool2",
            arguments={},
            result={"data": 2},
        )
        store.add_tool_result(
            iteration=2, server="github", tool="tool3", arguments={}, result={"data": 3}
        )
        store.add_tool_result(
            iteration=3,
            server="datarouter",
            tool="tool4",
            arguments={},
            result={"data": 4},
        )

        iteration_2_results = store.get_results_by_iteration(2)
        assert len(iteration_2_results) == 2
        assert all(r["iteration"] == 2 for r in iteration_2_results)

    def test_get_total_execution_time(self):
        """Test calculating total execution time."""
        store = AccumulatedKnowledge()

        store.add_tool_result(
            iteration=1,
            server="datarouter",
            tool="tool1",
            arguments={},
            result={},
            execution_time=0.5,
        )
        store.add_tool_result(
            iteration=2,
            server="datarouter",
            tool="tool2",
            arguments={},
            result={},
            execution_time=1.2,
        )
        store.add_tool_result(
            iteration=3,
            server="github",
            tool="tool3",
            arguments={},
            result={},
            execution_time=0.8,
        )

        total_time = store.get_total_execution_time()
        assert total_time == pytest.approx(2.5)

    def test_get_total_data_size(self):
        """Test calculating total data size."""
        store = AccumulatedKnowledge()

        store.add_tool_result(
            iteration=1,
            server="datarouter",
            tool="tool1",
            arguments={},
            result={},
            result_size_bytes=100,
        )
        store.add_tool_result(
            iteration=2,
            server="datarouter",
            tool="tool2",
            arguments={},
            result={},
            result_size_bytes=200,
        )
        store.add_tool_result(
            iteration=3,
            server="github",
            tool="tool3",
            arguments={},
            result={},
            result_size_bytes=150,
        )

        total_size = store.get_total_data_size()
        assert total_size == 450

    def test_to_dict_serialization(self):
        """Test dictionary serialization for JSON export."""
        store = AccumulatedKnowledge()

        store.add_tool_result(
            iteration=1,
            server="datarouter",
            tool="getTotalRequests",
            arguments={"numOfDays": "1"},
            result={"total": 15234},
            execution_time=0.5,
            result_size_bytes=100,
        )

        data = store.to_dict()

        assert "tool_results" in data
        assert "results_by_server" in data
        assert "results_by_tool" in data
        assert "total_tools_executed" in data
        assert "servers_used" in data
        assert "tools_used" in data
        assert "total_execution_time" in data
        assert "total_data_size_bytes" in data

        assert data["total_tools_executed"] == 1
        assert "datarouter" in data["servers_used"]
        assert "getTotalRequests" in data["tools_used"]
        assert data["total_execution_time"] == 0.5
        assert data["total_data_size_bytes"] == 100

    def test_from_dict_reconstruction(self):
        """Test reconstructing knowledge store from dictionary."""
        store = AccumulatedKnowledge()

        store.add_tool_result(
            iteration=1,
            server="datarouter",
            tool="tool1",
            arguments={"test": "value"},
            result={"data": 123},
        )
        store.add_tool_result(
            iteration=2,
            server="github",
            tool="tool2",
            arguments={},
            result=["item1", "item2"],
        )

        data = store.to_dict()
        reconstructed = AccumulatedKnowledge.from_dict(data)

        assert reconstructed.total_tools_executed == store.total_tools_executed
        assert len(reconstructed.tool_results) == len(store.tool_results)
        assert reconstructed.servers_used == store.servers_used
        assert reconstructed.tools_used == store.tools_used

    def test_preserve_full_data_not_compacted(self):
        """Test that full data is preserved without compression."""
        store = AccumulatedKnowledge()

        # Large result that would be compacted in conversation
        large_result = {
            "data": ["item" + str(i) for i in range(1000)],
            "metadata": {"count": 1000, "size": "large"},
        }

        store.add_tool_result(
            iteration=1,
            server="datarouter",
            tool="getLargeDataset",
            arguments={},
            result=large_result,
        )

        # Verify full data is preserved
        retrieved = store.get_latest_result("getLargeDataset")
        assert retrieved is not None
        assert len(retrieved["result"]["data"]) == 1000
        assert retrieved["result"]["metadata"]["count"] == 1000

    def test_multiple_servers_indexing(self):
        """Test that results are correctly indexed by multiple servers."""
        store = AccumulatedKnowledge()

        # Add results from multiple servers
        store.add_tool_result(
            iteration=1, server="datarouter", tool="tool1", arguments={}, result={}
        )
        store.add_tool_result(
            iteration=1, server="github", tool="tool2", arguments={}, result={}
        )
        store.add_tool_result(
            iteration=1, server="sourcebot", tool="tool3", arguments={}, result={}
        )
        store.add_tool_result(
            iteration=2, server="datarouter", tool="tool4", arguments={}, result={}
        )

        assert len(store.servers_used) == 3
        assert "datarouter" in store.servers_used
        assert "github" in store.servers_used
        assert "sourcebot" in store.servers_used

        # Verify indexing
        assert len(store.results_by_server["datarouter"]) == 2
        assert len(store.results_by_server["github"]) == 1
        assert len(store.results_by_server["sourcebot"]) == 1

    def test_repr_string(self):
        """Test string representation for debugging."""
        store = AccumulatedKnowledge()

        store.add_tool_result(
            iteration=1,
            server="datarouter",
            tool="tool1",
            arguments={},
            result={},
            result_size_bytes=100,
        )
        store.add_tool_result(
            iteration=2,
            server="github",
            tool="tool2",
            arguments={},
            result={},
            result_size_bytes=200,
        )

        repr_str = repr(store)

        assert "AccumulatedKnowledge" in repr_str
        assert "tools=2" in repr_str
        assert "datarouter" in repr_str
        assert "github" in repr_str
        assert "300bytes" in repr_str
