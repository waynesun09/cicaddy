"""Progressive memory architecture for maintaining detailed and summarized views."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from cicaddy.utils.logger import get_logger

from .tool_result_processor import ToolResultProcessor

logger = get_logger(__name__)


class ProgressiveMemory:
    """
    Two-tier memory architecture that maintains both detailed and summarized views.

    Tiers:
    - Tier 0: Raw storage (always preserved)
    - Tier 1: Per-tool summaries (context preserved)
    - Tier 2: Final aggregate (validated)
    - Metrics cache (never compressed)
    """

    def __init__(
        self,
        tier1_max_tokens: int = 1000,
        tier2_max_tokens: int = 500,
        keep_raw_storage: bool = True,
        max_entities_per_type: int = 100,
    ):
        """
        Initialize progressive memory.

        Args:
            tier1_max_tokens: Target token limit for per-tool summaries
            tier2_max_tokens: Target token limit for final aggregate
            keep_raw_storage: Whether to keep raw tool results
            max_entities_per_type: Maximum entities to track per type
        """
        self.tier1_max_tokens = tier1_max_tokens
        self.tier2_max_tokens = tier2_max_tokens
        self.keep_raw_storage = keep_raw_storage

        # Storage tiers
        self.raw_storage: Dict[str, str] = {}  # Full tool results
        self.tier1_summaries: Dict[str, Dict[str, Any]] = {}  # Per-tool summaries
        self.tier2_summary: Optional[Dict[str, Any]] = None  # Final aggregate
        self.metrics_cache: Dict[str, Dict[str, Any]] = {}  # Always preserved

        # Processor
        self.processor = ToolResultProcessor(
            max_entities_per_type=max_entities_per_type
        )

        # Metadata
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

        logger.info(
            f"ProgressiveMemory initialized with tier1={tier1_max_tokens}t, "
            f"tier2={tier2_max_tokens}t, raw_storage={keep_raw_storage}"
        )

    def add_tool_result(
        self,
        tool_id: str,
        result: str,
        tool_name: str = "unknown",
        tool_server: str = "",
    ) -> Dict[str, Any]:
        """
        Store and progressively summarize a tool result.

        Args:
            tool_id: Unique identifier for this tool execution
            result: Raw result string
            tool_name: Name of the tool
            tool_server: MCP server name

        Returns:
            Processed result with metrics
        """
        logger.debug(f"Adding tool result {tool_id} ({tool_name})")

        # Tier 0: Raw storage (always kept if enabled)
        if self.keep_raw_storage:
            self.raw_storage[tool_id] = result

        # Extract metrics BEFORE any AI summarization
        processed = self.processor.process(tool_name, result, tool_server)
        self.metrics_cache[tool_id] = {
            "metadata": processed["metadata"],
            "metrics": processed["metrics"],
            "entities": processed["entities"],
            "summary_stats": processed["summary_stats"],
            "tool_name": tool_name,
            "tool_server": tool_server,
            "timestamp": datetime.now().isoformat(),
        }

        # Tier 1: Store processed result as summary
        # (In a full implementation, this would call AI to create a summary)
        self.tier1_summaries[tool_id] = {
            "tool_id": tool_id,
            "tool_name": tool_name,
            "tool_server": tool_server,
            "metrics": processed["metrics"],
            "entities": processed["entities"],
            "summary_stats": processed["summary_stats"],
            "result_preview": self._create_preview(result, max_chars=500),
            "timestamp": datetime.now().isoformat(),
        }

        self.updated_at = datetime.now()

        logger.debug(
            f"Tool {tool_id} stored: {processed['summary_stats']['unique_repositories']} repos, "
            f"{processed['metrics']['counts']['match']} matches"
        )

        return processed

    def _create_preview(self, text: str, max_chars: int = 500) -> str:
        """
        Create a preview of text limited to max_chars.

        Args:
            text: Full text
            max_chars: Maximum characters in preview

        Returns:
            Preview string with ellipsis if truncated
        """
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated metrics across all stored tool results.

        Returns:
            Dictionary with aggregated metrics
        """
        processed_results = list(self.metrics_cache.values())
        aggregated = self.processor.aggregate_metrics(processed_results)

        return aggregated

    def get_final_summary(self, include_validation: bool = True) -> Dict[str, Any]:
        """
        Generate final summary with validation.

        Args:
            include_validation: Whether to include validation metadata

        Returns:
            Dictionary containing:
                - summary: Aggregated summary data
                - metrics: Aggregated metrics
                - tier1_summaries: Per-tool summaries
                - raw_available: List of tool IDs with raw data
                - validation_ready: Whether data is ready for validation
        """
        logger.debug("Generating final summary")

        aggregated_metrics = self.get_aggregated_metrics()

        # Tier 2: Create aggregate summary
        self.tier2_summary = {
            "total_tools": len(self.tier1_summaries),
            "tools_executed": list(self.tier1_summaries.keys()),
            "aggregated_metrics": aggregated_metrics,
            "tier1_summaries": self.tier1_summaries if include_validation else {},
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

        final = {
            "summary": self.tier2_summary,
            "metrics": aggregated_metrics,
            "tier1_summaries": self.tier1_summaries,
            "raw_available": list(self.raw_storage.keys())
            if self.keep_raw_storage
            else [],
            "validation_ready": len(self.metrics_cache) > 0,
        }

        logger.info(
            f"Final summary generated: {aggregated_metrics.get('total_tools', 0)} tools, "
            f"{aggregated_metrics.get('unique_repository_count', 0)} unique repos"
        )

        return final

    def get_raw_result(self, tool_id: str) -> Optional[str]:
        """
        Retrieve raw result for a specific tool.

        Args:
            tool_id: Tool identifier

        Returns:
            Raw result string or None if not found
        """
        return self.raw_storage.get(tool_id)

    def get_tool_metrics(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metrics for a specific tool.

        Args:
            tool_id: Tool identifier

        Returns:
            Metrics dictionary or None if not found
        """
        return self.metrics_cache.get(tool_id)

    def get_tools_by_criteria(
        self,
        min_matches: Optional[int] = None,
        min_repositories: Optional[int] = None,
        has_errors: Optional[bool] = None,
        tool_server: Optional[str] = None,
    ) -> List[str]:
        """
        Find tools matching specific criteria.

        Args:
            min_matches: Minimum number of code matches
            min_repositories: Minimum number of repositories
            has_errors: Filter by error presence
            tool_server: Filter by MCP server name

        Returns:
            List of tool IDs matching criteria
        """
        matching_tools = []

        for tool_id, metrics in self.metrics_cache.items():
            # Apply filters
            if min_matches is not None:
                if metrics["metrics"]["counts"].get("match", 0) < min_matches:
                    continue

            if min_repositories is not None:
                if (
                    metrics["summary_stats"].get("unique_repositories", 0)
                    < min_repositories
                ):
                    continue

            if has_errors is not None:
                if metrics["metadata"].get("has_errors", False) != has_errors:
                    continue

            if tool_server is not None:
                if metrics.get("tool_server", "") != tool_server:
                    continue

            matching_tools.append(tool_id)

        return matching_tools

    def get_top_results(
        self, n: int = 5, sort_by: str = "matches"
    ) -> List[Dict[str, Any]]:
        """
        Get top N tool results sorted by a metric.

        Args:
            n: Number of results to return
            sort_by: Metric to sort by ('matches', 'repositories', 'files')

        Returns:
            List of top tool summaries
        """
        sort_key_map = {
            "matches": lambda x: x[1]["metrics"]["counts"].get("match", 0),
            "repositories": lambda x: x[1]["summary_stats"].get(
                "unique_repositories", 0
            ),
            "files": lambda x: x[1]["summary_stats"].get("unique_files", 0),
        }

        sort_key = sort_key_map.get(sort_by, sort_key_map["matches"])

        sorted_tools = sorted(self.metrics_cache.items(), key=sort_key, reverse=True)

        top_results = []
        for tool_id, metrics in sorted_tools[:n]:
            top_results.append(
                {
                    "tool_id": tool_id,
                    "tool_name": metrics.get("tool_name"),
                    "tool_server": metrics.get("tool_server"),
                    "metrics": metrics["metrics"],
                    "summary_stats": metrics["summary_stats"],
                }
            )

        return top_results

    def clear(self):
        """Clear all stored data."""
        self.raw_storage.clear()
        self.tier1_summaries.clear()
        self.tier2_summary = None
        self.metrics_cache.clear()
        self.updated_at = datetime.now()

        logger.info("Progressive memory cleared")

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about memory usage.

        Returns:
            Dictionary with memory statistics
        """
        raw_size = sum(len(v) for v in self.raw_storage.values())

        return {
            "total_tools": len(self.metrics_cache),
            "raw_storage_size_bytes": raw_size,
            "raw_storage_enabled": self.keep_raw_storage,
            "tier1_summaries_count": len(self.tier1_summaries),
            "tier2_summary_exists": self.tier2_summary is not None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
