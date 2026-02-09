"""Generic knowledge accumulation layer for MCP tool results."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AccumulatedKnowledge:
    """
    Persistent store for MCP tool results across iterations.

    Design: Generic and works for ANY MCP server combination (DataRouter, GitHub,
    Sourcebot, SonarQube, etc.). Tool results are never compacted, ensuring complete
    data preservation for report generation and notifications.

    This solves the data loss problem where aggressive context compaction at iteration
    limits would discard valuable tool result data, leaving reports and Slack
    notifications with minimal or useless information.
    """

    # All tool results in execution order (never compacted)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)

    # Indexed by server for easy access (e.g., "datarouter", "github", "sourcebot")
    results_by_server: Dict[str, List[Dict]] = field(default_factory=dict)

    # Indexed by tool name for easy access (e.g., "getTotalRequestNumberFromPastDays")
    results_by_tool: Dict[str, List[Dict]] = field(default_factory=dict)

    # Execution metadata
    total_tools_executed: int = 0
    servers_used: Set[str] = field(default_factory=set)
    tools_used: Set[str] = field(default_factory=set)

    # Inference tracking: which inference we're currently in (1=initial, 2+=recovery)
    current_inference_id: int = 1

    def add_tool_result(
        self,
        iteration: int,
        server: str,
        tool: str,
        arguments: Dict[str, Any],
        result: Any,
        execution_time: float = 0.0,
        result_size_bytes: int = 0,
        inference_id: Optional[int] = None,
    ):
        """
        Add a tool result to the knowledge store.

        Generic method that works for ANY MCP tool from ANY server. Tool results are
        stored with full fidelity and never compacted, ensuring complete data is
        available for report generation and notifications.

        Args:
            iteration: Which iteration this tool was executed in (within current inference)
            server: MCP server name (e.g., "datarouter", "github", "sourcebot")
            tool: Tool name (e.g., "getTotalRequestNumberFromPastDays")
            arguments: Arguments passed to the tool
            result: The actual result returned by the tool (ANY type, never compacted)
            execution_time: How long the tool took to execute (seconds)
            result_size_bytes: Size of the result in bytes
            inference_id: Which inference this tool belongs to (1=initial, 2+=recovery).
                         If not provided, uses current_inference_id.
        """
        # Use current inference_id if not provided
        inf_id = inference_id if inference_id is not None else self.current_inference_id

        entry = {
            "inference_id": inf_id,
            "iteration": iteration,
            "unique_ref": f"inf{inf_id}_i{iteration}",
            "server": server,
            "tool": tool,
            "arguments": arguments,
            "result": result,  # FULL result, never compacted
            "execution_time": execution_time,
            "result_size_bytes": result_size_bytes,
            "timestamp": time.time(),
        }

        # Store in all indices for flexible access
        self.tool_results.append(entry)

        # Index by server
        if server not in self.results_by_server:
            self.results_by_server[server] = []
        self.results_by_server[server].append(entry)

        # Index by tool
        if tool not in self.results_by_tool:
            self.results_by_tool[tool] = []
        self.results_by_tool[tool].append(entry)

        # Update metadata
        self.total_tools_executed += 1
        self.servers_used.add(server)
        self.tools_used.add(tool)

        # Warn on large individual results
        if result_size_bytes > 10 * 1024 * 1024:  # 10MB
            logger.warning(
                f"Large tool result detected: {tool} from {server} "
                f"({result_size_bytes / (1024 * 1024):.2f} MB)"
            )

        # Warn when total size grows large (check every 10 tools to avoid spam)
        if self.total_tools_executed % 10 == 0:
            total_size = self.get_total_data_size()
            if total_size > 50 * 1024 * 1024:  # 50MB
                logger.warning(
                    f"Accumulated knowledge store is large: "
                    f"{total_size / (1024 * 1024):.2f} MB across {self.total_tools_executed} tools"
                )

    def start_new_inference(self):
        """
        Increment inference ID when starting recovery or new inference.

        This method should be called when the execution engine begins a recovery
        inference due to context limits, premature completion, or other triggers.
        It increments the inference ID so that subsequent tool calls are tagged
        with the new inference number.
        """
        self.current_inference_id += 1
        logger.info(f"Started new inference: inference_id={self.current_inference_id}")

    def get_results_for_server(self, server: str) -> List[Dict[str, Any]]:
        """Get all results for a specific MCP server."""
        return self.results_by_server.get(server, [])

    def get_results_for_tool(self, tool: str) -> List[Dict[str, Any]]:
        """Get all results for a specific tool."""
        return self.results_by_tool.get(tool, [])

    def get_latest_result(self, tool: str) -> Optional[Dict[str, Any]]:
        """Get the most recent result for a specific tool."""
        results = self.get_results_for_tool(tool)
        return results[-1] if results else None

    def get_results_by_iteration(self, iteration: int) -> List[Dict[str, Any]]:
        """Get all tool results from a specific iteration."""
        return [r for r in self.tool_results if r["iteration"] == iteration]

    def get_minimal_tool_calls_for_recovery(
        self, max_content_length: int = 10000
    ) -> List[Dict[str, Any]]:
        """
        Extract minimal representation of tool calls for token-limited recovery.

        This method creates a compact version of tool_results suitable for
        inclusion in a recovery prompt, saving context tokens by excluding:
        - timestamps
        - execution_time
        - result_size_bytes
        - inference_id (redundant with unique_ref)

        Args:
            max_content_length: Maximum characters to keep per tool result.
                Default 10000 matches settings.recovery_content_truncation_length.

        Returns:
            List of minimal tool call dictionaries with:
            - unique_ref: Unique reference (e.g., "inf1_i3")
            - tool_name: Name of the tool
            - tool_server: MCP server name
            - arguments: Tool arguments (as-is)
            - status: "success", "error", or "unknown"
            - content: The result content (truncated if very large)
        """
        minimal_calls = []

        for tool_result in self.tool_results:
            result = tool_result.get("result", "")
            result_str = str(result)

            # Use existing status field from tool_result if available
            # Falls back to "unknown" if not set by the MCP transport layer
            status = tool_result.get("status", "unknown")

            # Truncate very large results for recovery context
            # Full results are preserved in knowledge_store.tool_results
            if len(result_str) > max_content_length:
                content = (
                    result_str[:max_content_length]
                    + f"... [truncated, full size: {len(result_str)} chars]"
                )
            else:
                content = result_str

            minimal_call = {
                "unique_ref": tool_result.get(
                    "unique_ref", f"i{tool_result.get('iteration', 0)}"
                ),
                "tool_name": tool_result.get("tool", "unknown"),
                "tool_server": tool_result.get("server", "unknown"),
                "arguments": tool_result.get("arguments", {}),
                "status": status,
                "content": content,
            }
            minimal_calls.append(minimal_call)

        return minimal_calls

    def get_tool_calls_summary_for_prompt(self) -> str:
        """
        Format minimal tool calls as a string for inclusion in recovery prompts.

        Returns:
            Formatted string with all tool calls and their results.
        """
        minimal_calls = self.get_minimal_tool_calls_for_recovery()

        if not minimal_calls:
            return "No tool calls have been executed yet."

        lines = [f"## Tool Execution History ({len(minimal_calls)} calls)\n"]

        for call in minimal_calls:
            lines.append(
                f"### [{call['unique_ref']}] {call['tool_name']} ({call['tool_server']})"
            )
            lines.append(f"**Status:** {call['status']}")
            lines.append(f"**Arguments:** {call['arguments']}")
            lines.append(f"**Result:**\n```\n{call['content']}\n```\n")

        return "\n".join(lines)

    def get_total_execution_time(self) -> float:
        """Calculate total execution time for all tools."""
        return sum(r["execution_time"] for r in self.tool_results)

    def get_total_data_size(self) -> int:
        """Calculate total size of all tool results in bytes."""
        return sum(r["result_size_bytes"] for r in self.tool_results)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns complete knowledge store including all tool results, indices, and
        metadata. This dictionary can be included in analysis results for report
        generation and notification formatting.
        """
        return {
            "tool_results": self.tool_results,
            "results_by_server": self.results_by_server,
            "results_by_tool": self.results_by_tool,
            "total_tools_executed": self.total_tools_executed,
            "servers_used": list(self.servers_used),
            "tools_used": list(self.tools_used),
            "total_execution_time": self.get_total_execution_time(),
            "total_data_size_bytes": self.get_total_data_size(),
            "current_inference_id": self.current_inference_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AccumulatedKnowledge":
        """
        Reconstruct knowledge store from dictionary.

        Args:
            data: Dictionary representation from to_dict()

        Returns:
            Reconstructed AccumulatedKnowledge instance
        """
        knowledge = cls()
        knowledge.tool_results = data.get("tool_results", [])
        knowledge.results_by_server = data.get("results_by_server", {})
        knowledge.results_by_tool = data.get("results_by_tool", {})
        knowledge.total_tools_executed = data.get("total_tools_executed", 0)
        knowledge.servers_used = set(data.get("servers_used", []))
        knowledge.tools_used = set(data.get("tools_used", []))
        # Backward compatibility: default to 1 if not present in old data
        knowledge.current_inference_id = data.get("current_inference_id", 1)
        return knowledge

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"AccumulatedKnowledge("
            f"tools={self.total_tools_executed}, "
            f"servers={list(self.servers_used)}, "
            f"size={self.get_total_data_size()}bytes)"
        )
