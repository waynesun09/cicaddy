"""Append-only event log for CI job context preservation.

Phase 1 KV-Cache Optimization: EventLog replaces volatile conversation context
with persistent, append-only JSONL files that preserve execution history without
consuming token budget or breaking KV-cache with dynamic content.

Inspired by Lumino Design's append-only event log pattern.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)


class EventLog:
    """
    Append-only event log for CI/CD pipeline context preservation.

    Features:
    - JSONL format (one JSON object per line)
    - Never modifies existing lines (append-only)
    - Deterministic JSON with sort_keys=True for KV-cache optimization
    - Automatic directory creation and error handling
    - Thread-safe appends (each write is atomic)
    """

    def __init__(self, report_id: str, log_dir: str = ".."):
        """
        Initialize event log for a specific job.

        Args:
            report_id: Unique report identifier (e.g., "cron_custom_20251105_084748")
            log_dir: Base directory for logs (default: ".." for parent directory, same as HTML reports)
        """
        self.report_id = report_id
        # Resolve to absolute path to avoid context dependency issues
        self.log_dir = Path(log_dir).resolve()

        # Generate short UUID to prevent collisions when multiple AI calls happen in same job
        short_uuid = str(uuid.uuid4())[:8]

        # Event log filename follows same pattern as HTML/JSON reports: {report_id}_{uuid}.jsonl
        self.log_path = self.log_dir / f"{report_id}_{short_uuid}.jsonl"

        # Create directory if it doesn't exist
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"EventLog initialized: {self.log_path}")
        except Exception as e:
            logger.error(f"Failed to create event log directory {self.log_dir}: {e}")
            raise

    def append_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Append event to log.

        Events are written as single-line JSON objects with deterministic key ordering
        for KV-cache optimization.

        Args:
            event_type: Type of event (tool_execution, ai_inference, etc.)
            data: Event-specific data to log
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "report_id": self.report_id,
            **data,
        }

        try:
            # Write single line with deterministic JSON (sort_keys=True)
            # This ensures consistent serialization for KV-cache hits
            json_line = json.dumps(event, sort_keys=True, ensure_ascii=False)

            # Atomic append operation
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json_line + "\n")
                f.flush()  # Ensure immediate write for debugging

        except Exception as e:
            logger.error(f"Failed to append event to log: {e}")
            # Don't raise - logging failures should not break execution

    def append_tool_execution(
        self,
        iteration: int,
        server: str,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        execution_time: float,
        result_size_bytes: int,
    ) -> None:
        """
        Log tool execution event.

        Args:
            iteration: Current iteration number
            server: MCP server name
            tool_name: Name of tool executed
            arguments: Tool arguments
            result: Tool result (will be converted to string)
            execution_time: Execution time in seconds
            result_size_bytes: Size of result in bytes
        """
        # Convert result to string for logging (may be dict, list, etc.)
        result_str = str(result) if not isinstance(result, str) else result

        # Truncate very large results for log efficiency
        max_result_length = 10000  # 10KB limit for log entries
        if len(result_str) > max_result_length:
            result_str = (
                result_str[:max_result_length]
                + f"... [truncated, full size: {len(result_str)} chars]"
            )

        self.append_event(
            "tool_execution",
            {
                "iteration": iteration,
                "server": server,
                "tool": tool_name,
                "arguments": arguments,
                "result": result_str,
                "result_size_bytes": result_size_bytes,
                "execution_time_seconds": round(execution_time, 3),
            },
        )

    def append_ai_inference(
        self,
        iteration: int,
        input_tokens: int,
        output_tokens: int,
        response: str,
        tool_calls: Optional[List[Dict]] = None,
    ) -> None:
        """
        Log AI inference event.

        Args:
            iteration: Current iteration number
            input_tokens: Number of input tokens consumed
            output_tokens: Number of output tokens generated
            response: AI response text
            tool_calls: Optional list of tool calls made by AI
        """
        # Truncate very long responses for log efficiency
        max_response_length = 5000  # 5KB limit for responses
        if len(response) > max_response_length:
            response = (
                response[:max_response_length]
                + f"... [truncated, full size: {len(response)} chars]"
            )

        event_data = {
            "iteration": iteration,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "response": response,
        }

        if tool_calls:
            event_data["tool_calls_count"] = len(tool_calls)
            # Log tool names only, not full arguments (already logged in tool_execution)
            event_data["tool_names"] = [
                tc.get("tool_name") or tc.get("name") for tc in tool_calls
            ]

        self.append_event("ai_inference", event_data)

    def append_iteration_start(
        self, iteration: int, total_tokens_used: int, current_iteration_tokens: int
    ) -> None:
        """
        Log iteration start event.

        Args:
            iteration: Current iteration number
            total_tokens_used: Total tokens used so far
            current_iteration_tokens: Tokens used in current iteration
        """
        self.append_event(
            "iteration_start",
            {
                "iteration": iteration,
                "total_tokens_used": total_tokens_used,
                "current_iteration_tokens": current_iteration_tokens,
            },
        )

    def append_compaction_event(
        self,
        iteration: int,
        compression_ratio: float,
        information_preserved: float,
        trigger: str,
    ) -> None:
        """
        Log context compaction event.

        Args:
            iteration: Current iteration number
            compression_ratio: Compression ratio achieved
            information_preserved: Information preservation ratio (0.0-1.0)
            trigger: What triggered compaction
        """
        self.append_event(
            "context_compaction",
            {
                "iteration": iteration,
                "compression_ratio": round(compression_ratio, 2),
                "information_preserved": round(information_preserved, 2),
                "trigger": trigger,
            },
        )

    def get_recent_events(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Read last N events from log.

        Args:
            count: Number of recent events to retrieve

        Returns:
            List of event dictionaries (most recent last)
        """
        if not self.log_path.exists():
            return []

        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Parse last N lines
            recent_lines = lines[-count:] if count < len(lines) else lines
            events = []
            for line in recent_lines:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse event line: {e}")
                    continue

            return events

        except Exception as e:
            logger.error(f"Failed to read recent events: {e}")
            return []

    def get_event_count(self) -> int:
        """
        Get total number of events in log.

        Returns:
            Number of events logged
        """
        if not self.log_path.exists():
            return 0

        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except Exception as e:
            logger.error(f"Failed to count events: {e}")
            return 0

    def get_log_size_bytes(self) -> int:
        """
        Get size of log file in bytes.

        Returns:
            Log file size in bytes
        """
        if not self.log_path.exists():
            return 0

        try:
            return self.log_path.stat().st_size
        except Exception as e:
            logger.error(f"Failed to get log size: {e}")
            return 0
