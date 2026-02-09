"""Generic result formatter with LlamaStack-inspired structured boundaries and prioritization."""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ResultPriority:
    """Priority levels for result content."""

    CRITICAL = 1  # Essential findings, errors, key insights
    HIGH = 2  # Important data, summaries, primary results
    MEDIUM = 3  # Supporting details, context, additional info
    LOW = 4  # Verbose output, raw data, debug info


@dataclass
class FormattedResult:
    """Structured result with metadata following LlamaStack patterns."""

    content: str
    priority: int = ResultPriority.MEDIUM
    size_bytes: int = 0
    tool_name: str = ""
    server_name: str = ""
    execution_time: float = 0.0
    truncated: bool = False
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.size_bytes = len(self.content.encode("utf-8"))


class GenericResultFormatter:
    """
    Generic result formatter for any MCP tool responses.

    Implements LlamaStack-inspired patterns:
    - Structured boundaries (BEGIN/END markers)
    - Priority-based content preservation
    - Intelligent truncation and summarization
    - Token-aware formatting
    """

    def __init__(self, max_result_size: int = 2048):
        self.max_result_size = max_result_size

    def format_tool_result(
        self,
        tool_name: str,
        server_name: str,
        result: Any,
        execution_time: float = 0.0,
        status: str = "success",
        arguments: Optional[Dict[str, Any]] = None,
    ) -> FormattedResult:
        """
        Format a single tool result with structured boundaries.

        Follows LlamaStack pattern of clear BEGIN/END markers for tools.
        """
        arguments = arguments or {}

        try:
            # Determine content and priority based on result type and status
            if status == "error":
                content = self._format_error_result(tool_name, result)
                priority = ResultPriority.CRITICAL
            else:
                content, priority = self._format_success_result(tool_name, result)

            # Apply structured boundaries (LlamaStack pattern)
            structured_content = self._apply_structured_boundaries(
                tool_name, content, server_name, execution_time, arguments
            )

            formatted_result = FormattedResult(
                content=structured_content,
                priority=priority,
                tool_name=tool_name,
                server_name=server_name,
                execution_time=execution_time,
                metadata={
                    "status": status,
                    "arguments": arguments,
                    "original_type": type(result).__name__,
                },
            )

            # Apply size limits with intelligent truncation
            if formatted_result.size_bytes > self.max_result_size:
                formatted_result = self._apply_intelligent_truncation(formatted_result)
            # Ensure very large simple strings are marked truncated even if boundaries skew byte count
            if (
                not formatted_result.truncated
                and isinstance(result, str)
                and len(result.encode("utf-8")) > self.max_result_size
            ):
                overflow = len(result.encode("utf-8")) - self.max_result_size
                formatted_result.content = (
                    formatted_result.content
                    + f"\n[TRUNCATED: {overflow} bytes over limit]"
                )
                formatted_result.truncated = True
                formatted_result.size_bytes = len(
                    formatted_result.content.encode("utf-8")
                )

            logger.debug(
                f"Formatted {tool_name} result: {formatted_result.size_bytes} bytes, "
                f"priority {formatted_result.priority}"
            )

            return formatted_result

        except Exception as e:
            logger.error(f"Failed to format result for {tool_name}: {e}")
            return FormattedResult(
                content=f"[FORMATTING ERROR: {e}]",
                priority=ResultPriority.CRITICAL,
                tool_name=tool_name,
                server_name=server_name,
            )

    def _format_success_result(self, tool_name: str, result: Any) -> tuple[str, int]:
        """Format successful tool result and determine priority."""
        if isinstance(result, dict):
            return self._format_dict_result(result)
        elif isinstance(result, list):
            return self._format_list_result(result)
        elif isinstance(result, str):
            return self._format_string_result(result)
        else:
            # Convert other types to JSON
            try:
                # Phase 1 KV-Cache Optimization: Use sort_keys=True for deterministic JSON
                json_result = json.dumps(
                    result, sort_keys=True, indent=2, ensure_ascii=False
                )
                return json_result, ResultPriority.MEDIUM
            except (TypeError, ValueError):
                return str(result), ResultPriority.MEDIUM

    def _format_dict_result(self, result: dict) -> tuple[str, int]:
        """Format dictionary results with priority detection."""
        # Check for common error patterns
        if "error" in result and result.get("error"):
            return f"ERROR: {result['error']}", ResultPriority.CRITICAL

        # Check for common data patterns
        if "data" in result:
            # Prioritize actual data content
            data = result["data"]
            if isinstance(data, list) and len(data) > 0:
                priority = ResultPriority.HIGH
            elif isinstance(data, dict) and data:
                priority = ResultPriority.HIGH
            else:
                priority = ResultPriority.MEDIUM

            # Phase 1 KV-Cache Optimization: Use sort_keys=True for deterministic JSON
            formatted = json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False)
            return formatted, priority

        # Check for content fields
        if "content" in result:
            content = result["content"]
            if isinstance(content, str) and len(content) > 100:
                priority = ResultPriority.HIGH
            else:
                priority = ResultPriority.MEDIUM
            # Phase 1 KV-Cache Optimization: Use sort_keys=True for deterministic JSON
            formatted = json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False)
            return formatted, priority

        # Default JSON formatting
        # Phase 1 KV-Cache Optimization: Use sort_keys=True for deterministic JSON
        formatted = json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False)

        # Prioritize based on size and content richness
        if len(formatted) > 500:
            priority = ResultPriority.HIGH
        elif len(formatted) > 50:
            priority = ResultPriority.MEDIUM
        else:
            priority = ResultPriority.LOW

        return formatted, priority

    def _format_list_result(self, result: list) -> tuple[str, int]:
        """Format list results with priority based on content."""
        if not result:
            return "[]", ResultPriority.LOW

        # Check if list contains complex objects
        has_complex_objects = not all(
            isinstance(item, (str, int, float, bool)) for item in result
        )

        # Prioritize based on list size and content complexity
        if len(result) > 10:
            priority = ResultPriority.HIGH
        elif len(result) > 2:
            priority = ResultPriority.MEDIUM
        elif has_complex_objects:
            # Complex objects get at least MEDIUM priority even if small
            priority = ResultPriority.MEDIUM
        else:
            priority = ResultPriority.LOW

        # Format with numbered items for readability
        if all(isinstance(item, str) for item in result):
            # Simple string list
            formatted = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(result))
        else:
            # Complex list - use JSON with line breaks
            # Phase 1 KV-Cache Optimization: Use sort_keys=True for deterministic JSON
            formatted = json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False)

        return formatted, priority

    def _format_string_result(self, result: str) -> tuple[str, int]:
        """Format string results with priority based on content analysis."""
        if not result or not result.strip():
            return result, ResultPriority.LOW

        # Check for error patterns
        error_patterns = [
            r"\berror\b",
            r"\bfailed?\b",
            r"\bexception\b",
            r"\bwarning\b",
            r"\balert\b",
            r"\bissue\b",
        ]
        if any(re.search(pattern, result, re.IGNORECASE) for pattern in error_patterns):
            return result, ResultPriority.CRITICAL

        # Check for structured content patterns
        if result.strip().startswith("{") and result.strip().endswith("}"):
            # Looks like JSON
            try:
                parsed = json.loads(result)
                return self._format_dict_result(parsed)
            except json.JSONDecodeError:
                pass

        # Prioritize based on content length and information density
        lines = result.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        if (
            len(non_empty_lines) > 20
            or len(result.encode("utf-8")) > self.max_result_size
        ):
            priority = ResultPriority.HIGH
        elif len(non_empty_lines) > 5:
            priority = ResultPriority.MEDIUM
        elif len(result) > 500:
            priority = ResultPriority.MEDIUM
        else:
            priority = ResultPriority.LOW

        return result, priority

    def _format_error_result(self, tool_name: str, result: Any) -> str:
        """Format error results with clear error indicators."""
        error_msg = str(result)

        # Clean up common error patterns
        if error_msg.startswith("Tool execution failed: "):
            error_msg = error_msg[23:]  # Remove redundant prefix

        return f"❌ ERROR in {tool_name}: {error_msg}"

    def _apply_structured_boundaries(
        self,
        tool_name: str,
        content: str,
        server_name: str,
        execution_time: float,
        arguments: Dict[str, Any],
    ) -> str:
        """Apply LlamaStack-style structured boundaries to results."""
        # Create header with tool metadata
        header_parts = [f"=== {tool_name.upper()} RESULT ==="]

        if server_name:
            header_parts.append(f"Server: {server_name}")

        if execution_time > 0:
            header_parts.append(f"Execution Time: {execution_time:.2f}s")

        if arguments:
            # Summarize arguments (avoid too much detail)
            arg_summary = ", ".join(f"{k}={v}" for k, v in list(arguments.items())[:3])
            if len(arguments) > 3:
                arg_summary += f" (+{len(arguments) - 3} more)"
            header_parts.append(f"Arguments: {arg_summary}")

        header = "\n".join(header_parts)

        # Create structured format with clear boundaries
        structured = f"""
{header}

BEGIN_CONTENT
{content}
END_CONTENT

=== END {tool_name.upper()} ===
""".strip()

        return structured

    def _apply_intelligent_truncation(self, result: FormattedResult) -> FormattedResult:
        """Apply intelligent truncation to oversized results."""
        target_size = self.max_result_size
        content = result.content

        if result.size_bytes <= target_size:
            return result

        logger.warning(
            f"Truncating {result.tool_name} result from {result.size_bytes} to ~{target_size} bytes"
        )

        # Find the content section
        begin_match = re.search(r"BEGIN_CONTENT\s*\n", content)
        end_match = re.search(r"\nEND_CONTENT", content)

        if begin_match and end_match:
            # Extract header, content, and footer
            header = content[: begin_match.end()]
            footer = content[end_match.start() :]
            actual_content = content[begin_match.end() : end_match.start()]

            # Calculate available space for content
            overhead = (
                len(header.encode("utf-8")) + len(footer.encode("utf-8")) + 100
            )  # Extra space for notice
            available_space = target_size - overhead

            if available_space > 200:  # Need minimum space for meaningful content
                # Apply smart truncation to the content
                truncated_content = self._smart_truncate_content(
                    actual_content, available_space
                )
                truncated_full = f"{header}{truncated_content}{footer}"

                result.content = truncated_full
                result.truncated = True
                result.size_bytes = len(truncated_full.encode("utf-8"))
            else:
                # Extreme truncation - just keep essential info
                result.content = f"[TRUNCATED: {result.tool_name} result too large ({result.size_bytes} bytes)]"
                result.truncated = True
                result.size_bytes = len(result.content.encode("utf-8"))
        else:
            # Fallback truncation for non-structured content
            truncated_content = self._smart_truncate_content(content, target_size - 100)
            result.content = truncated_content + "\n[TRUNCATED]"
            result.truncated = True
            result.size_bytes = len(result.content.encode("utf-8"))

        return result

    def _smart_truncate_content(self, content: str, max_size: int) -> str:
        """Apply smart truncation that preserves important content."""
        if len(content.encode("utf-8")) <= max_size:
            return content

        # Strategy 1: Try to keep complete lines
        lines = content.split("\n")
        kept_lines = []
        current_size = 0

        for line in lines:
            line_size = len(line.encode("utf-8")) + 1  # +1 for newline
            if (
                current_size + line_size > max_size * 0.8
            ):  # Leave space for truncation notice
                break
            kept_lines.append(line)
            current_size += line_size

        if kept_lines:
            # Add summary of what was truncated
            remaining_lines = len(lines) - len(kept_lines)
            truncation_notice = f"\n... [TRUNCATED: {remaining_lines} more lines, {len(content) - current_size} more bytes]"
            return "\n".join(kept_lines) + truncation_notice

        # Strategy 2: Character-based truncation with word boundaries
        target_chars = max_size - 50  # Leave space for truncation notice
        if target_chars > 100:
            # Find last complete word
            truncated = content[:target_chars]
            last_space = truncated.rfind(" ")
            if (
                last_space > target_chars * 0.8
            ):  # If we found a reasonable word boundary
                truncated = truncated[:last_space]

            return truncated + "...\n[TRUNCATED]"

        # Strategy 3: Extreme truncation
        return (
            f"[CONTENT TOO LARGE: {len(content)} characters, showing first 50]\n"
            + content[:50]
            + "..."
        )


class ResultPrioritizer:
    """Prioritize and manage multiple tool results based on importance and token budget."""

    def __init__(self, token_budget: int = 8000):
        self.token_budget = token_budget
        self.formatter = GenericResultFormatter()

    def prioritize_results(
        self, tool_results: List[Dict[str, Any]]
    ) -> List[FormattedResult]:
        """
        Prioritize and format multiple tool results within token budget.

        Returns results sorted by priority, with lower priority results
        truncated or summarized as needed.
        """
        formatted_results = []

        # Format all results first
        for result in tool_results:
            formatted = self.formatter.format_tool_result(
                tool_name=result.get("tool_name", "unknown"),
                server_name=result.get("tool_server", ""),
                result=result.get("result", ""),
                execution_time=result.get("execution_time", 0.0),
                status=result.get("status", "success"),
                arguments=result.get("arguments", {}),
            )
            formatted_results.append(formatted)

        # Sort by priority (lower number = higher priority)
        formatted_results.sort(key=lambda x: x.priority)

        # Apply token budget constraints
        return self._apply_token_budget(formatted_results)

    def _apply_token_budget(
        self, results: List[FormattedResult]
    ) -> List[FormattedResult]:
        """Apply token budget constraints to prioritized results."""
        total_tokens = 0
        final_results = []

        for result in results:
            # Estimate tokens (4 chars per token)
            result_tokens = result.size_bytes // 4

            if total_tokens + result_tokens <= self.token_budget:
                # Result fits within budget
                final_results.append(result)
                total_tokens += result_tokens
            elif result.priority <= ResultPriority.HIGH:
                # High priority result - try to summarize instead of dropping
                remaining_budget = max(self.token_budget - total_tokens, 0)
                # Always try to summarize high priority results; use up to remaining budget but at least 50 tokens if available
                if remaining_budget <= 0:
                    break
                target_budget = min(remaining_budget, max(50, remaining_budget))

                summarized = self._create_summary(
                    result, target_budget * 4
                )  # Convert back to bytes
                # Enforce hard cap: summarized must fit in remaining budget
                allowed_bytes = remaining_budget * 4
                if summarized.size_bytes > allowed_bytes:
                    raw = summarized.content.encode("utf-8")
                    trimmed = raw[:allowed_bytes]
                    summarized.content = trimmed.decode("utf-8", errors="ignore")
                    summarized.size_bytes = len(summarized.content.encode("utf-8"))
                    summarized.truncated = True
                final_results.append(summarized)
                # Budget is considered fully used after summarizing
                total_tokens = min(
                    self.token_budget, (total_tokens + (summarized.size_bytes // 4))
                )
                break
            else:
                # Low/medium priority - drop to save budget
                logger.info(
                    f"Dropping result {result.tool_name} (priority {result.priority}) "
                    "due to token budget"
                )
                continue

        logger.info(
            f"Token budget utilization: {total_tokens}/{self.token_budget} tokens "
            f"({total_tokens / self.token_budget:.1%})"
        )
        return final_results

    def _create_summary(
        self, result: FormattedResult, max_size_bytes: int
    ) -> FormattedResult:
        """Create a summary of a result that fits within the size limit."""
        summary_content = f"""
=== {result.tool_name.upper()} RESULT (SUMMARIZED) ===
Server: {result.server_name}
Status: {result.metadata.get("status", "unknown")}
Execution Time: {result.execution_time:.2f}s

SUMMARY: Large result truncated due to token budget constraints.
Original Size: {result.size_bytes} bytes
Content Preview: {result.content[:200]}...

[BUDGET TRUNCATED: Use higher token limits to see full results]
=== END {result.tool_name.upper()} ===
""".strip()
        # Enforce max size limit on summary
        if len(summary_content.encode("utf-8")) > max_size_bytes:
            # Trim content preview portion further to fit
            base_header = (
                f"=== {result.tool_name.upper()} RESULT (SUMMARIZED) ===\n"
                f"Server: {result.server_name}\n"
                f"Status: {result.metadata.get('status', 'unknown')}\n"
                f"Execution Time: {result.execution_time:.2f}s\n\n"
                f"SUMMARY: Large result truncated due to token budget constraints.\n"
                f"Original Size: {result.size_bytes} bytes\n"
                f"Content Preview: "
            )
            suffix = f"\n\n[BUDGET TRUNCATED: Use higher token limits to see full results]\n=== END {result.tool_name.upper()} ==="
            available = max_size_bytes - len((base_header + suffix).encode("utf-8"))
            preview = result.content.encode("utf-8")[: max(available, 0)].decode(
                "utf-8", errors="ignore"
            )
            summary_content = base_header + preview + "..." + suffix

        summarized = FormattedResult(
            content=summary_content,
            priority=result.priority,
            tool_name=result.tool_name,
            server_name=result.server_name,
            execution_time=result.execution_time,
            truncated=True,
            metadata=result.metadata,
        )
        return summarized
