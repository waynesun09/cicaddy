"""Progressive analysis with graceful degradation for token-aware execution."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

from cicaddy.ai_providers.base import BaseProvider
from cicaddy.utils.logger import get_logger

from .result_formatter import ResultPriority
from .token_aware_executor import ExecutionLimits, ExecutionState

logger = get_logger(__name__)


class AnalysisLevel(Enum):
    """Analysis depth levels for progressive degradation."""

    COMPREHENSIVE = 1  # Full analysis with all details
    DETAILED = 2  # Detailed analysis with key findings
    SUMMARY = 3  # High-level summary with main points
    CRITICAL_ONLY = 4  # Only critical findings and errors


@dataclass
class DegradationStrategy:
    """Strategy for progressive degradation based on resource constraints."""

    level: int
    max_tools_per_iteration: int
    max_result_size: int
    summary_threshold: int
    include_raw_data: bool
    include_debug_info: bool


class ProgressiveAnalyzer:
    """
    Progressive analysis engine that adapts to resource constraints.

    Implements graceful degradation by:
    1. Reducing analysis depth as limits approach
    2. Prioritizing critical findings over verbose details
    3. Generating intelligent summaries when truncation is needed
    4. Providing actionable recommendations despite constraints
    """

    def __init__(self, ai_provider: BaseProvider, limits: ExecutionLimits):
        self.ai_provider = ai_provider
        self.limits = limits

        # Define degradation strategies
        self.strategies = {
            AnalysisLevel.COMPREHENSIVE: DegradationStrategy(
                level=AnalysisLevel.COMPREHENSIVE.value,
                max_tools_per_iteration=limits.max_tools_per_iteration,
                max_result_size=limits.max_tokens_per_tool_result
                * 4,  # Convert tokens to chars
                summary_threshold=int(
                    1e9
                ),  # Very large int (effectively no summarization)
                include_raw_data=True,
                include_debug_info=True,
            ),
            AnalysisLevel.DETAILED: DegradationStrategy(
                level=AnalysisLevel.DETAILED.value,
                max_tools_per_iteration=limits.max_tools_per_iteration,
                max_result_size=limits.max_tokens_per_tool_result * 2,
                summary_threshold=2000,
                include_raw_data=True,
                include_debug_info=False,
            ),
            AnalysisLevel.SUMMARY: DegradationStrategy(
                level=AnalysisLevel.SUMMARY.value,
                max_tools_per_iteration=limits.max_tools_per_iteration,
                max_result_size=limits.max_tokens_per_tool_result,
                summary_threshold=1000,
                include_raw_data=False,
                include_debug_info=False,
            ),
            AnalysisLevel.CRITICAL_ONLY: DegradationStrategy(
                level=AnalysisLevel.CRITICAL_ONLY.value,
                max_tools_per_iteration=1,
                max_result_size=500,
                summary_threshold=300,
                include_raw_data=False,
                include_debug_info=False,
            ),
        }

    def determine_analysis_level(self, state: ExecutionState) -> AnalysisLevel:
        """
        Determine appropriate analysis level based on current resource utilization.

        Follows LlamaStack pattern of checking multiple resource dimensions.
        """
        # Calculate utilization ratios
        iter_ratio = state.current_iteration / self.limits.max_infer_iters
        token_ratio = state.total_tokens_used / self.limits.max_tokens_total
        tool_ratio = state.total_tools_executed / self.limits.max_total_tools
        time_ratio = state.get_elapsed_time() / self.limits.max_execution_time

        # Find the highest utilization
        max_utilization = max(iter_ratio, token_ratio, tool_ratio, time_ratio)

        logger.debug(
            f"Resource utilization - Iter: {iter_ratio:.1%}, Token: {token_ratio:.1%}, "
            f"Tool: {tool_ratio:.1%}, Time: {time_ratio:.1%}"
        )

        # Determine degradation level based on resource pressure
        if max_utilization >= 0.95:
            return AnalysisLevel.CRITICAL_ONLY
        elif max_utilization >= 0.85:
            return AnalysisLevel.SUMMARY
        elif max_utilization >= 0.70:
            return AnalysisLevel.DETAILED
        else:
            return AnalysisLevel.COMPREHENSIVE

    def apply_degradation_strategy(
        self,
        tool_results: List[Dict[str, Any]],
        analysis_level: AnalysisLevel,
        original_prompt: str = "",
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Apply degradation strategy to tool results and generate analysis.

        Returns:
            - Processed tool results (potentially summarized/filtered)
            - Analysis summary appropriate for the degradation level
        """
        strategy = self.strategies[analysis_level]

        logger.info(
            f"Applying degradation strategy: {analysis_level.name} "
            f"(max_tools: {strategy.max_tools_per_iteration}, "
            f"max_size: {strategy.max_result_size})"
        )

        # Filter and prioritize results
        processed_results = self._filter_results_by_priority(tool_results, strategy)

        # Generate appropriate analysis summary
        analysis_summary = self._generate_degraded_analysis(
            processed_results, analysis_level, original_prompt
        )

        return processed_results, analysis_summary

    def _filter_results_by_priority(
        self, tool_results: List[Dict[str, Any]], strategy: DegradationStrategy
    ) -> List[Dict[str, Any]]:
        """Filter and prioritize tool results based on degradation strategy."""
        # Classify results by importance
        critical_results = []
        important_results = []
        normal_results = []
        verbose_results = []

        for result in tool_results:
            content = str(result.get("result", ""))
            status = result.get("status", "success")

            # Classify based on content analysis
            priority = self._classify_result_priority(content, status)

            if priority == ResultPriority.CRITICAL:
                critical_results.append(result)
            elif priority == ResultPriority.HIGH:
                important_results.append(result)
            elif priority == ResultPriority.MEDIUM:
                normal_results.append(result)
            else:
                verbose_results.append(result)

        # Apply filtering based on strategy
        filtered_results = []

        # In COMPREHENSIVE mode, include all results without enforcing tool limits
        if strategy.level == AnalysisLevel.COMPREHENSIVE.value:
            filtered_results = (
                critical_results + important_results + normal_results + verbose_results
            )
        else:
            # Include critical results up to the tool limit
            critical_to_add = min(
                len(critical_results), strategy.max_tools_per_iteration
            )
            filtered_results.extend(critical_results[:critical_to_add])

            # Include other results based on available capacity
            remaining_capacity = strategy.max_tools_per_iteration - len(
                filtered_results
            )

            if remaining_capacity > 0 and strategy.level <= AnalysisLevel.SUMMARY.value:
                # Include important results
                to_add = min(remaining_capacity, len(important_results))
                filtered_results.extend(important_results[:to_add])
                remaining_capacity -= to_add

            if (
                remaining_capacity > 0
                and strategy.level <= AnalysisLevel.DETAILED.value
            ):
                # Include normal results
                to_add = min(remaining_capacity, len(normal_results))
                filtered_results.extend(normal_results[:to_add])
                remaining_capacity -= to_add

            if remaining_capacity > 0 and strategy.level == AnalysisLevel.SUMMARY.value:
                # SUMMARY level does not include verbose results
                pass

        # Apply size limits to individual results
        for result in filtered_results:
            content = str(result.get("result", ""))
            if len(content) > strategy.max_result_size:
                # Truncate with smart summary
                truncated = self._smart_truncate_result(
                    content, strategy.max_result_size
                )
                result["result"] = truncated
                result["truncated"] = True

        logger.info(
            f"Filtered {len(tool_results)} -> {len(filtered_results)} results "
            f"(Critical: {len(critical_results)}, Important: {len(important_results)})"
        )

        return filtered_results

    def _classify_result_priority(self, content: str, status: str) -> int:
        """Classify result priority based on content analysis."""
        if status == "error" or not content:
            return ResultPriority.CRITICAL

        content_lower = content.lower()

        # Critical patterns
        critical_patterns = [
            r"\berror\b",
            r"\bfailed?\b",
            r"\bexception\b",
            r"\bcrash\b",
            r"\balert\b",
            r"\bwarning\b",
            r"\bissue\b",
            r"\bproblem\b",
        ]
        if any(re.search(pattern, content_lower) for pattern in critical_patterns):
            return ResultPriority.CRITICAL

        # High importance patterns
        important_patterns = [
            r"\bfound\b.*\d+",
            r"\bresults?\b",
            r"\bsummary\b",
            r"\btotal\b",
            r"\banalysis\b",
            r"\breport\b",
            r"\binsights?\b",
        ]
        if any(re.search(pattern, content_lower) for pattern in important_patterns):
            return ResultPriority.HIGH

        # Content richness indicators
        lines = content.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        if len(non_empty_lines) > 10 or len(content) > 1000:
            return ResultPriority.HIGH
        elif len(non_empty_lines) > 3 or len(content) > 200:
            return ResultPriority.MEDIUM
        else:
            return ResultPriority.LOW

    def _smart_truncate_result(self, content: str, max_size: int) -> str:
        """Apply smart truncation that preserves key information."""
        if len(content) <= max_size:
            return content

        # Strategy: Keep beginning and end, summarize middle
        if max_size > 200:
            keep_start = max_size // 3
            keep_end = max_size // 3
            truncation_notice = (
                f"\n... [TRUNCATED: {len(content) - keep_start - keep_end} chars] ...\n"
            )

            start_content = content[:keep_start]
            end_content = content[-keep_end:]

            return start_content + truncation_notice + end_content
        else:
            # Simple truncation for very small limits
            return content[: max_size - 15] + "...[TRUNCATED]"

    def _generate_degraded_analysis(
        self,
        tool_results: List[Dict[str, Any]],
        analysis_level: AnalysisLevel,
        original_prompt: str,
    ) -> str:
        """Generate analysis summary appropriate for the degradation level."""
        if not tool_results:
            return "No tool results available for analysis."

        # Count results by status
        successful_tools = [r for r in tool_results if r.get("status") == "success"]
        failed_tools = [r for r in tool_results if r.get("status") == "error"]

        summary_parts = []

        # Level-appropriate header
        if analysis_level == AnalysisLevel.CRITICAL_ONLY:
            summary_parts.append("🚨 CRITICAL ANALYSIS (Resource Constrained)")
        elif analysis_level == AnalysisLevel.SUMMARY:
            summary_parts.append("📊 SUMMARY ANALYSIS (Progressive Degradation Active)")
        elif analysis_level == AnalysisLevel.DETAILED:
            summary_parts.append("📈 DETAILED ANALYSIS")
        else:
            summary_parts.append("📋 COMPREHENSIVE ANALYSIS")

        # Execution summary
        summary_parts.append(
            f"Tools Executed: {len(successful_tools)} successful, {len(failed_tools)} failed"
        )

        # Key findings based on analysis level
        if analysis_level.value <= AnalysisLevel.SUMMARY.value:
            key_findings = self._extract_key_findings(successful_tools)
            if key_findings:
                summary_parts.append("Key Findings:")
                summary_parts.extend(f"• {finding}" for finding in key_findings[:5])

        # Error summary (always included)
        if failed_tools:
            summary_parts.append("⚠️ Errors Encountered:")
            for tool in failed_tools[:3]:  # Limit error details
                tool_name = tool.get("tool_name", "Unknown")
                error_msg = str(tool.get("result", ""))[:100]
                summary_parts.append(f"• {tool_name}: {error_msg}")

        # Degradation notice
        if analysis_level.value > AnalysisLevel.COMPREHENSIVE.value:
            strategy = self.strategies[analysis_level]
            summary_parts.append(
                f"ℹ️ Analysis limited due to resource constraints "
                f"(Level: {analysis_level.name}, Max tools: {strategy.max_tools_per_iteration})"
            )

        # Recommendations based on level
        if analysis_level.value <= AnalysisLevel.DETAILED.value:
            recommendations = self._generate_recommendations(
                tool_results, analysis_level
            )
            if recommendations:
                summary_parts.append("💡 Recommendations:")
                summary_parts.extend(f"• {rec}" for rec in recommendations[:3])

        return "\n\n".join(summary_parts)

    def _extract_key_findings(
        self, successful_tools: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract key findings from successful tool results."""
        findings = []

        for tool in successful_tools:
            result = str(tool.get("result", ""))
            tool_name = tool.get("tool_name", "Unknown")

            # Extract numerical findings
            numbers = re.findall(r"\b\d+\b", result)
            if numbers and len(numbers) >= 2:
                findings.append(f"{tool_name} found {len(numbers)} numerical values")

            # Extract list findings
            lines = [line.strip() for line in result.split("\n") if line.strip()]
            if len(lines) > 5:
                findings.append(f"{tool_name} returned {len(lines)} items")

            # Extract specific patterns
            if "total" in result.lower() or "count" in result.lower():
                # Extract the line containing total/count
                for line in lines:
                    if "total" in line.lower() or "count" in line.lower():
                        findings.append(f"{tool_name}: {line[:100]}")
                        break

        return findings[:10]  # Limit findings

    def _generate_recommendations(
        self, tool_results: List[Dict[str, Any]], analysis_level: AnalysisLevel
    ) -> List[str]:
        """Generate actionable recommendations based on results and degradation level."""
        recommendations = []

        # Analyze failure patterns
        failed_tools = [r for r in tool_results if r.get("status") == "error"]
        if failed_tools:
            unique_errors = set()
            for tool in failed_tools:
                error_msg = str(tool.get("result", "")).lower()
                if "timeout" in error_msg or "timed out" in error_msg:
                    unique_errors.add("timeout")
                elif "not found" in error_msg or "404" in error_msg:
                    unique_errors.add("not_found")
                elif "authentication" in error_msg or "401" in error_msg:
                    unique_errors.add("auth")
                else:
                    unique_errors.add("general")

            if "timeout" in unique_errors:
                recommendations.append(
                    "Increase tool timeout limits or optimize tool queries"
                )
            if "not_found" in unique_errors:
                recommendations.append("Verify tool parameters and data availability")
            if "auth" in unique_errors:
                recommendations.append(
                    "Check authentication and permissions for failed tools"
                )

        # Resource optimization recommendations
        if analysis_level.value > AnalysisLevel.COMPREHENSIVE.value:
            recommendations.append(
                "Increase token/iteration limits for more comprehensive analysis"
            )

        # Data quality recommendations
        successful_tools = [r for r in tool_results if r.get("status") == "success"]
        if len(successful_tools) > 0:
            avg_result_size = sum(
                len(str(r.get("result", ""))) for r in successful_tools
            ) / len(successful_tools)
            if avg_result_size > 2000:
                recommendations.append(
                    "Consider using more specific queries to reduce result sizes"
                )

        return recommendations
