"""Token usage extraction, aggregation, and management utilities."""

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, cast

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Note: httpx would be used for actual API calls to fetch dynamic limits
# For now, we use enhanced static mappings with the latest model information
HTTPX_AVAILABLE = False


class TokenUsageExtractor:
    """Utility class for extracting and aggregating AI token usage from analysis results."""

    @staticmethod
    def extract_token_usage(
        analysis_result: Dict[str, Any],
        default_provider: Optional[str] = None,
        include_model_info: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract and aggregate token usage from analysis result execution steps.

        Args:
            analysis_result: The analysis result containing execution steps
            default_provider: Default provider to use if not found in analysis result
            include_model_info: Whether to include model_used in the result

        Returns:
            Dict containing aggregated token usage statistics and metadata
        """
        token_summary: Dict[str, Any] = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "inference_calls": 0,
            "provider": default_provider or "unknown",
        }

        if include_model_info:
            token_summary["model_used"] = None

        # Extract token data from execution steps
        execution_steps = analysis_result.get("execution_steps", [])

        for step in execution_steps:
            # Only process inference steps that have usage stats
            if (
                step.get("step_type") == "inference"
                and step.get("usage_stats")
                and isinstance(step.get("usage_stats"), dict)
            ):
                usage_stats = step["usage_stats"]
                token_summary["inference_calls"] += 1

                # Aggregate token counts
                token_summary["total_tokens"] += usage_stats.get("total_tokens", 0)
                token_summary["prompt_tokens"] += usage_stats.get("prompt_tokens", 0)
                token_summary["completion_tokens"] += usage_stats.get(
                    "completion_tokens", 0
                )

                # Capture the model used (use the last one if multiple models)
                if include_model_info and step.get("model_used"):
                    token_summary["model_used"] = step["model_used"]

        # Try to get provider from analysis result
        if "ai_provider" in analysis_result:
            token_summary["provider"] = analysis_result["ai_provider"]
        elif "provider" in analysis_result:
            token_summary["provider"] = analysis_result["provider"]

        # If no model was captured from steps, try to get it from analysis result
        if include_model_info and not token_summary.get("model_used"):
            token_summary["model_used"] = analysis_result.get("model_used")

        return token_summary

    @staticmethod
    def has_meaningful_usage(token_data: Dict[str, Any]) -> bool:
        """
        Check if token data contains meaningful AI usage information.

        Args:
            token_data: Token usage data from extract_token_usage()

        Returns:
            True if there is meaningful token usage data
        """
        return token_data["total_tokens"] > 0 and token_data["inference_calls"] > 0

    @staticmethod
    def format_compact_usage(token_data: Dict[str, Any]) -> str:
        """
        Format token usage data as compact string for footers/notifications.

        Args:
            token_data: Token usage data from extract_token_usage()

        Returns:
            Formatted string like "openai | 1,234 tokens | 3 calls"
        """
        if not TokenUsageExtractor.has_meaningful_usage(token_data):
            return ""

        provider = token_data.get("provider", "").lower()
        if not provider or provider == "unknown":
            return ""

        total_tokens = f"{token_data['total_tokens']:,}"
        inference_calls = token_data["inference_calls"]
        call_text = "call" if inference_calls == 1 else "calls"

        return f"{total_tokens} tokens | {inference_calls} {call_text}"

    @staticmethod
    def format_detailed_usage(token_data: Dict[str, Any]) -> str:
        """
        Format token usage data as detailed string for MR comments.

        Args:
            token_data: Token usage data from extract_token_usage()

        Returns:
            Formatted string like "📊 AI Usage: OpenAI GPT-4 | 1,234 tokens (890 input + 344 output) | 3 inference calls"
        """
        if not TokenUsageExtractor.has_meaningful_usage(token_data):
            return ""

        # Check if we have model info
        if not token_data.get("model_used"):
            return ""

        # Format the model name for display
        model_name = token_data.get("model_used") or "Unknown Model"
        provider = token_data.get("provider", "").title()

        # Create a readable model display name
        if model_name and model_name != "Unknown Model":
            if provider and provider.lower() not in model_name.lower():
                model_display = f"{provider} {model_name}"
            else:
                model_display = model_name
        else:
            model_display = provider if provider else "Unknown Model"

        # Format token counts with thousands separators
        total_tokens = f"{token_data['total_tokens']:,}"
        prompt_tokens = f"{token_data['prompt_tokens']:,}"
        completion_tokens = f"{token_data['completion_tokens']:,}"
        inference_calls = token_data["inference_calls"]

        # Build the detailed summary string
        call_text = "call" if inference_calls == 1 else "calls"
        summary = (
            f"📊 AI Usage: {model_display} | {total_tokens} tokens "
            f"({prompt_tokens} input + {completion_tokens} output) | "
            f"{inference_calls} inference {call_text}"
        )

        return summary


class TokenLimitManager:
    """Manages token limits across different AI providers with dynamic fetching."""

    # Cache for dynamic limits
    _cache: Dict[str, Dict[str, int]] = {}
    _cache_timestamps: Dict[str, float] = {}
    _cache_ttl = int(os.getenv("TOKEN_LIMIT_CACHE_TTL", "7200"))  # 2 hours default
    _dynamic_enabled = os.getenv("DYNAMIC_TOKEN_LIMITS", "true").lower() == "true"

    # Static fallback limits (used when dynamic fetch fails)
    STATIC_LIMITS = {
        "gemini": {
            "input": 1048576,  # 1,048,576 tokens (1024² = 2²⁰)
            "output": 65536,  # 64K tokens
            "models": {
                "gemini-3-flash-preview": {"input": 1048576, "output": 65536},
                "gemini-3-pro-preview": {"input": 1048576, "output": 65536},
                "gemini-2.5-flash": {"input": 1048576, "output": 65536},
                "gemini-2.5-pro": {"input": 1048576, "output": 65536},
                "gemini-1.5-pro": {"input": 2097152, "output": 8192},
            },
        },
        "openai": {
            "input": 128000,  # Default for most models
            "output": 16384,  # Default for most models
            "models": {
                "gpt-4o": {"input": 128000, "output": 16384},
                "gpt-4o-mini": {"input": 128000, "output": 16384},
                "gpt-4-turbo": {"input": 128000, "output": 4096},
                "gpt-3.5-turbo": {"input": 16385, "output": 4096},
            },
        },
        "claude": {
            "input": 200000,  # ~200K tokens
            "output": 4096,  # Standard Claude output
            "models": {
                "claude-3-5-sonnet-latest": {"input": 200000, "output": 8192},
                "claude-3-5-haiku-latest": {"input": 200000, "output": 8192},
                "claude-3-opus-latest": {"input": 200000, "output": 4096},
            },
        },
    }

    @classmethod
    def get_limits(cls, provider: str, model: Optional[str] = None) -> Dict[str, int]:
        """Get token limits for a specific provider and model."""
        provider = provider.lower()
        cache_key = f"{provider}:{model or 'default'}"

        # Try dynamic limits first if enabled
        if cls._dynamic_enabled:
            # Check cache first
            if cls._is_cache_valid(cache_key):
                return cls._cache[cache_key]

            # Try to fetch dynamically
            try:
                limits = cls._fetch_dynamic_limits(provider, model)
                if limits:
                    cls._update_cache(cache_key, limits)
                    return limits
            except Exception as e:
                # Log error but continue to fallback
                import logging

                logging.getLogger(__name__).warning(
                    f"Dynamic limit fetch failed for {provider}:{model}: {e}"
                )

        # Fall back to static limits
        return cls._get_static_limits(provider, model)

    @classmethod
    def _get_static_limits(
        cls, provider: str, model: Optional[str] = None
    ) -> Dict[str, int]:
        """Get static fallback limits."""
        if provider not in cls.STATIC_LIMITS:
            # Default fallback limits
            return {"input": 4096, "output": 1024}

        provider_config: Dict[str, Any] = cls.STATIC_LIMITS[provider]
        models_config: Dict[str, Dict[str, int]] = provider_config.get("models", {})

        if model and model in models_config:
            return models_config[model]

        return {
            "input": cast(int, provider_config["input"]),
            "output": cast(int, provider_config["output"]),
        }

    @classmethod
    def _is_cache_valid(cls, cache_key: str) -> bool:
        """Check if cached value is still valid."""
        if cache_key not in cls._cache or cache_key not in cls._cache_timestamps:
            return False

        age = time.time() - cls._cache_timestamps[cache_key]
        return age < cls._cache_ttl

    @classmethod
    def _update_cache(cls, cache_key: str, limits: Dict[str, int]) -> None:
        """Update cache with new limits."""
        cls._cache[cache_key] = limits
        cls._cache_timestamps[cache_key] = time.time()

    @classmethod
    def _fetch_dynamic_limits(
        cls, provider: str, model: Optional[str] = None
    ) -> Optional[Dict[str, int]]:
        """Fetch current token limits from provider APIs."""
        if not HTTPX_AVAILABLE:
            return None

        if provider == "openai":
            return cls._fetch_openai_limits(model)
        elif provider == "claude":
            return cls._fetch_claude_limits(model)
        elif provider == "gemini":
            return cls._fetch_gemini_limits(model)

        return None

    @classmethod
    def _fetch_openai_limits(
        cls, model: Optional[str] = None
    ) -> Optional[Dict[str, int]]:
        """Fetch OpenAI model limits."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return None

            # OpenAI doesn't provide token limits in their models API
            # Use known mappings for common models
            model_limits = {
                "gpt-4o": {"input": 128000, "output": 16384},
                "gpt-4o-mini": {"input": 128000, "output": 16384},
                "gpt-4-turbo": {"input": 128000, "output": 4096},
                "gpt-3.5-turbo": {"input": 16385, "output": 4096},
                "o1-preview": {"input": 128000, "output": 32768},
                "o1-mini": {"input": 128000, "output": 65536},
            }

            if model and model in model_limits:
                return model_limits[model]

            # Default for unknown OpenAI models
            return {"input": 128000, "output": 16384}

        except Exception:
            return None

    @classmethod
    def _fetch_claude_limits(
        cls, model: Optional[str] = None
    ) -> Optional[Dict[str, int]]:
        """Fetch Claude model limits."""
        try:
            # Claude model limits (Anthropic doesn't expose these via API yet)
            model_limits = {
                "claude-3-5-sonnet-latest": {"input": 200000, "output": 8192},
                "claude-3-5-haiku-latest": {"input": 200000, "output": 8192},
                "claude-3-opus-latest": {"input": 200000, "output": 4096},
                "claude-3-sonnet-20240229": {"input": 200000, "output": 4096},
                "claude-3-haiku-20240307": {"input": 200000, "output": 4096},
            }

            if model and model in model_limits:
                return model_limits[model]

            # Default for unknown Claude models
            return {"input": 200000, "output": 8192}

        except Exception:
            return None

    @classmethod
    def _fetch_gemini_limits(
        cls, model: Optional[str] = None
    ) -> Optional[Dict[str, int]]:
        """Fetch Gemini model limits."""
        try:
            # Gemini model limits (from official documentation)
            model_limits = {
                "gemini-3-flash-preview": {"input": 1048576, "output": 65536},
                "gemini-3-pro-preview": {"input": 1048576, "output": 65536},
                "gemini-2.5-flash": {"input": 1048576, "output": 65536},
                "gemini-2.5-pro": {"input": 1048576, "output": 65536},
                "gemini-2.0-flash": {"input": 1048576, "output": 8192},
                "gemini-1.5-pro": {"input": 2097152, "output": 8192},
                "gemini-1.5-flash": {"input": 1048576, "output": 8192},
            }

            if model and model in model_limits:
                return model_limits[model]

            # Default for unknown Gemini models
            return {"input": 1048576, "output": 65536}

        except Exception:
            return None

    @classmethod
    def validate_input_tokens(
        cls, provider: str, model: Optional[str], token_count: int
    ) -> bool:
        """Check if input token count is within limits."""
        limits = cls.get_limits(provider, model)
        return token_count <= limits["input"]

    @classmethod
    def get_max_input_tokens(cls, provider: str, model: Optional[str] = None) -> int:
        """Get maximum input tokens for a provider/model."""
        limits = cls.get_limits(provider, model)
        return limits["input"]

    @classmethod
    def get_max_output_tokens(cls, provider: str, model: Optional[str] = None) -> int:
        """Get maximum output tokens for a provider/model."""
        limits = cls.get_limits(provider, model)
        return limits["output"]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the token limits cache."""
        cls._cache.clear()
        cls._cache_timestamps.clear()

    @classmethod
    def set_cache_ttl(cls, ttl_seconds: int) -> None:
        """Set cache TTL in seconds."""
        cls._cache_ttl = ttl_seconds

    @classmethod
    def enable_dynamic_limits(cls, enabled: bool = True) -> None:
        """Enable or disable dynamic limit fetching."""
        cls._dynamic_enabled = enabled


class TokenCounter:
    """Universal token counting across different providers."""

    @classmethod
    def count_tokens(cls, text: str, provider: str, model: Optional[str] = None) -> int:
        """Count tokens for any provider."""
        provider = provider.lower()

        if provider in ["openai", "claude"] and TIKTOKEN_AVAILABLE:
            return cls._count_tiktoken(text, provider, model)
        elif provider == "gemini":
            return cls._count_gemini_tokens(text)
        else:
            # Fallback: estimate based on characters
            return cls._estimate_tokens(text)

    @classmethod
    def _count_tiktoken(
        cls, text: str, provider: str, model: Optional[str] = None
    ) -> int:
        """Count tokens using tiktoken for OpenAI/Claude."""
        try:
            # Choose appropriate encoding
            if provider == "openai":
                if model and "gpt-3.5" in model:
                    encoding = tiktoken.get_encoding("cl100k_base")
                else:
                    encoding = tiktoken.get_encoding("o200k_base")  # For GPT-4o
            else:  # claude
                encoding = tiktoken.get_encoding("cl100k_base")

            return len(encoding.encode(text))
        except Exception:
            return cls._estimate_tokens(text)

    @classmethod
    def _count_gemini_tokens(cls, text: str) -> int:
        """Count tokens for Gemini (approximation based on chars)."""
        # Gemini: ~4 characters per token
        return len(text) // 4

    @classmethod
    def _estimate_tokens(cls, text: str) -> int:
        """Fallback token estimation."""
        # Rough estimation: 4 characters per token
        return len(text) // 4


class PromptTruncator:
    """Intelligent prompt truncation with content prioritization."""

    # Truncation notice constant (shorter and more concise)
    TRUNCATION_NOTICE = "\n\n[Content truncated due to token limits]"

    @classmethod
    def truncate_for_provider(
        cls,
        prompt: str,
        provider: str,
        model: Optional[str] = None,
        safety_margin: int = 1000,
    ) -> Tuple[str, bool]:
        """Truncate prompt to fit provider limits.

        Returns:
            Tuple of (truncated_prompt, was_truncated)
        """
        max_tokens = TokenLimitManager.get_max_input_tokens(provider, model)
        current_tokens = TokenCounter.count_tokens(prompt, provider, model)

        if current_tokens <= max_tokens - safety_margin:
            return prompt, False

        # Need to truncate
        target_tokens = max_tokens - safety_margin
        return cls._intelligent_truncate(prompt, target_tokens, provider, model)

    @classmethod
    def _intelligent_truncate(
        cls, prompt: str, target_tokens: int, provider: str, model: Optional[str]
    ) -> Tuple[str, bool]:
        """Perform intelligent truncation prioritizing important content."""

        # Split into sections
        sections = cls._split_into_sections(prompt)

        # Priority order: system > instructions > examples > data
        priority_order = [
            "system",
            "instructions",
            "task",
            "context",
            "examples",
            "data",
            "analysis",
            "details",
        ]

        truncated_sections = []
        current_tokens = 0

        # Add sections by priority until we hit the limit
        for priority in priority_order:
            for section in sections:
                if cls._get_section_type(section) == priority:
                    section_tokens = TokenCounter.count_tokens(section, provider, model)

                    if current_tokens + section_tokens <= target_tokens:
                        truncated_sections.append(section)
                        current_tokens += section_tokens
                    else:
                        # Try to partially include this section
                        remaining_tokens = target_tokens - current_tokens
                        if remaining_tokens > 100:  # Only if meaningful space left
                            partial = cls._truncate_section(
                                section, remaining_tokens, provider, model
                            )
                            truncated_sections.append(partial)
                        break

            if current_tokens >= target_tokens:
                break

        # Add truncation notice
        result = "\n\n".join(truncated_sections)
        result += cls.TRUNCATION_NOTICE

        return result, True

    @classmethod
    def _split_into_sections(cls, prompt: str) -> List[str]:
        """Split prompt into logical sections."""
        # Look for common section markers
        section_markers = [
            r"System:",
            r"Instructions:",
            r"Task:",
            r"Context:",
            r"Examples:",
            r"Analysis:",
            r"Data:",
            r"Tool Execution Details:",
            r"Results:",
        ]

        sections = []
        current_section = ""

        for line in prompt.split("\n"):
            if any(
                re.match(marker, line.strip(), re.IGNORECASE)
                for marker in section_markers
            ):
                if current_section.strip():
                    sections.append(current_section.strip())
                current_section = line
            else:
                current_section += "\n" + line

        if current_section.strip():
            sections.append(current_section.strip())

        return sections if sections else [prompt]

    @classmethod
    def _get_section_type(cls, section: str) -> str:
        """Determine the type/priority of a section."""
        section_lower = section.lower()

        if "system:" in section_lower or section_lower.startswith("you are"):
            return "system"
        elif "instruction" in section_lower or "task:" in section_lower:
            return "instructions"
        elif "example" in section_lower:
            return "examples"
        elif "analysis" in section_lower or "result" in section_lower:
            return "analysis"
        elif "data:" in section_lower or "tool execution" in section_lower:
            return "data"
        elif "context" in section_lower:
            return "context"
        else:
            return "details"

    @classmethod
    def _truncate_section(
        cls, section: str, target_tokens: int, provider: str, model: Optional[str]
    ) -> str:
        """Truncate a single section to fit token limit."""
        current_tokens = TokenCounter.count_tokens(section, provider, model)

        if current_tokens <= target_tokens:
            return section

        # Binary search for the right length
        lines = section.split("\n")
        left, right = 0, len(lines)
        best_result = lines[0] if lines else ""

        while left <= right:
            mid = (left + right) // 2
            candidate = (
                "\n".join(lines[:mid]) + "\n[... truncated]"
                if mid < len(lines)
                else "\n".join(lines[:mid])
            )

            tokens = TokenCounter.count_tokens(candidate, provider, model)

            if tokens <= target_tokens:
                best_result = candidate
                left = mid + 1
            else:
                right = mid - 1

        return best_result


class TokenAwareResponse:
    """Helper for handling token-limited responses."""

    @staticmethod
    def create_truncated_summary(
        tool_calls: List[Dict[str, Any]], reason: str = "token_limit"
    ) -> str:
        """Create a summary when analysis is truncated due to token limits."""
        if not tool_calls:
            return "Analysis was truncated due to token limits. No tool executions were completed."

        summary_parts = [
            f"Analysis was truncated due to {reason.replace('_', ' ')}.",
            f"Successfully executed {len(tool_calls)} tools before truncation:",
        ]

        # Summarize tool calls
        for i, tool_call in enumerate(tool_calls, 1):
            tool_name = tool_call.get("tool_name", "Unknown Tool")
            status = tool_call.get("status", "unknown")

            if status == "success":
                summary_parts.append(f"  {i}. ✅ {tool_name}: Completed successfully")
            else:
                summary_parts.append(f"  {i}. ❌ {tool_name}: Failed or incomplete")

        summary_parts.append(
            "\nRecommendation: Consider reducing the scope of analysis or implementing data pagination."
        )

        return "\n".join(summary_parts)

    @staticmethod
    def should_summarize_existing_calls(
        tool_calls: List[Dict[str, Any]], min_calls: int = 1
    ) -> bool:
        """Determine if we should summarize existing tool calls when truncated."""
        successful_calls = len(
            [tc for tc in tool_calls if tc.get("status") == "success"]
        )
        return successful_calls >= min_calls

    @staticmethod
    def create_intelligent_truncated_summary(
        tool_calls: List[Dict[str, Any]],
        original_prompt: Optional[str] = None,
        reason: str = "token_limit",
    ) -> str:
        """
        Create an intelligent summary focusing on actual findings rather than technical limitations.

        Args:
            tool_calls: List of tool execution results
            original_prompt: Original analysis prompt to understand objectives
            reason: Reason for truncation

        Returns:
            Intelligent summary based on actual data collected
        """
        if not tool_calls:
            return "Analysis was truncated due to token limits. No tool executions were completed."

        # Extract insights from tool results
        insights = TokenAwareResponse._extract_universal_insights(tool_calls)

        # Determine analysis objectives from prompt
        objectives = TokenAwareResponse._extract_analysis_objectives(original_prompt)

        # Build intelligent summary
        summary_parts = []

        # Start with context about what was accomplished
        successful_tools = len(
            [tc for tc in tool_calls if tc.get("status") == "success"]
        )
        summary_parts.append(
            f"Analysis completed {successful_tools} operations before reaching token limits."
        )

        # Add key findings based on actual data
        if insights.get("key_findings"):
            summary_parts.append("\n## Key Findings:")
            for finding in insights["key_findings"]:
                summary_parts.append(f"- {finding}")

        # Add data summary if available
        if insights.get("data_summary"):
            summary_parts.append("\n## Data Summary:")
            summary_parts.append(insights["data_summary"])

        # Add metrics if found
        if insights.get("metrics"):
            summary_parts.append("\n## Metrics Discovered:")
            for metric, value in insights["metrics"].items():
                summary_parts.append(f"- {metric}: {value}")

        # Add recommendation based on objectives
        if objectives:
            summary_parts.append("\n## Analysis Status:")
            summary_parts.append(
                f"Successfully gathered data for: {', '.join(objectives)}"
            )
            summary_parts.append(
                "Recommendation: Review findings above for key insights."
            )
        else:
            summary_parts.append(
                "\nRecommendation: Key insights have been extracted from available data."
            )

        return "\n".join(summary_parts)

    @staticmethod
    def _extract_universal_insights(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract universal insights from any MCP tool results using pattern recognition.

        Args:
            tool_calls: List of tool execution results

        Returns:
            Dictionary containing extracted insights
        """
        key_findings: List[str] = []
        patterns: List[str] = []
        insights: Dict[str, Any] = {
            "key_findings": key_findings,
            "data_summary": "",
            "metrics": {},
            "patterns": patterns,
        }

        total_items = 0
        categories: Dict[str, int] = {}
        status_counts: Dict[str, int] = {"success": 0, "error": 0, "warning": 0}

        for tool_call in tool_calls:
            if tool_call.get("status") != "success":
                continue

            # Extract data from tool results
            result = tool_call.get("result", "")
            if not result:
                continue

            # Analyze result content for patterns
            result_str = str(result)

            # Count items (repositories, files, records, etc.)
            item_counts = TokenAwareResponse._count_items_in_result(result_str)
            total_items += item_counts

            # Extract categories and types
            found_categories = TokenAwareResponse._extract_categories(result_str)
            for category, count in found_categories.items():
                categories[category] = categories.get(category, 0) + count

            # Extract status indicators
            statuses = TokenAwareResponse._extract_status_indicators(result_str)
            for status, count in statuses.items():
                status_counts[status] = status_counts.get(status, 0) + count

            # Extract key patterns
            found_patterns = TokenAwareResponse._extract_data_patterns(result_str)
            patterns.extend(found_patterns)

        # Build insights
        if total_items > 0:
            key_findings.append(
                f"Analyzed {total_items} total items across {len(tool_calls)} operations"
            )

        if categories:
            top_categories = sorted(
                categories.items(), key=lambda x: x[1], reverse=True
            )[:3]
            category_summary = ", ".join(
                [f"{cat} ({count})" for cat, count in top_categories]
            )
            key_findings.append(f"Primary categories: {category_summary}")

        if status_counts["success"] > 0 or status_counts["error"] > 0:
            total_status = sum(status_counts.values())
            if total_status > 0:
                success_rate = (status_counts["success"] / total_status) * 100
                insights["metrics"]["Success Rate"] = f"{success_rate:.1f}%"

        # Create data summary
        if total_items > 0:
            insights["data_summary"] = (
                f"Successfully processed {total_items} items with "
                f"{len(categories)} distinct categories identified."
            )

        return insights

    @staticmethod
    def _count_items_in_result(result_str: str) -> int:
        """Count items in result using common patterns."""
        import re

        # Look for common counting patterns
        patterns = [
            r"(\d+)\s+(repositories|repos|files|items|records|entries|results)",
            r"found\s+(\d+)",
            r"total[:\s]+(\d+)",
            r"(\d+)\s+matches?",
            r"(\d+)\s+found",
        ]

        total = 0
        for pattern in patterns:
            matches = re.findall(pattern, result_str.lower())
            for match in matches:
                if isinstance(match, tuple):
                    try:
                        total += int(match[0])
                    except (ValueError, IndexError):
                        pass
                else:
                    try:
                        total += int(match)
                    except ValueError:
                        pass

        # Fallback: count JSON arrays or lists
        if total == 0:
            json_arrays = re.findall(r"\[[^\]]+\]", result_str)
            for array in json_arrays:
                # Rough count of items in array
                items = (
                    array.count(",") + 1
                    if "," in array
                    else (1 if array.strip() != "[]" else 0)
                )
                total += items

        return total

    @staticmethod
    def _extract_categories(result_str: str) -> Dict[str, int]:
        """Extract categories from result using pattern recognition."""
        import re

        categories = {}

        # Common technology patterns
        tech_patterns = {
            "Python": r"\b(python|\.py|django|flask|fastapi)\b",
            "JavaScript": r"\b(javascript|js|node|npm|react|vue|angular)\b",
            "Java": r"\b(java|\.java|spring|maven|gradle)\b",
            "Go": r"\b(golang|go|\.go)\b",
            "Rust": r"\b(rust|\.rs|cargo)\b",
            "TypeScript": r"\b(typescript|\.ts|\.tsx)\b",
            "Docker": r"\b(docker|dockerfile|container)\b",
            "Kubernetes": r"\b(k8s|kubernetes|kubectl)\b",
            "Configuration": r"\b(config|yaml|json|toml|env)\b",
            "Documentation": r"\b(readme|docs|documentation|\.md)\b",
        }

        for category, pattern in tech_patterns.items():
            matches = len(re.findall(pattern, result_str.lower()))
            if matches > 0:
                categories[category] = matches

        return categories

    @staticmethod
    def _extract_status_indicators(result_str: str) -> Dict[str, int]:
        """Extract status indicators from result."""
        import re

        status_patterns = {
            "success": r"\b(success|ok|passed|completed|active|healthy)\b",
            "error": r"\b(error|failed|failure|critical|down|broken)\b",
            "warning": r"\b(warning|caution|deprecated|slow)\b",
        }

        statuses = {}
        for status, pattern in status_patterns.items():
            matches = len(re.findall(pattern, result_str.lower()))
            if matches > 0:
                statuses[status] = matches

        return statuses

    @staticmethod
    def _extract_data_patterns(result_str: str) -> List[str]:
        """Extract interesting data patterns from result."""
        import re

        patterns = []

        # Look for percentage patterns
        percentages = re.findall(r"(\d+(?:\.\d+)?%)", result_str)
        if percentages:
            patterns.append(f"Metrics include: {', '.join(percentages[:3])}")

        # Look for large numbers (could be metrics)
        large_numbers = re.findall(r"\b(\d{4,})\b", result_str)
        if large_numbers:
            patterns.append(f"Notable values: {', '.join(large_numbers[:3])}")

        return patterns

    @staticmethod
    def _extract_analysis_objectives(prompt: str) -> List[str]:
        """Extract analysis objectives from the original prompt."""
        if not prompt:
            return []

        objectives = []
        prompt_lower = prompt.lower()

        # Common analysis objectives
        objective_patterns = [
            ("Repository Analysis", ["repository", "repo", "codebase"]),
            ("Code Search", ["search", "find", "pattern", "code"]),
            ("Technology Stack", ["technology", "framework", "language", "stack"]),
            ("Metrics Analysis", ["metrics", "performance", "usage", "statistics"]),
            ("Health Check", ["health", "status", "monitoring", "uptime"]),
            ("Configuration Review", ["config", "settings", "environment"]),
            ("Security Analysis", ["security", "vulnerability", "audit"]),
            ("Documentation Review", ["documentation", "readme", "docs"]),
        ]

        for objective, keywords in objective_patterns:
            if any(keyword in prompt_lower for keyword in keywords):
                objectives.append(objective)

        return objectives[:5]  # Limit to top 5 objectives
