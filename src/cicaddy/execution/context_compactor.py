"""Stateless context compacting for ephemeral pipeline environments."""

import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from cicaddy.ai_providers.base import BaseProvider, ProviderMessage
from cicaddy.utils.logger import get_logger
from cicaddy.utils.token_utils import TokenCounter

logger = get_logger(__name__)


class CompressionStrategy(Enum):
    """Compression strategy levels based on token utilization."""

    NONE = "none"  # No compression needed
    LIGHT = "light"  # Compress verbose data only
    MODERATE = "moderate"  # Compress older iterations
    AGGRESSIVE = "aggressive"  # Keep only essentials


@dataclass
class ContextCompactorConfig:
    """Configuration for context compacting behavior."""

    use_ai_summarization: bool = True  # Use AI for intelligent summarization (Phase 2)
    ai_summary_max_tokens: int = 500  # Max tokens for AI-generated summaries
    fallback_to_rules: bool = True  # Fallback to rule-based if AI fails
    preserve_recent_messages: int = 2  # Number of recent messages to preserve
    min_messages_to_compress: int = 3  # Minimum messages before compression


@dataclass
class CompressionResult:
    """Result of a compression operation."""

    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    strategy_used: CompressionStrategy
    compression_time_ms: float
    information_preserved: float  # Percentage of info preserved (measured in Phase 2)


class ContextCompactor:
    """
    Stateless context compacting for ephemeral GitLab pipeline environments.

    Implements Phase 1 + Phase 2 compression strategies:
    1. Pre-execution prompt compression (rule-based)
    2. Iterative context summarization (AI-powered + rule-based fallback)
    3. Tool result streaming compression (rule-based)
    4. Adaptive information preservation measurement (Phase 2)
    """

    def __init__(
        self,
        provider: str = "gemini",
        model: Optional[str] = None,
        ai_provider: Optional[BaseProvider] = None,
        config: Optional[ContextCompactorConfig] = None,
        tokenizer: Optional[Any] = None,
    ):
        """
        Initialize context compactor.

        Args:
            provider: AI provider name for token counting
            model: Model name for accurate token estimation
            ai_provider: AI provider instance for Phase 2 AI-powered summarization
            config: Configuration for compacting behavior
            tokenizer: Optional tokenizer for accurate token counting (Phase 4)
        """
        self.provider = provider
        self.model = model
        self.ai_provider = ai_provider
        self.config = config or ContextCompactorConfig()
        self.tokenizer = tokenizer  # Phase 4: Optional tokenizer for improved accuracy

        if self.config.use_ai_summarization and not ai_provider:
            logger.warning(
                "AI summarization enabled but no AI provider provided - will use rule-based fallback"
            )
            self.config.use_ai_summarization = False

    def determine_compression_strategy(
        self,
        token_utilization: float,
        iteration_count: int = 0,
        has_large_results: bool = False,
    ) -> CompressionStrategy:
        """
        Determine appropriate compression strategy based on multiple factors.

        Phase 3 enhancement: Multi-factor strategy selection considers:
        - Token utilization (primary factor)
        - Iteration count (early vs late in execution)
        - Result size patterns (large tool outputs)

        Args:
            token_utilization: Current token usage as percentage (0.0 to 1.0)
            iteration_count: Current iteration number (0 = not provided)
            has_large_results: Whether recent tool results were unusually large

        Returns:
            Appropriate compression strategy
        """
        # Override: Critical token utilization always uses AGGRESSIVE compression
        if token_utilization >= 0.85:
            return CompressionStrategy.AGGRESSIVE

        # Factor 1: Iteration-based strategy (if iteration count provided)
        if iteration_count > 0:
            if iteration_count <= 3:
                # Early iterations: Be conservative, use LIGHT compression only
                base_strategy = CompressionStrategy.LIGHT
            elif iteration_count <= 6:
                # Mid iterations: Use MODERATE compression
                base_strategy = CompressionStrategy.MODERATE
            else:
                # Late iterations (7+): Use AGGRESSIVE compression
                base_strategy = CompressionStrategy.AGGRESSIVE
        else:
            # No iteration info, use token utilization only (backward compatibility)
            if token_utilization < 0.5:
                base_strategy = CompressionStrategy.NONE
            elif token_utilization < 0.7:
                base_strategy = CompressionStrategy.LIGHT
            else:
                base_strategy = CompressionStrategy.MODERATE

        # Factor 2: Adjust for large results (upgrade compression level)
        if has_large_results and base_strategy == CompressionStrategy.LIGHT:
            base_strategy = CompressionStrategy.MODERATE

        # Factor 3: Token utilization override (downgrade if very low utilization)
        if token_utilization < 0.5 and base_strategy != CompressionStrategy.AGGRESSIVE:
            # Very low token usage, can use less aggressive compression
            return (
                CompressionStrategy.NONE
                if iteration_count <= 3
                else CompressionStrategy.LIGHT
            )

        return base_strategy

    def estimate_tokens(self, text: str, use_tokenizer: bool = True) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to estimate tokens for
            use_tokenizer: If True and tokenizer available, use it for accurate counting

        Returns:
            Estimated token count

        Phase 4: Supports optional tokenizer for improved accuracy (from LangChain).
        Falls back to TokenCounter approximation if tokenizer unavailable.
        """
        if use_tokenizer and self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text)
                return len(tokens)
            except Exception as e:
                logger.warning(f"Tokenizer failed, falling back to TokenCounter: {e}")

        # Fallback to TokenCounter (provider-specific approximation)
        return TokenCounter.count_tokens(text, self.provider, self.model)

    def compress_prompt_before_send(
        self, prompt: str, token_budget: int, reserve_for_response: int = 2000
    ) -> Tuple[str, CompressionResult]:
        """
        Compress prompt before sending to AI to fit within token budget.

        Strategy:
        - Reserve space for response (default 2000 tokens)
        - Target 60-70% of available budget for prompt
        - Priority: system > task > critical_data > examples > verbose_data

        Args:
            prompt: Original prompt text
            token_budget: Total token budget available
            reserve_for_response: Tokens to reserve for AI response

        Returns:
            Tuple of (compressed_prompt, compression_result)
        """
        start_time = time.time()

        original_tokens = self.estimate_tokens(prompt, use_tokenizer=True)
        available_for_prompt = token_budget - reserve_for_response
        target_tokens = int(available_for_prompt * 0.7)  # Use 70% of available

        if original_tokens <= target_tokens:
            # No compression needed
            result = CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                strategy_used=CompressionStrategy.NONE,
                compression_time_ms=0,
                information_preserved=1.0,
            )
            return prompt, result

        # Perform intelligent compression
        compressed_prompt = self._compress_by_sections(prompt, target_tokens)

        compressed_tokens = self.estimate_tokens(compressed_prompt, use_tokenizer=True)
        compression_time = (time.time() - start_time) * 1000  # Convert to ms

        result = CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=original_tokens / max(compressed_tokens, 1),
            strategy_used=CompressionStrategy.MODERATE,
            compression_time_ms=compression_time,
            information_preserved=0.85,  # Estimated for section-based truncation (Phase 1 heuristic)
        )

        logger.info(
            f"Compressed prompt: {original_tokens} -> {compressed_tokens} tokens "
            f"(ratio: {result.compression_ratio:.2f}x, time: {compression_time:.1f}ms)"
        )

        return compressed_prompt, result

    async def compact_iteration_context(
        self,
        conversation_messages: List[ProviderMessage],
        available_budget: int,
        current_iteration: int,
        tool_pairs: Optional[Dict[str, tuple]] = None,
    ) -> Tuple[List[ProviderMessage], CompressionResult]:
        """
        Compact conversation context between iterations.

        Strategy (Phase 2):
        - Keep current iteration in full detail
        - Use AI to intelligently summarize previous iterations (with rule-based fallback)
        - Calculate actual information preservation
        - Replace verbose tool results with key findings

        Phase 4 Enhancement (Tool Pair Preservation):
        - Preserve incomplete tool pairs (request without response or vice versa)
        - Complete tool pairs (both present) can be safely summarized together

        Args:
            conversation_messages: List of conversation messages
            available_budget: Available token budget
            current_iteration: Current iteration number
            tool_pairs: Optional dict mapping tool_id -> (request_idx, response_idx)
                        Used to preserve tool pair integrity during compaction

        Returns:
            Tuple of (compacted_messages, compression_result)
        """
        start_time = time.time()

        if len(conversation_messages) <= self.config.min_messages_to_compress:
            # Not enough context to compress
            original_tokens = self._count_message_tokens(conversation_messages)
            result = CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                strategy_used=CompressionStrategy.NONE,
                compression_time_ms=0,
                information_preserved=1.0,
            )
            return conversation_messages, result

        original_tokens = self._count_message_tokens(conversation_messages)
        original_content = " ".join([msg.content for msg in conversation_messages])

        # Phase 4: Identify messages that must be preserved (incomplete tool pairs)
        preserve_indices = self._identify_incomplete_tool_pair_messages(
            conversation_messages, tool_pairs
        )

        # Keep first message (system/task) and last N messages (current iteration)
        compacted = []
        if conversation_messages:
            compacted.append(conversation_messages[0])  # Keep initial prompt

        # Separate messages to summarize vs preserve
        if len(conversation_messages) > self.config.min_messages_to_compress:
            # Middle section (normally summarized)
            middle_start = 1
            middle_end = (
                len(conversation_messages) - self.config.preserve_recent_messages
            )

            # Split middle messages into "can summarize" and "must preserve"
            middle_messages = []
            preserved_middle = []

            for i in range(middle_start, middle_end):
                if i in preserve_indices:
                    preserved_middle.append((i, conversation_messages[i]))
                else:
                    middle_messages.append(conversation_messages[i])

            # Summarize only the messages that can be safely summarized
            if middle_messages:  # Only summarize if there are messages to summarize
                # Try AI-powered summarization first (Phase 2)
                if self.config.use_ai_summarization and self.ai_provider:
                    try:
                        summary = await self._summarize_messages_with_ai(
                            middle_messages, current_iteration
                        )
                        logger.debug("Using AI-powered summarization")
                    except Exception as e:
                        logger.warning(
                            f"AI summarization failed: {e}, falling back to rule-based"
                        )
                        if self.config.fallback_to_rules:
                            summary = self._summarize_messages(
                                middle_messages, current_iteration
                            )
                        else:
                            raise
                else:
                    # Fall back to rule-based summarization (Phase 1)
                    summary = self._summarize_messages(
                        middle_messages, current_iteration
                    )
                    logger.debug("Using rule-based summarization")

                compacted.append(ProviderMessage(content=summary, role="user"))

            # Phase 4: Add back preserved middle messages (incomplete tool pairs) in original order
            if preserved_middle:
                logger.debug(
                    f"Preserving {len(preserved_middle)} messages (incomplete tool pairs)"
                )
                # Sort by original index to maintain conversation order
                preserved_middle.sort(key=lambda x: x[0])
                compacted.extend([msg for _, msg in preserved_middle])

        # Keep last N messages (most recent iteration)
        if len(conversation_messages) >= self.config.preserve_recent_messages:
            compacted.extend(
                conversation_messages[-self.config.preserve_recent_messages :]
            )

        compressed_tokens = self._count_message_tokens(compacted)
        compressed_content = " ".join([msg.content for msg in compacted])
        compression_time = (time.time() - start_time) * 1000

        # Calculate actual information preservation (Phase 2)
        info_preserved = self._calculate_information_preserved(
            original_content, compressed_content
        )

        result = CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=original_tokens / max(compressed_tokens, 1),
            strategy_used=CompressionStrategy.MODERATE,
            compression_time_ms=compression_time,
            information_preserved=info_preserved,
        )

        logger.info(
            f"Compacted conversation: {len(conversation_messages)} -> {len(compacted)} messages, "
            f"{original_tokens} -> {compressed_tokens} tokens "
            f"(ratio: {result.compression_ratio:.2f}x, preserved: {info_preserved:.1%})"
        )

        return compacted, result

    def compress_tool_result_realtime(
        self, tool_result: Dict[str, Any], max_tokens: int
    ) -> Tuple[Dict[str, Any], CompressionResult]:
        """
        Compress tool result in real-time to fit token budget.

        Strategy:
        - Extract structured data (numbers, statuses, summaries)
        - Discard verbose debug output and redundant information
        - Preserve error messages and critical findings

        Args:
            tool_result: Tool execution result
            max_tokens: Maximum tokens allowed for result

        Returns:
            Tuple of (compressed_result, compression_result)
        """
        start_time = time.time()

        result_content = str(tool_result.get("result", ""))
        original_tokens = self.estimate_tokens(result_content, use_tokenizer=True)

        if original_tokens <= max_tokens:
            # No compression needed
            result = CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                strategy_used=CompressionStrategy.NONE,
                compression_time_ms=0,
                information_preserved=1.0,
            )
            return tool_result, result

        # Compress the result content
        compressed_content = self._compress_tool_content(result_content, max_tokens)

        # Create compressed tool result
        compressed_result = tool_result.copy()
        compressed_result["result"] = compressed_content
        compressed_result["compressed"] = True

        compressed_tokens = self.estimate_tokens(compressed_content, use_tokenizer=True)
        compression_time = (time.time() - start_time) * 1000

        result = CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=original_tokens / max(compressed_tokens, 1),
            strategy_used=CompressionStrategy.LIGHT,
            compression_time_ms=compression_time,
            information_preserved=0.75,  # Estimated for aggressive tool result compression (Phase 1 heuristic)
        )

        logger.debug(
            f"Compressed tool result: {original_tokens} -> {compressed_tokens} tokens "
            f"(ratio: {result.compression_ratio:.2f}x)"
        )

        return compressed_result, result

    # Private helper methods

    def _identify_incomplete_tool_pair_messages(
        self, messages: List[ProviderMessage], tool_pairs: Optional[Dict[str, tuple]]
    ) -> Set[int]:
        """
        Identify message indices that are part of incomplete tool pairs.

        Phase 4: Tool Pair Preservation (from Goose).
        Incomplete pairs (request without response or vice versa) must be preserved
        during compaction to maintain conversation coherence.

        Also preserves messages marked with [TOOL_RESULTS:DO_NOT_COMPACT] to prevent
        AI from fabricating analysis for entities it never actually queried.

        Args:
            messages: List of conversation messages
            tool_pairs: Dict mapping tool_id -> (request_idx, response_idx)

        Returns:
            Set of message indices that must be preserved
        """
        preserve_indices = set()

        # First, check for messages with DO_NOT_COMPACT marker
        for idx, message in enumerate(messages):
            if "[TOOL_RESULTS:DO_NOT_COMPACT]" in message.content:
                preserve_indices.add(idx)
                logger.debug(
                    f"Preserving tool result message at index {idx} (marked DO_NOT_COMPACT)"
                )

        if not tool_pairs:
            return preserve_indices

        # Then, identify incomplete tool pairs
        for tool_id, (req_idx, resp_idx) in tool_pairs.items():
            # Case 1: Request without response (awaiting completion)
            if req_idx is not None and resp_idx is None:
                preserve_indices.add(req_idx)
                logger.debug(
                    f"Preserving incomplete tool request at index {req_idx} (tool_id: {tool_id})"
                )

            # Case 2: Response without request (orphaned response)
            elif req_idx is None and resp_idx is not None:
                preserve_indices.add(resp_idx)
                logger.debug(
                    f"Preserving orphaned tool response at index {resp_idx} (tool_id: {tool_id})"
                )

            # Case 3: Complete pair where one is in "preserve_recent" zone
            # and the other is in middle zone - preserve both
            elif req_idx is not None and resp_idx is not None:
                middle_end = len(messages) - self.config.preserve_recent_messages
                # Check if pair is split across middle/recent boundary
                req_in_middle = 0 < req_idx < middle_end
                resp_in_middle = 0 < resp_idx < middle_end

                if req_in_middle != resp_in_middle:
                    # Pair is split - preserve both
                    preserve_indices.add(req_idx)
                    preserve_indices.add(resp_idx)
                    logger.debug(
                        f"Preserving split tool pair at indices {req_idx},{resp_idx} (tool_id: {tool_id})"
                    )

        return preserve_indices

    def _count_message_tokens(self, messages: List[ProviderMessage]) -> int:
        """Count total tokens in message list. Phase 4: Uses tokenizer when available."""
        total = 0
        for msg in messages:
            total += self.estimate_tokens(msg.content, use_tokenizer=True)
        return total

    def _compress_by_sections(self, prompt: str, target_tokens: int) -> str:
        """
        Compress prompt by prioritizing important sections.

        Priority order: system > instructions > task > context > examples > data > debug
        """
        sections = self._split_into_sections(prompt)
        priority_order = [
            "system",
            "instruction",
            "task",
            "context",
            "example",
            "data",
            "analysis",
            "debug",
        ]

        compressed_sections = []
        current_tokens = 0
        # Reserve tokens for the compression notice
        notice_tokens = 20
        effective_target = max(target_tokens - notice_tokens, 10)

        # Add sections by priority until we hit target
        for priority in priority_order:
            for section in sections:
                section_type = self._get_section_type(section)
                if section_type == priority:
                    section_tokens = self.estimate_tokens(section, use_tokenizer=True)

                    if current_tokens + section_tokens <= effective_target:
                        compressed_sections.append(section)
                        current_tokens += section_tokens
                    else:
                        # Partially include this section if there's any space left
                        remaining_tokens = effective_target - current_tokens
                        if (
                            remaining_tokens > 10
                        ):  # Lower threshold - include even small portions
                            partial = self._truncate_section(section, remaining_tokens)
                            compressed_sections.append(partial)
                        break

            if current_tokens >= effective_target:
                break

        # Ensure we have at least some content
        if not compressed_sections and sections:
            # Emergency: include at least the first section, heavily truncated
            compressed_sections.append(
                self._truncate_section(sections[0], effective_target)
            )

        result = "\n\n".join(compressed_sections)
        result += (
            "\n\n[Note: Prompt compressed to fit token budget. Some details omitted.]"
        )
        return result

    def _split_into_sections(self, text: str) -> List[str]:
        """
        Split text into logical sections.

        Recognizes analysis directive markers (Objective, Deliverable, Required Analysis, Phase markers)
        to ensure they are treated as separate sections for preservation during compaction.
        """
        section_markers = [
            r"System:",
            r"Instructions?:",
            r"Objective:",  # Analysis directive
            r"Deliverable:",  # Analysis directive
            r"Required Analysis:",  # Analysis directive
            r"Phase \d+",  # Phase markers (Phase 1, Phase 2, Phase 3, etc.)
            r"\*\*MANDATORY\*\*",  # Mandatory directive marker
            r"Task:",
            r"Context:",
            r"Examples?:",
            r"Analysis:",
            r"Data:",
            r"Tool.*Results?:",
            r"Debug:",
        ]

        sections = []
        current_section = ""

        for line in text.split("\n"):
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

        return sections if sections else [text]

    def _get_section_type(self, section: str) -> str:
        """
        Determine section type for prioritization.

        Analysis directives (Objective, Deliverable, Required Analysis, Phase markers)
        are classified as "instruction" to preserve them during compaction, preventing
        loss of critical analysis requirements in iterative execution.
        """
        section_lower = section.lower()

        # NEW: Preserve analysis instructions (highest priority)
        # Identify sections containing analysis directives that must not be compressed
        if any(
            keyword in section_lower
            for keyword in [
                "objective:",
                "deliverable:",
                "required analysis:",
                "phase 1",
                "phase 2",
                "phase 3",
                "mandatory",
            ]
        ):
            return "instruction"  # Never compress - these are analysis requirements

        if "system:" in section_lower or section_lower.startswith("you are"):
            return "system"
        elif "instruction" in section_lower:
            return "instruction"
        elif "task:" in section_lower:
            return "task"
        elif "context" in section_lower:
            return "context"
        elif "example" in section_lower:
            return "example"
        elif "analysis" in section_lower or "result" in section_lower:
            return "analysis"
        elif "data:" in section_lower or "tool" in section_lower:
            return "data"
        elif "debug" in section_lower:
            return "debug"
        else:
            return "other"

    def _truncate_section(self, section: str, target_tokens: int) -> str:
        """Truncate a section to fit target tokens. Phase 4: Uses tokenizer when available."""
        current_tokens = self.estimate_tokens(section, use_tokenizer=True)

        if current_tokens <= target_tokens:
            return section

        # Binary search for optimal length
        lines = section.split("\n")
        left, right = 0, len(lines)
        best_result = lines[0] if lines else ""

        while left <= right:
            mid = (left + right) // 2
            candidate = "\n".join(lines[:mid])
            if mid < len(lines):
                candidate += "\n... [truncated]"

            tokens = self.estimate_tokens(candidate, use_tokenizer=True)

            if tokens <= target_tokens:
                best_result = candidate
                left = mid + 1
            else:
                right = mid - 1

        return best_result

    def _summarize_messages(
        self, messages: List[ProviderMessage], iteration: int
    ) -> str:
        """
        Summarize message history into compact form.

        Extracts key information from previous iterations.
        """
        tool_executions = []
        findings = []

        for msg in messages:
            content = msg.content

            # Extract tool execution summaries
            if "Tool" in content and (
                "success" in content.lower() or "failed" in content.lower()
            ):
                # Parse tool results
                tool_lines = [line for line in content.split("\n") if "Tool" in line]
                tool_executions.extend(tool_lines[:3])  # Keep first 3

            # Extract findings and errors
            if any(
                keyword in content.lower()
                for keyword in ["error", "warning", "found", "critical"]
            ):
                # Extract key findings
                lines = content.split("\n")
                for line in lines:
                    if any(
                        keyword in line.lower()
                        for keyword in ["error", "warning", "found", "critical"]
                    ):
                        findings.append(line[:150])  # Limit length

        summary_parts = [f"[Summary of iterations 1-{iteration - 1}]"]

        if tool_executions:
            summary_parts.append(f"Tools executed: {len(tool_executions)}")
            summary_parts.extend(tool_executions[:5])  # Top 5 tools

        if findings:
            summary_parts.append("Key findings:")
            summary_parts.extend(findings[:5])  # Top 5 findings

        summary_parts.append("[End summary - current iteration follows]")

        return "\n".join(summary_parts)

    def _compress_tool_content(self, content: str, max_tokens: int) -> str:
        """
        Compress tool result content using general-purpose truncation.

        Strategy (inspired by LlamaStack/DeepAgents patterns):
        - Preserve beginning of content (most important for analysis)
        - Use binary search to maximize content within token budget
        - Add clear truncation markers
        - Works for ALL data types: code search, metrics, JSON, logs, etc.

        This replaces keyword extraction with structure-preserving truncation
        to handle diverse tool outputs uniformly.
        """
        current_tokens = self.estimate_tokens(content, use_tokenizer=True)

        logger.debug(
            f"Compressing tool content: {current_tokens} tokens, budget: {max_tokens} tokens"
        )

        if current_tokens <= max_tokens:
            return content

        # Binary search for optimal truncation point
        lines = content.split("\n")
        total_lines = len(lines)

        # Reserve tokens for footer message
        footer_tokens = 50
        effective_budget = max(max_tokens - footer_tokens, 100)

        logger.debug(
            f"Binary search: total_lines={total_lines}, effective_budget={effective_budget}"
        )

        # Handle single-line results (common for JSON/structured data)
        # by splitting into chunks instead of lines
        if total_lines <= 2:
            logger.debug(
                "Single-line result detected, using character-based truncation"
            )
            # Estimate ~4 chars per token as rough heuristic
            target_chars = effective_budget * 4
            if len(content) > target_chars:
                truncated = content[:target_chars]
                # Try to break at a reasonable point (space, comma, etc.)
                for sep in ["\n", ". ", ", ", " "]:
                    last_sep = truncated.rfind(sep)
                    if last_sep > target_chars * 0.8:  # Accept if within 80% of target
                        truncated = content[: last_sep + len(sep)]
                        break
                result_tokens = self.estimate_tokens(truncated, use_tokenizer=True)
                logger.debug(
                    f"Character truncation: {len(truncated)}/{len(content)} chars, {result_tokens}/{current_tokens} tokens"
                )
                result = (
                    truncated
                    + f"\n\n[Truncated to fit {max_tokens} token budget: preserved ~{result_tokens}/{current_tokens} tokens]"
                )
                return result

        # Normal line-based binary search for multi-line content
        left, right = 0, len(lines)
        best_result = ""

        while left <= right:
            mid = (left + right) // 2
            if mid == 0:
                candidate = ""
            else:
                candidate = "\n".join(lines[:mid])

            tokens = self.estimate_tokens(candidate, use_tokenizer=True)

            if tokens <= effective_budget:
                best_result = candidate
                left = mid + 1
                logger.debug(
                    f"Binary search: mid={mid}, tokens={tokens} <= {effective_budget}, accepting"
                )
            else:
                right = mid - 1
                logger.debug(
                    f"Binary search: mid={mid}, tokens={tokens} > {effective_budget}, rejecting"
                )

        # Add informative footer showing compression statistics
        lines_preserved = len(best_result.split("\n")) if best_result else 0
        result_tokens = self.estimate_tokens(best_result, use_tokenizer=True)
        logger.debug(
            f"Compression result: preserved {lines_preserved}/{total_lines} lines, {result_tokens}/{current_tokens} tokens"
        )
        footer = (
            f"\n\n[Preserved {lines_preserved}/{total_lines} lines "
            f"({result_tokens}/{current_tokens} tokens) to fit {max_tokens} token budget]"
        )
        result = best_result + footer

        return result

    # Phase 2: AI-Powered Summarization Methods

    async def _summarize_messages_with_ai(
        self, messages: List[ProviderMessage], iteration: int
    ) -> str:
        """
        Use AI to generate intelligent summary of conversation history.

        Phase 2 enhancement: Uses the AI provider to create context-aware summaries
        that preserve critical information while dramatically reducing token count.

        Args:
            messages: Message history to summarize
            iteration: Current iteration number

        Returns:
            AI-generated summary of conversation history
        """
        if not self.ai_provider:
            raise ValueError("AI provider not available for summarization")

        # Prepare conversation history for AI summarization
        conversation_text = "\n\n".join(
            [f"[{msg.role}]: {msg.content}" for msg in messages]
        )

        # Create summarization prompt
        summary_prompt = f"""Summarize the following conversation history from iterations 1-{iteration - 1}.
Focus on preserving:
1. Key tool executions and their results
2. Critical errors, warnings, or findings
3. Important decisions or discoveries
4. Numerical data and metrics

Provide a concise summary in {self.config.ai_summary_max_tokens} tokens or less.

Conversation history:
{conversation_text}

Summary:"""

        try:
            # Use AI provider to generate summary
            summary_message = ProviderMessage(content=summary_prompt, role="user")
            response = await self.ai_provider.chat_completion([summary_message])

            if not response or not response.content:
                raise ValueError("AI provider returned empty summary")

            # Format as summary with markers
            formatted_summary = f"[Summary of iterations 1-{iteration - 1} (AI-generated)]\n{response.content}\n[End summary - current iteration follows]"

            logger.debug(f"AI-generated summary: {len(response.content)} chars")
            return formatted_summary

        except Exception as e:
            logger.error(f"AI summarization failed: {e}")
            raise

    def _calculate_information_preserved(self, original: str, compressed: str) -> float:
        """
        Calculate actual information preservation ratio.

        Phase 2 enhancement: Measures semantic preservation by comparing
        key entities and important keywords before and after compression.

        Args:
            original: Original content
            compressed: Compressed content

        Returns:
            Float between 0.0 and 1.0 representing preservation ratio
        """
        # Extract key entities from both versions
        original_entities = self._extract_key_entities(original)
        compressed_entities = self._extract_key_entities(compressed)

        if not original_entities:
            return 1.0  # No entities to preserve

        # Calculate preservation as intersection over original
        preserved_entities = original_entities & compressed_entities
        preservation_ratio = len(preserved_entities) / len(original_entities)

        logger.debug(
            f"Information preservation: {len(preserved_entities)}/{len(original_entities)} "
            f"key entities preserved ({preservation_ratio:.1%})"
        )

        return preservation_ratio

    def _extract_key_entities(self, text: str) -> Set[str]:
        """
        Extract important keywords and entities for preservation measurement.

        Extracts:
        - Tool names (capitalized words followed by execution/call/result)
        - Error/warning keywords
        - Numerical values with context
        - Important domain terms (CRITICAL, WARNING, SUCCESS, FAILED, etc.)

        Args:
            text: Text to extract entities from

        Returns:
            Set of key entities found in text
        """
        entities = set()

        # Extract tool names (words before "execution", "call", "result", "success", "failed")
        tool_pattern = r"(\w+)\s+(?:execution|call|result|success|failed|error)"
        tool_matches = re.findall(tool_pattern, text, re.IGNORECASE)
        entities.update([t.lower() for t in tool_matches])

        # Extract error/warning keywords with context
        error_pattern = r"(error|warning|critical|failed|failure)[\s:]+([^\n]{0,50})"
        error_matches = re.findall(error_pattern, text, re.IGNORECASE)
        entities.update([f"{e[0].lower()}:{e[1][:20]}" for e in error_matches])

        # Extract numerical values with context (e.g., "100 requests", "50% success")
        number_pattern = (
            r"(\d+\.?\d*)\s*(%|requests?|errors?|tokens?|ms|seconds?|bytes?|MB|KB)"
        )
        number_matches = re.findall(number_pattern, text, re.IGNORECASE)
        entities.update([f"{n[0]}{n[1].lower()}" for n in number_matches])

        # Extract important status keywords
        status_keywords = [
            "success",
            "failed",
            "completed",
            "timeout",
            "critical",
            "warning",
            "error",
        ]
        for keyword in status_keywords:
            if keyword.lower() in text.lower():
                entities.add(keyword.lower())

        # Extract capitalized important terms (potential tool/system names)
        capitalized_pattern = r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)*)\b"
        capitalized_matches = re.findall(capitalized_pattern, text)
        # Only keep if they appear multiple times (likely important)
        for match in capitalized_matches:
            if text.count(match) >= 2:
                entities.add(match.lower())

        return entities
