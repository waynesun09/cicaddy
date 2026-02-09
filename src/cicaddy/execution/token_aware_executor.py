"""Token-aware execution engine with LlamaStack-inspired multi-level safety valves."""

import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from cicaddy.ai_providers.base import BaseProvider, ProviderMessage, ProviderResponse
from cicaddy.execution.context_compactor import (
    ContextCompactor,
    ContextCompactorConfig,
)
from cicaddy.execution.error_classifier import (
    ClassifiedError,
    ErrorType,
    has_continuation_indicator,
)
from cicaddy.execution.event_log import EventLog
from cicaddy.execution.knowledge_store import AccumulatedKnowledge
from cicaddy.execution.recovery import RecoveryManager, RecoveryResult
from cicaddy.mcp_client.client import OfficialMCPClientManager
from cicaddy.tools import ToolRegistry
from cicaddy.utils.logger import get_logger
from cicaddy.utils.tool_converter import convert_openai_tool_calls_to_mcp

logger = get_logger(__name__)

# Phase 4: Conservative Context Estimation (from Goose)
SYSTEM_PROMPT_TOKEN_OVERHEAD = 3000  # Estimated tokens for system prompt
TOOLS_TOKEN_OVERHEAD = 5000  # Estimated tokens for tool definitions
# Note: CONTEXT_SAFETY_FACTOR is now configurable via settings.context_safety_factor


class StopReason(Enum):
    """Execution stop reasons following LlamaStack patterns."""

    end_of_turn = "end_of_turn"  # Complete response generated
    end_of_message = "end_of_message"  # Partial response (usually tool call)
    out_of_tokens = "out_of_tokens"  # Token budget exhausted
    max_iterations = "max_iterations"  # Iteration limit reached
    max_tools = "max_tools"  # Tool limit per iteration reached
    max_result_size = "max_result_size"  # Result size limit reached
    timeout = "timeout"  # Time limit exceeded
    error = "error"  # Execution error occurred


@dataclass
class ExecutionLimits:
    """Multi-level execution limits following LlamaStack patterns."""

    # Iteration limits (LlamaStack max_infer_iters pattern)
    max_infer_iters: int = 10

    # Token limits (LlamaStack out_of_tokens pattern)
    # Defaults are conservative fallbacks - BaseAIAgent dynamically configures these based on model capabilities
    # Default values accommodate most modern LLMs (Claude, GPT-4, Gemini) with reasonable safety margins
    max_tokens_total: int = (
        128000  # Total token budget - conservative fallback (most models support 128K+)
    )
    max_tokens_per_iteration: int = (
        8000  # Per-iteration budget - dynamically set to ~6.25% of model's input
    )
    max_tokens_per_tool_result: int = (
        4000  # Per-tool budget - dynamically set to 25% of model's output
    )

    # Tool execution limits
    max_tools_per_iteration: int = 5  # Max tool calls per iteration
    max_total_tools: int = 50  # Max total tool calls

    # Result size limits (bytes)
    max_result_size_bytes: int = 1024 * 1024  # 1MB max per tool result
    max_total_result_size_bytes: int = 10 * 1024 * 1024  # 10MB total

    # Time limits (seconds)
    max_execution_time: int = 600  # 10 minutes total execution
    max_tool_timeout: int = 300  # 5 minutes absolute maximum per tool call
    max_tool_idle_timeout: int = (
        60  # 1 minute idle timeout (time between progress updates/data)
    )

    # Progressive degradation thresholds (percentage of limits)
    warning_threshold: float = 0.8  # Warn at 80% of any limit
    degradation_threshold: float = 0.9  # Start degrading at 90% of any limit

    # Recovery mechanism configuration (Early Break Recovery)
    # When enabled, degradation is skipped for iteration/token limits
    # to allow execution to continue until the actual limit is hit,
    # at which point recovery handles continuation with fresh context
    recovery_enabled: bool = True  # Enable recovery from early stops
    max_recovery_attempts: int = 3  # Max recovery attempts per error pattern
    recovery_ai_timeout: int = 30  # Shorter timeout for recovery AI calls

    def __post_init__(self):
        """Override max_execution_time from MAX_EXECUTION_TIME environment variable if set."""
        # Allow overriding max_execution_time for long-running database queries
        # Valid range: 60-7200 seconds (1 minute to 2 hours)
        if env_max_time := os.getenv("MAX_EXECUTION_TIME"):
            try:
                timeout = int(env_max_time)
                # Validate range: 60s (1 min) to 7200s (2 hours)
                if not (60 <= timeout <= 7200):
                    logger.warning(
                        f"MAX_EXECUTION_TIME {timeout}s outside safe range [60-7200], "
                        f"using default: {self.max_execution_time}s"
                    )
                else:
                    self.max_execution_time = timeout
                    logger.info(
                        f"Overriding max_execution_time from environment: {self.max_execution_time}s"
                    )
            except ValueError:
                logger.warning(
                    f"Invalid MAX_EXECUTION_TIME value: {env_max_time}, using default: {self.max_execution_time}s"
                )


@dataclass
class ExecutionState:
    """Current execution state tracking all limits."""

    # Iteration tracking
    current_iteration: int = 0

    # Token usage tracking
    total_tokens_used: int = 0  # Per-inference tokens (resets on fresh context)
    current_iteration_tokens: int = 0
    cumulative_tokens_all_inferences: int = (
        0  # Total across all inferences (never lost)
    )

    # Tool execution tracking
    total_tools_executed: int = 0  # Per-inference tools (resets on fresh context)
    current_iteration_tools: int = 0
    cumulative_tools_all_inferences: int = (
        0  # Total tools across all inferences (never lost)
    )

    # Result size tracking
    total_result_size_bytes: int = 0

    # Time tracking
    start_time: float = field(default_factory=time.time)
    last_iteration_time: float = field(default_factory=time.time)

    # State flags
    warnings_issued: List[str] = field(default_factory=list)
    degradation_active: bool = False
    stop_reason: Optional[StopReason] = None

    # Phase 3: Compression tracking (quality metrics)
    compression_count: int = 0  # Number of compressions performed
    total_compression_ratio: float = 0.0  # Sum of compression ratios
    total_information_preserved: float = 0.0  # Sum of information preservation ratios
    compression_triggers: List[str] = field(
        default_factory=list
    )  # Triggers that caused compaction

    # Early Break Recovery tracking
    recovery_attempts: int = (
        0  # Total recovery attempts (each is an additional AI call)
    )
    successful_recoveries: int = 0  # Number of successful recoveries
    last_recovery_error_type: Optional[str] = (
        None  # Last error type that triggered recovery
    )

    # Max tokens recovery tracking (separate from other recovery types)
    max_tokens_recovery_attempts: int = 0  # Track fresh context recovery attempts
    max_tokens_recovery_limit: int = 3  # Maximum allowed recovery inferences

    # Max iterations recovery tracking (parallel to max_tokens_recovery)
    max_iterations_recovery_attempts: int = 0  # Track iteration limit recovery attempts
    max_iterations_recovery_limit: int = 2  # More conservative than tokens (2 vs 3)

    # Cumulative iterations tracking (never reset - for total metrics across all inferences)
    cumulative_iterations_all_inferences: int = 0

    # Fresh context recovery mode - when True, AI is expected to synthesize not explore
    # Used to prevent false AI_PREMATURE_COMPLETION detection after recovery
    in_fresh_context_recovery: bool = False

    # Internal flag to prevent repeated logging of degradation skip message
    _degradation_skip_logged: bool = False

    def reset_iteration(self):
        """Reset per-iteration counters."""
        self.current_iteration_tokens = 0
        self.current_iteration_tools = 0
        self.last_iteration_time = time.time()

    def reset_for_fresh_context(self):
        """Reset per-inference counters for fresh context recovery.

        Called when starting a new inference with fresh context after
        MAX_TOKENS_EXCEEDED or MAX_ITERATIONS_EXCEEDED. Preserves cumulative
        metrics while resetting per-inference counters.

        Token tracking strategy:
        - cumulative_tokens_all_inferences: Never reset, accumulates across all inferences
        - total_tokens_used: Reset to 0 for fresh context limit checking

        Iteration tracking strategy:
        - cumulative_iterations_all_inferences: Never reset, accumulates for total metrics
        - current_iteration: Reset to 0 for fresh context (will increment to 1 at loop start)

        Tool tracking strategy:
        - cumulative_tools_all_inferences: Never reset, accumulates for total metrics/billing
        - total_tools_executed: Reset to 0 for fresh context limit checking
        """
        # Accumulate current inference tokens before resetting (preserves metrics)
        self.cumulative_tokens_all_inferences += self.total_tokens_used

        # Accumulate current iterations before resetting (preserves total count)
        self.cumulative_iterations_all_inferences += self.current_iteration

        # Accumulate current inference tools before resetting (preserves total count)
        self.cumulative_tools_all_inferences += self.total_tools_executed

        # Reset per-inference counters for limit checking
        self.total_tokens_used = 0
        self.current_iteration_tokens = 0
        self.current_iteration_tools = 0
        self.current_iteration = 0  # Reset iteration counter for fresh context
        self.total_tools_executed = 0  # Reset per-inference tool counter
        self.last_iteration_time = time.time()

        # Enter fresh context recovery mode - AI should synthesize, not explore
        # This flag prevents false AI_PREMATURE_COMPLETION detection from ultra-short responses
        # because after recovery, a short synthesis response is actually correct behavior
        self.in_fresh_context_recovery = True

        # Note: We preserve total_result_size_bytes as it represents accumulated data size

    def get_total_tokens_all_inferences(self) -> int:
        """Get total tokens consumed across all inferences (for reporting/billing).

        Returns:
            Total tokens from current inference + all previous inferences
        """
        return self.cumulative_tokens_all_inferences + self.total_tokens_used

    def get_total_iterations_all_inferences(self) -> int:
        """Get total iterations executed across all inferences (for reporting).

        Returns:
            Total iterations from current inference + all previous inferences
        """
        return self.cumulative_iterations_all_inferences + self.current_iteration

    def get_total_tools_all_inferences(self) -> int:
        """Get total tools executed across all inferences (for reporting).

        Returns:
            Total tools from current inference + all previous inferences
        """
        return self.cumulative_tools_all_inferences + self.total_tools_executed

    def add_tokens(self, count: int):
        """Add token usage to current tracking."""
        self.total_tokens_used += count
        self.current_iteration_tokens += count

    def add_tool_execution(self, result_size_bytes: int = 0):
        """Add tool execution to current tracking."""
        self.total_tools_executed += 1
        self.current_iteration_tools += 1
        self.total_result_size_bytes += result_size_bytes

    def get_elapsed_time(self) -> float:
        """Get total elapsed execution time."""
        return time.time() - self.start_time

    def get_iteration_time(self) -> float:
        """Get time since last iteration started."""
        return time.time() - self.last_iteration_time

    def record_compression(
        self, compression_ratio: float, information_preserved: float, trigger: str
    ):
        """Record compression event for quality tracking."""
        self.compression_count += 1
        self.total_compression_ratio += compression_ratio
        self.total_information_preserved += information_preserved
        self.compression_triggers.append(trigger)


class TokenAwareExecutor:
    """
    Generic token-aware execution engine implementing LlamaStack patterns.

    Provides multi-level safety valves and progressive degradation for any MCP tools.
    """

    def __init__(
        self,
        ai_provider: BaseProvider,
        mcp_manager: Optional[OfficialMCPClientManager] = None,
        local_tool_registry: Optional[ToolRegistry] = None,
        limits: Optional[ExecutionLimits] = None,
        session_id: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        context_safety_factor: float = 0.7,  # NEW: Configurable via CONTEXT_SAFETY_FACTOR env var
    ):
        self.ai_provider = ai_provider
        self.mcp_manager = mcp_manager
        self.local_tool_registry = local_tool_registry
        self.limits = limits or ExecutionLimits()
        self.session_id = session_id or str(uuid.uuid4())
        self.tokenizer = (
            tokenizer  # Phase 4: Optional tokenizer for accurate token counting
        )
        self.context_safety_factor = (
            context_safety_factor  # NEW: Store as instance variable
        )

        # Initialize state
        self.state = ExecutionState()

        # Initialize knowledge store for MCP tool results (data preservation)
        self.knowledge_store = AccumulatedKnowledge()

        # Phase 1 KV-Cache Optimization: Initialize EventLog for append-only context tracking
        # Generate report_id following same pattern as HTML/JSON reports
        # Include milliseconds for finer granularity in high-concurrency scenarios
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
            :-3
        ]  # Truncate to milliseconds
        job_name = os.getenv("CI_JOB_NAME", "")

        # Sanitize and truncate job_name to prevent excessively long filenames
        sanitized_job_name = "".join(
            c for c in job_name if c.isalnum() or c in ("-", "_")
        )
        if len(sanitized_job_name) > 50:  # Keep reasonable length for readability
            sanitized_job_name = sanitized_job_name[:50]

        # Generate report_id: session_{job_name}_{timestamp} or session_{timestamp}
        if sanitized_job_name:
            report_id = f"session_{sanitized_job_name}_{timestamp}"
        else:
            report_id = f"session_{timestamp}"

        self.event_log: Optional[EventLog] = None
        try:
            self.event_log = EventLog(report_id)
            logger.info(f"EventLog enabled for session {report_id}")
        except Exception as e:
            logger.warning(f"EventLog initialization failed, continuing without: {e}")

        # Initialize Phase 2 context compactor with AI-powered summarization
        # Phase 4: Pass tokenizer for improved accuracy
        provider_name = getattr(ai_provider, "provider_name", "gemini")
        model_name = getattr(ai_provider, "model_id", None)
        self.compactor = ContextCompactor(
            provider=provider_name,
            model=model_name,
            ai_provider=ai_provider,
            config=ContextCompactorConfig(),
            tokenizer=tokenizer,
        )

        # Phase 4: Calculate and store effective token budget (Conservative Context Estimation)
        # Store both original limit and effective budget for transparency
        self._original_token_limit = self.limits.max_tokens_total
        self._effective_token_budget = self._calculate_effective_token_budget()

        # Phase 5: Early Break Recovery mechanism
        self.recovery_manager: Optional[RecoveryManager] = None
        self.recovery_steps: List = []  # Collect InferenceSteps from recovery AI calls
        if self.limits.recovery_enabled:
            self.recovery_manager = RecoveryManager(
                ai_provider=ai_provider,
                event_log=self.event_log,
                max_recovery_attempts=self.limits.max_recovery_attempts,
            )
            logger.info("Early Break Recovery mechanism enabled")

        logger.info(
            f"TokenAwareExecutor initialized with limits: "
            f"max_infer_iters={self.limits.max_infer_iters}, "
            f"max_tokens_total={self._effective_token_budget} (original: {self._original_token_limit}, "
            f"effective budget with {int(self.context_safety_factor * 100)}% safety factor and {SYSTEM_PROMPT_TOKEN_OVERHEAD + TOOLS_TOKEN_OVERHEAD} overhead), "
            f"max_tools_per_iteration={self.limits.max_tools_per_iteration}, "
            f"AI-powered compression enabled, "
            f"knowledge store enabled for data preservation, "
            f"tokenizer={'available' if tokenizer else 'not available'}"
        )

    def check_limits(self) -> Optional[StopReason]:
        """
        Check all execution limits and return stop reason if any limit exceeded.

        Follows LlamaStack pattern: check iteration limit first, then token limit.

        Phase 4: Uses effective token budget with conservative context estimation.
        """
        # Check iteration limit (LlamaStack max_infer_iters pattern)
        if self.state.current_iteration >= self.limits.max_infer_iters:
            logger.info(
                f"Maximum iterations reached: {self.state.current_iteration}/{self.limits.max_infer_iters}"
            )
            return StopReason.max_iterations

        # Check token limits (LlamaStack out_of_tokens pattern)
        # Phase 4: Use effective budget instead of raw limit
        if self.state.total_tokens_used >= self._effective_token_budget:
            logger.info(
                f"Total token budget exhausted: {self.state.total_tokens_used}/{self._effective_token_budget} "
                f"(effective budget, original limit: {self._original_token_limit})"
            )
            return StopReason.out_of_tokens

        if self.state.current_iteration_tokens >= self.limits.max_tokens_per_iteration:
            logger.info(
                f"Iteration token budget exhausted: "
                f"{self.state.current_iteration_tokens}/{self.limits.max_tokens_per_iteration}"
            )
            return StopReason.out_of_tokens

        # Check tool limits
        if self.state.current_iteration_tools >= self.limits.max_tools_per_iteration:
            logger.info(
                f"Tool limit per iteration reached: "
                f"{self.state.current_iteration_tools}/{self.limits.max_tools_per_iteration}"
            )
            return StopReason.max_tools

        if self.state.total_tools_executed >= self.limits.max_total_tools:
            logger.info(
                f"Total tool limit reached: "
                f"{self.state.total_tools_executed}/{self.limits.max_total_tools}"
            )
            return StopReason.max_tools

        # Check result size limits
        if (
            self.state.total_result_size_bytes
            >= self.limits.max_total_result_size_bytes
        ):
            logger.info(
                f"Total result size limit reached: "
                f"{self.state.total_result_size_bytes}/{self.limits.max_total_result_size_bytes} bytes"
            )
            return StopReason.max_result_size

        # Check time limits
        elapsed_time = self.state.get_elapsed_time()
        if elapsed_time >= self.limits.max_execution_time:
            logger.info(
                f"Execution time limit reached: {elapsed_time:.1f}/{self.limits.max_execution_time}s"
            )
            return StopReason.timeout

        return None

    def check_warnings(self) -> List[str]:
        """
        Check for approaching limits and issue warnings.

        Phase 4: Uses effective token budget for accurate warnings.
        """
        warnings = []

        # Check iteration warning
        iter_ratio = self.state.current_iteration / self.limits.max_infer_iters
        if iter_ratio >= self.limits.warning_threshold:
            warning = (
                f"Approaching iteration limit: "
                f"{self.state.current_iteration}/{self.limits.max_infer_iters} ({iter_ratio:.1%})"
            )
            if warning not in self.state.warnings_issued:
                warnings.append(warning)
                self.state.warnings_issued.append(warning)

        # Check token warning (Phase 4: use effective budget)
        token_ratio = self.state.total_tokens_used / self._effective_token_budget
        if token_ratio >= self.limits.warning_threshold:
            warning = f"Approaching token limit: {self.state.total_tokens_used}/{self._effective_token_budget} ({token_ratio:.1%})"
            if warning not in self.state.warnings_issued:
                warnings.append(warning)
                self.state.warnings_issued.append(warning)

        # Check tool warning
        tool_ratio = self.state.total_tools_executed / self.limits.max_total_tools
        if tool_ratio >= self.limits.warning_threshold:
            warning = (
                f"Approaching tool limit: "
                f"{self.state.total_tools_executed}/{self.limits.max_total_tools} ({tool_ratio:.1%})"
            )
            if warning not in self.state.warnings_issued:
                warnings.append(warning)
                self.state.warnings_issued.append(warning)

        # Check if we should activate degradation mode
        # When recovery is enabled, skip degradation for iteration/token limits
        # (recovery will handle those - let execution continue until actual limit)
        # Tool limit still triggers degradation as it's not recoverable
        should_degrade_iter_token = (
            iter_ratio >= self.limits.degradation_threshold
            or token_ratio >= self.limits.degradation_threshold
        ) and not self.limits.recovery_enabled

        should_degrade_tools = tool_ratio >= self.limits.degradation_threshold

        # Log when degradation would have activated but recovery is enabled
        if (
            (
                iter_ratio >= self.limits.degradation_threshold
                or token_ratio >= self.limits.degradation_threshold
            )
            and self.limits.recovery_enabled
            and not self.state.degradation_active
            and not self.state._degradation_skip_logged
        ):
            logger.info(
                "Skipping degradation for iteration/token limits - recovery enabled"
            )
            self.state._degradation_skip_logged = True

        if should_degrade_iter_token or should_degrade_tools:
            if not self.state.degradation_active:
                self.state.degradation_active = True
                reason = (
                    "tools" if should_degrade_tools else "limits (recovery disabled)"
                )
                warnings.append("Activating progressive degradation mode")
                logger.warning(
                    f"Progressive degradation mode activated due to approaching {reason}"
                )

        return warnings

    def estimate_tokens(self, text: str, use_tokenizer: bool = False) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to estimate tokens for
            use_tokenizer: If True and tokenizer available, use it for accurate counting

        Returns:
            Estimated token count

        Phase 4: Supports optional tokenizer for improved accuracy (from LangChain).
        Falls back to conservative 4 characters per token estimate.
        """
        if use_tokenizer and hasattr(self, "tokenizer") and self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text)
                return len(tokens)
            except Exception as e:
                logger.warning(f"Tokenizer failed, falling back to approximation: {e}")

        # Fallback to approximation (1 token per 4 characters)
        return max(1, len(text) // 4)

    def _calculate_effective_token_budget(self) -> int:
        """
        Calculate effective token budget after accounting for overhead.

        Phase 4: Conservative Context Estimation (from Goose).
        Accounts for system prompt and tools overhead that aren't included
        in message token counts.

        Returns:
            Effective usable token budget
        """
        # Use original limit stored during initialization
        # (or current value if not yet initialized)
        model_limit = getattr(
            self, "_original_token_limit", self.limits.max_tokens_total
        )

        # Apply safety factor (configurable via CONTEXT_SAFETY_FACTOR env var)
        safe_limit = int(model_limit * self.context_safety_factor)
        logger.debug(
            f"Applying context_safety_factor={self.context_safety_factor} to model limit {model_limit}"
        )

        # Subtract overhead for system prompt and tools
        overhead = SYSTEM_PROMPT_TOKEN_OVERHEAD + TOOLS_TOKEN_OVERHEAD
        effective_budget = safe_limit - overhead

        # Ensure minimum usable budget
        return max(effective_budget, 1000)

    def _identify_tool_pairs(
        self, messages: List[ProviderMessage]
    ) -> Dict[str, tuple[Optional[int], Optional[int]]]:
        """
        Map tool IDs to (request_index, response_index) pairs.

        Phase 4: Tool Pair Preservation (from Goose).
        Ensures tool requests and responses are never split during compaction.

        Args:
            messages: List of conversation messages

        Returns:
            Dictionary mapping tool_id -> (request_idx, response_idx)
            response_idx may be None if response hasn't been received yet
        """
        tool_pairs: Dict[str, tuple[Optional[int], Optional[int]]] = {}

        for i, msg in enumerate(messages):
            # Identify tool requests (assistant messages with tool_calls)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_id = tool_call.get("id")
                    if tool_id:
                        # Record request index, response not yet seen
                        tool_pairs[tool_id] = (i, None)

            # Identify tool responses (messages with tool_call_id)
            if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                tool_id = msg.tool_call_id
                if tool_id in tool_pairs:
                    request_idx, _ = tool_pairs[tool_id]
                    # Update with response index
                    tool_pairs[tool_id] = (request_idx, i)
                else:
                    # Response without known request (orphaned)
                    tool_pairs[tool_id] = (None, i)

        return tool_pairs

    def _validate_conversation_start(
        self, messages: List[ProviderMessage]
    ) -> List[ProviderMessage]:
        """
        Ensure conversation starts with System or User message.

        Phase 4: Message Validation (from LangChain).
        Prevents AI model errors from malformed conversation history.

        Args:
            messages: Conversation messages to validate

        Returns:
            Validated conversation messages (may have system message prepended)
        """
        if not messages:
            return messages

        first_msg = messages[0]

        # Valid starts: system, user
        # Invalid starts: assistant, tool
        if first_msg.role not in ["system", "user"]:
            logger.warning(
                f"Invalid conversation start (role={first_msg.role}), adding system message"
            )
            # Add system message at start
            system_msg = ProviderMessage(
                role="system", content="Previous conversation context summary."
            )
            return [system_msg] + messages

        return messages

    def should_truncate_results(self) -> bool:
        """
        Determine if tool results should be truncated to save tokens.

        Phase 4: Uses effective token budget for accurate thresholds.
        """
        return (
            self.state.degradation_active
            or self.state.total_tokens_used / self._effective_token_budget
            > self.limits.degradation_threshold
        )

    def _calculate_conversation_tokens(self, messages: List[ProviderMessage]) -> int:
        """
        Calculate total token count for conversation messages.

        Args:
            messages: List of conversation messages

        Returns:
            Estimated total token count

        Phase 4: Uses tokenizer when available for improved accuracy.
        """
        total_tokens = 0
        for msg in messages:
            total_tokens += self.estimate_tokens(msg.content, use_tokenizer=True)
        return total_tokens

    def _should_compact_conversation(
        self,
        conversation_messages: List[ProviderMessage],
        last_tool_result_tokens: int = 0,
    ) -> bool:
        """
        Multi-factor trigger logic to determine if conversation should be compacted.

        Triggers compaction if ANY of:
        1. Iteration > 3 (original behavior for safety)
        2. Conversation tokens > 50% of max_tokens_per_iteration (handles large tool results early)
        3. Total tokens used > 60% AND projected usage > 80% (proactive management)
        4. Single tool result > 30% of max_tokens_per_iteration (immediate large result)

        Args:
            conversation_messages: Current conversation messages
            last_tool_result_tokens: Token count of most recent tool result (if any)

        Returns:
            True if compaction should be triggered
        """
        # Trigger 1: Iteration > 3 (original safety behavior)
        if self.state.current_iteration > 3:
            logger.debug(
                f"Compaction trigger 1: iteration {self.state.current_iteration} > 3"
            )
            return True

        # Calculate conversation size
        conversation_tokens = self._calculate_conversation_tokens(conversation_messages)

        # Trigger 2: Conversation exceeds 50% of iteration budget
        iteration_threshold = self.limits.max_tokens_per_iteration * 0.5
        if conversation_tokens > iteration_threshold:
            logger.debug(
                f"Compaction trigger 2: conversation size {conversation_tokens} tokens "
                f"> 50% iteration budget ({iteration_threshold})"
            )
            return True

        # Trigger 3: Proactive management - high utilization with concerning trajectory
        token_utilization = self.state.total_tokens_used / self.limits.max_tokens_total
        if token_utilization > 0.6:
            # Estimate next iteration cost based on current iteration
            avg_tokens_per_iteration = self.state.total_tokens_used / max(
                1, self.state.current_iteration
            )
            projected_usage = (
                self.state.total_tokens_used + avg_tokens_per_iteration
            ) / self.limits.max_tokens_total

            if projected_usage > 0.8:
                logger.debug(
                    f"Compaction trigger 3: token utilization {token_utilization:.1%} > 60% "
                    f"AND projected {projected_usage:.1%} > 80%"
                )
                return True

        # Trigger 4: Single large tool result (immediate action needed)
        if last_tool_result_tokens > 0:
            large_result_threshold = self.limits.max_tokens_per_iteration * 0.3
            if last_tool_result_tokens > large_result_threshold:
                logger.debug(
                    f"Compaction trigger 4: tool result {last_tool_result_tokens} tokens "
                    f"> 30% iteration budget ({large_result_threshold})"
                )
                return True

        return False

    def _get_compaction_trigger_reason(
        self,
        conversation_messages: List[ProviderMessage],
        last_tool_result_tokens: int = 0,
    ) -> str:
        """
        Determine which trigger caused compaction (for metrics and logging).

        Returns a descriptive string identifying the trigger.
        """
        # Check triggers in same order as _should_compact_conversation
        if self.state.current_iteration > 3:
            return f"iteration_{self.state.current_iteration}"

        conversation_tokens = self._calculate_conversation_tokens(conversation_messages)
        iteration_threshold = self.limits.max_tokens_per_iteration * 0.5

        if conversation_tokens > iteration_threshold:
            return f"conversation_size_{conversation_tokens}tokens"

        token_utilization = self.state.total_tokens_used / self.limits.max_tokens_total
        if token_utilization > 0.6:
            avg_tokens_per_iteration = self.state.total_tokens_used / max(
                1, self.state.current_iteration
            )
            projected_usage = (
                self.state.total_tokens_used + avg_tokens_per_iteration
            ) / self.limits.max_tokens_total

            if projected_usage > 0.8:
                return f"projected_usage_{projected_usage:.1%}"

        if last_tool_result_tokens > 0:
            large_result_threshold = self.limits.max_tokens_per_iteration * 0.3
            if last_tool_result_tokens > large_result_threshold:
                return f"large_tool_result_{last_tool_result_tokens}tokens"

        return "unknown_trigger"

    async def _check_conversation_size_and_compact(
        self,
        conversation_messages: List[ProviderMessage],
        last_tool_result: Optional[str] = None,
    ) -> List[ProviderMessage]:
        """
        Check conversation size after tool execution and compact if needed.

        This enables early compaction (iteration 1+) when MCP tools return large results,
        instead of waiting until iteration 3.

        Phase 4 Enhancement: Preserves tool pair integrity during compaction.

        Args:
            conversation_messages: Current conversation messages
            last_tool_result: Most recent tool result message (if any)

        Returns:
            Possibly compacted conversation messages
        """
        # Calculate size of last tool result if provided
        last_tool_result_tokens = (
            self.estimate_tokens(last_tool_result) if last_tool_result else 0
        )

        # Check if compaction is needed using multi-factor logic
        if self._should_compact_conversation(
            conversation_messages, last_tool_result_tokens
        ):
            # Determine which trigger caused compaction for metrics
            trigger_reason = self._get_compaction_trigger_reason(
                conversation_messages, last_tool_result_tokens
            )

            # Phase 4: Identify tool pairs before compaction to preserve them
            tool_pairs = self._identify_tool_pairs(conversation_messages)

            logger.info(
                f"Triggering compaction at iteration {self.state.current_iteration} (trigger: {trigger_reason})"
            )
            (
                conversation_messages,
                compression_result,
            ) = await self.compactor.compact_iteration_context(
                conversation_messages,
                self.limits.max_tokens_per_iteration,
                self.state.current_iteration,
                tool_pairs,
            )

            # Record compression metrics
            self.state.record_compression(
                compression_result.compression_ratio,
                compression_result.information_preserved,
                trigger_reason,
            )

            logger.info(
                f"Conversation compacted: {compression_result.compression_ratio:.2f}x compression, "
                f"{compression_result.information_preserved:.1%} information preserved "
                f"(trigger: {trigger_reason})"
            )

            # Phase 4: Validate conversation after compaction
            conversation_messages = self._validate_conversation_start(
                conversation_messages
            )

        return conversation_messages

    async def execute_with_limits(
        self,
        messages: List[ProviderMessage],
        available_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Execute AI agent with token-aware limits following LlamaStack patterns.

        This is the main execution method that implements the dual-limit approach:
        1. Check max_infer_iters (iteration limit)
        2. Check out_of_tokens (token budget)
        3. Progressive degradation as limits approach
        """
        logger.info(f"Starting token-aware execution for session {self.session_id}")
        conversation_messages = messages.copy()
        tool_results: List[Dict[str, Any]] = []

        try:
            while True:
                # Start new iteration
                self.state.current_iteration += 1
                self.state.reset_iteration()

                logger.info(
                    f"Starting iteration {self.state.current_iteration}/{self.limits.max_infer_iters}"
                )

                # Check limits before proceeding (LlamaStack pattern)
                stop_reason = self.check_limits()
                if stop_reason:
                    # === RECOVERY HOOK: Max Tokens Exceeded ===
                    if (
                        stop_reason == StopReason.out_of_tokens
                        and self.recovery_manager
                    ):
                        # Check if we've exhausted recovery attempts (max 3)
                        if (
                            self.state.max_tokens_recovery_attempts
                            >= self.state.max_tokens_recovery_limit
                        ):
                            logger.warning(
                                f"Max token recovery limit reached: "
                                f"{self.state.max_tokens_recovery_attempts}/"
                                f"{self.state.max_tokens_recovery_limit}. "
                                f"Triggering final synthesis."
                            )
                            # Trigger final synthesis instead of just stopping
                            synthesis_response = await self._perform_final_synthesis(
                                messages[0].content if messages else "",
                                "token budget",
                                conversation_messages,
                            )
                            if synthesis_response:
                                # Add synthesis as final assistant message
                                conversation_messages.append(
                                    ProviderMessage(
                                        content=synthesis_response, role="assistant"
                                    )
                                )
                            self.state.stop_reason = stop_reason
                            break

                        logger.info(
                            f"Token budget exhausted at iteration "
                            f"{self.state.current_iteration}. "
                            f"Tokens used: {self.state.total_tokens_used}/"
                            f"{self._effective_token_budget}. "
                            f"Attempting fresh context recovery "
                            f"({self.state.max_tokens_recovery_attempts + 1}/"
                            f"{self.state.max_tokens_recovery_limit})..."
                        )

                        error = ClassifiedError(
                            error_type=ErrorType.MAX_TOKENS_EXCEEDED,
                            message=(
                                f"Token budget exhausted: "
                                f"{self.state.total_tokens_used}/"
                                f"{self._effective_token_budget} tokens used"
                            ),
                            iteration=self.state.current_iteration,
                        )

                        recovery_result = await self._attempt_recovery_from_token_limit(
                            error,
                            messages[0].content if messages else "",
                            available_tools,
                        )

                        if recovery_result and recovery_result.should_continue:
                            # Start new inference with fresh context
                            self.knowledge_store.start_new_inference()
                            self.state.max_tokens_recovery_attempts += 1

                            # Log captured tool calls from recovery AI (for debugging)
                            if recovery_result.recovery_tool_calls:
                                logger.info(
                                    f"Recovery AI made {len(recovery_result.recovery_tool_calls)} "
                                    f"tool calls: {[tc['name'] for tc in recovery_result.recovery_tool_calls]}"
                                )

                            logger.info(
                                f"Recovery successful for MAX_TOKENS_EXCEEDED. "
                                f"Starting inference "
                                f"{self.knowledge_store.current_inference_id} "
                                f"with fresh context."
                            )

                            # Replace conversation with fresh context
                            # CRITICAL: recovery_message now contains recovery_prompt with
                            # original task + knowledge store summary, not empty response.content
                            conversation_messages = [
                                ProviderMessage(
                                    content=recovery_result.recovery_message or "",
                                    role="user",
                                )
                            ]

                            # Reset token budget AND iteration counter for fresh context
                            # This is critical for:
                            # 1. Allowing full token budget for new inference
                            # 2. Preventing compaction from triggering immediately (iter > 3)
                            # 3. Preventing false AI_PREMATURE_COMPLETION detection
                            prev_iteration = self.state.current_iteration
                            self.state.reset_for_fresh_context()

                            logger.info(
                                f"Fresh context reset: tokens 0/{self._effective_token_budget}, "
                                f"iterations reset (was {prev_iteration}, cumulative: "
                                f"{self.state.cumulative_iterations_all_inferences})"
                            )

                            # Continue execution loop with fresh context
                            continue
                        else:
                            logger.warning(
                                f"MAX_TOKENS_EXCEEDED recovery failed: "
                                f"{recovery_result.reason if recovery_result else 'no result'}"
                            )

                    # === RECOVERY HOOK: Max Iterations Exceeded ===
                    elif (
                        stop_reason == StopReason.max_iterations
                        and self.recovery_manager
                    ):
                        # Check if we've exhausted recovery attempts (max 2)
                        if (
                            self.state.max_iterations_recovery_attempts
                            >= self.state.max_iterations_recovery_limit
                        ):
                            logger.warning(
                                f"Max iterations recovery limit reached: "
                                f"{self.state.max_iterations_recovery_attempts}/"
                                f"{self.state.max_iterations_recovery_limit}. "
                                f"Triggering final synthesis."
                            )
                            # Trigger final synthesis instead of just stopping
                            synthesis_response = await self._perform_final_synthesis(
                                messages[0].content if messages else "",
                                "iteration limit",
                                conversation_messages,
                            )
                            if synthesis_response:
                                # Add synthesis as final assistant message
                                conversation_messages.append(
                                    ProviderMessage(
                                        content=synthesis_response, role="assistant"
                                    )
                                )
                            self.state.stop_reason = stop_reason
                            break

                        # Check remaining token budget - only recover if tokens available
                        remaining_tokens = (
                            self._effective_token_budget - self.state.total_tokens_used
                        )
                        if (
                            remaining_tokens < 10000
                        ):  # Need at least 10k tokens for recovery
                            logger.warning(
                                f"Insufficient token budget for iteration recovery: "
                                f"{remaining_tokens} tokens remaining. "
                                f"Triggering final synthesis."
                            )
                            # Trigger final synthesis instead of just stopping
                            synthesis_response = await self._perform_final_synthesis(
                                messages[0].content if messages else "",
                                "iteration limit (insufficient tokens)",
                                conversation_messages,
                            )
                            if synthesis_response:
                                # Add synthesis as final assistant message
                                conversation_messages.append(
                                    ProviderMessage(
                                        content=synthesis_response, role="assistant"
                                    )
                                )
                            self.state.stop_reason = stop_reason
                            break

                        logger.info(
                            f"Iteration limit reached at iteration "
                            f"{self.state.current_iteration}. "
                            f"Token budget remaining: {remaining_tokens}. "
                            f"Attempting fresh context recovery "
                            f"({self.state.max_iterations_recovery_attempts + 1}/"
                            f"{self.state.max_iterations_recovery_limit})..."
                        )

                        error = ClassifiedError(
                            error_type=ErrorType.MAX_ITERATIONS_EXCEEDED,
                            message=(
                                f"Iteration limit reached: "
                                f"{self.state.current_iteration}/"
                                f"{self.limits.max_infer_iters} iterations"
                            ),
                            iteration=self.state.current_iteration,
                        )

                        recovery_result = (
                            await self._attempt_recovery_from_iteration_limit(
                                error,
                                messages[0].content if messages else "",
                                available_tools,
                            )
                        )

                        if recovery_result and recovery_result.should_continue:
                            # Start new inference with fresh context
                            self.knowledge_store.start_new_inference()
                            self.state.max_iterations_recovery_attempts += 1

                            # Log captured tool calls from recovery AI (for debugging)
                            if recovery_result.recovery_tool_calls:
                                logger.info(
                                    f"Recovery AI made {len(recovery_result.recovery_tool_calls)} "
                                    f"tool calls: {[tc['name'] for tc in recovery_result.recovery_tool_calls]}"
                                )

                            logger.info(
                                f"Recovery successful for MAX_ITERATIONS_EXCEEDED. "
                                f"Starting inference "
                                f"{self.knowledge_store.current_inference_id} "
                                f"with fresh context."
                            )

                            # Replace conversation with fresh context
                            # CRITICAL: recovery_message now contains recovery_prompt with
                            # original task + knowledge store summary, not empty response.content
                            conversation_messages = [
                                ProviderMessage(
                                    content=recovery_result.recovery_message or "",
                                    role="user",
                                )
                            ]

                            # Reset token budget AND iteration counter for fresh context
                            # This is critical for:
                            # 1. Allowing fresh iteration budget for new inference
                            # 2. Preventing compaction from triggering immediately (iter > 3)
                            # 3. Preventing false AI_PREMATURE_COMPLETION detection
                            prev_iteration = self.state.current_iteration
                            self.state.reset_for_fresh_context()

                            logger.info(
                                f"Fresh context reset: tokens 0/{self._effective_token_budget}, "
                                f"iterations reset (was {prev_iteration}, cumulative: "
                                f"{self.state.cumulative_iterations_all_inferences})"
                            )

                            # Continue execution loop with fresh context
                            continue
                        else:
                            logger.warning(
                                f"MAX_ITERATIONS_EXCEEDED recovery failed: "
                                f"{recovery_result.reason if recovery_result else 'no result'}"
                            )

                    # If no recovery or recovery failed, set stop reason and break
                    self.state.stop_reason = stop_reason
                    break

                # Check for warnings and progressive degradation
                warnings = self.check_warnings()
                for warning in warnings:
                    logger.warning(warning)

                # Phase 3: Check if conversation should be compacted (multi-factor trigger logic)
                # This replaces the hardcoded "iteration > 3" check with intelligent triggers
                conversation_messages = await self._check_conversation_size_and_compact(
                    conversation_messages
                )

                # AI inference step
                ai_response = await self._execute_ai_inference(
                    conversation_messages, available_tools
                )
                if not ai_response:
                    # === RECOVERY HOOK: AI Inference Failure ===
                    if self.recovery_manager:
                        error = ClassifiedError(
                            error_type=ErrorType.AI_INFERENCE_FAILURE,
                            message="AI inference returned empty response",
                            iteration=self.state.current_iteration,
                        )
                        recovery_result = await self._attempt_recovery(
                            error,
                            messages[0].content if messages else "",
                            conversation_messages,
                            available_tools,
                        )
                        if recovery_result and recovery_result.should_continue:
                            # Increment inference ID before starting recovery inference
                            self.knowledge_store.start_new_inference()

                            # Inject recovery message and continue
                            conversation_messages.append(
                                ProviderMessage(
                                    content=recovery_result.recovery_message or "",
                                    role="user",
                                )
                            )
                            logger.info(
                                "Recovery successful for AI_INFERENCE_FAILURE, continuing..."
                            )
                            continue  # Retry the iteration

                    # If AI inference fails and no recovery, mark as error and generate minimal response
                    self.state.stop_reason = StopReason.error
                    break

                # Add AI response to conversation
                conversation_messages.append(
                    ProviderMessage(content=ai_response.content, role="assistant")
                )

                # Parse and execute tool calls if any
                # Convert OpenAI format tool calls to MCP format before execution
                openai_tool_calls = ai_response.tool_calls or []
                mcp_tool_calls = (
                    convert_openai_tool_calls_to_mcp(openai_tool_calls)
                    if openai_tool_calls
                    else []
                )
                tool_calls = await self._parse_and_execute_tools(
                    mcp_tool_calls, available_tools
                )

                if tool_calls:
                    # Clear fresh context recovery mode - AI is exploring, not synthesizing
                    # If AI made tool calls after recovery, it's no longer in synthesis mode
                    if self.state.in_fresh_context_recovery:
                        logger.info(
                            f"AI made {len(tool_calls)} tool calls after fresh context recovery - "
                            f"exiting synthesis mode (now in exploration mode)"
                        )
                        self.state.in_fresh_context_recovery = False

                    # Ensure performance counters are updated even if tool execution was mocked in tests
                    for tr in tool_calls:
                        if not tr.get("metrics_applied"):
                            self.state.add_tool_execution(
                                int(tr.get("result_size_bytes", 0) or 0)
                            )
                    # Add tool results to conversation for next iteration
                    tool_results_message = self._format_tool_results_for_conversation(
                        tool_calls
                    )
                    conversation_messages.append(
                        ProviderMessage(content=tool_results_message, role="user")
                    )
                    tool_results.extend(tool_calls)

                    # Phase 3: Immediate compaction check after adding tool results
                    # This handles large MCP tool results in iteration 1+ instead of waiting until iteration 3
                    conversation_messages = (
                        await self._check_conversation_size_and_compact(
                            conversation_messages, last_tool_result=tool_results_message
                        )
                    )

                    logger.info(
                        f"Executed {len(tool_calls)} tools in iteration {self.state.current_iteration}"
                    )
                    continue
                else:
                    # No tools to execute - check if this is premature completion
                    # === RECOVERY HOOK: AI Premature Completion (MOST COMMON CASE - 25% of jobs) ===
                    # Two detection patterns:
                    # 1. AI says "I will continue..." but makes NO tool calls
                    # 2. AI gives ULTRA-SHORT response (<500 chars) with no tool calls after iteration 2
                    response_text = (
                        ai_response.content if hasattr(ai_response, "content") else ""
                    )
                    response_length = len(response_text)

                    # Pattern 1: Continuation phrase detection (improved logic)
                    # Check LAST 200 chars for continuation indicators - more accurate than full text
                    # Continuation phrases at the END indicate AI is about to do more work
                    last_portion = (
                        response_text[-200:]
                        if len(response_text) > 200
                        else response_text
                    )
                    has_continuation_phrase = (
                        self.recovery_manager
                        and has_continuation_indicator(last_portion)
                    )

                    # Pattern 2: Ultra-short response detection (new logic)
                    # Detects silent failures where AI stops without continuation phrases
                    # Criteria:
                    # - Very short response (< 800 chars = ~120-160 words, ~163-200 tokens)
                    # - After iteration 2 (not 1, to avoid code review false positives)
                    # - No tool calls made
                    # - NOT in fresh context recovery mode (where short synthesis is correct)
                    # This catches cases like the 163-token response in job 43829989
                    #
                    # IMPORTANT: in_fresh_context_recovery mode disables this check because
                    # after MAX_TOKENS/MAX_ITERATIONS recovery, the AI is expected to
                    # synthesize existing tool results, not make more tool calls.
                    # A short but complete synthesis response is the correct behavior.
                    is_ultra_short_response = (
                        self.recovery_manager
                        and response_length < 800
                        and self.state.current_iteration >= 2
                        and not self.state.in_fresh_context_recovery  # Skip in recovery mode
                    )

                    if has_continuation_phrase or is_ultra_short_response:
                        # Determine detection reason for logging
                        if has_continuation_phrase and is_ultra_short_response:
                            detection_reason = (
                                "continuation phrase + ultra-short response"
                            )
                        elif has_continuation_phrase:
                            detection_reason = "continuation phrase"
                        else:
                            detection_reason = "ultra-short response"

                        # Check if this is actually premature completion or a false positive
                        # FALSE POSITIVE PREVENTION:
                        # Don't trigger recovery on iteration 1 - code reviews often complete in first iteration
                        if self.state.current_iteration == 1:
                            logger.info(
                                f"Premature completion indicator detected in iteration 1 ({response_length} chars) "
                                f"- allowing completion (code reviews typically finish in iteration 1)"
                            )
                        else:
                            # Premature completion detected via continuation phrase or ultra-short response
                            logger.info(
                                f"Detected AI_PREMATURE_COMPLETION at iteration {self.state.current_iteration}: "
                                f"{detection_reason}, no tool calls ({response_length} chars)"
                            )
                            error = ClassifiedError(
                                error_type=ErrorType.AI_PREMATURE_COMPLETION,
                                message=f"AI stopped prematurely: {detection_reason}, no tool calls",
                                iteration=self.state.current_iteration,
                                last_response_preview=response_text[:500]
                                if response_text
                                else "",
                            )

                            # Log for context
                            if self.event_log:
                                self.event_log.append_event(
                                    "premature_completion",
                                    {
                                        "iteration": self.state.current_iteration,
                                        "response_preview": response_text[:200]
                                        if response_text
                                        else "",
                                        "response_length": response_length,
                                        "detection_reason": detection_reason,
                                    },
                                )

                            recovery_result = await self._attempt_recovery(
                                error,
                                messages[0].content if messages else "",
                                conversation_messages,
                                available_tools,
                            )

                            if recovery_result and recovery_result.should_continue:
                                # Increment inference ID before starting recovery inference
                                self.knowledge_store.start_new_inference()

                                # Inject recovery message to remind AI to make actual tool calls
                                conversation_messages.append(
                                    ProviderMessage(
                                        content=recovery_result.recovery_message or "",
                                        role="user",
                                    )
                                )
                                logger.info(
                                    f"Recovery successful for AI_PREMATURE_COMPLETION ({detection_reason}), continuing..."
                                )
                                continue  # Retry - force AI to make tool calls

                    # No recovery needed or recovery failed - AI provided final response
                    logger.info(
                        f"No tool calls found, completing execution at iteration {self.state.current_iteration}"
                    )
                    self.state.stop_reason = StopReason.end_of_turn
                    break

            # Generate final summary
            final_response = await self._generate_final_summary(
                conversation_messages, tool_results
            )

        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            self.state.stop_reason = StopReason.error
            final_response = f"Execution failed: {e}"

        # Reset recovery state for next session
        if self.recovery_manager:
            self.recovery_manager.reset_recovery_state()

        return {
            "final_response": final_response,
            "tool_results": tool_results,
            "execution_summary": self.get_execution_summary(),
            "conversation_messages": conversation_messages,
        }

    async def _attempt_recovery(
        self,
        error: ClassifiedError,
        original_prompt: str,
        conversation_messages: List[ProviderMessage],
        available_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[RecoveryResult]:
        """
        Attempt to recover from an execution error using secondary AI call.

        Wraps RecoveryManager.attempt_recovery with state tracking.

        Args:
            error: Classified error with context
            original_prompt: Original task prompt
            conversation_messages: Current conversation history
            available_tools: Available MCP tools (for recovery AI context)

        Returns:
            RecoveryResult if recovery succeeded, None otherwise
        """
        if not self.recovery_manager:
            return None

        try:
            self.state.recovery_attempts += 1
            self.state.last_recovery_error_type = error.error_type.value

            recovery_result = await self.recovery_manager.attempt_recovery(
                error,
                original_prompt,
                conversation_messages,
                available_tools,
                self.recovery_steps,
            )

            if recovery_result and recovery_result.should_continue:
                self.state.successful_recoveries += 1
                logger.info(
                    f"Recovery attempt {self.state.recovery_attempts} succeeded "
                    f"for {error.error_type.value}"
                )
            else:
                logger.warning(
                    f"Recovery attempt {self.state.recovery_attempts} failed "
                    f"for {error.error_type.value}: {recovery_result.reason if recovery_result else 'no result'}"
                )

            return recovery_result

        except Exception as e:
            logger.error(f"Recovery attempt failed with exception: {e}")
            return RecoveryResult(
                should_continue=False,
                reason=f"recovery_exception: {e}",
            )

    async def _attempt_recovery_from_token_limit(
        self,
        error: ClassifiedError,
        original_prompt: str,
        available_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[RecoveryResult]:
        """
        Attempt recovery from MAX_TOKENS_EXCEEDED using fresh context.

        This is a specialized recovery that:
        1. Passes knowledge_store to recovery manager for tool results
        2. Uses fresh context mode (no conversation history)
        3. Allows AI to continue with tool calls

        Args:
            error: Classified error with MAX_TOKENS_EXCEEDED type
            original_prompt: Original task prompt
            available_tools: Available MCP tools

        Returns:
            RecoveryResult if recovery succeeded, None otherwise
        """
        if not self.recovery_manager:
            return None

        try:
            self.state.recovery_attempts += 1
            self.state.last_recovery_error_type = error.error_type.value

            # Pass knowledge_store for tool call history
            recovery_result = await self.recovery_manager.attempt_recovery(
                error,
                original_prompt,
                [],  # Empty conversation - fresh context mode
                available_tools,
                self.recovery_steps,
                knowledge_store=self.knowledge_store,  # Pass knowledge store
            )

            if recovery_result and recovery_result.should_continue:
                self.state.successful_recoveries += 1
                logger.info(
                    f"MAX_TOKENS_EXCEEDED recovery attempt "
                    f"{self.state.max_tokens_recovery_attempts + 1}/"
                    f"{self.state.max_tokens_recovery_limit} succeeded. "
                    f"Tool results preserved: {self.knowledge_store.total_tools_executed}"
                )
            else:
                logger.warning(
                    f"MAX_TOKENS_EXCEEDED recovery attempt "
                    f"{self.state.max_tokens_recovery_attempts + 1}/"
                    f"{self.state.max_tokens_recovery_limit} failed: "
                    f"{recovery_result.reason if recovery_result else 'no result'}"
                )

            return recovery_result

        except Exception as e:
            logger.error(f"MAX_TOKENS_EXCEEDED recovery failed with exception: {e}")
            return RecoveryResult(
                should_continue=False,
                reason=f"recovery_exception: {e}",
            )

    async def _attempt_recovery_from_iteration_limit(
        self,
        error: ClassifiedError,
        original_prompt: str,
        available_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[RecoveryResult]:
        """
        Attempt recovery from MAX_ITERATIONS_EXCEEDED using fresh context.

        This is a specialized recovery that:
        1. Uses knowledge_store to preserve tool results across inferences
        2. Provides summary of completed work to avoid duplicating tool calls
        3. Resets iteration counter while preserving token budget tracking
        4. Is more conservative than token recovery (2 attempts vs 3)

        Args:
            error: Classified error with MAX_ITERATIONS_EXCEEDED type
            original_prompt: Original task prompt
            available_tools: Available MCP tools

        Returns:
            RecoveryResult if recovery succeeded, None otherwise
        """
        if not self.recovery_manager:
            return None

        try:
            self.state.recovery_attempts += 1
            self.state.last_recovery_error_type = error.error_type.value

            # The recovery manager will handle MAX_ITERATIONS_EXCEEDED using
            # _recover_from_iteration_limit which builds the prompt from
            # RECOVERY_PROMPTS[ErrorType.MAX_ITERATIONS_EXCEEDED]
            recovery_result = await self.recovery_manager.attempt_recovery(
                error,
                original_prompt,
                [],  # Empty conversation - fresh context mode
                available_tools,
                self.recovery_steps,
                knowledge_store=self.knowledge_store,  # Pass knowledge store
            )

            if recovery_result and recovery_result.should_continue:
                self.state.successful_recoveries += 1
                logger.info(
                    f"MAX_ITERATIONS_EXCEEDED recovery attempt "
                    f"{self.state.max_iterations_recovery_attempts + 1}/"
                    f"{self.state.max_iterations_recovery_limit} succeeded. "
                    f"Tool results preserved: {self.knowledge_store.total_tools_executed}"
                )
            else:
                logger.warning(
                    f"MAX_ITERATIONS_EXCEEDED recovery attempt "
                    f"{self.state.max_iterations_recovery_attempts + 1}/"
                    f"{self.state.max_iterations_recovery_limit} failed: "
                    f"{recovery_result.reason if recovery_result else 'no result'}"
                )

            return recovery_result

        except Exception as e:
            logger.error(f"MAX_ITERATIONS_EXCEEDED recovery failed with exception: {e}")
            return RecoveryResult(
                should_continue=False,
                reason=f"recovery_exception: {e}",
            )

    async def _perform_final_synthesis(
        self,
        original_prompt: str,
        limit_type: str,
        conversation_messages: Optional[List[ProviderMessage]] = None,
    ) -> Optional[str]:
        """
        Perform final synthesis when recovery limits are exhausted.

        This triggers a dedicated synthesis inference with tools=None to prevent
        further exploration. The AI must produce its final deliverable using
        only the accumulated tool results.

        Args:
            original_prompt: Original task prompt
            limit_type: Type of limit reached ("token budget" or "iteration limit")
            conversation_messages: Conversation history for context (truncated)

        Returns:
            Synthesized response text or None if failed
        """
        if not self.recovery_manager:
            logger.warning("Cannot perform synthesis: recovery_manager not available")
            return None

        logger.info(
            f"Triggering final synthesis after exhausting {limit_type}. "
            f"Accumulated tool results: {self.knowledge_store.total_tools_executed}"
        )

        return await self.recovery_manager.perform_final_synthesis(
            original_prompt=original_prompt,
            knowledge_store=self.knowledge_store,
            limit_type=limit_type,
            recovery_steps_collector=self.recovery_steps,
            conversation_messages=conversation_messages,
        )

    async def _execute_ai_inference(
        self,
        messages: List[ProviderMessage],
        available_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[ProviderResponse]:
        """Execute AI inference with token tracking and timeout safety."""
        try:
            logger.debug(f"Requesting AI inference with {len(messages)} messages")

            # Check time budget before expensive AI inference
            elapsed_total = self.state.get_elapsed_time()
            if elapsed_total >= self.limits.max_execution_time:
                logger.warning(
                    f"Total execution time limit exceeded before AI inference: {elapsed_total:.1f}s"
                )
                return None

            # Estimate input tokens
            input_text = " ".join([msg.content for msg in messages])
            input_tokens = self.estimate_tokens(input_text)
            logger.debug(f"Estimated input tokens: {input_tokens}")

            # Phase 2: Compress last message if token utilization > 70% and message is large
            token_utilization = (
                self.state.total_tokens_used / self.limits.max_tokens_total
            )
            if token_utilization > 0.7 and len(messages) > 0:
                last_message = messages[-1]
                if self.estimate_tokens(last_message.content) > 1000:
                    logger.debug(
                        "High token utilization detected, compressing prompt before send"
                    )
                    compressed_prompt, compression_result = (
                        self.compactor.compress_prompt_before_send(
                            last_message.content,
                            token_budget=self.limits.max_tokens_per_iteration,
                        )
                    )
                    # Update last message with compressed content
                    messages[-1] = ProviderMessage(
                        content=compressed_prompt, role=last_message.role
                    )
                    logger.info(
                        f"Prompt compressed: {compression_result.compression_ratio:.2f}x compression, "
                        f"{compression_result.information_preserved:.1%} information preserved"
                    )
                    # Recalculate input tokens after compression
                    input_text = " ".join([msg.content for msg in messages])
                    input_tokens = self.estimate_tokens(input_text)

            # Check if we have enough token budget for this inference
            if (
                self.state.total_tokens_used + input_tokens
            ) > self.limits.max_tokens_total:
                logger.warning("Insufficient token budget for AI inference")
                return None

            # Execute inference
            # Note: AI provider should handle its own timeouts (e.g., httpx client timeout)
            # Using asyncio.wait_for() can cause issues with anyio-based async libraries
            try:
                response = await self.ai_provider.chat_completion(
                    messages, tools=available_tools
                )
            except TimeoutError as e:
                logger.error(f"AI inference timed out: {e}")
                return None
            except Exception as e:
                # Check for timeout-related errors from HTTP clients
                if "timeout" in str(e).lower() or "Timeout" in type(e).__name__:
                    logger.error(
                        f"AI inference timed out (detected from exception): {e}"
                    )
                    return None
                # Re-raise other exceptions to be handled by outer try/except
                raise
            # A response is valid if it has content OR tool calls (or both)
            if not response or (not response.content and not response.tool_calls):
                logger.error("AI provider returned empty response")
                return None

            # Track token usage
            output_tokens = (
                self.estimate_tokens(response.content) if response.content else 0
            )
            total_tokens = input_tokens + output_tokens

            self.state.add_tokens(total_tokens)

            logger.debug(
                f"AI inference completed: {output_tokens} output tokens, {total_tokens} total"
            )

            # Phase 1 KV-Cache Optimization: Log to EventLog for persistent context
            if self.event_log:
                self.event_log.append_ai_inference(
                    iteration=self.state.current_iteration,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    response=response.content if response.content else "",
                    tool_calls=response.tool_calls if response.tool_calls else None,
                )

            return response

        except Exception as e:
            logger.error(f"AI inference failed: {e}")
            return None

    async def _parse_and_execute_tools(
        self, tool_calls: List[Dict[str, Any]], available_tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute tool calls with limits."""
        if not available_tools or not self.mcp_manager or not tool_calls:
            return []

        try:
            # Create a mapping from tool name to tool server for quick lookup
            tool_name_to_server = {
                tool["name"]: tool["server"] for tool in available_tools
            }

            # Populate tool_server field for each tool call
            for tool_call in tool_calls:
                tool_name = tool_call.get("tool_name")
                if tool_name in tool_name_to_server:
                    tool_call["tool_server"] = tool_name_to_server[tool_name]
                else:
                    logger.warning(
                        f"Tool '{tool_name}' not found in available tools, skipping"
                    )
                    continue

            # Filter out tool calls that don't have a matching server
            valid_tool_calls = [tc for tc in tool_calls if tc.get("tool_server")]

            # Check tool limits
            if (
                self.state.current_iteration_tools + len(valid_tool_calls)
                > self.limits.max_tools_per_iteration
            ):
                excess_tools = (
                    self.state.current_iteration_tools + len(valid_tool_calls)
                ) - self.limits.max_tools_per_iteration
                logger.warning(
                    f"Truncating {excess_tools} tool calls due to iteration limit"
                )
                valid_tool_calls = valid_tool_calls[
                    : self.limits.max_tools_per_iteration
                    - self.state.current_iteration_tools
                ]

            # Execute tool calls with result size tracking
            executed_tools = []
            for tool_call in valid_tool_calls:
                if (
                    self.state.current_iteration_tools
                    >= self.limits.max_tools_per_iteration
                ):
                    logger.warning("Stopping tool execution due to iteration limit")
                    break

                result = await self._execute_single_tool(tool_call)
                executed_tools.append(result)

            return executed_tools

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return []

    async def _execute_single_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool call with size and time limits."""
        start_time = time.time()

        try:
            # Check if we have enough time budget remaining
            elapsed_total = self.state.get_elapsed_time()
            if elapsed_total >= self.limits.max_execution_time:
                raise TimeoutError(
                    f"Total execution time limit exceeded: {elapsed_total:.1f}s"
                )

            # Execute tool via local registry or MCP manager
            # Note: Timeouts are handled by the MCP transport layer (HTTP client timeout configuration)
            # Using asyncio.wait_for() here causes incompatibility with anyio-based MCP library cleanup
            try:
                tool_server = tool_call["tool_server"]
                tool_name = tool_call["tool_name"]
                arguments = tool_call["arguments"]

                # Check if this is a local tool (server="local")
                if tool_server == "local" and self.local_tool_registry:
                    result = await self.local_tool_registry.call_tool(
                        tool_name, arguments
                    )
                elif self.mcp_manager:
                    result = await self.mcp_manager.call_tool(
                        tool_server,
                        tool_name,
                        arguments,
                    )
                else:
                    raise ValueError(
                        f"No tool executor available for server '{tool_server}': "
                        f"MCP manager not configured and tool is not local"
                    )
            except TimeoutError as e:
                # MCP transport raised timeout - propagate with context
                raise TimeoutError(
                    f"Tool {tool_call['tool_name']} exceeded timeout: {e}"
                ) from e
            except Exception as e:
                # Check if it's a timeout-related error from httpx/httpcore
                if "timeout" in str(e).lower() or "ReadTimeout" in type(e).__name__:
                    raise TimeoutError(
                        f"Tool {tool_call['tool_name']} timed out: {e}"
                    ) from e
                # Otherwise re-raise the original exception
                raise

            # Calculate result size and execution time
            result_str = str(result)
            result_size_bytes = len(result_str.encode("utf-8"))
            execution_time = time.time() - start_time

            # Check if execution exceeded recommended total timeout (log warning but don't fail)
            exceeded_total_timeout = execution_time > self.limits.max_tool_timeout
            if exceeded_total_timeout:
                logger.warning(
                    f"Tool {tool_call['tool_name']} exceeded recommended total timeout "
                    f"({execution_time:.1f}s > {self.limits.max_tool_timeout}s) but completed successfully. "
                    f"Consider increasing max_tool_timeout or using idle_timeout for progress-reporting tools."
                )

            # Store FULL result in knowledge store (before any compression)
            # This ensures complete data preservation for reports and notifications
            self.knowledge_store.add_tool_result(
                iteration=self.state.current_iteration,
                server=tool_call["tool_server"],
                tool=tool_call["tool_name"],
                arguments=tool_call["arguments"],
                result=result,  # FULL, uncompacted result
                execution_time=execution_time,
                result_size_bytes=result_size_bytes,
            )

            # Log tool addition with unique reference
            latest_tool = self.knowledge_store.tool_results[-1]
            logger.info(
                f"Tool [{latest_tool['unique_ref']}]: {tool_call['tool_name']} added to knowledge store. "
                f"Total tools: {len(self.knowledge_store.tool_results)}"
            )

            # Phase 1 KV-Cache Optimization: Log to EventLog for persistent context
            if self.event_log:
                self.event_log.append_tool_execution(
                    iteration=self.state.current_iteration,
                    server=tool_call["tool_server"],
                    tool_name=tool_call["tool_name"],
                    arguments=tool_call["arguments"],
                    result=result,  # FULL result
                    execution_time=execution_time,
                    result_size_bytes=result_size_bytes,
                )

            # Phase 2: Use intelligent compression for large tool results
            # (for conversation efficiency - knowledge store has full data)
            tool_result_obj = {
                "tool_name": tool_call["tool_name"],
                "tool_server": tool_call["tool_server"],
                "arguments": tool_call["arguments"],
                "result": result,
                "status": "success",
            }

            # Compress if result is large (>4000 tokens estimated)
            result_tokens = self.estimate_tokens(result_str)
            if result_tokens > self.limits.max_tokens_per_tool_result:
                logger.debug(
                    f"Tool result large ({result_tokens} tokens), compressing to fit budget"
                )
                tool_result_obj, compression_result = (
                    self.compactor.compress_tool_result_realtime(
                        tool_result_obj,
                        max_tokens=self.limits.max_tokens_per_tool_result,
                    )
                )
                logger.info(
                    f"Tool result compressed: {compression_result.compression_ratio:.2f}x compression, "
                    f"{compression_result.information_preserved:.1%} information preserved"
                )
                # Recalculate size after compression
                result_str = str(tool_result_obj["result"])
                result_size_bytes = len(result_str.encode("utf-8"))

            # Check result size limits (after compression)
            if result_size_bytes > self.limits.max_result_size_bytes:
                logger.warning(
                    f"Tool result still exceeds size limit after compression: {result_size_bytes}/{self.limits.max_result_size_bytes} bytes"
                )
                # Emergency truncation
                truncated_result = result_str[: self.limits.max_result_size_bytes // 2]
                tool_result_obj["result"] = (
                    f"{truncated_result}...\n[TRUNCATED: Result exceeded {self.limits.max_result_size_bytes} byte limit]"
                )
                result_size_bytes = len(str(tool_result_obj["result"]).encode("utf-8"))

            # Track execution
            self.state.add_tool_execution(result_size_bytes)

            logger.debug(
                f"Tool {tool_call['tool_name']} executed in {execution_time:.2f}s, {result_size_bytes} bytes"
            )

            return {
                "tool_name": tool_result_obj["tool_name"],
                "tool_server": tool_result_obj["tool_server"],
                "arguments": tool_result_obj["arguments"],
                "result": tool_result_obj["result"],
                "execution_time": execution_time,
                "result_size_bytes": result_size_bytes,
                "status": tool_result_obj["status"],
                "metrics_applied": True,
                "compressed": tool_result_obj.get("compressed", False),
                "exceeded_total_timeout": exceeded_total_timeout,
            }

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Tool {tool_call['tool_name']} failed after {execution_time:.2f}s: {e}"
            )

            self.state.add_tool_execution(0)  # Count the attempt even if failed

            return {
                "tool_name": tool_call["tool_name"],
                "tool_server": tool_call["tool_server"],
                "arguments": tool_call["arguments"],
                "result": f"Tool execution failed: {e}",
                "execution_time": execution_time,
                "result_size_bytes": 0,
                "status": "error",
                "metrics_applied": True,
                "exceeded_total_timeout": False,
            }

    def _format_tool_results_for_conversation(
        self, tool_results: List[Dict[str, Any]]
    ) -> str:
        """
        Format tool results for adding to conversation context.

        Adds special marker to prevent context compaction from discarding tool data.
        This ensures AI always has access to queried entity information.
        """
        # Add marker at start to prevent compaction of tool results
        marker = "[TOOL_RESULTS:DO_NOT_COMPACT]\n"

        if self.should_truncate_results():
            # Progressive degradation: provide summarized results
            return marker + self._format_summarized_tool_results(tool_results)
        else:
            # Full results
            formatted_results = []
            for result in tool_results:
                if result["status"] == "success":
                    formatted_results.append(
                        f"Tool {result['tool_name']}: {result['result']}"
                    )
                else:
                    formatted_results.append(
                        f"Tool {result['tool_name']} failed: {result['result']}"
                    )

            return marker + "Tool execution results:\n" + "\n".join(formatted_results)

    def _format_summarized_tool_results(
        self, tool_results: List[Dict[str, Any]]
    ) -> str:
        """Format tool results in summarized form to save tokens."""
        summary_parts = []
        successful_tools = [r for r in tool_results if r["status"] == "success"]
        failed_tools = [r for r in tool_results if r["status"] == "error"]

        if successful_tools:
            tool_names = [r["tool_name"] for r in successful_tools]
            summary_parts.append(
                f"Successfully executed {len(successful_tools)} tools: {', '.join(tool_names)}"
            )

            # Include abbreviated results for successful tools
            for result in successful_tools:
                result_preview = (
                    str(result["result"])[:200] + "..."
                    if len(str(result["result"])) > 200
                    else str(result["result"])
                )
                summary_parts.append(f"- {result['tool_name']}: {result_preview}")

        if failed_tools:
            tool_names = [r["tool_name"] for r in failed_tools]
            summary_parts.append(
                f"Failed tools ({len(failed_tools)}): {', '.join(tool_names)}"
            )

        summary_parts.append(
            f"[SUMMARIZED: {len(tool_results)} total tools executed in degradation mode]"
        )

        return "Tool execution summary:\n" + "\n".join(summary_parts)

    async def _generate_final_summary(
        self,
        conversation_messages: List[ProviderMessage],
        tool_results: List[Dict[str, Any]],
    ) -> str:
        """Generate final summary based on execution state."""
        # Get the last assistant message as the final response
        final_response = None
        for msg in reversed(conversation_messages):
            if msg.role == "assistant":
                final_response = msg.content
                break

        if not final_response:
            final_response = "Analysis completed"

        # Add execution summary if degradation was active or limits were hit
        if self.state.degradation_active or self.state.stop_reason in [
            StopReason.max_iterations,
            StopReason.out_of_tokens,
        ]:
            summary_parts = [final_response]

            # Add limit information
            if self.state.stop_reason == StopReason.max_iterations:
                summary_parts.append(
                    f"\n\n[EXECUTION SUMMARY: Completed {self.state.current_iteration} iterations (limit reached)]"
                )
            elif self.state.stop_reason == StopReason.out_of_tokens:
                summary_parts.append(
                    f"\n\n[EXECUTION SUMMARY: Token budget exhausted ({self.state.total_tokens_used} tokens used)]"
                )

            if tool_results:
                successful_tools = len(
                    [r for r in tool_results if r["status"] == "success"]
                )
                summary_parts.append(
                    f"\nTools executed: {successful_tools}/{len(tool_results)} successful"
                )

            final_response = "".join(summary_parts)

        return final_response

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution summary with all metrics."""
        elapsed_time = self.state.get_elapsed_time()

        return {
            "session_id": self.session_id,
            "stop_reason": self.state.stop_reason.value
            if self.state.stop_reason
            else None,
            "execution_time": elapsed_time,
            "iterations": {
                "completed": self.state.current_iteration,
                "max_allowed": self.limits.max_infer_iters,
                "utilization": self.state.current_iteration
                / self.limits.max_infer_iters,
            },
            "tokens": {
                "total_used": self.state.total_tokens_used,
                "cumulative_all_inferences": self.state.get_total_tokens_all_inferences(),
                "max_allowed": self.limits.max_tokens_total,
                "utilization": self.state.total_tokens_used
                / self.limits.max_tokens_total,
                "current_iteration": self.state.current_iteration_tokens,
            },
            "tools": {
                "total_executed": self.state.total_tools_executed,
                "max_allowed": self.limits.max_total_tools,
                "utilization": self.state.total_tools_executed
                / self.limits.max_total_tools,
                "current_iteration": self.state.current_iteration_tools,
            },
            "results": {
                "total_size_bytes": self.state.total_result_size_bytes,
                "max_allowed_bytes": self.limits.max_total_result_size_bytes,
                "utilization": self.state.total_result_size_bytes
                / self.limits.max_total_result_size_bytes,
            },
            "compression": {
                "count": self.state.compression_count,
                "avg_compression_ratio": (
                    self.state.total_compression_ratio / self.state.compression_count
                    if self.state.compression_count > 0
                    else 0.0
                ),
                "avg_information_preserved": (
                    self.state.total_information_preserved
                    / self.state.compression_count
                    if self.state.compression_count > 0
                    else 0.0
                ),
                "triggers": self.state.compression_triggers.copy()
                if self.state.compression_triggers
                else [],
            },
            "recovery": {
                "attempts": self.state.recovery_attempts,
                "successful": self.state.successful_recoveries,
                "success_rate": (
                    self.state.successful_recoveries / self.state.recovery_attempts
                    if self.state.recovery_attempts > 0
                    else 0.0
                ),
                "last_error_type": self.state.last_recovery_error_type,
                "manager_stats": self.recovery_manager.get_recovery_stats()
                if self.recovery_manager
                else {},
                "max_tokens_recovery": {
                    "attempts": self.state.max_tokens_recovery_attempts,
                    "limit": self.state.max_tokens_recovery_limit,
                },
            },
            "degradation_active": self.state.degradation_active,
            "warnings_count": len(self.state.warnings_issued),
        }
