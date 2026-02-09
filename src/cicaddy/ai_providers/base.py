"""Base provider interface following Llama Stack patterns."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from cicaddy.utils.logger import get_logger
from cicaddy.utils.retry import RetryConfig, retry_async
from cicaddy.utils.token_utils import (
    PromptTruncator,
    TokenAwareResponse,
    TokenCounter,
    TokenLimitManager,
)

logger = get_logger(__name__)


class StopReason(Enum):
    """Stop reasons following Llama Stack patterns."""

    end_of_turn = "end_of_turn"
    end_of_message = "end_of_message"
    out_of_tokens = "out_of_tokens"


@dataclass
class ProviderMessage:
    """Message format for provider interactions."""

    content: str
    role: str = "user"


@dataclass
class ProviderResponse:
    """Response format from providers."""

    content: str
    model: str
    stop_reason: Optional[StopReason] = None
    usage: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ProviderError(Exception):
    """Base exception for provider errors."""

    pass


class ContentExtractionError(ProviderError):
    """Error when content cannot be extracted from response."""

    pass


class TemporaryServiceError(ProviderError):
    """Error indicating temporary service unavailability that should be retried.

    Attributes:
        retry_count: Number of retries already attempted
        suggested_delay: API-suggested retry delay in seconds (e.g., from 429 rate limit response)
    """

    def __init__(
        self,
        message: str,
        retry_count: int = 0,
        suggested_delay: Optional[float] = None,
    ):
        super().__init__(message)
        self.retry_count = retry_count
        self.suggested_delay = suggested_delay


class BaseProvider(ABC):
    """Abstract base class for AI providers following Llama Stack patterns."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_id = config.get("model_id", "default")

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider connection."""
        pass

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[ProviderMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ProviderResponse:
        """Generate chat completion with optional tool calls."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Cleanup provider resources."""
        pass

    def _safe_extract_content(
        self, response: Any, fallback_message: Optional[str] = None
    ) -> str:
        """
        Safely extract content from provider response following Llama Stack patterns.

        This method provides a standard way to extract content with fallbacks,
        following the robust error handling patterns seen in Llama Stack.
        """
        if fallback_message is None:
            fallback_message = "Error: No content available in response. This may be due to safety filters or content restrictions."

        # Try various common content extraction patterns
        content_attempts = [
            lambda: getattr(response, "content", None),
            lambda: getattr(response, "text", None),
            lambda: response.get("content") if hasattr(response, "get") else None,
            lambda: response.get("text") if hasattr(response, "get") else None,
            lambda: str(response) if response else None,
        ]

        for attempt in content_attempts:
            try:
                content = attempt()
                if content and isinstance(content, str) and content.strip():
                    return content.strip()
            except Exception as e:
                logger.debug(f"Content extraction attempt failed: {e}")
                continue

        logger.warning("Failed to extract content from provider response")
        return fallback_message

    def _convert_finish_reason_to_stop_reason(self, finish_reason: Any) -> StopReason:
        """
        Convert provider-specific finish reasons to standard stop reasons.

        Following Llama Stack's approach to standardize stop reasons across providers.
        """
        if finish_reason is None:
            return StopReason.end_of_turn

        # Convert to string for comparison
        reason_str = str(finish_reason).lower()

        # Map common finish reasons to stop reasons
        if reason_str in ["stop", "eos", "eos_token", "end", "end_turn"]:
            return StopReason.end_of_turn
        elif reason_str in ["length", "max_tokens", "token_limit", "out_of_tokens"]:
            return StopReason.out_of_tokens
        elif reason_str in ["tool_calls", "function_call", "tool_use"]:
            return StopReason.end_of_message
        else:
            # Unknown finish reason - log and default to end_of_turn
            logger.debug(
                f"Unknown finish_reason: {finish_reason}, defaulting to end_of_turn"
            )
            return StopReason.end_of_turn

    async def _retry_with_exponential_backoff(
        self,
        func,
        *args,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        max_retry_time_seconds: float = 300.0,
        jitter: bool = True,
        context_name: str = "AI provider operation",
        **kwargs,
    ):
        """
        Retry a function with exponential backoff and time limit.

        Uses the centralized retry_async utility from cicaddy.utils.retry.

        Args:
            func: Async function to retry
            *args: Arguments to pass to func
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds (doubled each retry)
            max_delay: Maximum delay between individual retries
            max_retry_time_seconds: Maximum total time for all retry attempts (0=disabled)
            jitter: Add random jitter to prevent thundering herd
            context_name: Name of the operation for logging
            **kwargs: Keyword arguments to pass to func

        Returns:
            Result of successful function call

        Raises:
            RetryTimeoutError: If max_retry_time_seconds exceeded
            RetryExhaustedError: If all retries are exhausted
            Exception: Non-retryable errors are re-raised immediately
        """
        # Build retry config from parameters
        config = RetryConfig(
            max_retries=max_retries,
            initial_delay=base_delay,
            max_delay=max_delay,
            max_retry_time_seconds=max_retry_time_seconds,
            backoff_factor=2.0,
            jitter=jitter,
            retry_on_5xx=True,
            retry_on_timeout=True,
            retry_on_connection_error=True,
            # Add TemporaryServiceError to retryable exceptions
            retryable_exceptions=(TemporaryServiceError,),
        )

        return await retry_async(
            func,
            config,
            context_name,
            *args,
            **kwargs,
        )

    def _should_retry_on_finish_reason(self, finish_reason: Any) -> bool:
        """
        Determine if a finish reason indicates a retryable error.

        Args:
            finish_reason: The finish reason to check

        Returns:
            True if this finish reason should trigger a retry
        """
        # Finish reason 12 = Gemini service not available
        if finish_reason == 12:
            return True

        # Add other retryable finish reasons here as needed
        return False

    def _validate_token_limits(
        self, prompt: str, provider_name: str
    ) -> tuple[str, bool]:
        """
        Validate and truncate prompt if it exceeds token limits.

        Returns:
            Tuple of (processed_prompt, was_truncated)
        """
        try:
            # Count tokens for this provider
            token_count = TokenCounter.count_tokens(
                prompt, provider_name, self.model_id
            )

            # Check if within limits
            if TokenLimitManager.validate_input_tokens(
                provider_name, self.model_id, token_count
            ):
                logger.debug(
                    f"Prompt within limits: {token_count} tokens for {provider_name}"
                )
                return prompt, False

            # Truncate if too long
            logger.warning(
                f"Prompt exceeds limits: {token_count} tokens for {provider_name}, truncating"
            )
            truncated_prompt, was_truncated = PromptTruncator.truncate_for_provider(
                prompt, provider_name, self.model_id
            )

            return truncated_prompt, was_truncated

        except Exception as e:
            logger.error(f"Error validating token limits: {e}")
            # Return original prompt if validation fails
            return prompt, False

    def _handle_max_tokens_response(
        self,
        content: str,
        tool_calls: Optional[List[Any]] = None,
        original_prompt: Optional[str] = None,
    ) -> str:
        """
        Handle responses that were truncated due to max tokens.

        Creates an intelligent summary based on successful tool calls and their results,
        focusing on the original analysis objectives rather than just technical limitations.

        Args:
            content: Partial response content before truncation
            tool_calls: List of tool execution results
            original_prompt: The original analysis prompt to understand objectives
        """
        if not tool_calls:
            tool_calls = []

        # Check if we should summarize existing calls
        if TokenAwareResponse.should_summarize_existing_calls(tool_calls):
            summary = TokenAwareResponse.create_intelligent_truncated_summary(
                tool_calls, original_prompt, "max_tokens"
            )

            # Combine partial content with intelligent summary if content exists
            if content and content.strip():
                return f"{content.strip()}\n\n{summary}"
            else:
                return summary

        # If no tool calls to summarize, just note the truncation
        if content and content.strip():
            return f"{content.strip()}\n\n[Note: Response was truncated due to maximum token limit.]"
        else:
            return "Response was truncated due to maximum token limit. Please try with a more focused query."

    def _get_dynamic_max_tokens(self, provider_name: str, prompt_length: int) -> int:
        """
        Calculate dynamic max_tokens based on provider limits and prompt length.

        Ensures we don't exceed total token limits while allowing reasonable output.
        """
        # Get provider limits
        max_output = TokenLimitManager.get_max_output_tokens(
            provider_name, self.model_id
        )
        max_input = TokenLimitManager.get_max_input_tokens(provider_name, self.model_id)

        # Reserve space for prompt and safety margin
        safety_margin = 1000
        available_output = min(max_output, max_input - prompt_length - safety_margin)

        # Ensure we have at least some reasonable output space
        return max(available_output, 512)
