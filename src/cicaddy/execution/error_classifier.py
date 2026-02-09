"""Error classification system for early break recovery mechanism.

Classifies execution errors to enable scenario-specific recovery prompts.
Based on real-world analysis of 12 jobs across 4 pipelines showing:
- AI_PREMATURE_COMPLETION: 25% (most common recoverable error)
- TIMEOUT/CONTINUED: 42% (often legitimate, may not need recovery)
- TOOL_ERROR: 8% (recoverable with alternative approaches)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ErrorType(Enum):
    """Classification of execution errors for recovery handling."""

    # AI said "I will continue..." but made no tool calls (MOST COMMON - 25% of jobs)
    # Discovered from real pipeline analysis: jobs 43268795, 43268799, 43067065
    AI_PREMATURE_COMPLETION = "ai_premature_completion"

    # Model called tool with unsupported arguments (e.g., update operations on read-only tools)
    INVALID_TOOL_CALL = "invalid_tool_call"

    # AI provider returned empty/error response
    AI_INFERENCE_FAILURE = "ai_inference_failure"

    # MCP tool execution failed or timed out
    TOOL_EXECUTION_ERROR = "tool_execution_error"

    # Same error occurred multiple times - needs strategy change
    REPEATED_FAILURE = "repeated_failure"

    # Token budget exhausted - needs fresh context recovery
    MAX_TOKENS_EXCEEDED = "max_tokens_exceeded"

    # Iteration limit reached - needs fresh context recovery
    MAX_ITERATIONS_EXCEEDED = "max_iterations_exceeded"


@dataclass
class ClassifiedError:
    """Structured error with classification and context for recovery."""

    error_type: ErrorType
    message: str
    tool_name: Optional[str] = None
    tool_server: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    iteration: int = 0
    retry_count: int = 0
    last_response_preview: Optional[str] = None  # For AI_PREMATURE_COMPLETION

    def get_error_key(self) -> str:
        """
        Generate unique key for this error pattern.

        Used to track recovery attempts and prevent infinite loops.
        """
        if self.error_type == ErrorType.TOOL_EXECUTION_ERROR and self.tool_name:
            return f"{self.error_type.value}:{self.tool_name}:{self.tool_server}"
        elif self.error_type == ErrorType.AI_PREMATURE_COMPLETION:
            return f"{self.error_type.value}:iteration_{self.iteration}"
        else:
            return f"{self.error_type.value}:{self.iteration}"


def has_continuation_indicator(response_text: str) -> bool:
    """
    Check if AI response contains continuation indicators.

    Designed to work with last 200 chars of response (end portion).
    Detects simple intent markers that indicate AI plans to continue with tool calls.

    Args:
        response_text: The AI response text to check (typically last 200 chars)

    Returns:
        True if response contains phrases indicating intent to make tool calls
    """
    if not response_text:
        return False

    response_lower = response_text.lower()

    # Simple intent markers that indicate future tool actions
    # These appear at the END of responses when AI plans to continue
    # Note: "i will" catches "first, i will", "i will pivot", "i will attempt", etc.
    # Note: "next step is to" catches "my next step is to", "the next step is to", etc.
    INTENT_MARKERS = [
        "i will",
        "i'll",
        "let me",
        "next step is to",
        "i need to",
        "proceeding to",
        "continuing with",
        "to overcome this",
    ]

    return any(marker in response_lower for marker in INTENT_MARKERS)


def classify_tool_error(error_message: str) -> ErrorType:
    """
    Classify tool error type based on error message patterns.

    Args:
        error_message: The error message from tool execution

    Returns:
        Appropriate ErrorType for the error
    """
    error_lower = error_message.lower()

    # Check for invalid operation errors (unsupported tool usage)
    invalid_keywords = [
        "not supported",
        "invalid",
        "not allowed",
        "update",
        "write",
        "modify",
    ]
    if any(keyword in error_lower for keyword in invalid_keywords):
        return ErrorType.INVALID_TOOL_CALL

    # Check for timeout/connection errors
    timeout_keywords = ["timeout", "connection", "refused", "deadline exceeded"]
    if any(keyword in error_lower for keyword in timeout_keywords):
        return ErrorType.TOOL_EXECUTION_ERROR

    # Default to general tool execution error
    return ErrorType.TOOL_EXECUTION_ERROR
