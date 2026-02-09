"""Retry utilities for MCP client connections with exponential backoff and jitter.

This module provides MCP-specific retry utilities that build on the centralized
retry framework in utils.retry. It includes:
- Factory function to create RetryConfig from MCPServerConfig
- MCP-specific error detection for websockets, JSON decode errors
- RetryableMixin for transport classes
- Connection error wrapper decorator
"""

import asyncio
import time
from typing import Any, Callable, Optional

import httpx
import websockets

from cicaddy.utils.logger import get_logger
from cicaddy.utils.retry import RetryConfig, RetryTimeoutError
from cicaddy.utils.retry import calculate_delay as base_calculate_delay

logger = get_logger(__name__)


# Re-export RetryConfig and RetryTimeoutError from cicaddy.utils.retry for backward compatibility
# These are imported directly above and will be available to consumers of this module
__all__ = [
    "RetryConfig",
    "RetryTimeoutError",
    "RetryableError",
    "ConnectionRetryableError",
    "TimeoutRetryableError",
    "ServerRetryableError",
    "calculate_delay",
    "should_retry",
    "retry_async",
    "retry_config_from_mcp",
    "RetryableMixin",
    "wrap_connection_errors",
]


def retry_config_from_mcp(config) -> RetryConfig:
    """Create RetryConfig from MCPServerConfig.

    Args:
        config: MCPServerConfig instance with retry settings

    Returns:
        RetryConfig configured for MCP server connection
    """
    return RetryConfig(
        max_retries=config.retry_count,
        initial_delay=config.retry_delay,
        max_delay=config.retry_max_delay,
        max_retry_time_seconds=getattr(config, "max_retry_time_seconds", 300.0),
        backoff_factor=config.retry_backoff_factor,
        jitter=config.retry_jitter,
        retry_on_5xx=config.retry_on_server_error,
        retry_on_timeout=config.retry_on_timeout,
        retry_on_connection_error=config.retry_on_connection_error,
    )


class RetryableError(Exception):
    """Base class for errors that should trigger a retry."""

    pass


class ConnectionRetryableError(RetryableError):
    """Connection-related errors that should trigger a retry."""

    pass


class TimeoutRetryableError(RetryableError):
    """Timeout-related errors that should trigger a retry."""

    pass


class ServerRetryableError(RetryableError):
    """Server-related errors that should trigger a retry."""

    pass


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate retry delay using centralized implementation.

    Args:
        attempt: The current attempt number (0-based)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    return base_calculate_delay(attempt, config)


def should_retry(exception: BaseException, config: RetryConfig) -> bool:
    """
    Determine if an exception should trigger a retry based on configuration.

    Args:
        exception: The exception that occurred (Exception or BaseException like CancelledError)
        config: Retry configuration

    Returns:
        True if the exception should trigger a retry
    """
    # Always retry RetryableError subclasses
    if isinstance(exception, RetryableError):
        return True

    # Check error message for retryable patterns
    error_str = str(exception).lower()
    error_type = type(exception).__name__

    # Check for JSON decode errors (HTML error pages instead of JSON responses)
    # This commonly occurs when APIs return HTML error pages due to rate limiting
    if "unexpected token" in error_str or error_type == "JSONDecodeError":
        return True

    # Check for rate limiting errors
    if "rate limit" in error_str or "too many requests" in error_str:
        return True

    # Check for service unavailable / gateway errors
    if (
        "service unavailable" in error_str
        or "gateway timeout" in error_str
        or "gateway time-out" in error_str
        or "502" in error_str
        or "503" in error_str
        or "504" in error_str
    ):
        return True

    # Check for async generator cleanup errors (common with streaming HTTP connections)
    # These can occur when HTTP errors (like 504) happen during streaming responses
    # and the cleanup of the async generator fails
    if "async" in error_str and (
        "generator" in error_str or "async_generator" in error_str
    ):
        logger.warning(
            f"Detected async generator cleanup error, will retry: {error_str[:200]}"
        )
        return True

    # Check for anyio/asyncio TaskGroup exceptions (common during HTTP streaming cleanup)
    if (
        error_type in ("BaseExceptionGroup", "ExceptionGroup")
        or "ExceptionGroup" in error_type
    ):
        logger.warning(
            f"Detected ExceptionGroup error (likely from streaming cleanup), will retry: {error_str[:200]}"
        )
        return True

    # Check for RuntimeError patterns from MCP SDK async cleanup issues
    # These occur when streamablehttp_client cleanup fails due to anyio task group constraints:
    # - "Attempted to exit cancel scope in a different task than it was entered in"
    # - "aclose(): asynchronous generator is already running"
    # See: https://github.com/modelcontextprotocol/python-sdk/issues/915
    # See: https://github.com/modelcontextprotocol/python-sdk/issues/521
    if error_type == "RuntimeError":
        # Check for cancel scope cross-task exit error
        if "cancel scope" in error_str and "different task" in error_str:
            logger.warning(
                f"Detected RuntimeError from cancel scope cross-task exit (MCP SDK cleanup), will retry: {error_str[:200]}"
            )
            return True
        # Check for async generator already running error
        if "aclose" in error_str and "already running" in error_str:
            logger.warning(
                f"Detected RuntimeError from aclose() during async generator cleanup (MCP SDK), will retry: {error_str[:200]}"
            )
            return True
        # Check for generic async generator cleanup RuntimeError
        if "asynchronous generator" in error_str:
            logger.warning(
                f"Detected RuntimeError from async generator (MCP SDK cleanup), will retry: {error_str[:200]}"
            )
            return True

    # Check for CancelledError from HTTP streaming cleanup (NOT for clean user cancellations)
    # When 504 Gateway Timeout happens inside MCP's streamablehttp_client, it triggers
    # async generator cleanup which raises RuntimeError, and the final exception
    # bubbling up is CancelledError with message "Cancelled by cancel scope XXXXXXXX".
    # Only retry if the message contains "cancel scope" (anyio internal cleanup pattern).
    # Clean user-initiated cancellations typically have no message or different patterns.
    if error_type == "CancelledError" or isinstance(exception, asyncio.CancelledError):
        # Only retry if this looks like anyio's internal cancel scope cleanup
        if "cancel scope" in error_str:
            logger.warning(
                f"Detected CancelledError from cancel scope (HTTP streaming cleanup), will retry: {error_str[:200]}"
            )
            return True
        # Don't retry clean cancellations (user-initiated, shutdown, etc.)
        logger.debug(
            f"CancelledError without cancel scope pattern, not retrying: {error_str[:100]}"
        )
        return False

    # Check for timeout errors first (TimeoutError is also an OSError)
    if config.retry_on_timeout:
        timeout_error_types = (
            TimeoutError,
            asyncio.TimeoutError,
            httpx.TimeoutException,
            httpx.ReadTimeout,
        )

        if isinstance(exception, timeout_error_types):
            return True

        # Check for timeout in error message
        if "timeout" in error_str or "timed out" in error_str:
            return True

    # Check for connection errors (but not timeout errors)
    if config.retry_on_connection_error:
        # Exclude timeout errors which are handled separately
        if isinstance(exception, (TimeoutError, asyncio.TimeoutError)):
            return False

        connection_error_types = (
            ConnectionError,
            OSError,
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.NetworkError,
            httpx.PoolTimeout,
            websockets.ConnectionClosed,
            websockets.InvalidURI,
        )

        if isinstance(exception, connection_error_types):
            return True

        # Check for connection in error message
        if "connection" in error_str:
            return True

    # Check for server errors (5xx HTTP status codes)
    # Note: use retry_on_5xx from cicaddy.utils.retry.RetryConfig
    if config.retry_on_5xx:
        if isinstance(exception, httpx.HTTPStatusError):
            return 500 <= exception.response.status_code < 600

    return False


# RetryTimeoutError is imported from cicaddy.utils.retry for consistency


async def retry_async(
    func: Callable[..., Any],
    config: RetryConfig,
    context_name: str = "operation",
    *args,
    **kwargs,
) -> Any:
    """
    Retry an async function with exponential backoff and time limit.

    Args:
        func: The async function to retry
        config: Retry configuration
        context_name: Name of the operation for logging
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the successful function call

    Raises:
        RetryTimeoutError: If max_retry_time_seconds is exceeded
        The last exception if all retries are exhausted
    """
    last_exception: Optional[BaseException] = None
    retry_start_time = time.time()
    time_limit_enabled = config.max_retry_time_seconds > 0

    for attempt in range(config.max_retries + 1):
        # Check if we've exceeded the maximum retry time before attempting
        if time_limit_enabled:
            elapsed_total = time.time() - retry_start_time
            if elapsed_total >= config.max_retry_time_seconds:
                logger.error(
                    f"Max retry time ({config.max_retry_time_seconds}s) exceeded for {context_name} "
                    f"after {attempt} attempts and {elapsed_total:.2f}s"
                )
                if last_exception:
                    raise RetryTimeoutError(
                        f"Max retry time exceeded for {context_name}: {last_exception}",
                        elapsed_time=elapsed_total,
                        attempts=attempt,
                    ) from last_exception
                raise RetryTimeoutError(
                    f"Max retry time exceeded for {context_name}",
                    elapsed_time=elapsed_total,
                    attempts=attempt,
                )

        try:
            if attempt > 0:
                delay = calculate_delay(attempt, config)

                # Adjust delay if it would exceed the time limit
                if time_limit_enabled:
                    remaining_time = config.max_retry_time_seconds - (
                        time.time() - retry_start_time
                    )
                    if remaining_time <= 0:
                        logger.warning(
                            f"No time remaining for retry of {context_name}, stopping"
                        )
                        break
                    if delay > remaining_time:
                        delay = max(
                            0.1, remaining_time * 0.8
                        )  # Use 80% of remaining time
                        logger.info(
                            f"Reduced retry delay to {delay:.2f}s due to time limit for {context_name}"
                        )

                logger.info(
                    f"Retrying {context_name} (attempt {attempt + 1}/{config.max_retries + 1}) "
                    f"after {delay:.2f}s delay"
                )
                # Shield sleep from lingering cancel scopes (MCP SDK cleanup can leak cancellation)
                try:
                    await asyncio.shield(asyncio.sleep(delay))
                except asyncio.CancelledError:
                    # If shielded sleep still gets cancelled, wait in a new context
                    logger.warning(
                        "Retry sleep cancelled by lingering cancel scope, creating isolated delay"
                    )
                    await asyncio.get_event_loop().run_in_executor(
                        None, time.sleep, min(delay, 2.0)
                    )

            # Attempt the operation
            attempt_start_time = time.time()
            # Shield the operation from lingering cancel scopes
            try:
                result = await asyncio.shield(func(*args, **kwargs))
            except asyncio.CancelledError as shield_error:
                # Re-raise to be handled by the CancelledError handler below
                raise shield_error

            if attempt > 0:
                duration = time.time() - attempt_start_time
                total_elapsed = time.time() - retry_start_time
                logger.info(
                    f"Successfully retried {context_name} after {attempt} attempts "
                    f"(attempt took {duration:.2f}s, total {total_elapsed:.2f}s)"
                )

            return result

        except asyncio.CancelledError as e:
            # CancelledError inherits from BaseException in Python 3.8+, not Exception
            # Must catch explicitly to run through should_retry() logic
            last_exception = e
            duration = (
                time.time() - attempt_start_time
                if "attempt_start_time" in locals()
                else 0
            )

            if attempt == config.max_retries:
                total_elapsed = time.time() - retry_start_time
                logger.error(
                    f"Final attempt for {context_name} failed after {config.max_retries} retries "
                    f"(CancelledError, took {duration:.2f}s, total {total_elapsed:.2f}s): {e}"
                )
                break

            if should_retry(e, config):
                logger.warning(
                    f"Attempt {attempt + 1} for {context_name} failed with CancelledError "
                    f"(took {duration:.2f}s), will retry: {e}"
                )
                continue
            else:
                # Clean cancellation - don't retry
                logger.info(
                    f"CancelledError for {context_name} is not retryable, stopping: {e}"
                )
                break

        except Exception as e:
            last_exception = e
            duration = (
                time.time() - attempt_start_time
                if "attempt_start_time" in locals()
                else 0
            )

            if attempt == config.max_retries:
                # Final attempt failed
                total_elapsed = time.time() - retry_start_time
                logger.error(
                    f"Final attempt for {context_name} failed after {config.max_retries} retries "
                    f"(attempt took {duration:.2f}s, total {total_elapsed:.2f}s): {e}"
                )
                break

            if should_retry(e, config):
                logger.warning(
                    f"Attempt {attempt + 1} for {context_name} failed (took {duration:.2f}s), "
                    f"will retry: {e}"
                )
                continue
            else:
                # Not a retryable error
                logger.error(f"Non-retryable error in {context_name}: {e}")
                break

    # Re-raise the last exception
    if last_exception:
        raise last_exception
    else:
        raise RuntimeError(
            f"Unexpected error: no exception recorded for {context_name}"
        )


class RetryableMixin:
    """Mixin class to add retry functionality to transport classes.

    Note: Classes using this mixin must have a `config` attribute with retry settings.
    """

    _retry_config: Optional[RetryConfig]
    config: Any  # Must be provided by the class using this mixin

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._retry_config = None

    def _get_retry_config(self) -> RetryConfig:
        """Get retry configuration from MCP config."""
        if self._retry_config is None:
            self._retry_config = retry_config_from_mcp(self.config)
        return self._retry_config

    async def _retry_operation(
        self,
        func: Callable[..., Any],
        context_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Retry an operation with the configured retry policy."""
        return await retry_async(
            func, self._get_retry_config(), context_name, *args, **kwargs
        )


def wrap_connection_errors(func):
    """Decorator to wrap common connection errors as RetryableError."""

    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except (ConnectionError, OSError, TimeoutError) as e:
            raise ConnectionRetryableError(f"Connection error: {e}") from e
        except asyncio.TimeoutError as e:
            raise TimeoutRetryableError(f"Timeout error: {e}") from e

    return wrapper
