"""Reusable exponential retry utilities for HTTP requests and async operations.

This module provides a generic retry mechanism with exponential backoff that can be used
by AI providers, HTTP clients, and other components that need resilient request handling.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Set, Tuple, Type

import httpx

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior with exponential backoff.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay between retries in seconds (default: 1.0)
        max_delay: Maximum delay between individual retries in seconds (default: 60.0)
        max_retry_time_seconds: Maximum total time for all retry attempts (default: 300.0, 0=disabled)
        backoff_factor: Exponential backoff multiplier (default: 2.0)
        jitter: Add random jitter to retry delays (default: True)
        retry_on_5xx: Retry on 5xx HTTP status codes (default: True)
        retry_on_timeout: Retry on timeout errors (default: True)
        retry_on_connection_error: Retry on connection errors (default: True)
        retryable_status_codes: Set of HTTP status codes to retry on (default: 500-599)
        retryable_exceptions: Tuple of exception types to retry on
    """

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    max_retry_time_seconds: float = 300.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retry_on_5xx: bool = True
    retry_on_timeout: bool = True
    retry_on_connection_error: bool = True
    retryable_status_codes: Set[int] = field(
        default_factory=lambda: set(range(500, 600))
    )
    retryable_exceptions: Tuple[Type[Exception], ...] = field(
        default_factory=lambda: (
            TimeoutError,
            asyncio.TimeoutError,
            ConnectionError,
            OSError,
        )
    )


class RetryTimeoutError(Exception):
    """Raised when retry attempts exceed the maximum retry time limit."""

    def __init__(self, message: str, elapsed_time: float, attempts: int):
        super().__init__(message)
        self.elapsed_time = elapsed_time
        self.attempts = attempts


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(
        self, message: str, attempts: int, last_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """
    Calculate the delay for a retry attempt using exponential backoff with optional jitter.

    Args:
        attempt: The current attempt number (0-based, 0 = first retry after initial attempt)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    if attempt <= 0:
        return 0  # No delay for first attempt

    # Calculate exponential backoff delay
    delay = config.initial_delay * (config.backoff_factor ** (attempt - 1))

    # Cap at maximum delay
    delay = min(delay, config.max_delay)

    # Add jitter if enabled
    if config.jitter:
        # Add random jitter between 0-25% of the delay
        jitter_amount = delay * 0.25 * random.random()  # nosec B311 - jitter, not crypto
        delay += jitter_amount

    return delay


def is_retryable_status_code(status_code: int, config: RetryConfig) -> bool:
    """Check if an HTTP status code should trigger a retry."""
    if config.retry_on_5xx and 500 <= status_code < 600:
        return True
    return status_code in config.retryable_status_codes


def is_retryable_exception(exception: Exception, config: RetryConfig) -> bool:
    """Check if an exception should trigger a retry."""
    # Check for httpx-specific exceptions
    if isinstance(exception, httpx.HTTPStatusError):
        return is_retryable_status_code(exception.response.status_code, config)

    if config.retry_on_timeout:
        timeout_types = (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException)
        if isinstance(exception, timeout_types):
            return True

    if config.retry_on_connection_error:
        connection_types = (
            ConnectionError,
            OSError,
            httpx.ConnectError,
            httpx.NetworkError,
        )
        if isinstance(exception, connection_types):
            return True

    # Check configured retryable exceptions
    if isinstance(exception, config.retryable_exceptions):
        return True

    # Check error message for common retryable patterns
    error_str = str(exception).lower()
    retryable_patterns = [
        "rate limit",
        "too many requests",
        "service unavailable",
        "gateway timeout",
        "502",
        "503",
        "504",
    ]
    return any(pattern in error_str for pattern in retryable_patterns)


async def retry_async(
    func: Callable[..., Any],
    config: Optional[RetryConfig] = None,
    context_name: str = "operation",
    *args,
    **kwargs,
) -> Any:
    """
    Retry an async function with exponential backoff and time limit.

    Args:
        func: The async function to retry
        config: Retry configuration (uses defaults if not provided)
        context_name: Name of the operation for logging
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the successful function call

    Raises:
        RetryTimeoutError: If max_retry_time_seconds is exceeded
        RetryExhaustedError: If all retries are exhausted
        Exception: The last exception if not retryable
    """
    if config is None:
        config = RetryConfig()

    last_exception: Optional[Exception] = None
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
                raise RetryTimeoutError(
                    f"Max retry time exceeded for {context_name}",
                    elapsed_time=elapsed_total,
                    attempts=attempt,
                ) from last_exception

        try:
            if attempt > 0:
                # Check if last exception has API-suggested retry delay
                suggested = (
                    getattr(last_exception, "suggested_delay", None)
                    if last_exception
                    else None
                )
                if suggested and suggested > 0:
                    # Use API-suggested delay with 1s safety margin, capped at max_delay
                    delay = min(suggested + 1.0, config.max_delay)
                    logger.info(
                        f"Using API-suggested delay of {delay:.2f}s "
                        f"(suggested: {suggested:.2f}s) for {context_name}"
                    )
                else:
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
                        delay = max(0.1, remaining_time * 0.8)
                        logger.info(
                            f"Reduced retry delay to {delay:.2f}s due to time limit for {context_name}"
                        )

                logger.info(
                    f"Retrying {context_name} (attempt {attempt + 1}/{config.max_retries + 1}) "
                    f"after {delay:.2f}s delay"
                )
                await asyncio.sleep(delay)

            # Attempt the operation
            attempt_start_time = time.time()
            result = await func(*args, **kwargs)

            if attempt > 0:
                duration = time.time() - attempt_start_time
                total_elapsed = time.time() - retry_start_time
                logger.info(
                    f"Successfully retried {context_name} after {attempt} attempts "
                    f"(attempt took {duration:.2f}s, total {total_elapsed:.2f}s)"
                )

            return result

        except Exception as e:
            last_exception = e
            duration = (
                time.time() - attempt_start_time
                if "attempt_start_time" in locals()
                else 0
            )

            # Log if exception has suggested retry delay for use in next attempt
            suggested_delay = getattr(e, "suggested_delay", None)
            if suggested_delay and suggested_delay > 0:
                logger.debug(
                    f"Exception has suggested_delay={suggested_delay:.2f}s for {context_name}"
                )

            if attempt == config.max_retries:
                total_elapsed = time.time() - retry_start_time
                logger.error(
                    f"Final attempt for {context_name} failed after {config.max_retries} retries "
                    f"(attempt took {duration:.2f}s, total {total_elapsed:.2f}s): {e}"
                )
                break

            if is_retryable_exception(e, config):
                logger.warning(
                    f"Attempt {attempt + 1} for {context_name} failed (took {duration:.2f}s), "
                    f"will retry: {e}"
                )
                continue
            else:
                logger.error(f"Non-retryable error in {context_name}: {e}")
                raise

    # All retries exhausted
    if last_exception:
        raise last_exception
    raise RetryExhaustedError(
        f"All {config.max_retries + 1} attempts failed for {context_name}",
        attempts=config.max_retries + 1,
        last_exception=last_exception,
    )


def retry_sync(
    func: Callable[..., Any],
    config: Optional[RetryConfig] = None,
    context_name: str = "operation",
    *args,
    **kwargs,
) -> Any:
    """
    Retry a synchronous function with exponential backoff and time limit.

    Args:
        func: The synchronous function to retry
        config: Retry configuration (uses defaults if not provided)
        context_name: Name of the operation for logging
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the successful function call

    Raises:
        RetryTimeoutError: If max_retry_time_seconds is exceeded
        RetryExhaustedError: If all retries are exhausted
        Exception: The last exception if not retryable
    """
    if config is None:
        config = RetryConfig()

    last_exception: Optional[Exception] = None
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
                raise RetryTimeoutError(
                    f"Max retry time exceeded for {context_name}",
                    elapsed_time=elapsed_total,
                    attempts=attempt,
                ) from last_exception

        try:
            if attempt > 0:
                # Check if last exception has API-suggested retry delay
                suggested = (
                    getattr(last_exception, "suggested_delay", None)
                    if last_exception
                    else None
                )
                if suggested and suggested > 0:
                    # Use API-suggested delay with 1s safety margin, capped at max_delay
                    delay = min(suggested + 1.0, config.max_delay)
                    logger.info(
                        f"Using API-suggested delay of {delay:.2f}s "
                        f"(suggested: {suggested:.2f}s) for {context_name}"
                    )
                else:
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
                        delay = max(0.1, remaining_time * 0.8)
                        logger.info(
                            f"Reduced retry delay to {delay:.2f}s due to time limit for {context_name}"
                        )

                logger.info(
                    f"Retrying {context_name} (attempt {attempt + 1}/{config.max_retries + 1}) "
                    f"after {delay:.2f}s delay"
                )
                time.sleep(delay)

            # Attempt the operation
            attempt_start_time = time.time()
            result = func(*args, **kwargs)

            if attempt > 0:
                duration = time.time() - attempt_start_time
                total_elapsed = time.time() - retry_start_time
                logger.info(
                    f"Successfully retried {context_name} after {attempt} attempts "
                    f"(attempt took {duration:.2f}s, total {total_elapsed:.2f}s)"
                )

            return result

        except Exception as e:
            last_exception = e
            duration = (
                time.time() - attempt_start_time
                if "attempt_start_time" in locals()
                else 0
            )

            # Log if exception has suggested retry delay for use in next attempt
            suggested_delay = getattr(e, "suggested_delay", None)
            if suggested_delay and suggested_delay > 0:
                logger.debug(
                    f"Exception has suggested_delay={suggested_delay:.2f}s for {context_name}"
                )

            if attempt == config.max_retries:
                total_elapsed = time.time() - retry_start_time
                logger.error(
                    f"Final attempt for {context_name} failed after {config.max_retries} retries "
                    f"(attempt took {duration:.2f}s, total {total_elapsed:.2f}s): {e}"
                )
                break

            if is_retryable_exception(e, config):
                logger.warning(
                    f"Attempt {attempt + 1} for {context_name} failed (took {duration:.2f}s), "
                    f"will retry: {e}"
                )
                continue
            else:
                logger.error(f"Non-retryable error in {context_name}: {e}")
                raise

    # All retries exhausted
    if last_exception:
        raise last_exception
    raise RetryExhaustedError(
        f"All {config.max_retries + 1} attempts failed for {context_name}",
        attempts=config.max_retries + 1,
        last_exception=last_exception,
    )


class RetryableHTTPClient:
    """HTTP client wrapper with built-in retry logic for resilient requests."""

    def __init__(
        self,
        client: Optional[httpx.AsyncClient] = None,
        config: Optional[RetryConfig] = None,
    ):
        """
        Initialize the retryable HTTP client.

        Args:
            client: Optional httpx.AsyncClient to use (creates one if not provided)
            config: Retry configuration (uses defaults if not provided)
        """
        self._client = client
        self._owns_client = client is None
        self.config = config or RetryConfig()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    async def request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> httpx.Response:
        """
        Make an HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional arguments passed to httpx.AsyncClient.request

        Returns:
            httpx.Response object

        Raises:
            RetryTimeoutError: If max retry time exceeded
            httpx.HTTPStatusError: If response has error status after all retries
        """
        client = await self._get_client()

        async def _make_request():
            response = await client.request(method, url, **kwargs)
            # Raise for 5xx errors to trigger retry
            if self.config.retry_on_5xx and 500 <= response.status_code < 600:
                response.raise_for_status()
            return response

        return await retry_async(
            _make_request,
            config=self.config,
            context_name=f"{method} {url}",
        )

    async def get(self, url: str, **kwargs) -> httpx.Response:
        """Make a GET request with retry logic."""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> httpx.Response:
        """Make a POST request with retry logic."""
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs) -> httpx.Response:
        """Make a PUT request with retry logic."""
        return await self.request("PUT", url, **kwargs)

    async def patch(self, url: str, **kwargs) -> httpx.Response:
        """Make a PATCH request with retry logic."""
        return await self.request("PATCH", url, **kwargs)

    async def delete(self, url: str, **kwargs) -> httpx.Response:
        """Make a DELETE request with retry logic."""
        return await self.request("DELETE", url, **kwargs)

    async def close(self) -> None:
        """Close the HTTP client if we own it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "RetryableHTTPClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
