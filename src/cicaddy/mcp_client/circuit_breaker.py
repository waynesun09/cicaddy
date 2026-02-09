"""Circuit breaker pattern implementation for MCP client fault tolerance."""

import asyncio
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service is back


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[BaseException] = Exception,
        success_threshold: int = 3,
        timeout: float = 10.0,
        monitor_calls: int = 10,
        success_rate_threshold: float = 50.0,
    ):
        """
        Initialize circuit breaker configuration.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before transitioning to half-open
            expected_exception: Exception type to count as failure
            success_threshold: Successful calls needed in half-open to close circuit
            timeout: Timeout for individual calls
            monitor_calls: Number of calls to monitor for success rate
            success_rate_threshold: Minimum success rate to keep circuit closed
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception: Type[BaseException] = expected_exception
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.monitor_calls = monitor_calls
        self.success_rate_threshold = success_rate_threshold


class CircuitBreakerStats:
    """Statistics tracking for circuit breaker."""

    def __init__(self, monitor_calls: int = 10):
        self.monitor_calls = monitor_calls
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None
        self.state_changes = 0
        self.created_at = time.time()

        # Rolling window for success rate calculation
        self._recent_calls: List[Tuple[float, bool]] = []

    def record_success(self):
        """Record a successful call."""
        current_time = time.time()
        self.total_calls += 1
        self.successful_calls += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = current_time

        # Update rolling window
        self._recent_calls.append((current_time, True))
        self._cleanup_old_calls()

    def record_failure(self):
        """Record a failed call."""
        current_time = time.time()
        self.total_calls += 1
        self.failed_calls += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = current_time

        # Update rolling window
        self._recent_calls.append((current_time, False))
        self._cleanup_old_calls()

    def record_state_change(self):
        """Record a state change."""
        self.state_changes += 1

    def get_success_rate(self) -> float:
        """Get current success rate as percentage."""
        if not self._recent_calls:
            return 100.0

        successful = sum(1 for _, success in self._recent_calls if success)
        total = len(self._recent_calls)
        return (successful / total) * 100.0

    def _cleanup_old_calls(self):
        """Remove old calls to maintain rolling window size."""
        if len(self._recent_calls) > self.monitor_calls:
            self._recent_calls = self._recent_calls[-self.monitor_calls :]

    @property
    def overall_success_rate(self) -> float:
        """Get overall success rate since creation."""
        if self.total_calls == 0:
            return 100.0
        return (self.successful_calls / self.total_calls) * 100.0

    @property
    def uptime(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self.created_at


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.

        Args:
            name: Name for logging and identification
            config: Configuration object, uses defaults if None
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats(self.config.monitor_calls)
        self._state_changed_at = time.time()
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function call through the circuit breaker.

        Args:
            func: Function to execute
            *args, **kwargs: Arguments for the function

        Returns:
            Result of the function call

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception from the wrapped function
        """
        async with self._lock:
            await self._update_state()

            if self.state == CircuitState.OPEN:
                logger.warning(f"Circuit breaker {self.name} is OPEN, failing fast")
                raise CircuitBreakerError(f"Circuit breaker {self.name} is open")

        # Execute the function with timeout
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs), timeout=self.config.timeout
                )
            else:
                result = func(*args, **kwargs)

            # Record success
            async with self._lock:
                self.stats.record_success()
                await self._check_half_open_success()

            execution_time = time.time() - start_time
            logger.debug(
                f"Circuit breaker {self.name} call succeeded in {execution_time:.3f}s"
            )
            return result

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.warning(
                f"Circuit breaker {self.name} call timed out after {execution_time:.3f}s"
            )
            async with self._lock:
                self.stats.record_failure()
                await self._check_failure_threshold()
            raise

        except self.config.expected_exception as e:
            execution_time = time.time() - start_time
            logger.warning(
                f"Circuit breaker {self.name} call failed after {execution_time:.3f}s: {e}"
            )
            async with self._lock:
                self.stats.record_failure()
                await self._check_failure_threshold()
            raise

        except Exception as e:
            # Unexpected exceptions are not counted as circuit breaker failures
            execution_time = time.time() - start_time
            logger.error(
                f"Circuit breaker {self.name} unexpected error after {execution_time:.3f}s: {e}"
            )
            raise

    async def _update_state(self):
        """Update circuit breaker state based on current conditions."""
        current_time = time.time()

        if self.state == CircuitState.OPEN:
            # Check if we should transition to half-open
            if current_time - self._state_changed_at >= self.config.recovery_timeout:
                await self._transition_to_half_open()

        elif self.state == CircuitState.CLOSED:
            # Check if success rate is too low
            if (
                self.stats.total_calls >= self.config.monitor_calls
                and self.stats.get_success_rate() < self.config.success_rate_threshold
            ):
                await self._transition_to_open()

    async def _check_failure_threshold(self):
        """Check if failure threshold is exceeded and open circuit if needed."""
        if (
            self.state == CircuitState.CLOSED
            and self.stats.consecutive_failures >= self.config.failure_threshold
        ):
            await self._transition_to_open()
        elif (
            self.state == CircuitState.HALF_OPEN
            and self.stats.consecutive_failures
            >= 1  # Any failure in half-open goes back to open
        ):
            await self._transition_to_open()

    async def _check_half_open_success(self):
        """Check if half-open state has enough successes to close circuit."""
        if (
            self.state == CircuitState.HALF_OPEN
            and self.stats.consecutive_successes >= self.config.success_threshold
        ):
            await self._transition_to_closed()

    async def _transition_to_open(self):
        """Transition circuit breaker to OPEN state."""
        old_state = self.state
        self.state = CircuitState.OPEN
        self._state_changed_at = time.time()
        self.stats.record_state_change()

        logger.warning(
            f"Circuit breaker {self.name} transitioned from {old_state.value} to OPEN "
            f"(failures: {self.stats.consecutive_failures}, "
            f"success_rate: {self.stats.get_success_rate():.1f}%)"
        )

    async def _transition_to_half_open(self):
        """Transition circuit breaker to HALF_OPEN state."""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self._state_changed_at = time.time()
        self.stats.record_state_change()

        # Reset consecutive counters for half-open testing
        self.stats.consecutive_failures = 0
        self.stats.consecutive_successes = 0

        logger.info(
            f"Circuit breaker {self.name} transitioned from {old_state.value} to HALF_OPEN (testing recovery)"
        )

    async def _transition_to_closed(self):
        """Transition circuit breaker to CLOSED state."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self._state_changed_at = time.time()
        self.stats.record_state_change()

        logger.info(
            f"Circuit breaker {self.name} transitioned from {old_state.value} to CLOSED "
            f"(successes: {self.stats.consecutive_successes})"
        )

    async def reset(self):
        """Reset circuit breaker to CLOSED state (for manual recovery)."""
        async with self._lock:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self._state_changed_at = time.time()
            self.stats.consecutive_failures = 0
            self.stats.consecutive_successes = 0
            self.stats.record_state_change()

            logger.info(
                f"Circuit breaker {self.name} manually reset from {old_state.value} to CLOSED"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.stats.total_calls,
            "successful_calls": self.stats.successful_calls,
            "failed_calls": self.stats.failed_calls,
            "consecutive_failures": self.stats.consecutive_failures,
            "consecutive_successes": self.stats.consecutive_successes,
            "current_success_rate": self.stats.get_success_rate(),
            "overall_success_rate": self.stats.overall_success_rate,
            "state_changes": self.stats.state_changes,
            "uptime": self.stats.uptime,
            "time_in_current_state": time.time() - self._state_changed_at,
            "last_failure_time": self.stats.last_failure_time,
            "last_success_time": self.stats.last_success_time,
        }

    @property
    def is_available(self) -> bool:
        """Check if circuit breaker is available for calls."""
        return self.state != CircuitState.OPEN


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._default_config = CircuitBreakerConfig()

    def get_breaker(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """
        Get or create a circuit breaker.

        Args:
            name: Circuit breaker name
            config: Configuration, uses default if None

        Returns:
            CircuitBreaker instance
        """
        if name not in self._breakers:
            breaker_config = config or self._default_config
            self._breakers[name] = CircuitBreaker(name, breaker_config)
            logger.info(f"Created circuit breaker: {name}")

        return self._breakers[name]

    def remove_breaker(self, name: str):
        """Remove a circuit breaker from registry."""
        if name in self._breakers:
            del self._breakers[name]
            logger.info(f"Removed circuit breaker: {name}")

    async def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            await breaker.reset()
        logger.info("Reset all circuit breakers")

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}

    def get_available_breakers(self) -> Dict[str, CircuitBreaker]:
        """Get all circuit breakers that are available (not open)."""
        return {
            name: breaker
            for name, breaker in self._breakers.items()
            if breaker.is_available
        }

    def get_unavailable_breakers(self) -> Dict[str, CircuitBreaker]:
        """Get all circuit breakers that are unavailable (open)."""
        return {
            name: breaker
            for name, breaker in self._breakers.items()
            if not breaker.is_available
        }


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()


def circuit_breaker(
    name: str, config: Optional[CircuitBreakerConfig] = None
) -> Callable[[Callable], Callable]:
    """
    Decorator to wrap functions with circuit breaker protection.

    Args:
        name: Circuit breaker name
        config: Circuit breaker configuration

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        breaker = circuit_breaker_registry.get_breaker(name, config)

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await breaker.call(func, *args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        setattr(
            wrapper, "_circuit_breaker", breaker
        )  # Use setattr for dynamic attribute
        return wrapper

    return decorator
