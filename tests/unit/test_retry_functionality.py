"""Tests for retry functionality and error handling in MCP transports."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from cicaddy.config.settings import MCPServerConfig
from cicaddy.mcp_client.retry import (
    ConnectionRetryableError,
    RetryConfig,
    ServerRetryableError,
    TimeoutRetryableError,
    calculate_delay,
    retry_async,
    retry_config_from_mcp,
    should_retry,
)
from cicaddy.mcp_client.transports.base import BaseMCPTransport


class TestRetryConfig:
    """Test cases for retry configuration."""

    def test_default_retry_config(self):
        """Test default retry configuration from centralized utils.retry."""
        config = RetryConfig()

        assert config.max_retries == 3  # Default from utils.retry.RetryConfig
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_factor == 2.0
        assert config.jitter is True
        assert config.retry_on_connection_error is True
        assert config.retry_on_timeout is True
        assert config.retry_on_5xx is True  # Renamed from retry_on_server_error

    def test_from_mcp_config(self):
        """Test creating RetryConfig from MCPServerConfig using factory function."""
        mcp_config = MCPServerConfig(
            name="test-server",
            protocol="http",
            endpoint="https://test.com",
            retry_count=5,
            retry_delay=2.0,
            retry_max_delay=120.0,
            retry_backoff_factor=1.5,
            retry_jitter=False,
            retry_on_connection_error=False,
            retry_on_timeout=False,
            retry_on_server_error=True,
        )

        config = retry_config_from_mcp(mcp_config)

        assert config.max_retries == 5
        assert config.initial_delay == 2.0
        assert config.max_delay == 120.0
        assert config.backoff_factor == 1.5
        assert config.jitter is False
        assert config.retry_on_connection_error is False
        assert config.retry_on_timeout is False
        assert config.retry_on_5xx is True  # Mapped from retry_on_server_error


class TestDelayCalculation:
    """Test cases for delay calculation with exponential backoff."""

    def test_calculate_delay_first_attempt(self):
        """Test that first attempt has no delay."""
        config = RetryConfig(initial_delay=2.0)
        delay = calculate_delay(0, config)
        assert delay == 0

    def test_calculate_delay_exponential_backoff(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(
            initial_delay=1.0, backoff_factor=2.0, max_delay=60.0, jitter=False
        )

        # Second attempt (attempt 1)
        delay1 = calculate_delay(1, config)
        assert delay1 == 1.0

        # Third attempt (attempt 2)
        delay2 = calculate_delay(2, config)
        assert delay2 == 2.0

        # Fourth attempt (attempt 3)
        delay3 = calculate_delay(3, config)
        assert delay3 == 4.0

    def test_calculate_delay_max_cap(self):
        """Test that delay is capped at maximum."""
        config = RetryConfig(
            initial_delay=1.0, backoff_factor=2.0, max_delay=5.0, jitter=False
        )

        # Large attempt number should be capped
        delay = calculate_delay(10, config)
        assert delay == 5.0

    def test_calculate_delay_with_jitter(self):
        """Test that jitter adds randomness to delay."""
        config = RetryConfig(
            initial_delay=4.0, backoff_factor=2.0, max_delay=60.0, jitter=True
        )

        # Generate multiple delays and check they vary due to jitter
        delays = [calculate_delay(1, config) for _ in range(10)]

        # All delays should be >= base delay (4.0) and <= base delay + 25% jitter (5.0)
        for delay in delays:
            assert 4.0 <= delay <= 5.0

        # Should have some variation due to jitter
        assert len(set(delays)) > 1


class TestShouldRetry:
    """Test cases for retry decision logic."""

    def test_should_retry_retryable_errors(self):
        """Test that RetryableError subclasses trigger retries."""
        config = RetryConfig()

        assert should_retry(ConnectionRetryableError("test"), config) is True
        assert should_retry(TimeoutRetryableError("test"), config) is True
        assert should_retry(ServerRetryableError("test"), config) is True

    def test_should_retry_connection_errors(self):
        """Test connection error retry logic."""
        config = RetryConfig(retry_on_connection_error=True)

        assert should_retry(ConnectionError("test"), config) is True
        assert should_retry(OSError("test"), config) is True
        assert should_retry(TimeoutError("test"), config) is True

    def test_should_not_retry_connection_errors_when_disabled(self):
        """Test that connection errors don't retry when disabled."""
        config = RetryConfig(retry_on_connection_error=False)

        assert should_retry(ConnectionError("test"), config) is False
        assert should_retry(OSError("test"), config) is False

    def test_should_retry_timeout_errors(self):
        """Test timeout error retry logic."""
        config = RetryConfig(retry_on_timeout=True)

        assert should_retry(TimeoutError("test"), config) is True
        assert should_retry(asyncio.TimeoutError("test"), config) is True

    def test_should_not_retry_timeout_errors_when_disabled(self):
        """Test that timeout errors don't retry when disabled."""
        config = RetryConfig(retry_on_timeout=False)

        assert should_retry(TimeoutError("test"), config) is False
        assert should_retry(asyncio.TimeoutError("test"), config) is False

    @pytest.mark.skipif(True, reason="httpx not required for core functionality")
    def test_should_retry_httpx_errors(self):
        """Test httpx error retry logic (when httpx is available)."""
        try:
            import httpx

            config = RetryConfig(
                retry_on_connection_error=True,
                retry_on_timeout=True,
                retry_on_server_error=True,
            )

            # Connection errors
            assert should_retry(httpx.ConnectError("test"), config) is True
            assert should_retry(httpx.NetworkError("test"), config) is True

            # Timeout errors
            assert should_retry(httpx.TimeoutException("test"), config) is True

            # Server errors (5xx)
            response_mock = Mock()
            response_mock.status_code = 500
            server_error = httpx.HTTPStatusError(
                "test", request=Mock(), response=response_mock
            )
            assert should_retry(server_error, config) is True

            # Client errors (4xx) should not retry
            response_mock.status_code = 404
            client_error = httpx.HTTPStatusError(
                "test", request=Mock(), response=response_mock
            )
            assert should_retry(client_error, config) is False

        except ImportError:
            pytest.skip("httpx not available")

    def test_should_retry_cancelled_errors_from_cancel_scope(self):
        """Test that CancelledError from anyio cancel scope triggers retries."""
        config = RetryConfig()

        # CancelledError with "cancel scope" pattern should trigger retry
        # This occurs when 504 Gateway Timeout happens during HTTP streaming cleanup
        assert (
            should_retry(
                asyncio.CancelledError("Cancelled by cancel scope 7fab7226ef90"), config
            )
            is True
        )

    def test_should_not_retry_clean_cancelled_errors(self):
        """Test that clean user-initiated CancelledError does NOT trigger retries."""
        config = RetryConfig()

        # Clean cancellations (no cancel scope pattern) should NOT retry
        assert should_retry(asyncio.CancelledError(), config) is False
        assert should_retry(asyncio.CancelledError("Task cancelled"), config) is False
        assert should_retry(asyncio.CancelledError("User cancelled"), config) is False

    def test_should_not_retry_non_retryable_errors(self):
        """Test that non-retryable errors don't trigger retries."""
        config = RetryConfig()

        assert should_retry(ValueError("test"), config) is False
        assert should_retry(RuntimeError("test"), config) is False
        assert should_retry(KeyError("test"), config) is False


class TestRetryAsync:
    """Test cases for async retry functionality."""

    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self):
        """Test successful operation on first attempt."""
        mock_func = AsyncMock(return_value="success")
        config = RetryConfig(max_retries=3)

        result = await retry_async(mock_func, config, "test_operation")

        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_success_after_retries(self):
        """Test successful operation after retries."""
        mock_func = AsyncMock(
            side_effect=[ConnectionError("fail"), TimeoutError("fail"), "success"]
        )
        config = RetryConfig(max_retries=3, initial_delay=0.01)  # Fast test

        result = await retry_async(mock_func, config, "test_operation")

        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test that retries are exhausted and last exception is raised."""
        mock_func = AsyncMock(side_effect=ConnectionError("persistent_failure"))
        config = RetryConfig(max_retries=2, initial_delay=0.01)

        with pytest.raises(ConnectionError, match="persistent_failure"):
            await retry_async(mock_func, config, "test_operation")

        assert mock_func.call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_retry_non_retryable_error(self):
        """Test that non-retryable errors are not retried."""
        mock_func = AsyncMock(side_effect=ValueError("not_retryable"))
        config = RetryConfig(max_retries=3)

        with pytest.raises(ValueError, match="not_retryable"):
            await retry_async(mock_func, config, "test_operation")

        assert mock_func.call_count == 1  # Only initial attempt

    @pytest.mark.asyncio
    async def test_retry_with_arguments(self):
        """Test retry functionality with function arguments."""
        mock_func = AsyncMock(side_effect=[ConnectionError("fail"), "success"])
        config = RetryConfig(max_retries=2, initial_delay=0.01)

        result = await retry_async(
            mock_func, config, "test_operation", "arg1", kwarg1="value1"
        )

        assert result == "success"
        assert mock_func.call_count == 2
        mock_func.assert_called_with("arg1", kwarg1="value1")


class MockMCPTransport(BaseMCPTransport):
    """Mock MCP transport for testing."""

    async def _connect_impl(self):
        """Mock connect implementation."""
        pass

    async def _disconnect_impl(self):
        """Mock disconnect implementation."""
        pass

    async def _send_request_impl(self, method, params=None):
        """Mock send request implementation."""
        return {"result": "success"}


class TestBaseMCPTransportRetry:
    """Test retry functionality in BaseMCPTransport."""

    @pytest.fixture
    def transport_config(self):
        """Transport configuration for testing."""
        return MCPServerConfig(
            name="test-transport",
            protocol="http",
            endpoint="https://test.com",
            retry_count=2,
            retry_delay=0.01,  # Fast test
        )

    @pytest.fixture
    def transport(self, transport_config):
        """Create test transport."""
        return MockMCPTransport(transport_config)

    @pytest.mark.asyncio
    async def test_connect_with_retry(self, transport):
        """Test connection with retry logic."""
        # Mock _connect_impl to fail once then succeed
        transport._connect_impl = AsyncMock(
            side_effect=[ConnectionError("connection failed"), None]
        )  # Success

        await transport.connect()

        assert transport.connected is True
        assert transport._connect_impl.call_count == 2

    @pytest.mark.asyncio
    async def test_send_request_with_retry(self, transport):
        """Test send request with retry logic."""
        transport.connected = True
        transport._send_request_impl = AsyncMock(
            side_effect=[TimeoutError("timeout"), {"result": "success"}]
        )

        result = await transport.send_request("test_method")

        assert result == {"result": "success"}
        assert transport._send_request_impl.call_count == 2

    @pytest.mark.asyncio
    async def test_health_check(self, transport):
        """Test connection health checking."""
        transport.connected = True
        transport._send_request_impl = AsyncMock(return_value={"tools": []})

        is_healthy = await transport.check_health()

        assert is_healthy is True
        assert transport.is_healthy is True
        transport._send_request_impl.assert_called_with("tools/list")

    @pytest.mark.asyncio
    async def test_health_check_failure(self, transport):
        """Test health check failure handling."""
        transport.connected = True
        transport._send_request_impl = AsyncMock(
            side_effect=ConnectionError("health check failed")
        )

        is_healthy = await transport.check_health()

        assert is_healthy is False
        assert transport.is_healthy is False

    @pytest.mark.asyncio
    async def test_list_tools_with_fallback(self, transport):
        """Test list_tools with error handling and fallback."""
        transport.connected = True
        transport.config.tools = ["tool1", "tool2"]
        transport.send_request = AsyncMock(side_effect=ConnectionError("failed"))

        tools = await transport.list_tools()

        # Should return configured tools as fallback
        assert len(tools) == 2
        assert tools[0]["name"] == "tool1"
        assert tools[1]["name"] == "tool2"
        assert all(tool["server"] == "test-transport" for tool in tools)

    @pytest.mark.asyncio
    async def test_call_tool_error_handling(self, transport):
        """Test call_tool error handling and health status updates."""
        transport.connected = True
        transport.send_request = AsyncMock(side_effect=RuntimeError("tool failed"))

        with pytest.raises(RuntimeError, match="tool failed"):
            await transport.call_tool("test_tool", {})

        # Connection should be marked as unhealthy
        assert transport._connection_health_ok is False


class SuggestedDelayError(Exception):
    """Exception with suggested_delay attribute for testing."""

    def __init__(self, message: str, suggested_delay: float = None):
        super().__init__(message)
        self.suggested_delay = suggested_delay


class TestSuggestedDelayRetry:
    """Test cases for API-suggested retry delay functionality."""

    @pytest.mark.asyncio
    async def test_retry_uses_suggested_delay(self):
        """Test that retry uses suggested_delay from exception when available."""
        # Create an exception with suggested_delay attribute
        call_count = 0
        suggested_delay = 5.0

        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise SuggestedDelayError(
                    "rate limited", suggested_delay=suggested_delay
                )
            return "success"

        config = RetryConfig(max_retries=3, initial_delay=1.0, max_delay=60.0)

        # Run with suggested delay - the actual sleep is done by asyncio.sleep
        # We just verify the function succeeds after retry
        result = await retry_async(failing_func, config, "test_operation")

        assert result == "success"
        assert call_count == 2  # Initial attempt + 1 retry

    @pytest.mark.asyncio
    async def test_retry_respects_max_delay_for_suggested(self):
        """Test that suggested_delay is capped at max_delay."""
        call_count = 0
        suggested_delay = 100.0  # Very long suggested delay

        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise SuggestedDelayError(
                    "rate limited", suggested_delay=suggested_delay
                )
            return "success"

        # max_delay is 10, so suggested_delay + 1 (101) should be capped to 10
        config = RetryConfig(max_retries=3, initial_delay=0.1, max_delay=10.0)

        result = await retry_async(failing_func, config, "test_operation")

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_falls_back_to_exponential_without_suggested(self):
        """Test that exponential backoff is used when no suggested_delay."""
        call_count = 0

        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("connection failed")  # No suggested_delay
            return "success"

        config = RetryConfig(max_retries=3, initial_delay=0.01, max_delay=60.0)

        result = await retry_async(failing_func, config, "test_operation")

        assert result == "success"
        assert call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_retry_with_zero_suggested_delay(self):
        """Test that zero suggested_delay falls back to exponential backoff."""
        call_count = 0

        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise SuggestedDelayError("rate limited", suggested_delay=0)
            return "success"

        config = RetryConfig(max_retries=3, initial_delay=0.01, max_delay=60.0)

        result = await retry_async(failing_func, config, "test_operation")

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_with_negative_suggested_delay(self):
        """Test that negative suggested_delay falls back to exponential backoff."""
        call_count = 0

        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise SuggestedDelayError("rate limited", suggested_delay=-5.0)
            return "success"

        config = RetryConfig(max_retries=3, initial_delay=0.01, max_delay=60.0)

        result = await retry_async(failing_func, config, "test_operation")

        assert result == "success"
        assert call_count == 2


class TestGeminiParseRetryDelay:
    """Test cases for parsing retry delay from Gemini error messages."""

    def test_parse_retry_delay_with_seconds(self):
        """Test parsing retry delay from standard Gemini 429 message."""
        from cicaddy.ai_providers.gemini import GeminiProvider

        provider = GeminiProvider({"model_id": "gemini-2.5-flash"})

        # Standard Gemini format: "Please retry in 35.088331533s"
        delay = provider._parse_retry_delay("Please retry in 35.088331533s")
        assert delay is not None
        assert abs(delay - 35.088331533) < 0.001

    def test_parse_retry_delay_without_s_suffix(self):
        """Test parsing retry delay without 's' suffix."""
        from cicaddy.ai_providers.gemini import GeminiProvider

        provider = GeminiProvider({"model_id": "gemini-2.5-flash"})

        delay = provider._parse_retry_delay("retry in 10.5")
        assert delay is not None
        assert abs(delay - 10.5) < 0.001

    def test_parse_retry_delay_integer(self):
        """Test parsing integer retry delay."""
        from cicaddy.ai_providers.gemini import GeminiProvider

        provider = GeminiProvider({"model_id": "gemini-2.5-flash"})

        delay = provider._parse_retry_delay("Please retry in 30s")
        assert delay is not None
        assert delay == 30.0

    def test_parse_retry_delay_case_insensitive(self):
        """Test parsing is case insensitive."""
        from cicaddy.ai_providers.gemini import GeminiProvider

        provider = GeminiProvider({"model_id": "gemini-2.5-flash"})

        delay = provider._parse_retry_delay("PLEASE RETRY IN 25.5S")
        assert delay is not None
        assert abs(delay - 25.5) < 0.001

    def test_parse_retry_delay_no_match(self):
        """Test parsing returns None when no pattern found."""
        from cicaddy.ai_providers.gemini import GeminiProvider

        provider = GeminiProvider({"model_id": "gemini-2.5-flash"})

        delay = provider._parse_retry_delay("Rate limit exceeded")
        assert delay is None

    def test_parse_retry_delay_empty_string(self):
        """Test parsing returns None for empty string."""
        from cicaddy.ai_providers.gemini import GeminiProvider

        provider = GeminiProvider({"model_id": "gemini-2.5-flash"})

        delay = provider._parse_retry_delay("")
        assert delay is None

    def test_parse_retry_delay_full_429_message(self):
        """Test parsing from a full 429 error message."""
        from cicaddy.ai_providers.gemini import GeminiProvider

        provider = GeminiProvider({"model_id": "gemini-2.5-flash"})

        full_message = (
            "429 You exceeded your current quota, please check your plan and billing "
            "details. For more information on this error, head to "
            "https://ai.google.dev/gemini-api/docs/troubleshooting#billing-quota-errors. "
            "Please retry in 35.088331533s"
        )
        delay = provider._parse_retry_delay(full_message)
        assert delay is not None
        assert abs(delay - 35.088331533) < 0.001
