"""Tests for Phase 2 enterprise features."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from cicaddy.config.advanced_settings import AdvancedMCPConfig, Environment
from cicaddy.config.settings import MCPServerConfig
from cicaddy.mcp_client.cache import CacheMiddleware, MCPCache
from cicaddy.mcp_client.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
)
from cicaddy.mcp_client.connection_pool import MCPConnectionPool
from cicaddy.mcp_client.enterprise_client import EnterpriseMCPClient
from cicaddy.mcp_client.metrics import MCPMetrics
from cicaddy.mcp_client.telemetry import GitLabActionTelemetry


class TestConnectionPool:
    """Test cases for connection pool functionality."""

    @pytest.fixture
    def pool_config(self):
        """Connection pool configuration."""
        return {
            "pool_size": 3,
            "max_retries": 2,
            "load_balancing_strategy": "round_robin",
            "health_check_interval": 10.0,
            "cleanup_interval": 60.0,
            "max_idle_time": 120.0,
        }

    @pytest.fixture
    def mock_client_factory(self):
        """Mock client factory."""

        def factory():
            client = AsyncMock()
            client.config.name = "test-server"
            client.connected = True
            client.transport.check_health.return_value = True
            return client

        return factory

    @pytest.mark.asyncio
    async def test_pool_initialization(self, pool_config):
        """Test connection pool initialization."""
        pool = MCPConnectionPool(**pool_config)

        assert pool.pool_size == 3
        assert pool.max_retries == 2
        assert pool.load_balancing_strategy == "round_robin"
        assert not pool._running

    @pytest.mark.asyncio
    async def test_pool_start_stop(self, pool_config):
        """Test pool start and stop."""
        pool = MCPConnectionPool(**pool_config)

        await pool.start()
        assert pool._running

        await pool.stop()
        assert not pool._running

    @pytest.mark.asyncio
    async def test_get_connection(self, pool_config, mock_client_factory):
        """Test getting connection from pool."""
        pool = MCPConnectionPool(**pool_config)
        await pool.start()

        try:
            # Get first connection (should create new)
            client = await pool.get_connection("test-server", mock_client_factory)
            assert client is not None
            assert client.config.name == "test-server"

            # Return connection
            await pool.return_connection(client)

            # Get connection again (should reuse)
            client2 = await pool.get_connection("test-server", mock_client_factory)
            assert client2 is not None

        finally:
            await pool.stop()

    @pytest.mark.asyncio
    async def test_execute_with_pool(self, pool_config, mock_client_factory):
        """Test executing operation with pool."""
        pool = MCPConnectionPool(**pool_config)
        await pool.start()

        try:

            async def mock_operation(client, tool_name, args):
                return {"result": f"called {tool_name} with {args}"}

            result = await pool.execute_with_pool(
                "test-server",
                mock_client_factory,
                mock_operation,
                "test_tool",
                {"param": "value"},
            )

            assert result["result"] == "called test_tool with {'param': 'value'}"

        finally:
            await pool.stop()

    @pytest.mark.asyncio
    async def test_pool_stats(self, pool_config, mock_client_factory):
        """Test pool statistics."""
        pool = MCPConnectionPool(**pool_config)
        await pool.start()

        try:
            # Get connection to populate stats
            client = await pool.get_connection("test-server", mock_client_factory)

            stats = pool.get_pool_stats("test-server")
            assert "server_name" in stats
            assert "pool_size" in stats
            assert "available_connections" in stats
            assert stats["server_name"] == "test-server"

            await pool.return_connection(client)

        finally:
            await pool.stop()


class TestCircuitBreaker:
    """Test cases for circuit breaker functionality."""

    @pytest.fixture
    def breaker_config(self):
        """Circuit breaker configuration."""
        return CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,  # Short timeout for testing
            success_threshold=2,
            timeout=0.5,
            monitor_calls=5,
            success_rate_threshold=60.0,
        )

    @pytest.mark.asyncio
    async def test_circuit_breaker_initialization(self, breaker_config):
        """Test circuit breaker initialization."""
        breaker = CircuitBreaker("test-breaker", breaker_config)

        assert breaker.name == "test-breaker"
        assert breaker.config.failure_threshold == 3
        assert breaker.state.value == "closed"
        assert breaker.is_available

    @pytest.mark.asyncio
    async def test_successful_call(self, breaker_config):
        """Test successful function call through circuit breaker."""
        breaker = CircuitBreaker("test-breaker", breaker_config)

        async def successful_func():
            return "success"

        result = await breaker.call(successful_func)
        assert result == "success"
        assert breaker.stats.successful_calls == 1
        assert breaker.stats.consecutive_successes == 1

    @pytest.mark.asyncio
    async def test_failed_call(self, breaker_config):
        """Test failed function call through circuit breaker."""
        breaker = CircuitBreaker("test-breaker", breaker_config)

        async def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await breaker.call(failing_func)

        assert breaker.stats.failed_calls == 1
        assert breaker.stats.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self, breaker_config):
        """Test circuit opens after threshold failures."""
        breaker = CircuitBreaker("test-breaker", breaker_config)

        async def failing_func():
            raise ValueError("Test error")

        # Fail enough times to open circuit
        for _ in range(breaker_config.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(failing_func)

        # Circuit should be open now
        assert breaker.state.value == "open"
        assert not breaker.is_available

        # Next call should fail fast
        with pytest.raises(CircuitBreakerError):
            await breaker.call(failing_func)

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open(self, breaker_config):
        """Test circuit transitions to half-open after recovery timeout."""
        breaker = CircuitBreaker("test-breaker", breaker_config)

        async def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        for _ in range(breaker_config.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(failing_func)

        assert breaker.state.value == "open"

        # Wait for recovery timeout
        await asyncio.sleep(breaker_config.recovery_timeout + 0.1)

        # Call should transition to half-open
        async def successful_func():
            return "success"

        result = await breaker.call(successful_func)
        assert result == "success"
        assert breaker.state.value == "half_open"

    @pytest.mark.asyncio
    async def test_circuit_closes_after_successes_in_half_open(self, breaker_config):
        """Test circuit closes after enough successes in half-open state."""
        breaker = CircuitBreaker("test-breaker", breaker_config)

        # Manually set to half-open
        breaker.state = breaker.state.HALF_OPEN

        async def successful_func():
            return "success"

        # Succeed enough times to close circuit
        for _ in range(breaker_config.success_threshold):
            result = await breaker.call(successful_func)
            assert result == "success"

        assert breaker.state.value == "closed"
        assert breaker.is_available

    @pytest.mark.asyncio
    async def test_circuit_breaker_stats(self, breaker_config):
        """Test circuit breaker statistics."""
        breaker = CircuitBreaker("test-breaker", breaker_config)

        async def successful_func():
            return "success"

        await breaker.call(successful_func)

        stats = breaker.get_stats()
        assert stats["name"] == "test-breaker"
        assert stats["state"] == "closed"
        assert stats["total_calls"] == 1
        assert stats["successful_calls"] == 1
        assert stats["failed_calls"] == 0


class TestCache:
    """Test cases for caching functionality."""

    @pytest.fixture
    def cache(self):
        """Cache instance."""
        return MCPCache(default_ttl=60.0, namespace="test")

    @pytest.mark.asyncio
    async def test_cache_miss_and_set(self, cache):
        """Test cache miss and set operation."""
        server_name = "test-server"
        method = "tools/list"
        params = {"filter": "test"}

        # Should be cache miss initially
        result = await cache.get_cached_response(server_name, method, params)
        assert result is None

        # Cache a response
        response = {"tools": [{"name": "test_tool"}]}
        success = await cache.cache_response(server_name, method, params, response)
        assert success

        # Should be cache hit now
        cached_result = await cache.get_cached_response(server_name, method, params)
        assert cached_result == response

    @pytest.mark.asyncio
    async def test_cache_expiration(self, cache):
        """Test cache entry expiration."""
        server_name = "test-server"
        method = "tools/list"

        # Cache with very short TTL
        response = {"tools": []}
        await cache.cache_response(server_name, method, None, response, ttl=0.1)

        # Should be available immediately
        cached_result = await cache.get_cached_response(server_name, method, None)
        assert cached_result == response

        # Wait for expiration
        await asyncio.sleep(0.2)

        # Should be expired now
        expired_result = await cache.get_cached_response(server_name, method, None)
        assert expired_result is None

    def test_cache_key_generation(self, cache):
        """Test cache key generation consistency."""
        server_name = "test-server"
        method = "tools/list"
        params1 = {"a": 1, "b": 2}
        params2 = {"b": 2, "a": 1}  # Same params, different order

        key1 = cache._make_key(server_name, method, params1)
        key2 = cache._make_key(server_name, method, params2)

        # Should generate same key for same params regardless of order
        assert key1 == key2

    def test_is_cacheable(self, cache):
        """Test cacheable method detection."""
        # Cacheable methods
        assert cache.is_cacheable("tools/list")
        assert cache.is_cacheable("resources/list")
        assert cache.is_cacheable("prompts/list")

        # Non-cacheable methods
        assert not cache.is_cacheable("tools/call")
        assert not cache.is_cacheable("notifications/send")

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, cache):
        """Test cache invalidation."""
        server_name = "test-server"
        method = "tools/list"

        # Cache some responses
        await cache.cache_response(server_name, method, None, {"tools": ["tool1"]})
        await cache.cache_response(server_name, "other/method", None, {"data": "test"})

        # Verify cached
        cached = await cache.get_cached_response(server_name, method, None)
        assert cached is not None

        # Invalidate specific method
        invalidated = await cache.invalidate(server_name, method)
        assert invalidated > 0

        # Should be gone now
        cached_after = await cache.get_cached_response(server_name, method, None)
        assert cached_after is None

    @pytest.mark.asyncio
    async def test_cache_middleware(self, cache):
        """Test cache middleware functionality."""
        middleware = CacheMiddleware(cache)

        call_count = 0

        async def mock_operation():
            nonlocal call_count
            call_count += 1
            return {"result": f"call_{call_count}"}

        # First call should execute operation
        result1 = await middleware.execute_with_cache(
            "test-server", "tools/list", None, mock_operation
        )
        assert result1["result"] == "call_1"
        assert call_count == 1

        # Second call should use cache
        result2 = await middleware.execute_with_cache(
            "test-server", "tools/list", None, mock_operation
        )
        assert result2["result"] == "call_1"  # Same result from cache
        assert call_count == 1  # Operation not called again

        # Force refresh should execute operation again
        result3 = await middleware.execute_with_cache(
            "test-server", "tools/list", None, mock_operation, force_refresh=True
        )
        assert result3["result"] == "call_2"
        assert call_count == 2


class TestMetrics:
    """Test cases for metrics functionality with OpenTelemetry."""

    def test_mcp_metrics_integration(self):
        """Test MCP metrics integration with OpenTelemetry."""
        metrics = MCPMetrics()

        # Record some operations
        metrics.record_request("server1", "tools/list", success=True, duration=0.1)
        metrics.record_request("server1", "tools/call", success=False, duration=0.5)
        metrics.record_connection("server1", success=True)

        # Get summary stats
        summary = metrics.get_summary_stats()
        assert "total_requests" in summary
        assert "success_rate" in summary
        assert "active_connections" in summary
        assert summary["total_requests"] == 2
        assert summary["successful_requests"] == 1
        assert summary["failed_requests"] == 1

        # Get all metrics (now returns different structure for OpenTelemetry)
        all_metrics = metrics.get_all_metrics()
        assert "request_stats" in all_metrics
        assert "connection_stats" in all_metrics
        assert "cache_stats" in all_metrics

    def test_telemetry_initialization(self):
        """Test OpenTelemetry telemetry initialization."""
        # Test with mock config
        from cicaddy.config.advanced_settings import TelemetrySettings

        config = TelemetrySettings(
            enabled=True, service_name="test-service", environment="testing"
        )

        telemetry = GitLabActionTelemetry(config)
        assert telemetry.config == config

        # Test stats
        stats = telemetry.get_stats()
        assert "enabled" in stats
        assert "service_name" in stats

    def test_metrics_export_summary(self):
        """Test metrics export for GitLab Action logs."""
        metrics = MCPMetrics()

        # Record some test data
        metrics.record_request("server1", "tools/list", success=True, duration=0.1)
        metrics.record_cache_hit("server1", "tools/list")

        # Get summary export
        summary = metrics.export_opentelemetry_summary()
        assert "MCP Action Telemetry Summary" in summary
        assert "Total Requests" in summary
        assert "Success Rate" in summary


class TestAdvancedConfiguration:
    """Test cases for advanced configuration management."""

    def test_config_initialization(self):
        """Test advanced configuration initialization."""
        config = AdvancedMCPConfig()

        assert config.environment == Environment.DEVELOPMENT
        assert config.enable_retry is True
        assert config.enable_circuit_breaker is True
        assert config.enable_connection_pool is True
        assert config.enable_cache is True
        assert config.enable_telemetry is True

    def test_environment_overrides(self):
        """Test environment-specific configuration overrides."""
        # Production config
        prod_config = AdvancedMCPConfig(environment=Environment.PRODUCTION)
        prod_config.apply_environment_overrides()

        assert prod_config.debug is False
        assert prod_config.connection_pool.pool_size == 10
        assert prod_config.cache.max_size == 5000
        assert prod_config.security.ssl_verify is True

        # Development config
        dev_config = AdvancedMCPConfig(environment=Environment.DEVELOPMENT)
        dev_config.apply_environment_overrides()

        assert dev_config.debug is True
        assert dev_config.connection_pool.pool_size == 2
        assert dev_config.cache.max_size == 100
        assert dev_config.security.ssl_verify is False

    def test_config_validation(self):
        """Test configuration validation."""
        config = AdvancedMCPConfig()

        # Valid config should have no issues
        from cicaddy.config.advanced_settings import ConfigurationManager

        manager = ConfigurationManager()
        issues = manager.validate_config(config)
        assert len(issues) == 0

        # Invalid config should report issues
        config.max_memory_usage = 32 * 1024 * 1024  # Too low
        config.max_cpu_usage = 150.0  # Too high

        issues = manager.validate_config(config)
        assert len(issues) > 0

    def test_feature_flags(self):
        """Test feature flag functionality."""
        config = AdvancedMCPConfig()
        flags = config.get_feature_flags()

        assert "retry" in flags
        assert "circuit_breaker" in flags
        assert "connection_pool" in flags
        assert "cache" in flags
        assert "telemetry" in flags

        # Test disabling features
        config.enable_cache = False
        config.enable_telemetry = False

        flags = config.get_feature_flags()
        assert flags["cache"] is False
        assert flags["telemetry"] is False


class TestEnterpriseMCPClient:
    """Test cases for enterprise MCP client."""

    @pytest.fixture
    def server_configs(self):
        """Test server configurations."""
        return [
            MCPServerConfig(
                name="test-server-1",
                protocol="sse",
                endpoint="https://test1.example.com/mcp/sse",
                timeout=30,
            ),
            MCPServerConfig(
                name="test-server-2",
                protocol="websocket",
                endpoint="wss://test2.example.com/ws",
                timeout=30,
            ),
        ]

    @pytest.fixture
    def advanced_config(self):
        """Advanced configuration for testing."""
        config = AdvancedMCPConfig(environment=Environment.TESTING)
        config.apply_environment_overrides()
        return config

    def test_enterprise_client_initialization(self, server_configs, advanced_config):
        """Test enterprise client initialization."""
        client = EnterpriseMCPClient(
            server_configs=server_configs,
            advanced_config=advanced_config,
            ssl_verify=False,  # Explicit for testing
        )

        assert len(client.server_configs) == 2
        assert client.config.environment == Environment.TESTING
        assert client.ssl_verify is False
        assert not client.is_initialized
        assert not client.is_running

    def test_enterprise_client_configuration(self, server_configs, advanced_config):
        """Test enterprise client configuration retrieval."""
        client = EnterpriseMCPClient(
            server_configs=server_configs,
            advanced_config=advanced_config,
        )

        config_dict = client.get_configuration()
        assert "advanced_config" in config_dict
        assert "server_configs" in config_dict
        assert "feature_flags" in config_dict
        assert "ssl_verify" in config_dict

        # Test server listing methods
        server_names = client.get_server_names()
        assert "test-server-1" in server_names
        assert "test-server-2" in server_names

        sse_servers = client.get_servers_by_protocol("sse")
        assert "test-server-1" in sse_servers
        assert "test-server-2" not in sse_servers

    @pytest.mark.asyncio
    async def test_enterprise_client_lifecycle(self, server_configs, advanced_config):
        """Test enterprise client initialization and shutdown."""
        client = EnterpriseMCPClient(
            server_configs=server_configs,
            advanced_config=advanced_config,
        )

        # Mock the client manager to avoid actual connections
        client.client_manager = AsyncMock()
        client.client_manager.initialize = AsyncMock()
        client.client_manager.cleanup = AsyncMock()
        client.client_manager.get_server_names.return_value = [
            "test-server-1",
            "test-server-2",
        ]

        # Test initialization
        await client.initialize()
        assert client.is_initialized
        assert client.is_running
        client.client_manager.initialize.assert_called_once()

        # Test shutdown
        await client.shutdown()
        assert not client.is_initialized
        assert not client.is_running
        client.client_manager.cleanup.assert_called_once()

    def test_enterprise_client_metrics_export(self, server_configs, advanced_config):
        """Test metrics export functionality with OpenTelemetry."""
        client = EnterpriseMCPClient(
            server_configs=server_configs,
            advanced_config=advanced_config,
        )

        # Test OpenTelemetry summary export
        summary_metrics = client.export_metrics_summary()
        assert isinstance(summary_metrics, str)

        # Test performance metrics
        perf_metrics = client.get_performance_metrics()
        assert "summary" in perf_metrics
        assert "detailed" in perf_metrics
