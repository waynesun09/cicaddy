"""Enterprise-grade MCP client with all Phase 2 advanced features integrated."""

import time
from typing import Any, Dict, List, Optional

from cicaddy.config.advanced_settings import AdvancedMCPConfig, config_manager
from cicaddy.config.settings import MCPServerConfig
from cicaddy.utils.logger import get_logger

from .cache import CacheMiddleware, MCPCache
from .circuit_breaker import CircuitBreakerConfig, circuit_breaker_registry
from .client import MCPClient, MCPClientManager
from .connection_pool import MCPConnectionPool
from .metrics import mcp_metrics, performance_monitor
from .telemetry import initialize_telemetry, shutdown_telemetry

logger = get_logger(__name__)


class EnterpriseMCPClient:
    """
    Enterprise-grade MCP client with advanced features:
    - Connection pooling and load balancing
    - Circuit breaker pattern for fault tolerance
    - Request/response caching
    - Comprehensive metrics and monitoring
    - Advanced configuration management
    """

    def __init__(
        self,
        server_configs: List[MCPServerConfig],
        advanced_config: Optional[AdvancedMCPConfig] = None,
        ssl_verify: bool = True,
    ):
        """
        Initialize enterprise MCP client.

        Args:
            server_configs: List of MCP server configurations
            advanced_config: Advanced configuration settings
            ssl_verify: SSL verification setting
        """
        self.server_configs = server_configs
        self.ssl_verify = ssl_verify
        self.config = advanced_config or config_manager.load_config()

        # Core components
        self.client_manager = MCPClientManager(server_configs, ssl_verify)

        # Advanced components (initialized based on config)
        self.connection_pool: Optional[MCPConnectionPool] = None
        self.cache: Optional[MCPCache] = None
        self.cache_middleware: Optional[CacheMiddleware] = None

        # State tracking
        self._initialized = False
        self._running = False

        # Initialize components based on configuration
        self._initialize_components()

    def _initialize_components(self):
        """Initialize advanced components based on configuration."""

        # Initialize connection pool
        if self.config.enable_connection_pool and self.config.connection_pool.enabled:
            self.connection_pool = MCPConnectionPool(
                pool_size=self.config.connection_pool.pool_size,
                max_retries=self.config.connection_pool.max_retries,
                load_balancing_strategy=self.config.connection_pool.load_balancing_strategy,
                health_check_interval=self.config.connection_pool.health_check_interval,
                cleanup_interval=self.config.connection_pool.cleanup_interval,
                max_idle_time=self.config.connection_pool.max_idle_time,
            )
            logger.info("Connection pool initialized")

        # Initialize cache
        if self.config.enable_cache and self.config.cache.enabled:
            self.cache = MCPCache(
                default_ttl=self.config.cache.default_ttl,
                enable_compression=self.config.cache.enable_compression,
                namespace=self.config.cache.namespace,
            )
            self.cache_middleware = CacheMiddleware(self.cache)
            logger.info("Cache system initialized")

        # Initialize circuit breakers
        if self.config.enable_circuit_breaker and self.config.circuit_breaker.enabled:
            cb_config = CircuitBreakerConfig(
                failure_threshold=self.config.circuit_breaker.failure_threshold,
                recovery_timeout=self.config.circuit_breaker.recovery_timeout,
                success_threshold=self.config.circuit_breaker.success_threshold,
                timeout=self.config.circuit_breaker.timeout,
                monitor_calls=self.config.circuit_breaker.monitor_calls,
                success_rate_threshold=self.config.circuit_breaker.success_rate_threshold,
            )

            # Create circuit breakers for each server
            for server_config in self.server_configs:
                circuit_breaker_registry.get_breaker(server_config.name, cb_config)

            logger.info(
                f"Circuit breakers initialized for {len(self.server_configs)} servers"
            )

    async def initialize(self):
        """Initialize the enterprise client and all components."""
        if self._initialized:
            return

        logger.info("Initializing enterprise MCP client...")

        # Start connection pool
        if self.connection_pool:
            await self.connection_pool.start()

        # Initialize telemetry
        if self.config.enable_telemetry and self.config.telemetry.enabled:
            initialize_telemetry(self.config.telemetry)

        # Initialize base client manager
        await self.client_manager.initialize()

        # Start metrics monitoring (simplified for GitLab Actions)
        if self.config.enable_telemetry and self.config.telemetry.enabled:
            await performance_monitor.start_monitoring(
                self.config.telemetry.metrics_export_interval
            )

        self._initialized = True
        self._running = True
        logger.info("Enterprise MCP client initialized successfully")

    async def shutdown(self):
        """Shutdown the enterprise client and cleanup resources."""
        if not self._running:
            return

        logger.info("Shutting down enterprise MCP client...")

        self._running = False

        # Stop monitoring
        if self.config.enable_telemetry:
            await performance_monitor.stop_monitoring()

        # Stop connection pool
        if self.connection_pool:
            await self.connection_pool.stop()

        # Cleanup client manager
        await self.client_manager.cleanup()

        # Shutdown telemetry and export final metrics
        if self.config.enable_telemetry and self.config.telemetry.enabled:
            # Print summary to GitLab Action logs
            summary = mcp_metrics.export_opentelemetry_summary()
            logger.info(f"\n{summary}")

            # Shutdown telemetry
            shutdown_telemetry()

        self._initialized = False
        logger.info("Enterprise MCP client shutdown complete")

    async def list_tools(
        self, server_name: str, force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List tools from a specific server with enterprise features.

        Args:
            server_name: Name of the MCP server
            force_refresh: Force cache refresh

        Returns:
            List of available tools
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        method = "tools/list"

        try:
            # Record request start
            mcp_metrics.record_request(server_name, method, success=True)

            # Use cache middleware if enabled
            if self.cache_middleware and self.config.enable_cache:

                async def operation():
                    return await self._execute_with_pool_or_direct(
                        server_name, lambda client: client.list_tools()
                    )

                result = await self.cache_middleware.execute_with_cache(
                    server_name=server_name,
                    method=method,
                    params=None,
                    operation_func=operation,
                    force_refresh=force_refresh,
                )
            else:
                result = await self._execute_with_pool_or_direct(
                    server_name, lambda client: client.list_tools()
                )

            # Record metrics
            duration = time.time() - start_time
            mcp_metrics.record_request(
                server_name, method, success=True, duration=duration
            )

            if self.cache and self.config.enable_cache:
                mcp_metrics.record_cache_hit(server_name, method)

            return result

        except Exception as e:
            duration = time.time() - start_time
            mcp_metrics.record_request(
                server_name, method, success=False, duration=duration
            )
            mcp_metrics.record_error(server_name, type(e).__name__)
            logger.error(f"Failed to list tools from {server_name}: {e}")
            raise

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Call a tool on a specific server with enterprise features.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            timeout: Custom timeout for this request

        Returns:
            Tool execution result
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        method = "tools/call"

        try:
            # Use circuit breaker if enabled
            if (
                self.config.enable_circuit_breaker
                and self.config.circuit_breaker.enabled
            ):
                breaker = circuit_breaker_registry.get_breaker(server_name)

                async def protected_operation():
                    return await self._execute_with_pool_or_direct(
                        server_name,
                        lambda client: client.call_tool(tool_name, arguments),
                    )

                result = await breaker.call(protected_operation)
            else:
                result = await self._execute_with_pool_or_direct(
                    server_name, lambda client: client.call_tool(tool_name, arguments)
                )

            # Record metrics
            duration = time.time() - start_time
            mcp_metrics.record_request(
                server_name, method, success=True, duration=duration
            )

            return result

        except Exception as e:
            duration = time.time() - start_time
            mcp_metrics.record_request(
                server_name, method, success=False, duration=duration
            )
            mcp_metrics.record_error(server_name, type(e).__name__)

            # Record circuit breaker trip if applicable
            if self.config.enable_circuit_breaker:
                mcp_metrics.record_circuit_breaker_trip(server_name)

            logger.error(f"Failed to call tool {tool_name} on {server_name}: {e}")
            raise

    async def _execute_with_pool_or_direct(self, server_name: str, operation) -> Any:
        """Execute operation with connection pool if available, otherwise use direct client."""
        if self.connection_pool and self.config.enable_connection_pool:

            def client_factory():
                config = next(
                    (c for c in self.server_configs if c.name == server_name), None
                )
                if not config:
                    raise ValueError(f"Server config not found: {server_name}")
                return MCPClient(config, self.ssl_verify)

            return await self.connection_pool.execute_with_pool(
                server_name=server_name,
                client_factory=client_factory,
                operation=operation,
            )
        else:
            # Use direct client manager
            client = self.client_manager.clients.get(server_name)
            if not client:
                raise ValueError(f"MCP server {server_name} not found or not connected")
            return await operation(client)

    async def get_server_health(self, server_name: str) -> Dict[str, Any]:
        """
        Get health status for a specific server.

        Args:
            server_name: Name of the MCP server

        Returns:
            Health status information
        """
        try:
            client = self.client_manager.clients.get(server_name)
            if not client:
                return {"healthy": False, "error": "Server not found or not connected"}

            healthy = await client.transport.check_health()
            mcp_metrics.update_server_health(server_name, healthy)

            health_info = {
                "healthy": healthy,
                "connected": client.connected,
                "server_name": server_name,
                "protocol": client.config.protocol,
                "timestamp": time.time(),
            }

            # Add circuit breaker state if enabled
            if self.config.enable_circuit_breaker:
                breaker = circuit_breaker_registry.get_breaker(server_name)
                health_info["circuit_breaker"] = breaker.get_stats()

            # Add pool stats if enabled
            if self.connection_pool:
                health_info["pool"] = self.connection_pool.get_pool_stats(server_name)

            return health_info

        except Exception as e:
            mcp_metrics.record_error(server_name, "health_check_error")
            return {
                "healthy": False,
                "error": str(e),
                "server_name": server_name,
                "timestamp": time.time(),
            }

    async def get_all_server_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all servers."""
        health_status = {}
        for server_name in self.client_manager.get_server_names():
            health_status[server_name] = await self.get_server_health(server_name)
        return health_status

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics: Dict[str, Any] = {
            "summary": mcp_metrics.get_summary_stats(),
            "detailed": mcp_metrics.get_all_metrics(),
        }

        # Add cache metrics if enabled
        if self.cache and self.config.enable_cache:
            metrics["cache"] = self.cache.get_cache_stats()

        # Add pool metrics if enabled
        if self.connection_pool and self.config.enable_connection_pool:
            metrics["pools"] = self.connection_pool.get_all_stats()

        # Add circuit breaker metrics if enabled
        if self.config.enable_circuit_breaker:
            metrics["circuit_breakers"] = circuit_breaker_registry.get_all_stats()

        # Add recent alerts
        if self.config.enable_telemetry:
            metrics["alerts"] = performance_monitor.get_recent_alerts()

        return metrics

    def export_metrics_summary(self) -> str:
        """Export metrics summary for GitLab Action logs."""
        if not self.config.enable_telemetry:
            return ""
        return mcp_metrics.export_opentelemetry_summary()

    async def invalidate_cache(
        self, server_name: Optional[str] = None, method: Optional[str] = None
    ) -> int:
        """
        Invalidate cache entries.

        Args:
            server_name: Specific server to invalidate, or None for all
            method: Specific method to invalidate, or None for all

        Returns:
            Number of entries invalidated
        """
        if not self.cache or not self.config.enable_cache:
            return 0

        if server_name:
            return await self.cache.invalidate(server_name, method)
        else:
            # Invalidate all servers
            total_invalidated = 0
            for config in self.server_configs:
                total_invalidated += await self.cache.invalidate(config.name, method)
            return total_invalidated

    async def reset_circuit_breakers(self, server_name: Optional[str] = None):
        """
        Reset circuit breakers.

        Args:
            server_name: Specific server to reset, or None for all
        """
        if not self.config.enable_circuit_breaker:
            return

        if server_name:
            breaker = circuit_breaker_registry.get_breaker(server_name)
            await breaker.reset()
            logger.info(f"Reset circuit breaker for {server_name}")
        else:
            await circuit_breaker_registry.reset_all()
            logger.info("Reset all circuit breakers")

    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            "advanced_config": self.config.to_dict(),
            "server_configs": [config.model_dump() for config in self.server_configs],
            "feature_flags": self.config.get_feature_flags(),
            "ssl_verify": self.ssl_verify,
        }

    def get_server_names(self) -> List[str]:
        """Get list of configured server names."""
        return [config.name for config in self.server_configs]

    def get_servers_by_protocol(self, protocol: str) -> List[str]:
        """Get servers using a specific protocol."""
        return [
            config.name for config in self.server_configs if config.protocol == protocol
        ]

    @property
    def is_healthy(self) -> bool:
        """Check if the enterprise client is healthy."""
        if not self._initialized or not self._running:
            return False

        # Check if at least one server is healthy
        for server_name in self.client_manager.get_server_names():
            client = self.client_manager.clients.get(server_name)
            if client and client.connected:
                return True

        return False

    @property
    def is_initialized(self) -> bool:
        """Check if client is initialized."""
        return self._initialized

    @property
    def is_running(self) -> bool:
        """Check if client is running."""
        return self._running
