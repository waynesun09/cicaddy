"""Connection pooling and load balancing for MCP clients."""

import asyncio
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from cicaddy.utils.logger import get_logger

from .client import MCPClient

logger = get_logger(__name__)


class ConnectionStats:
    """Statistics for connection performance tracking."""

    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.avg_response_time = 0.0
        self.last_request_time = 0.0
        self.connection_errors = 0
        self.consecutive_failures = 0
        self.created_at = time.time()

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0

    @property
    def error_rate(self) -> float:
        """Calculate error rate as percentage."""
        return 100.0 - self.success_rate

    def record_request(self, success: bool, response_time: float):
        """Record request statistics."""
        self.total_requests += 1
        self.last_request_time = time.time()

        if success:
            self.successful_requests += 1
            self.consecutive_failures = 0
            # Update average response time with exponential moving average
            alpha = 0.1  # Smoothing factor
            self.avg_response_time = (alpha * response_time) + (
                (1 - alpha) * self.avg_response_time
            )
        else:
            self.failed_requests += 1
            self.consecutive_failures += 1

    def record_connection_error(self):
        """Record connection error."""
        self.connection_errors += 1
        self.consecutive_failures += 1


class LoadBalancer:
    """Load balancing strategies for MCP connections."""

    @staticmethod
    def round_robin(
        connections: List[MCPClient], last_used_index: int
    ) -> tuple[MCPClient, int]:
        """Round-robin load balancing."""
        if not connections:
            raise ValueError("No available connections")

        next_index = (last_used_index + 1) % len(connections)
        return connections[next_index], next_index

    @staticmethod
    def least_connections(
        connections: List[MCPClient], stats: Dict[str, ConnectionStats]
    ) -> MCPClient:
        """Select connection with least active requests."""
        if not connections:
            raise ValueError("No available connections")

        # For simplicity, we'll use request count as proxy for active connections
        return min(
            connections,
            key=lambda c: stats.get(c.config.name, ConnectionStats()).total_requests,
        )

    @staticmethod
    def weighted_random(
        connections: List[MCPClient], stats: Dict[str, ConnectionStats]
    ) -> MCPClient:
        """Weighted random selection based on success rate and response time."""
        if not connections:
            raise ValueError("No available connections")

        if len(connections) == 1:
            return connections[0]

        # Calculate weights based on success rate and inverse response time
        weights = []
        for client in connections:
            client_stats = stats.get(client.config.name, ConnectionStats())
            # Higher success rate and lower response time = higher weight
            success_weight = client_stats.success_rate / 100.0
            # Avoid division by zero, use 1ms as minimum
            time_weight = 1.0 / max(client_stats.avg_response_time, 0.001)
            # Combine weights with bias towards success rate
            weight = (success_weight * 0.7) + (time_weight * 0.3)
            weights.append(max(weight, 0.1))  # Minimum weight

        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(connections)  # nosec B311 - not cryptographic

        r = random.uniform(0, total_weight)  # nosec B311 - load balancing, not crypto
        cumulative = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return connections[i]

        return connections[-1]  # Fallback


class MCPConnectionPool:
    """Connection pool for managing multiple MCP client connections."""

    def __init__(
        self,
        pool_size: int = 5,
        max_retries: int = 3,
        load_balancing_strategy: str = "weighted_random",
        health_check_interval: float = 30.0,
        cleanup_interval: float = 300.0,  # 5 minutes
        max_idle_time: float = 600.0,  # 10 minutes
    ):
        self.pool_size = pool_size
        self.max_retries = max_retries
        self.load_balancing_strategy = load_balancing_strategy
        self.health_check_interval = health_check_interval
        self.cleanup_interval = cleanup_interval
        self.max_idle_time = max_idle_time

        # Connection management
        self._pools: Dict[str, List[MCPClient]] = defaultdict(list)
        self._available: Dict[str, Set[MCPClient]] = defaultdict(set)
        self._busy: Dict[str, Set[MCPClient]] = defaultdict(set)
        self._stats: Dict[str, ConnectionStats] = defaultdict(ConnectionStats)

        # Synchronization for connection availability
        self._connection_available_condition: Dict[str, asyncio.Condition] = (
            defaultdict(lambda: asyncio.Condition())
        )

        # Load balancing state
        self._round_robin_index: Dict[str, int] = defaultdict(int)

        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """Start the connection pool and background tasks."""
        if self._running:
            return

        self._running = True

        # Start background tasks
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info(
            f"Started MCP connection pool with {self.pool_size} connections per server, "
            f"strategy: {self.load_balancing_strategy}"
        )

    async def stop(self):
        """Stop the connection pool and cleanup resources."""
        if not self._running:
            return

        self._running = False

        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        for server_name in list(self._pools.keys()):
            await self._close_pool(server_name)

        logger.info("Stopped MCP connection pool")

    async def get_connection(self, server_name: str, client_factory) -> MCPClient:
        """
        Get a connection from the pool.

        Args:
            server_name: Name of the MCP server
            client_factory: Function to create new MCPClient instances

        Returns:
            MCPClient instance
        """
        if not self._running:
            await self.start()

        # Try to get available connection
        if self._available[server_name]:
            client = self._available[server_name].pop()
            self._busy[server_name].add(client)
            return client

        # Create new connection if pool not full
        if len(self._pools[server_name]) < self.pool_size:
            client = await self._create_connection(server_name, client_factory)
            self._pools[server_name].append(client)
            self._busy[server_name].add(client)
            return client

        # Pool is full, use load balancing to select from available connections
        available_clients = list(self._available[server_name])
        if available_clients:
            client = self._select_connection(available_clients, server_name)
            self._available[server_name].discard(client)
            self._busy[server_name].add(client)
            return client

        # No available connections, wait for one to become available
        logger.warning(f"No available connections for {server_name}, waiting...")
        async with self._connection_available_condition[server_name]:
            await self._connection_available_condition[server_name].wait()
        return await self.get_connection(server_name, client_factory)

    async def return_connection(self, client: MCPClient):
        """Return a connection to the pool."""
        server_name = client.config.name

        if client in self._busy[server_name]:
            self._busy[server_name].discard(client)

            # Check if connection is still healthy
            if client.connected and await client.transport.check_health():
                self._available[server_name].add(client)
                # Notify waiting tasks that a connection is available
                async with self._connection_available_condition[server_name]:
                    self._connection_available_condition[server_name].notify()
            else:
                # Remove unhealthy connection
                await self._remove_connection(client, server_name)

    async def execute_with_pool(
        self, server_name: str, client_factory, operation, *args, **kwargs
    ):
        """
        Execute an operation using a pooled connection.

        Args:
            server_name: Name of the MCP server
            client_factory: Function to create new MCPClient instances
            operation: Function to execute (e.g., client.call_tool)
            *args, **kwargs: Arguments for the operation

        Returns:
            Result of the operation
        """
        start_time = time.time()
        last_exception = None

        for attempt in range(self.max_retries + 1):
            client = None
            try:
                client = await self.get_connection(server_name, client_factory)

                # Execute operation
                result = await operation(client, *args, **kwargs)

                # Record success
                response_time = time.time() - start_time
                self._stats[server_name].record_request(True, response_time)

                return result

            except Exception as e:
                last_exception = e
                response_time = time.time() - start_time
                self._stats[server_name].record_request(False, response_time)

                if attempt < self.max_retries:
                    logger.warning(
                        f"Operation failed on {server_name} (attempt {attempt + 1}), retrying: {e}"
                    )
                    await asyncio.sleep(0.5 * (2**attempt))  # Exponential backoff
                else:
                    logger.error(
                        f"Operation failed on {server_name} after {self.max_retries} retries: {e}"
                    )

            finally:
                if client:
                    await self.return_connection(client)

        # All retries exhausted
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Operation failed after {self.max_retries} retries")

    def get_pool_stats(self, server_name: str) -> Dict[str, Any]:
        """Get statistics for a server's connection pool."""
        pool = self._pools[server_name]
        available = self._available[server_name]
        busy = self._busy[server_name]
        stats = self._stats[server_name]

        return {
            "server_name": server_name,
            "pool_size": len(pool),
            "available_connections": len(available),
            "busy_connections": len(busy),
            "total_requests": stats.total_requests,
            "successful_requests": stats.successful_requests,
            "failed_requests": stats.failed_requests,
            "success_rate": stats.success_rate,
            "error_rate": stats.error_rate,
            "avg_response_time": stats.avg_response_time,
            "connection_errors": stats.connection_errors,
            "consecutive_failures": stats.consecutive_failures,
            "uptime": time.time() - stats.created_at,
        }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all server pools."""
        return {
            server_name: self.get_pool_stats(server_name)
            for server_name in self._pools.keys()
        }

    async def _create_connection(self, server_name: str, client_factory) -> MCPClient:
        """Create and initialize a new connection."""
        try:
            client = client_factory()
            await client.connect()
            logger.debug(f"Created new connection for {server_name}")
            return client
        except Exception as e:
            self._stats[server_name].record_connection_error()
            logger.error(f"Failed to create connection for {server_name}: {e}")
            raise

    async def _remove_connection(self, client: MCPClient, server_name: str):
        """Remove a connection from the pool."""
        try:
            await client.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting client for {server_name}: {e}")

        # Remove from all tracking sets
        pool = self._pools[server_name]
        if client in pool:
            pool.remove(client)

        self._available[server_name].discard(client)
        self._busy[server_name].discard(client)

        logger.debug(f"Removed connection for {server_name}")

    def _select_connection(
        self, connections: List[MCPClient], server_name: str
    ) -> MCPClient:
        """Select a connection using the configured load balancing strategy."""
        if self.load_balancing_strategy == "round_robin":
            client, index = LoadBalancer.round_robin(
                connections, self._round_robin_index[server_name]
            )
            self._round_robin_index[server_name] = index
            return client
        elif self.load_balancing_strategy == "least_connections":
            return LoadBalancer.least_connections(connections, self._stats)
        elif self.load_balancing_strategy == "weighted_random":
            return LoadBalancer.weighted_random(connections, self._stats)
        else:
            # Default to random
            return random.choice(connections)  # nosec B311 - not cryptographic

    async def _health_check_loop(self):
        """Background task for periodic health checks."""
        while self._running:
            try:
                await asyncio.sleep(self.health_check_interval)

                for server_name in list(self._pools.keys()):
                    await self._health_check_pool(server_name)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

    async def _health_check_pool(self, server_name: str):
        """Perform health check on all connections in a pool."""
        pool = self._pools[server_name]
        unhealthy_clients = []

        for client in pool:
            try:
                if not client.connected or not await client.transport.check_health():
                    unhealthy_clients.append(client)
            except Exception as e:
                logger.warning(f"Health check failed for {server_name}: {e}")
                unhealthy_clients.append(client)

        # Remove unhealthy connections
        for client in unhealthy_clients:
            await self._remove_connection(client, server_name)

        if unhealthy_clients:
            logger.info(
                f"Removed {len(unhealthy_clients)} unhealthy connections for {server_name}"
            )

    async def _cleanup_loop(self):
        """Background task for periodic cleanup of idle connections."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)

                current_time = time.time()
                for server_name in list(self._pools.keys()):
                    await self._cleanup_idle_connections(server_name, current_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _cleanup_idle_connections(self, server_name: str, current_time: float):
        """Clean up idle connections that haven't been used recently."""
        stats = self._stats[server_name]
        if current_time - stats.last_request_time > self.max_idle_time:
            # Pool has been idle, reduce to minimum size
            available_clients = list(self._available[server_name])
            if len(available_clients) > 1:  # Keep at least one connection
                clients_to_remove = available_clients[1:]
                for client in clients_to_remove:
                    await self._remove_connection(client, server_name)

                logger.info(
                    f"Cleaned up {len(clients_to_remove)} idle connections for {server_name}"
                )

    async def _close_pool(self, server_name: str):
        """Close all connections in a pool."""
        if server_name not in self._pools:
            return

        pool = self._pools[server_name]
        for client in pool:
            try:
                await client.disconnect()
            except Exception as e:
                logger.warning(f"Error closing connection for {server_name}: {e}")

        # Clear all tracking (check existence before deleting)
        if server_name in self._pools:
            del self._pools[server_name]
        if server_name in self._available:
            del self._available[server_name]
        if server_name in self._busy:
            del self._busy[server_name]
        if server_name in self._stats:
            del self._stats[server_name]

        logger.debug(f"Closed connection pool for {server_name}")
