"""Base transport class for MCP communication."""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from cicaddy.config.settings import MCPServerConfig
from cicaddy.utils.logger import get_logger

from ..retry import RetryableMixin, wrap_connection_errors

logger = get_logger(__name__)


class BaseMCPTransport(RetryableMixin, ABC):
    """Abstract base class for all MCP transport implementations."""

    def __init__(self, config: MCPServerConfig):
        super().__init__()
        self.config = config
        self.connected = False
        self._request_id_counter = 0
        self._connection_health_ok = True
        self._last_heartbeat = None

    @abstractmethod
    async def _connect_impl(self) -> None:
        """Establish connection to MCP server (implementation-specific)."""
        pass

    @abstractmethod
    async def _disconnect_impl(self) -> None:
        """Close connection and cleanup resources (implementation-specific)."""
        pass

    @abstractmethod
    async def _send_request_impl(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send JSON-RPC request and return response (implementation-specific).

        Args:
            method: The JSON-RPC method name
            params: Optional parameters for the request

        Returns:
            Dict containing the response from the server

        Raises:
            ConnectionError: If not connected to server
            RuntimeError: If request fails
        """
        pass

    async def connect(self) -> None:
        """Establish connection to MCP server with retry logic."""
        if self.connected:
            return

        @wrap_connection_errors
        async def _connect():
            await self._connect_impl()
            self.connected = True
            self._connection_health_ok = True
            logger.info(f"Successfully connected to MCP server: {self.config.name}")

        await self._retry_operation(_connect, f"connection to {self.config.name}")

    async def disconnect(self) -> None:
        """Close connection and cleanup resources."""
        if not self.connected:
            return

        try:
            await self._disconnect_impl()
        except Exception as e:
            logger.warning(f"Error during disconnect from {self.config.name}: {e}")
        finally:
            self.connected = False
            self._connection_health_ok = False

    async def send_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send JSON-RPC request and return response with retry logic.

        Args:
            method: The JSON-RPC method name
            params: Optional parameters for the request

        Returns:
            Dict containing the response from the server

        Raises:
            ConnectionError: If not connected to server
            RuntimeError: If request fails
        """

        @wrap_connection_errors
        async def _send():
            # Check if we need to reconnect before sending
            # This handles cases where 5xx errors or cancel scope errors broke the connection
            if not self.connected or not self._connection_health_ok:
                logger.info(
                    f"Connection to {self.config.name} is broken/unhealthy, attempting to reconnect before retry"
                )
                # Shield reconnection from lingering cancel scopes
                try:
                    await asyncio.shield(self._reconnect())
                except asyncio.CancelledError as reconnect_error:
                    # Lingering cancel scope cancelled reconnection, try again in isolation
                    logger.warning(
                        f"Reconnection cancelled by lingering cancel scope, retrying: {reconnect_error}"
                    )
                    # Wait briefly to let cancel scope settle, then retry reconnection
                    import time as time_module

                    await asyncio.get_event_loop().run_in_executor(
                        None, time_module.sleep, 0.5
                    )
                    await self._reconnect()
            try:
                return await self._send_request_impl(method, params)
            except asyncio.CancelledError as e:
                # CancelledError from cancel scope indicates MCP SDK cleanup failure
                # Mark connection as broken to force reconnection on next retry
                error_str = str(e).lower()
                if "cancel scope" in error_str:
                    logger.warning(
                        f"Cancel scope CancelledError detected, marking connection as broken: {e}"
                    )
                    self.connected = False
                    self._connection_health_ok = False
                raise

        return await self._retry_operation(
            _send, f"request {method} to {self.config.name}"
        )

    async def _reconnect(self) -> None:
        """Reconnect to the MCP server after connection failure."""
        logger.info(f"Reconnecting to MCP server: {self.config.name}")

        # Clean up existing connection state
        try:
            await self._disconnect_impl()
        except Exception as e:
            logger.debug(
                f"Error during cleanup before reconnect for {self.config.name}: {e}"
            )

        # Reset connection state
        self.connected = False
        self._connection_health_ok = False

        # Reconnect
        await self._connect_impl()
        self.connected = True
        self._connection_health_ok = True
        logger.info(f"Successfully reconnected to MCP server: {self.config.name}")

    async def check_health(self) -> bool:
        """
        Check connection health.

        Returns:
            True if connection is healthy, False otherwise
        """
        if not self.connected:
            return False

        try:
            # Simple health check - attempt to list tools
            await self._send_request_impl("tools/list")
            self._connection_health_ok = True
            return True
        except Exception as e:
            logger.warning(f"Health check failed for {self.config.name}: {e}")
            self._connection_health_ok = False
            return False

    @property
    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        return self.connected and self._connection_health_ok

    def _generate_request_id(self) -> str:
        """Generate unique request ID for JSON-RPC."""
        self._request_id_counter += 1
        return f"{self.config.name}-{self._request_id_counter}-{uuid.uuid4().hex[:8]}"

    def _build_environment(self) -> Dict[str, str]:
        """Build environment variables for subprocess-based transports."""
        import os

        env = os.environ.copy()
        if self.config.env_vars:
            env.update(self.config.env_vars)
        return env

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools from the MCP server with retry logic.

        Returns:
            List of tool definitions
        """
        if not self.connected:
            await self.connect()

        try:
            logger.info(f"Listing tools from MCP server: {self.config.name}")
            response = await self.send_request("tools/list")

            tools = response.get("tools", [])

            # Add server name to each tool
            for tool in tools:
                tool["server"] = self.config.name

            # Filter tools if specified in config
            if self.config.tools:
                tools = [
                    tool for tool in tools if tool.get("name") in self.config.tools
                ]

            logger.info(f"MCP server {self.config.name} returned {len(tools)} tools")
            return tools

        except Exception as e:
            logger.error(f"Failed to list tools from {self.config.name}: {e}")
            # Mark connection as unhealthy
            self._connection_health_ok = False

            # Fallback: use configured tools if listing fails
            if self.config.tools:
                logger.info(
                    f"Using configured tools list for {self.config.name}: {self.config.tools}"
                )
                return [
                    {"name": tool_name, "server": self.config.name}
                    for tool_name in self.config.tools
                ]
            return []

    async def call_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call a tool on the MCP server with retry logic.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Dict containing the tool execution result
        """
        if not self.connected:
            await self.connect()

        logger.info(f"Calling tool {tool_name} on MCP server {self.config.name}")
        logger.debug(f"Arguments: {arguments}")

        try:
            response = await self.send_request(
                "tools/call", {"name": tool_name, "arguments": arguments}
            )

            # Format response in standard format
            result = {
                "tool": tool_name,
                "server": self.config.name,
                "status": "success",
            }

            # Extract content from response
            if "content" in response:
                if isinstance(response["content"], list) and response["content"]:
                    # Extract text from content items
                    text_content = []
                    for item in response["content"]:
                        if isinstance(item, dict) and "text" in item:
                            text_content.append(item["text"])
                        elif isinstance(item, str):
                            text_content.append(item)
                        else:
                            text_content.append(str(item))
                    result["content"] = "\n".join(text_content)
                elif isinstance(response["content"], str):
                    result["content"] = response["content"]
                else:
                    result["content"] = str(response["content"])
            elif "result" in response:
                result["content"] = str(response["result"])
            else:
                result["content"] = "Tool executed successfully"

            logger.info(f"Successfully called {tool_name} on {self.config.name}")
            return result

        except Exception as e:
            logger.error(f"Tool call failed for {tool_name} on {self.config.name}: {e}")
            # Mark connection as unhealthy
            self._connection_health_ok = False
            raise

    async def start_heartbeat(self) -> None:
        """Start heartbeat monitoring if configured."""
        if not self.config.heartbeat_interval or self.config.heartbeat_interval <= 0:
            return

        async def heartbeat_loop():
            while self.connected and self.config.heartbeat_interval:
                try:
                    await asyncio.sleep(self.config.heartbeat_interval)
                    if self.connected:
                        is_healthy = await self.check_health()
                        self._last_heartbeat = time.time()
                        if not is_healthy:
                            logger.warning(
                                f"Heartbeat failed for {self.config.name}, connection may be unhealthy"
                            )
                        else:
                            logger.debug(f"Heartbeat successful for {self.config.name}")
                except Exception as e:
                    logger.error(f"Heartbeat error for {self.config.name}: {e}")
                    self._connection_health_ok = False
                    break

        # Start heartbeat in background
        asyncio.create_task(heartbeat_loop())
        logger.info(
            f"Started heartbeat monitoring for {self.config.name} (interval: {self.config.heartbeat_interval}s)"
        )
