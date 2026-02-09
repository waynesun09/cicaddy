"""HTTP transport for MCP communication via HTTP streaming protocol."""

from typing import Any, Dict, Optional, cast

import httpx
from mcp import ClientSession as OfficialMCPClientSession
from mcp.client.streamable_http import streamablehttp_client as official_mcp_http_client
from mcp.shared._httpx_utils import McpHttpClientFactory

from cicaddy.config.settings import MCPServerConfig
from cicaddy.utils.logger import get_logger

from ..retry import ServerRetryableError
from .base import BaseMCPTransport

logger = get_logger(__name__)


class HTTPMCPTransport(BaseMCPTransport):
    """MCP transport using HTTP streaming protocol."""

    def __init__(self, config: MCPServerConfig, ssl_verify: bool = True):
        super().__init__(config)
        self.ssl_verify = ssl_verify
        self.session: Optional[OfficialMCPClientSession] = None
        self._session_context = None
        self.streams = None
        self.http_connection = None

    async def _connect_impl(self) -> None:
        """Establish HTTP connection to MCP server (implementation-specific)."""
        logger.info(f"Connecting to HTTP MCP server: {self.config.name}")

        if not self.config.endpoint:
            raise ValueError(
                f"No endpoint specified for HTTP server {self.config.name}"
            )

        # Create headers if specified in config
        headers = self.config.headers.copy() if self.config.headers else {}

        # Create custom HTTP client factory for SSL configuration
        http_client_factory = self._create_http_client_factory()

        # Use connection timeout from config
        connection_timeout = float(self.config.connection_timeout)
        # Priority: idle_timeout > read_timeout > timeout
        # idle_timeout is specifically for SSE/progress updates (time between events)
        read_timeout = float(
            self.config.idle_timeout or self.config.read_timeout or self.config.timeout
        )

        # Log actual timeout configuration being used
        logger.info(
            f"HTTP timeout config for {self.config.name}: "
            f"connection={connection_timeout}s, "
            f"read/idle={read_timeout}s "
            f"(from {'idle_timeout' if self.config.idle_timeout else 'read_timeout' if self.config.read_timeout else 'timeout'}), "
            f"total={self.config.timeout}s"
        )

        # Connect using the official MCP library
        self.streams = official_mcp_http_client(
            self.config.endpoint,
            headers=headers,
            timeout=connection_timeout,
            sse_read_timeout=read_timeout,
            httpx_client_factory=http_client_factory,
        )

        # Enter the context managers
        streams_cm = cast(Any, self.streams)  # MCP library's async context manager
        self.http_connection = await streams_cm.__aenter__()

        # Create session with proper timeout configuration
        # Cast to Any to avoid mypy "None is not iterable" error
        connection_tuple = cast(Any, self.http_connection)
        read_stream, write_stream, get_session_id = connection_tuple
        import datetime

        timeout_delta = datetime.timedelta(seconds=read_timeout)

        self._session_context = OfficialMCPClientSession(
            read_stream=read_stream,
            write_stream=write_stream,
            read_timeout_seconds=timeout_delta,
        )

        # Enter the session context manager
        session_cm = cast(
            Any, self._session_context
        )  # ClientSession async context manager
        self.session = await session_cm.__aenter__()

        # Initialize the session
        await self.session.initialize()

        # Start heartbeat monitoring if configured
        await self.start_heartbeat()

    async def _disconnect_impl(self) -> None:
        """Disconnect from HTTP MCP server (implementation-specific)."""
        logger.debug(f"Disconnecting from HTTP MCP server: {self.config.name}")

        # Step 1: Exit ClientSession context properly
        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error exiting ClientSession context: {e}")
            finally:
                self._session_context = None
                self.session = None

        # Step 2: Clean up http_connection and streams WITHOUT calling __aexit__
        # This prevents "Attempted to exit cancel scope in a different task" errors
        # Let the MCP library handle its own cleanup during program termination
        if self.http_connection:
            self.http_connection = None
        if self.streams:
            self.streams = None

    async def _send_request_impl(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send request via HTTP connection using the official MCP library (implementation-specific).

        Note: The underlying 'mcp' library directly exposes 'list_tools' and 'call_tool'
        methods on its session, rather than a generic JSON-RPC 'send_request'.
        This method acts as a dispatcher to map generic MCP method names to the
        library's specific API calls and formats their results to standard JSON-RPC.

        Args:
            method: The MCP method to call (e.g., "tools/list", "tools/call")
            params: Optional parameters for the request

        Returns:
            Dict containing the response from the server
        """
        if not self.connected or not self.session:
            raise ConnectionError(f"Not connected to HTTP server {self.config.name}")

        try:
            if method == "tools/list":
                result = await self.session.list_tools()
                tools = []
                for tool in result.tools:
                    tool_dict = {
                        "name": tool.name,
                        "description": tool.description,
                        "server": self.config.name,
                    }
                    # Add input schema if available
                    if hasattr(tool, "inputSchema") and tool.inputSchema:
                        tool_dict["inputSchema"] = tool.inputSchema
                    tools.append(tool_dict)
                return {"tools": tools}

            elif method == "tools/call":
                if not params or "name" not in params:
                    raise ValueError("tools/call requires 'name' parameter")

                tool_name = params["name"]
                arguments = params.get("arguments", {})

                logger.debug(
                    f"Calling tool {tool_name} with progress notification support enabled"
                )

                # Define progress callback to handle progress notifications
                def progress_callback(
                    progress_token: str, progress: float, total: float | None = None
                ):
                    """Handle progress notifications from the MCP server."""
                    try:
                        if total is not None:
                            percentage = (progress / total) * 100 if total > 0 else 0
                            logger.info(
                                f"Tool {tool_name} progress: {progress}/{total} "
                                f"({percentage:.1f}%) [token: {progress_token}]"
                            )
                        else:
                            logger.info(
                                f"Tool {tool_name} progress: {progress} [token: {progress_token}]"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Error in progress callback for {tool_name}: {e}"
                        )

                # Call the tool with progress notification support
                # Wrap in try/except to catch timeouts and 5xx errors BEFORE they propagate
                # to MCP library cleanup. This prevents asyncgen cleanup errors when
                # httpx.ReadTimeout or HTTPStatusError (5xx) occurs.
                try:
                    result = await self.session.call_tool(
                        tool_name, arguments, progress_callback=progress_callback
                    )
                except httpx.HTTPStatusError as e:
                    # Check if this is a 5xx server error (retryable)
                    if 500 <= e.response.status_code < 600:
                        logger.error(
                            f"Tool {tool_name} received server error {e.response.status_code}: {e}"
                        )
                        # Mark connection as broken to force reconnection on retry
                        self.connected = False
                        self._connection_health_ok = False
                        raise ServerRetryableError(
                            f"MCP server {self.config.name} returned {e.response.status_code}: {e}"
                        ) from e
                    # Non-5xx HTTP errors are not automatically retryable
                    raise
                except Exception as e:
                    # Check if this is a timeout error (httpx.ReadTimeout or similar)
                    error_type = type(e).__name__
                    error_str = str(e).lower()
                    if (
                        "timeout" in error_str
                        or "Timeout" in error_type
                        or "ReadTimeout" in error_type
                    ):
                        # Convert to TimeoutError immediately to prevent MCP cleanup issues
                        logger.error(f"Tool {tool_name} timed out: {e}")
                        # Mark connection as broken to force reconnection on retry
                        self.connected = False
                        self._connection_health_ok = False
                        raise TimeoutError(
                            f"MCP server {self.config.name} tool {tool_name} timed out: {e}"
                        ) from e

                    # Check for 5xx patterns in error messages (wrapped exceptions)
                    if any(code in error_str for code in ["502", "503", "504", "500"]):
                        logger.error(f"Tool {tool_name} received server error: {e}")
                        self.connected = False
                        self._connection_health_ok = False
                        raise ServerRetryableError(
                            f"MCP server {self.config.name} server error: {e}"
                        ) from e

                    # Check for gateway/service errors in message
                    if "gateway" in error_str or "service unavailable" in error_str:
                        logger.error(
                            f"Tool {tool_name} received gateway/service error: {e}"
                        )
                        self.connected = False
                        self._connection_health_ok = False
                        raise ServerRetryableError(
                            f"MCP server {self.config.name} gateway error: {e}"
                        ) from e

                    # Check for MCP SDK async cleanup errors (RuntimeError patterns)
                    # These are caused by anyio task group constraints during error cleanup
                    # See: https://github.com/modelcontextprotocol/python-sdk/issues/915
                    if error_type == "RuntimeError":
                        # Check for cancel scope cross-task exit error
                        if (
                            "cancel scope" in error_str
                            and "different task" in error_str
                        ):
                            logger.error(
                                f"Tool {tool_name} caused MCP SDK cancel scope error: {e}"
                            )
                            self.connected = False
                            self._connection_health_ok = False
                            # Re-raise as-is, retry logic will handle it
                            raise
                        # Check for async generator already running error
                        if "aclose" in error_str and "already running" in error_str:
                            logger.error(
                                f"Tool {tool_name} caused MCP SDK async generator error: {e}"
                            )
                            self.connected = False
                            self._connection_health_ok = False
                            raise
                        # Check for generic async generator cleanup RuntimeError
                        if "asynchronous generator" in error_str:
                            logger.error(
                                f"Tool {tool_name} caused MCP SDK async generator cleanup error: {e}"
                            )
                            self.connected = False
                            self._connection_health_ok = False
                            raise

                    # Check for CancelledError from cancel scope (anyio cleanup)
                    if error_type == "CancelledError" and "cancel scope" in error_str:
                        logger.error(
                            f"Tool {tool_name} caused CancelledError from cancel scope: {e}"
                        )
                        self.connected = False
                        self._connection_health_ok = False
                        raise

                    # Re-raise other exceptions
                    raise

                # Convert result to standard format
                if hasattr(result, "content"):
                    content_list = result.content
                    if content_list:
                        # Extract text content from the result
                        text_content = []
                        for content_item in content_list:
                            if hasattr(content_item, "text"):
                                text_content.append(content_item.text)
                            elif hasattr(content_item, "data"):
                                text_content.append(str(content_item.data))
                        return {"content": "\n".join(text_content)}
                    else:
                        return {
                            "content": "Tool executed successfully but returned no content"
                        }
                else:
                    return {"content": str(result)}
            else:
                raise ValueError(f"Unsupported method: {method}")

        except Exception as e:
            # Enhanced error handling for HTTP timeouts and connection errors
            error_type = type(e).__name__

            # Check for timeout-related errors
            if "timeout" in str(e).lower() or "Timeout" in error_type:
                logger.error(
                    f"Request timed out for HTTP server {self.config.name} (method: {method}): {e}"
                )
                raise TimeoutError(
                    f"MCP server {self.config.name} request timed out: {e}"
                ) from e

            # Check for connection errors
            if "connect" in str(e).lower() or "Connection" in error_type:
                logger.error(
                    f"Connection error for HTTP server {self.config.name} (method: {method}): {e}"
                )
                raise ConnectionError(
                    f"Failed to connect to MCP server {self.config.name}: {e}"
                ) from e

            # Generic error
            logger.error(
                f"Request failed for HTTP server {self.config.name} (method: {method}): {e}"
            )
            raise

    def _create_http_client_factory(self) -> McpHttpClientFactory:
        """Create HTTP client factory with SSL configuration and enhanced timeout handling."""

        def factory(
            headers: dict[str, str] | None = None,
            timeout: httpx.Timeout | None = None,
            auth: httpx.Auth | None = None,
        ) -> httpx.AsyncClient:
            # Merge headers
            merged_headers = {}
            if self.config.headers:
                merged_headers.update(self.config.headers)
            if headers:
                merged_headers.update(headers)

            # Create enhanced timeout configuration
            if timeout is None:
                # Priority for read timeout: idle_timeout > read_timeout > timeout
                effective_read_timeout = (
                    self.config.idle_timeout
                    or self.config.read_timeout
                    or self.config.timeout
                )
                timeout = httpx.Timeout(
                    timeout=self.config.timeout,  # Default timeout
                    connect=self.config.connection_timeout,  # Connection timeout
                    read=effective_read_timeout,  # Read/idle timeout (time between data chunks/SSE events)
                    write=self.config.timeout,  # Write timeout
                    pool=self.config.timeout,  # Pool timeout
                )

            # Create client with SSL configuration and enhanced timeouts
            return httpx.AsyncClient(
                headers=merged_headers,
                timeout=timeout,
                auth=auth,
                verify=self.ssl_verify,  # This controls SSL verification
            )

        return factory
