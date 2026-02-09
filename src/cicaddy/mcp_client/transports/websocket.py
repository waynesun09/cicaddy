"""WebSocket transport for MCP communication via WebSocket connections."""

from typing import Any, Dict, Optional, cast

from mcp import ClientSession as OfficialMCPClientSession

from cicaddy.config.settings import MCPServerConfig
from cicaddy.utils.logger import get_logger

from .base import BaseMCPTransport

logger = get_logger(__name__)


class WebSocketMCPTransport(BaseMCPTransport):
    """MCP transport using WebSocket connections."""

    def __init__(self, config: MCPServerConfig, ssl_verify: bool = True):
        super().__init__(config)
        self.ssl_verify = ssl_verify
        self.session: Optional[OfficialMCPClientSession] = None
        self._session_context = None
        self.streams = None
        self.ws_connection = None

    async def _connect_impl(self) -> None:
        """Establish WebSocket connection to MCP server (implementation-specific)."""
        logger.info(f"Connecting to WebSocket MCP server: {self.config.name}")

        if not self.config.endpoint:
            raise ValueError(
                f"No endpoint specified for WebSocket server {self.config.name}"
            )

        # Import the websocket client with proper error handling
        try:
            from mcp.client.websocket import (
                websocket_client as official_mcp_websocket_client,
            )
        except ImportError as e:
            if "websockets" in str(e):
                raise ValueError(
                    "WebSocket protocol requires additional dependencies. "
                    "Install with: pip install .cicaddy[websocket]. "
                    "or: pip install 'websockets>=10.0'"
                ) from e
            else:
                raise ValueError(f"WebSocket support not available: {e}") from e

        # Connect using the official MCP library
        self.streams = official_mcp_websocket_client(self.config.endpoint)

        # Enter the context managers
        # The streams object is an async context manager from the MCP library
        streams_cm = cast(Any, self.streams)  # MCP library's async context manager
        self.ws_connection = await streams_cm.__aenter__()
        self._session_context = OfficialMCPClientSession(*self.ws_connection)
        session_cm = cast(
            Any, self._session_context
        )  # ClientSession async context manager
        self.session = await session_cm.__aenter__()

        # Initialize the session
        await self.session.initialize()

        # Start heartbeat monitoring if configured
        await self.start_heartbeat()

    async def _disconnect_impl(self) -> None:
        """Disconnect from WebSocket MCP server (implementation-specific)."""
        logger.debug(f"Disconnecting from WebSocket MCP server: {self.config.name}")

        # Exit ClientSession context properly
        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error exiting ClientSession context: {e}")
            finally:
                self._session_context = None
                self.session = None

        # Clean up ws_connection and streams WITHOUT calling __aexit__
        # This prevents "Attempted to exit cancel scope in a different task" errors
        # Let the MCP library handle its own cleanup during program termination
        if self.ws_connection:
            self.ws_connection = None
        if self.streams:
            self.streams = None

    async def _send_request_impl(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send request via WebSocket connection using the official MCP library (implementation-specific).

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
            raise ConnectionError(
                f"Not connected to WebSocket server {self.config.name}"
            )

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
                result = await self.session.call_tool(
                    tool_name, arguments, progress_callback=progress_callback
                )

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
            logger.error(f"Request failed for WebSocket server {self.config.name}: {e}")
            raise
