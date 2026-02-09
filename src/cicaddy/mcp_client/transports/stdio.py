"""Stdio transport using official MCP SDK stdio_client."""

from typing import Any, Dict, Optional, cast

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from cicaddy.config.settings import MCPServerConfig
from cicaddy.utils.logger import get_logger

from ..security import SecurityValidator
from .base import BaseMCPTransport

logger = get_logger(__name__)


class SecurityError(Exception):
    """Exception raised when security validation fails."""

    pass


class StdioMCPTransport(BaseMCPTransport):
    """MCP transport using official MCP SDK stdio_client."""

    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        self._stdio_context = None
        self._session_context = None
        self.session: Optional[ClientSession] = None
        self._read_stream = None
        self._write_stream = None
        self._session_initialized = False

    async def _connect_impl(self) -> None:
        """Establish connection using official stdio_client."""
        logger.info(f"Connecting to stdio MCP server: {self.config.name}")
        logger.debug(
            f"Command: {self.config.command} {' '.join(self.config.args or [])}"
        )

        if not self.config.command:
            raise ValueError(
                f"No command specified for stdio server {self.config.name}"
            )

        # Security validation - validate command for dangerous patterns
        if not SecurityValidator.validate_command(
            self.config.command, self.config.args
        ):
            raise SecurityError(
                f"Command validation failed for {self.config.name}: potentially dangerous command"
            )

        # Validate and sanitize working directory
        validated_cwd = SecurityValidator.sanitize_working_directory(
            self.config.working_directory
        )
        if self.config.working_directory and not validated_cwd:
            raise SecurityError(
                f"Working directory validation failed for {self.config.name}: {self.config.working_directory}"
            )

        # Validate and sanitize environment variables
        validated_env_vars = SecurityValidator.validate_env_vars(
            self.config.env_vars or {}
        )

        # Log validation results
        if self.config.env_vars:
            filtered_count = len(self.config.env_vars) - len(validated_env_vars)
            if filtered_count > 0:
                filtered_vars = set(self.config.env_vars.keys()) - set(
                    validated_env_vars.keys()
                )
                logger.warning(
                    f"Filtered {filtered_count} dangerous environment variables for {self.config.name}: {filtered_vars}"
                )

        # Build server parameters for official SDK with validated values
        server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args or [],
            env=validated_env_vars,  # Use validated environment variables
            cwd=validated_cwd,  # Use sanitized working directory
        )

        logger.debug(
            f"Creating stdio_client with params: command={self.config.command}, "
            f"args={self.config.args}, cwd={self.config.working_directory}"
        )

        # Enter stdio_client context
        self._stdio_context = stdio_client(server_params)
        stdio_cm = cast(Any, self._stdio_context)  # MCP library's async context manager
        self._read_stream, self._write_stream = await stdio_cm.__aenter__()

        logger.debug(
            f"stdio_client connected, creating ClientSession for {self.config.name}"
        )

        # Enter ClientSession context
        self._session_context = ClientSession(self._read_stream, self._write_stream)
        session_cm = cast(
            Any, self._session_context
        )  # ClientSession async context manager
        self.session = await session_cm.__aenter__()

        logger.debug(f"Initializing MCP session for {self.config.name}")

        # Initialize the session
        await self.session.initialize()

        self._session_initialized = True
        logger.info(f"MCP session successfully initialized for {self.config.name}")

        # Verify session by listing tools
        try:
            logger.debug(
                f"Verifying session with tools list call for {self.config.name}"
            )
            tools_response = await self.session.list_tools()
            tools_count = len(tools_response.tools) if tools_response else 0
            logger.info(
                f"Session verification successful: {tools_count} tools available from {self.config.name}"
            )
        except Exception as verify_error:
            logger.warning(
                f"Session verification failed for {self.config.name}: {verify_error}"
            )
            # Don't fail initialization for this, but log the warning

    async def _disconnect_impl(self) -> None:
        """Close connection and cleanup resources."""
        logger.debug(f"Disconnecting from stdio MCP server: {self.config.name}")

        # Exit ClientSession context
        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error exiting ClientSession context: {e}")
            finally:
                self._session_context = None
                self.session = None

        # Clean up stdio_client context references without calling __aexit__
        # This prevents "Attempted to exit cancel scope in a different task" errors
        # Let the MCP library handle its own cleanup during program termination
        if self._stdio_context:
            self._stdio_context = None
            self._read_stream = None
            self._write_stream = None

        self._session_initialized = False
        logger.debug(f"Disconnected from stdio MCP server: {self.config.name}")

    async def _send_request_impl(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send JSON-RPC request via official ClientSession.

        Args:
            method: The JSON-RPC method name
            params: Optional parameters for the request

        Returns:
            Dict containing the response from the server
        """
        if not self.connected or not self.session:
            raise ConnectionError(f"Not connected to stdio server {self.config.name}")

        logger.debug(f"Sending {method} request to {self.config.name}")

        try:
            # Map common MCP methods to ClientSession methods
            if method == "tools/list":
                result = await self.session.list_tools()
                # Convert to dict format expected by our codebase
                tools = []
                for tool in result.tools:
                    # Handle inputSchema which might be dict or Pydantic model
                    input_schema = {}
                    if tool.inputSchema:
                        if hasattr(tool.inputSchema, "model_dump"):
                            input_schema = tool.inputSchema.model_dump()
                        elif isinstance(tool.inputSchema, dict):
                            input_schema = tool.inputSchema
                        else:
                            # Log warning for unexpected types and default to empty dict
                            logger.warning(
                                f"Unexpected type for tool.inputSchema: {type(tool.inputSchema)}. "
                                f"Defaulting to empty dict for tool {tool.name}"
                            )
                            input_schema = {}

                    tools.append(
                        {
                            "name": tool.name,
                            "description": tool.description or "",
                            "inputSchema": input_schema,
                        }
                    )
                return {"tools": tools}

            elif method == "tools/call":
                if not params or "name" not in params:
                    raise ValueError("tools/call requires 'name' parameter")

                tool_name = params["name"]
                arguments = params.get("arguments", {})

                logger.debug(f"Calling tool {tool_name} with arguments: {arguments}")

                result = await self.session.call_tool(tool_name, arguments=arguments)

                # Convert to dict format expected by our codebase
                content_items = []
                for item in result.content:
                    if hasattr(item, "text"):
                        content_items.append({"type": "text", "text": item.text})
                    elif hasattr(item, "data"):
                        content_items.append({"type": "resource", "data": item.data})

                return {
                    "content": content_items,
                    "isError": getattr(result, "isError", False),
                }

            elif method == "resources/list":
                result = await self.session.list_resources()
                resources = []
                for resource in result.resources:
                    resources.append(
                        {
                            "uri": str(resource.uri),
                            "name": resource.name or "",
                            "description": resource.description or "",
                            "mimeType": resource.mimeType,
                        }
                    )
                return {"resources": resources}

            elif method == "prompts/list":
                result = await self.session.list_prompts()
                prompts = []
                for prompt in result.prompts:
                    prompts.append(
                        {
                            "name": prompt.name,
                            "description": prompt.description or "",
                            "arguments": [arg.model_dump() for arg in prompt.arguments]
                            if prompt.arguments
                            else [],
                        }
                    )
                return {"prompts": prompts}

            else:
                # For other methods, we'd need to extend ClientSession or use raw JSON-RPC
                logger.warning(f"Unsupported method {method} for official stdio_client")
                raise NotImplementedError(
                    f"Method {method} not implemented in official stdio transport"
                )

        except Exception as e:
            logger.error(
                f"Request failed for {self.config.name} (method: {method}): {e}"
            )
            raise
