"""Unit tests for stdio MCP transport using official SDK."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cicaddy.config.settings import MCPServerConfig
from cicaddy.mcp_client.transports.stdio import StdioMCPTransport


class TestStdioMCPTransportOfficial:
    """Test cases for StdioMCPTransport using official SDK."""

    @pytest.fixture
    def stdio_config(self):
        """Mock stdio MCP server configuration."""
        return MCPServerConfig(
            name="test-stdio-server",
            protocol="stdio",
            command="uv",
            args=["--directory", "/opt/test-server", "run", "server.py"],
            working_directory="/opt/test-server",
            env_vars={"TEST_ENV": "test_value"},
            timeout=30,
        )

    @pytest.fixture
    def transport(self, stdio_config):
        """Create StdioMCPTransport instance."""
        return StdioMCPTransport(stdio_config)

    def test_transport_initialization(self, transport, stdio_config):
        """Test transport initialization."""
        assert transport.config == stdio_config
        assert transport.connected is False
        assert transport.session is None
        assert transport._session_initialized is False

    @pytest.mark.asyncio
    async def test_connect_success(self, transport):
        """Test successful connection to stdio server using official SDK."""
        # Mock the official SDK's stdio_client
        mock_read_stream = AsyncMock()
        mock_write_stream = AsyncMock()
        mock_stdio_context = AsyncMock()
        mock_stdio_context.__aenter__.return_value = (
            mock_read_stream,
            mock_write_stream,
        )

        # Mock the ClientSession
        mock_session = AsyncMock()
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session

        # Mock session.initialize()
        mock_session.initialize = AsyncMock()

        # Mock session.list_tools() for verification
        mock_tools_result = MagicMock()
        mock_tools_result.tools = []
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)

        with (
            patch(
                "cicaddy.mcp_client.transports.stdio.stdio_client",
                return_value=mock_stdio_context,
            ),
            patch(
                "cicaddy.mcp_client.transports.stdio.ClientSession",
                return_value=mock_session_context,
            ),
        ):
            await transport.connect()

            # Verify stdio_client was called with correct parameters
            # (Check happens inside the patch context)

            # Verify session was initialized
            assert transport.connected is True
            assert transport.session == mock_session
            assert transport._session_initialized is True

            # Verify initialize was called
            mock_session.initialize.assert_called_once()

            # Verify tools list was called for verification
            mock_session.list_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure(self, transport):
        """Test connection failure handling."""
        # Mock stdio_client to raise an exception
        with patch("cicaddy.mcp_client.transports.stdio.stdio_client") as mock_stdio:
            mock_stdio.side_effect = Exception("Failed to spawn process")

            with pytest.raises(Exception, match="Failed to spawn process"):
                await transport.connect()

            assert transport.connected is False
            assert transport.session is None

    @pytest.mark.asyncio
    async def test_disconnect_success(self, transport):
        """Test successful disconnection."""
        # Setup mock contexts
        mock_session_context = AsyncMock()
        mock_stdio_context = AsyncMock()

        transport._session_context = mock_session_context
        transport._stdio_context = mock_stdio_context
        transport.session = AsyncMock()
        transport.connected = True

        await transport.disconnect()

        # Verify session context was exited
        mock_session_context.__aexit__.assert_called_once()

        # Verify stdio_context.__aexit__ was NOT called (to avoid anyio cancel scope errors)
        # Instead, we just clear the reference
        mock_stdio_context.__aexit__.assert_not_called()

        # Verify state cleanup
        assert transport.connected is False
        assert transport.session is None
        assert transport._session_context is None
        assert transport._stdio_context is None

    @pytest.mark.asyncio
    async def test_list_tools_success(self, transport):
        """Test listing tools from MCP server."""
        # Setup mock session
        mock_session = AsyncMock()

        # Mock tool objects
        mock_tool1 = MagicMock()
        mock_tool1.name = "search_code"
        mock_tool1.description = "Search code in repository"
        mock_tool1.inputSchema = None

        mock_tool2 = MagicMock()
        mock_tool2.name = "list_repos"
        mock_tool2.description = "List repositories"
        mock_tool2.inputSchema = MagicMock()
        mock_tool2.inputSchema.model_dump.return_value = {
            "type": "object",
            "properties": {},
        }

        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool1, mock_tool2]

        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)

        transport.session = mock_session
        transport.connected = True

        tools = await transport.list_tools()

        # Verify tools were retrieved
        assert len(tools) == 2
        assert tools[0]["name"] == "search_code"
        assert tools[0]["description"] == "Search code in repository"
        assert tools[0]["server"] == "test-stdio-server"
        assert tools[1]["name"] == "list_repos"
        assert "inputSchema" in tools[1]

    @pytest.mark.asyncio
    async def test_list_tools_with_filter(self, transport):
        """Test listing tools with config filter."""
        # Update config to filter specific tools
        transport.config.tools = ["search_code"]

        # Setup mock session
        mock_session = AsyncMock()

        # Mock tool objects
        mock_tool1 = MagicMock()
        mock_tool1.name = "search_code"
        mock_tool1.description = "Search code"
        mock_tool1.inputSchema = None

        mock_tool2 = MagicMock()
        mock_tool2.name = "list_repos"
        mock_tool2.description = "List repos"
        mock_tool2.inputSchema = None

        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool1, mock_tool2]

        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)

        transport.session = mock_session
        transport.connected = True

        tools = await transport.list_tools()

        # Should only return filtered tool
        assert len(tools) == 1
        assert tools[0]["name"] == "search_code"

    @pytest.mark.asyncio
    async def test_call_tool_success(self, transport):
        """Test successful tool execution."""
        # Setup mock session
        mock_session = AsyncMock()

        # Mock tool result
        mock_content_item = MagicMock()
        mock_content_item.text = "Tool execution result"

        mock_result = MagicMock()
        mock_result.content = [mock_content_item]

        mock_session.call_tool = AsyncMock(return_value=mock_result)

        transport.session = mock_session
        transport.connected = True

        result = await transport.call_tool("search_code", {"query": "test"})

        # Verify tool was called
        mock_session.call_tool.assert_called_once_with(
            "search_code", arguments={"query": "test"}
        )

        # Verify result format
        assert result["tool"] == "search_code"
        assert result["server"] == "test-stdio-server"
        assert result["status"] == "success"
        assert result["content"] == "Tool execution result"

    @pytest.mark.asyncio
    async def test_call_tool_failure(self, transport):
        """Test tool execution failure handling with retry logic."""
        # Setup mock session
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(
            side_effect=Exception("Tool execution failed")
        )

        transport.session = mock_session
        transport.connected = True

        # The base class now raises exceptions after retry attempts
        with pytest.raises(Exception) as exc_info:
            await transport.call_tool("search_code", {"query": "test"})

        # Verify the exception message
        assert "Tool execution failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self, transport):
        """Test tool call when not connected - auto-connect with retry."""
        from cicaddy.mcp_client.retry import ConnectionRetryableError

        transport.connected = False
        transport.session = None

        # The base class tries to auto-connect when not connected
        # This will fail with retry logic since the test server doesn't exist
        with pytest.raises(ConnectionRetryableError):
            await transport.call_tool("search_code", {"query": "test"})

    @pytest.mark.asyncio
    async def test_send_request_tools_list(self, transport):
        """Test _send_request_impl for tools/list method."""
        # Setup mock session
        mock_session = AsyncMock()

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        mock_tool.inputSchema = None

        mock_result = MagicMock()
        mock_result.tools = [mock_tool]

        mock_session.list_tools = AsyncMock(return_value=mock_result)

        transport.session = mock_session
        transport.connected = True

        result = await transport._send_request_impl("tools/list")

        # Verify result structure
        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_send_request_tools_call(self, transport):
        """Test _send_request_impl for tools/call method."""
        # Setup mock session
        mock_session = AsyncMock()

        mock_content = MagicMock()
        mock_content.text = "Result text"

        mock_result = MagicMock()
        mock_result.content = [mock_content]

        mock_session.call_tool = AsyncMock(return_value=mock_result)

        transport.session = mock_session
        transport.connected = True

        result = await transport._send_request_impl(
            "tools/call", {"name": "test_tool", "arguments": {"arg1": "value1"}}
        )

        # Verify tool was called correctly
        mock_session.call_tool.assert_called_once_with(
            "test_tool", arguments={"arg1": "value1"}
        )

        # Verify result structure
        assert "content" in result
        assert len(result["content"]) == 1
        assert result["content"][0]["text"] == "Result text"

    @pytest.mark.asyncio
    async def test_send_request_not_connected(self, transport):
        """Test _send_request_impl when not connected."""
        transport.connected = False
        transport.session = None

        with pytest.raises(ConnectionError, match="Not connected"):
            await transport._send_request_impl("tools/list")

    @pytest.mark.asyncio
    async def test_send_request_unsupported_method(self, transport):
        """Test _send_request_impl with unsupported method."""
        transport.session = AsyncMock()
        transport.connected = True

        with pytest.raises(NotImplementedError, match="not implemented"):
            await transport._send_request_impl("custom/method")
