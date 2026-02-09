"""Unit tests for official MCP client."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from cicaddy.config.settings import MCPServerConfig
from cicaddy.mcp_client.client import OfficialMCPClient, OfficialMCPClientManager


class TestOfficialMCPClient:
    """Test cases for OfficialMCPClient."""

    @pytest.fixture
    def mock_config(self):
        """Mock MCP server configuration."""
        return MCPServerConfig(
            name="test-server",
            protocol="http",
            endpoint="https://test-mcp-server.com/mcp",
            tools=["test_tool_1", "test_tool_2"],
            timeout=30,
            headers={"Authorization": "Bearer test-token"},
        )

    @pytest.fixture
    def client(self, mock_config):
        """Create OfficialMCPClient instance."""
        return OfficialMCPClient(mock_config, ssl_verify=False)  # Explicit for testing

    @pytest.mark.asyncio
    async def test_client_initialization(self, client, mock_config):
        """Test client initialization."""
        assert client.config == mock_config
        assert client.ssl_verify is False
        assert client.session is None
        assert client.connected is False

    def test_http_client_factory(self, client):
        """Test HTTP client factory creation."""
        factory = client._create_http_client_factory()

        # Test factory function
        http_client = factory(headers={"Custom": "header"}, timeout=None, auth=None)

        assert http_client is not None
        # Check that headers are merged
        expected_headers = {"Authorization": "Bearer test-token", "Custom": "header"}
        for key, value in expected_headers.items():
            assert http_client.headers.get(key) == value

    @pytest.mark.asyncio
    async def test_connect_success(self, client):
        """Test successful connection to MCP server."""
        with (
            patch(
                "cicaddy.mcp_client.client.official_mcp_sse_client"
            ) as mock_sse_client,
            patch(
                "cicaddy.mcp_client.client.OfficialMCPClientSession"
            ) as mock_client_session,
        ):
            # Setup mocks
            mock_streams = AsyncMock()
            mock_sse_client.return_value = mock_streams

            mock_session_instance = AsyncMock()
            mock_client_session.return_value = mock_session_instance

            # Mock context manager behavior
            mock_streams.__aenter__ = AsyncMock(return_value=("stream1", "stream2"))
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.initialize = AsyncMock()

            # Test connection
            await client.connect()

            # Verify connection state
            assert client.connected is True
            assert client.session == mock_session_instance

            # Verify sse_client was called with correct parameters
            mock_sse_client.assert_called_once()
            call_args = mock_sse_client.call_args
            assert call_args[0][0] == client.config.endpoint
            assert "headers" in call_args[1]
            assert "httpx_client_factory" in call_args[1]

    @pytest.mark.asyncio
    async def test_connect_failure(self, client):
        """Test connection failure handling."""
        with patch(
            "cicaddy.mcp_client.client.official_mcp_sse_client"
        ) as mock_sse_client:
            mock_sse_client.side_effect = Exception("Connection failed")

            with pytest.raises(Exception, match="Connection failed"):
                await client.connect()

            assert client.connected is False

    @pytest.mark.asyncio
    async def test_list_tools_success(self, client):
        """Test successful tool listing."""
        # Mock successful connection
        client.connected = True
        client.session = AsyncMock()

        # Mock tools response
        mock_tool1 = Mock()
        mock_tool1.name = "test_tool_1"
        mock_tool1.description = "Test tool 1 description"

        mock_tool2 = Mock()
        mock_tool2.name = "test_tool_2"
        mock_tool2.description = "Test tool 2 description"

        mock_result = Mock()
        mock_result.tools = [mock_tool1, mock_tool2]

        client.session.list_tools.return_value = mock_result

        # Test tool listing
        tools = await client.list_tools()

        # Verify results
        assert len(tools) == 2
        assert tools[0]["name"] == "test_tool_1"
        assert tools[0]["description"] == "Test tool 1 description"
        assert tools[0]["server"] == "test-server"
        assert tools[1]["name"] == "test_tool_2"
        assert tools[1]["description"] == "Test tool 2 description"
        assert tools[1]["server"] == "test-server"

    @pytest.mark.asyncio
    async def test_list_tools_with_filtering(self, client):
        """Test tool listing with filtering based on config."""
        client.connected = True
        client.session = AsyncMock()

        # Mock tools response with more tools than configured
        mock_tools = []
        tool_names = [
            "test_tool_1",
            "test_tool_2",
            "other_tool_3",
            "other_tool_4",
            "other_tool_5",
        ]
        for i, name in enumerate(tool_names):
            tool = Mock()
            tool.name = name
            tool.description = f"Tool {i} description"
            mock_tools.append(tool)

        mock_result = Mock()
        mock_result.tools = mock_tools
        client.session.list_tools.return_value = mock_result

        # Test tool listing (should filter to only configured tools)
        tools = await client.list_tools()

        # Should only return tools that are in the config
        assert len(tools) == 2  # Only test_tool_1 and test_tool_2 from config
        tool_names = [tool["name"] for tool in tools]
        assert "test_tool_1" in tool_names
        assert "test_tool_2" in tool_names

    @pytest.mark.asyncio
    async def test_list_tools_failure_fallback(self, client):
        """Test tool listing failure with fallback to configured tools."""
        client.connected = True
        client.session = AsyncMock()
        client.session.list_tools.side_effect = Exception("Failed to list tools")

        # Test tool listing with failure
        tools = await client.list_tools()

        # Should return configured tools as fallback
        assert len(tools) == 2
        assert tools[0]["name"] == "test_tool_1"
        assert tools[0]["server"] == "test-server"
        assert tools[1]["name"] == "test_tool_2"
        assert tools[1]["server"] == "test-server"

    @pytest.mark.asyncio
    async def test_call_tool_success(self, client):
        """Test successful tool call."""
        client.connected = True
        client.session = AsyncMock()

        # Mock tool call response
        mock_content = Mock()
        mock_content.text = "Tool execution successful: result data"

        mock_result = Mock()
        mock_result.content = [mock_content]

        client.session.call_tool.return_value = mock_result

        # Test tool call
        result = await client.call_tool("test_tool_1", {"param1": "value1"})

        # Verify results
        assert result["content"] == "Tool execution successful: result data"
        assert result["tool"] == "test_tool_1"
        assert result["server"] == "test-server"
        assert result["status"] == "success"

        # Verify session.call_tool was called correctly (includes progress_callback)
        client.session.call_tool.assert_called_once()
        call_args = client.session.call_tool.call_args
        assert call_args[0] == ("test_tool_1", {"param1": "value1"})
        assert "progress_callback" in call_args[1]

    @pytest.mark.asyncio
    async def test_call_tool_no_content(self, client):
        """Test tool call with no content returned."""
        client.connected = True
        client.session = AsyncMock()

        mock_result = Mock()
        mock_result.content = []
        client.session.call_tool.return_value = mock_result

        # Test tool call
        result = await client.call_tool("test_tool_1", {})

        # Verify results
        assert result["content"] == "Tool executed successfully but returned no content"
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_call_tool_failure(self, client):
        """Test tool call failure."""
        client.connected = True
        client.session = AsyncMock()
        client.session.call_tool.side_effect = Exception("Tool call failed")

        # Test tool call failure
        with pytest.raises(Exception, match="Tool call failed"):
            await client.call_tool("test_tool_1", {})

    @pytest.mark.asyncio
    async def test_disconnect(self, client):
        """Test client disconnection."""
        # Setup mock session and connection
        mock_session = AsyncMock()
        mock_session.__aexit__ = AsyncMock()
        client.session = mock_session

        mock_streams = AsyncMock()
        mock_streams.__aexit__ = AsyncMock()
        client.streams = mock_streams
        client.sse_connection = Mock()

        client.connected = True

        # Test disconnection
        await client.disconnect()

        # Verify disconnection
        assert client.connected is False
        assert client.session is None
        mock_session.__aexit__.assert_called_once()
        mock_streams.__aexit__.assert_called_once()


class TestOfficialMCPClientManager:
    """Test cases for OfficialMCPClientManager."""

    @pytest.fixture
    def mock_configs(self):
        """Mock MCP server configurations."""
        return [
            MCPServerConfig(
                name="server1",
                protocol="http",
                endpoint="https://server1.com/mcp",
                tools=["tool1", "tool2"],
                timeout=30,
            ),
            MCPServerConfig(
                name="server2",
                protocol="http",
                endpoint="https://server2.com/mcp",
                tools=["tool3", "tool4"],
                timeout=30,
            ),
        ]

    @pytest.fixture
    def manager(self, mock_configs):
        """Create OfficialMCPClientManager instance."""
        return OfficialMCPClientManager(
            mock_configs, ssl_verify=False
        )  # Explicit for testing

    def test_manager_initialization(self, manager, mock_configs):
        """Test manager initialization."""
        assert manager.configs == mock_configs
        assert manager.ssl_verify is False
        assert manager.clients == {}

    @pytest.mark.asyncio
    async def test_initialize_success(self, manager):
        """Test successful initialization of all clients."""
        with patch("cicaddy.mcp_client.client.OfficialMCPClient") as mock_client_class:
            # Setup mock clients
            mock_client1 = AsyncMock()
            mock_client2 = AsyncMock()
            mock_client_class.side_effect = [mock_client1, mock_client2]

            # Test initialization
            await manager.initialize()

            # Verify clients were created and connected
            assert len(manager.clients) == 2
            assert "server1" in manager.clients
            assert "server2" in manager.clients

            mock_client1.connect.assert_called_once()
            mock_client2.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_partial_failure(self, manager):
        """Test initialization with one client failing."""
        with patch("cicaddy.mcp_client.client.OfficialMCPClient") as mock_client_class:
            # Setup mock clients - one succeeds, one fails
            mock_client1 = AsyncMock()
            mock_client2 = AsyncMock()
            mock_client2.connect.side_effect = Exception("Connection failed")

            mock_client_class.side_effect = [mock_client1, mock_client2]

            # Test initialization
            await manager.initialize()

            # Verify only successful client is added
            assert len(manager.clients) == 1
            assert "server1" in manager.clients
            assert "server2" not in manager.clients

    @pytest.mark.asyncio
    async def test_list_tools(self, manager):
        """Test listing tools from specific server."""
        # Setup mock client
        mock_client = AsyncMock()
        mock_tools = [{"name": "tool1", "server": "server1"}]
        mock_client.list_tools.return_value = mock_tools
        manager.clients["server1"] = mock_client

        # Test tool listing
        tools = await manager.list_tools("server1")

        # Verify results
        assert tools == mock_tools
        mock_client.list_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_tools_server_not_found(self, manager):
        """Test listing tools from non-existent server."""
        tools = await manager.list_tools("nonexistent")
        assert tools == []

    @pytest.mark.asyncio
    async def test_call_tool(self, manager):
        """Test calling tool on specific server."""
        # Setup mock client
        mock_client = AsyncMock()
        mock_result = {"content": "success", "status": "success"}
        mock_client.call_tool.return_value = mock_result
        manager.clients["server1"] = mock_client

        # Test tool call
        result = await manager.call_tool("server1", "tool1", {"param": "value"})

        # Verify results
        assert result == mock_result
        mock_client.call_tool.assert_called_once_with("tool1", {"param": "value"})

    @pytest.mark.asyncio
    async def test_call_tool_server_not_found(self, manager):
        """Test calling tool on non-existent server."""
        with pytest.raises(
            ValueError, match="Official MCP server nonexistent not found"
        ):
            await manager.call_tool("nonexistent", "tool1", {})

    def test_get_server_names(self, manager):
        """Test getting list of connected server names."""
        # Add some mock clients
        manager.clients["server1"] = Mock()
        manager.clients["server2"] = Mock()

        names = manager.get_server_names()
        assert sorted(names) == ["server1", "server2"]

    @pytest.mark.asyncio
    async def test_cleanup(self, manager):
        """Test cleanup of all clients."""
        # Setup mock clients
        mock_client1 = AsyncMock()
        mock_client2 = AsyncMock()
        manager.clients["server1"] = mock_client1
        manager.clients["server2"] = mock_client2

        # Test cleanup
        await manager.cleanup()

        # Verify cleanup
        mock_client1.disconnect.assert_called_once()
        mock_client2.disconnect.assert_called_once()
        assert manager.clients == {}
