"""Unit tests for unified MCP client."""

from unittest.mock import AsyncMock, patch

import pytest

from cicaddy.config.settings import MCPServerConfig
from cicaddy.mcp_client.client import MCPClient, MCPClientManager


class TestMCPClient:
    """Test cases for unified MCP client."""

    @pytest.fixture
    def sse_config(self):
        """SSE server configuration."""
        return MCPServerConfig(
            name="test-sse-server",
            protocol="sse",
            endpoint="https://test-server.com/mcp/sse",
            headers={"Authorization": "Bearer test-token"},
            timeout=30,
        )

    @pytest.fixture
    def stdio_config(self):
        """Stdio server configuration."""
        return MCPServerConfig(
            name="test-stdio-server",
            protocol="stdio",
            command="uv",
            args=["run", "server.py"],
            working_directory="/opt/test-server",
            timeout=30,
        )

    @pytest.fixture
    def websocket_config(self):
        """WebSocket server configuration."""
        return MCPServerConfig(
            name="test-websocket-server",
            protocol="websocket",
            endpoint="wss://test-server.com/ws",
            headers={"Authorization": "Bearer ws-token"},
            timeout=60,
        )

    @pytest.fixture
    def http_config(self):
        """HTTP server configuration."""
        return MCPServerConfig(
            name="test-http-server",
            protocol="http",
            endpoint="https://test-server.com/api/v1",
            headers={
                "Authorization": "Bearer test-fake-token-for-unit-test",  # gitleaks:allow
                "Content-Type": "application/json",
            },
            timeout=30,
        )

    def test_client_initialization_sse(self, sse_config):
        """Test client initialization with SSE transport."""
        client = MCPClient(sse_config, ssl_verify=False)  # Explicit for testing

        assert client.config == sse_config
        assert client.ssl_verify is False

        # Should create SSE transport
        from cicaddy.mcp_client.transports.sse import SSEMCPTransport

        assert isinstance(client.transport, SSEMCPTransport)

    def test_client_initialization_stdio(self, stdio_config):
        """Test client initialization with stdio transport."""
        client = MCPClient(stdio_config)

        assert client.config == stdio_config

        # Should create stdio transport
        from cicaddy.mcp_client.transports.stdio import StdioMCPTransport

        assert isinstance(client.transport, StdioMCPTransport)

    def test_client_initialization_websocket(self, websocket_config):
        """Test client initialization with WebSocket transport."""
        client = MCPClient(websocket_config, ssl_verify=True)

        assert client.config == websocket_config
        assert client.ssl_verify is True

        # Should create WebSocket transport
        from cicaddy.mcp_client.transports.websocket import WebSocketMCPTransport

        assert isinstance(client.transport, WebSocketMCPTransport)

    def test_client_initialization_http(self, http_config):
        """Test client initialization with HTTP transport."""
        client = MCPClient(http_config, ssl_verify=True)

        assert client.config == http_config
        assert client.ssl_verify is True

        # Should create HTTP transport
        from cicaddy.mcp_client.transports.http import HTTPMCPTransport

        assert isinstance(client.transport, HTTPMCPTransport)

    def test_client_initialization_unknown_protocol(self):
        """Test client initialization with unknown protocol."""
        # The protocol validation now happens at the Pydantic model level
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="String should match pattern"):
            MCPServerConfig(name="test-unknown-server", protocol="unknown", timeout=30)

    @pytest.mark.asyncio
    async def test_client_connect(self, sse_config):
        """Test client connect method."""
        client = MCPClient(sse_config)

        # Mock transport
        client.transport = AsyncMock()

        await client.connect()

        client.transport.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_disconnect(self, sse_config):
        """Test client disconnect method."""
        client = MCPClient(sse_config)

        # Mock transport
        client.transport = AsyncMock()

        await client.disconnect()

        client.transport.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_list_tools(self, sse_config):
        """Test client list_tools method."""
        client = MCPClient(sse_config)

        # Mock transport
        client.transport = AsyncMock()
        mock_tools = [{"name": "tool1", "server": "test-sse-server"}]
        client.transport.list_tools.return_value = mock_tools

        tools = await client.list_tools()

        assert tools == mock_tools
        client.transport.list_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_call_tool(self, sse_config):
        """Test client call_tool method."""
        client = MCPClient(sse_config)

        # Mock transport
        client.transport = AsyncMock()
        mock_result = {"content": "test result", "status": "success"}
        client.transport.call_tool.return_value = mock_result

        result = await client.call_tool("test_tool", {"param": "value"})

        assert result == mock_result
        client.transport.call_tool.assert_called_once_with(
            "test_tool", {"param": "value"}
        )

    def test_client_connected_property(self, sse_config):
        """Test client connected property."""
        client = MCPClient(sse_config)

        # Mock transport
        client.transport = AsyncMock()
        client.transport.connected = True

        assert client.connected is True


class TestMCPClientManager:
    """Test cases for MCP client manager."""

    @pytest.fixture
    def mixed_configs(self):
        """Mixed SSE, stdio, websocket, and HTTP configurations."""
        return [
            MCPServerConfig(
                name="sse-server",
                protocol="sse",
                endpoint="https://sse-server.com/mcp/sse",
                timeout=30,
            ),
            MCPServerConfig(
                name="stdio-server",
                protocol="stdio",
                command="python",
                args=["-m", "test_server"],
                timeout=30,
            ),
            MCPServerConfig(
                name="websocket-server",
                protocol="websocket",
                endpoint="wss://ws-server.com/ws",
                timeout=60,
            ),
            MCPServerConfig(
                name="http-server",
                protocol="http",
                endpoint="https://http-server.com/api/v1",
                timeout=30,
            ),
        ]

    def test_manager_initialization(self, mixed_configs):
        """Test manager initialization."""
        manager = MCPClientManager(mixed_configs, ssl_verify=True)

        assert manager.configs == mixed_configs
        assert manager.ssl_verify is True
        assert manager.clients == {}

    @pytest.mark.asyncio
    async def test_manager_initialize_success(self, mixed_configs):
        """Test successful initialization of all clients."""
        manager = MCPClientManager(mixed_configs)

        # Mock MCPClient class
        with patch("cicaddy.mcp_client.client.MCPClient") as mock_client_class:
            mock_client1 = AsyncMock()
            mock_client2 = AsyncMock()
            mock_client3 = AsyncMock()
            mock_client4 = AsyncMock()
            mock_client_class.side_effect = [
                mock_client1,
                mock_client2,
                mock_client3,
                mock_client4,
            ]

            await manager.initialize()

            # Verify clients were created and connected
            assert len(manager.clients) == 4
            assert "sse-server" in manager.clients
            assert "stdio-server" in manager.clients
            assert "websocket-server" in manager.clients
            assert "http-server" in manager.clients

            mock_client1.connect.assert_called_once()
            mock_client2.connect.assert_called_once()
            mock_client3.connect.assert_called_once()
            mock_client4.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_manager_initialize_partial_failure(self, mixed_configs):
        """Test initialization with one client failing."""
        manager = MCPClientManager(mixed_configs)

        with patch("cicaddy.mcp_client.client.MCPClient") as mock_client_class:
            mock_client1 = AsyncMock()
            mock_client2 = AsyncMock()
            mock_client2.connect.side_effect = Exception("Connection failed")

            mock_client_class.side_effect = [mock_client1, mock_client2]

            await manager.initialize()

            # Only successful client should be added
            assert len(manager.clients) == 1
            assert "sse-server" in manager.clients
            assert "stdio-server" not in manager.clients

    @pytest.mark.asyncio
    async def test_manager_list_tools(self, mixed_configs):
        """Test listing tools from specific server."""
        manager = MCPClientManager(mixed_configs)

        # Add mock client
        mock_client = AsyncMock()
        mock_tools = [{"name": "tool1", "server": "sse-server"}]
        mock_client.list_tools.return_value = mock_tools
        manager.clients["sse-server"] = mock_client

        tools = await manager.list_tools("sse-server")

        assert tools == mock_tools
        mock_client.list_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_manager_list_tools_server_not_found(self, mixed_configs):
        """Test listing tools from non-existent server."""
        manager = MCPClientManager(mixed_configs)

        tools = await manager.list_tools("nonexistent")

        assert tools == []

    @pytest.mark.asyncio
    async def test_manager_call_tool(self, mixed_configs):
        """Test calling tool on specific server."""
        manager = MCPClientManager(mixed_configs)

        # Add mock client
        mock_client = AsyncMock()
        mock_result = {"content": "success", "status": "success"}
        mock_client.call_tool.return_value = mock_result
        manager.clients["stdio-server"] = mock_client

        result = await manager.call_tool(
            "stdio-server", "test_tool", {"param": "value"}
        )

        assert result == mock_result
        mock_client.call_tool.assert_called_once_with("test_tool", {"param": "value"})

    @pytest.mark.asyncio
    async def test_manager_call_tool_server_not_found(self, mixed_configs):
        """Test calling tool on non-existent server."""
        manager = MCPClientManager(mixed_configs)

        with pytest.raises(ValueError, match="MCP server nonexistent not found"):
            await manager.call_tool("nonexistent", "tool", {})

    def test_manager_get_server_names(self, mixed_configs):
        """Test getting list of server names."""
        manager = MCPClientManager(mixed_configs)

        # Add mock clients
        manager.clients["server1"] = AsyncMock()
        manager.clients["server2"] = AsyncMock()

        names = manager.get_server_names()

        assert sorted(names) == ["server1", "server2"]

    def test_manager_get_servers_by_protocol(self, mixed_configs):
        """Test getting servers by protocol."""
        manager = MCPClientManager(mixed_configs)

        # Add mock clients with configs
        mock_client1 = AsyncMock()
        mock_client1.config = MCPServerConfig(
            name="sse1", protocol="sse", endpoint="test"
        )
        mock_client2 = AsyncMock()
        mock_client2.config = MCPServerConfig(
            name="stdio1", protocol="stdio", command="test"
        )
        mock_client3 = AsyncMock()
        mock_client3.config = MCPServerConfig(
            name="sse2", protocol="sse", endpoint="test2"
        )

        manager.clients["sse1"] = mock_client1
        manager.clients["stdio1"] = mock_client2
        manager.clients["sse2"] = mock_client3

        sse_servers = manager.get_servers_by_protocol("sse")
        stdio_servers = manager.get_servers_by_protocol("stdio")

        assert sorted(sse_servers) == ["sse1", "sse2"]
        assert stdio_servers == ["stdio1"]

    @pytest.mark.asyncio
    async def test_manager_cleanup(self, mixed_configs):
        """Test cleanup of all clients."""
        manager = MCPClientManager(mixed_configs)

        # Add mock clients
        mock_client1 = AsyncMock()
        mock_client2 = AsyncMock()
        manager.clients["client1"] = mock_client1
        manager.clients["client2"] = mock_client2

        await manager.cleanup()

        # Verify cleanup
        mock_client1.disconnect.assert_called_once()
        mock_client2.disconnect.assert_called_once()
        assert manager.clients == {}
