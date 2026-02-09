"""MCP client package with unified transport support."""

from .client import MCPClient, MCPClientManager
from .retry import (
    RetryConfig,
    RetryTimeoutError,
    RetryableError,
    retry_async,
    retry_config_from_mcp,
)
from .transports import (
    BaseMCPTransport,
    HTTPMCPTransport,
    SSEMCPTransport,
    StdioMCPTransport,
    WebSocketMCPTransport,
)

__all__ = [
    "MCPClient",
    "MCPClientManager",
    # Retry utilities
    "RetryConfig",
    "RetryTimeoutError",
    "RetryableError",
    "retry_async",
    "retry_config_from_mcp",
    # Backward compatibility aliases
    "OfficialMCPClient",
    "OfficialMCPClientManager",
    "BaseMCPTransport",
    "HTTPMCPTransport",
    "SSEMCPTransport",
    "StdioMCPTransport",
    "WebSocketMCPTransport",
]

# Backward compatibility aliases
OfficialMCPClient = MCPClient
OfficialMCPClientManager = MCPClientManager
