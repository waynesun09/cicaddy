"""Unit tests for agent factory MCP server selection."""

import json
import os
from unittest.mock import patch

from cicaddy.agent.factory import AgentFactory


class TestAgentFactoryMCPSelection:
    """Test cases for MCP server selection in agent factory."""

    def test_get_mcp_servers_no_config(self):
        """Test MCP server selection with no configuration."""
        with patch.dict(os.environ, {}, clear=True):
            servers = AgentFactory.get_mcp_servers_for_context("merge_request")

            # Should return empty list without MCP_SERVERS_CONFIG
            assert servers == []

    def test_get_mcp_servers_empty_config(self):
        """Test MCP server selection with empty JSON array."""
        with patch.dict(os.environ, {"MCP_SERVERS_CONFIG": "[]"}, clear=True):
            servers = AgentFactory.get_mcp_servers_for_context("merge_request")

            # Should return empty list for empty configuration
            assert servers == []

    def test_get_mcp_servers_single_server(self):
        """Test MCP server selection with single server configured."""
        mcp_config = [
            {
                "name": "test-server",
                "protocol": "sse",
                "endpoint": "https://test-mcp.company.com/mcp/sse",
                "timeout": 120,
            }
        ]

        with patch.dict(
            os.environ, {"MCP_SERVERS_CONFIG": json.dumps(mcp_config)}, clear=True
        ):
            servers = AgentFactory.get_mcp_servers_for_context("merge_request")

            assert len(servers) == 1
            assert servers[0].name == "test-server"
            assert servers[0].protocol == "sse"
            assert servers[0].endpoint == "https://test-mcp.company.com/mcp/sse"
            assert servers[0].timeout == 120

    def test_get_mcp_servers_multiple_servers(self):
        """Test MCP server selection with multiple servers configured."""
        mcp_config = [
            {
                "name": "monitoring-server",
                "protocol": "sse",
                "endpoint": "https://monitoring-mcp.company.com/mcp/sse",
                "timeout": 300,
            },
            {
                "name": "analysis-server",
                "protocol": "stdio",
                "command": "python",
                "args": ["-m", "analysis_server"],
                "working_directory": "/opt/analysis",
                "timeout": 180,
            },
        ]

        with patch.dict(
            os.environ, {"MCP_SERVERS_CONFIG": json.dumps(mcp_config)}, clear=True
        ):
            servers = AgentFactory.get_mcp_servers_for_context(
                "task", "full_project", "custom"
            )

            assert len(servers) == 2
            assert servers[0].name == "monitoring-server"
            assert servers[0].protocol == "sse"
            assert servers[1].name == "analysis-server"
            assert servers[1].protocol == "stdio"
            assert servers[1].command == "python"
            assert servers[1].args == ["-m", "analysis_server"]

    def test_get_mcp_servers_invalid_json(self):
        """Test MCP server selection with invalid JSON configuration."""
        with patch.dict(
            os.environ, {"MCP_SERVERS_CONFIG": "invalid json {"}, clear=True
        ):
            servers = AgentFactory.get_mcp_servers_for_context("merge_request")

            # Should return empty list for invalid JSON
            assert servers == []

    def test_get_mcp_servers_malformed_config(self):
        """Test MCP server selection with malformed server configuration."""
        mcp_config = [
            {
                "name": "valid-server",
                "protocol": "sse",
                "endpoint": "https://valid-mcp.company.com/mcp/sse",
                "timeout": 120,
            },
            {
                # Missing required fields
                "name": "invalid-server"
                # No protocol specified
            },
        ]

        with patch.dict(
            os.environ, {"MCP_SERVERS_CONFIG": json.dumps(mcp_config)}, clear=True
        ):
            servers = AgentFactory.get_mcp_servers_for_context("merge_request")

            # Should only include valid server configurations
            assert len(servers) == 1
            assert servers[0].name == "valid-server"

    def test_get_mcp_servers_agent_type_agnostic(self):
        """Test that MCP server selection is agent type agnostic."""
        mcp_config = [
            {
                "name": "generic-server",
                "protocol": "sse",
                "endpoint": "https://generic-mcp.company.com/mcp/sse",
                "timeout": 60,
            }
        ]

        with patch.dict(
            os.environ, {"MCP_SERVERS_CONFIG": json.dumps(mcp_config)}, clear=True
        ):
            # All agent types should get the same server configuration
            mr_servers = AgentFactory.get_mcp_servers_for_context("merge_request")
            task_servers = AgentFactory.get_mcp_servers_for_context(
                "task", "full_project", "security_audit"
            )
            branch_servers = AgentFactory.get_mcp_servers_for_context("branch_review")

            assert len(mr_servers) == 1
            assert len(task_servers) == 1
            assert len(branch_servers) == 1
            assert (
                mr_servers[0].name
                == task_servers[0].name
                == branch_servers[0].name
                == "generic-server"
            )
