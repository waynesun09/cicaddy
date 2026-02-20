"""Pytest configuration and shared fixtures."""

import asyncio
import sys
from pathlib import Path

import pytest

# Add src directory to Python path for tests
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables."""
    # Set test environment variables
    test_env = {
        "LOG_LEVEL": "DEBUG",
        "JSON_LOGS": "false",
        "SSL_VERIFY": "false",
        "CRON_MODE": "true",
        "TASK_TYPE": "custom",
        "TASK_SCOPE": "external_tools",
        "TASK_REPORT_FORMAT": "detailed",
    }

    for key, value in test_env.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def mock_mcp_server_config():
    """Mock MCP server configuration for testing."""
    return {
        "name": "test-mcp-server",
        "protocol": "http",
        "endpoint": "https://test-mcp-server.example.com/mcp",
        # Tools are auto-discovered by AI service
        "timeout": 300,
        "headers": {"Authorization": "Bearer test-token"},
    }
