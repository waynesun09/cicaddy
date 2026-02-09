"""Lightweight tool system for AI agents.

This module provides a decorator-based tool system inspired by LangChain,
but without external dependencies. Tools defined here integrate seamlessly
with the existing MCP-based tool infrastructure.

Example usage:
    from cicaddy.tools import tool, ToolRegistry

    @tool
    def my_tool(query: str, limit: int = 10) -> str:
        '''Search for something.

        Args:
            query: The search query
            limit: Maximum results to return
        '''
        return f"Found {limit} results for {query}"

    # Register and use
    registry = ToolRegistry()
    registry.register(my_tool)
    tools = registry.get_tools()  # MCP-compatible format

Local File Tools:
    from cicaddy.tools import create_local_file_registry

    # Create registry with glob_files and read_file tools
    registry = create_local_file_registry("/path/to/working/dir")
    tools = registry.get_tools()
"""

from .decorator import Tool, tool
from .file_tools import (
    create_local_file_registry,
    get_working_directory,
    glob_files,
    read_file,
    set_working_directory,
)
from .registry import ToolRegistry

__all__ = [
    # Core
    "tool",
    "Tool",
    "ToolRegistry",
    # Local file tools
    "glob_files",
    "read_file",
    "create_local_file_registry",
    "set_working_directory",
    "get_working_directory",
]
