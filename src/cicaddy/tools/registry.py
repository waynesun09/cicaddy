"""Tool registry for managing and executing tools.

Provides a central registry for tool registration, discovery,
and execution with MCP-compatible interfaces.
"""

from typing import Any, Callable, Dict, List, Optional, Union

from cicaddy.utils.logger import get_logger

from .decorator import Tool, tool

logger = get_logger(__name__)


class ToolRegistry:
    """Registry for managing agent tools.

    The registry provides:
    - Tool registration (decorated functions or Tool instances)
    - MCP-compatible tool listing
    - Tool execution by name
    - Integration with existing MCP tool infrastructure
    """

    def __init__(self, server_name: str = "local"):
        """Initialize the tool registry.

        Args:
            server_name: Name to use as the 'server' field in MCP format.
                        Defaults to 'local' for built-in tools.
        """
        self.server_name = server_name
        self._tools: Dict[str, Tool] = {}

    def register(self, tool_or_func: Union[Tool, Callable], **kwargs) -> Tool:
        """Register a tool with the registry.

        Args:
            tool_or_func: Either a Tool instance or a callable to wrap.
            **kwargs: Additional arguments passed to tool() decorator if
                     tool_or_func is a callable.

        Returns:
            The registered Tool instance.

        Example:
            registry = ToolRegistry()

            # Register a Tool instance
            @tool
            def my_tool(x: str) -> str:
                return x
            registry.register(my_tool)

            # Or register a function directly
            def another_tool(x: str) -> str:
                return x
            registry.register(another_tool, name="custom_name")
        """
        if isinstance(tool_or_func, Tool):
            t = tool_or_func
        else:
            t = tool(tool_or_func, **kwargs)

        if t.name in self._tools:
            logger.warning(f"Overwriting existing tool: {t.name}")

        self._tools[t.name] = t
        logger.debug(f"Registered tool: {t.name}")
        return t

    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry.

        Args:
            name: Name of the tool to remove.

        Returns:
            True if tool was removed, False if not found.
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")
            return True
        return False

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name.

        Args:
            name: Name of the tool.

        Returns:
            Tool instance or None if not found.
        """
        return self._tools.get(name)

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get all tools in MCP-compatible format.

        Returns:
            List of tool definitions with name, description,
            inputSchema, and server fields.
        """
        tools = [t.to_mcp_format(self.server_name) for t in self._tools.values()]
        # Sort alphabetically for consistent ordering (KV-cache optimization)
        tools.sort(key=lambda t: t.get("name", ""))
        return tools

    def list_tool_names(self) -> List[str]:
        """Get list of registered tool names.

        Returns:
            Sorted list of tool names.
        """
        return sorted(self._tools.keys())

    async def call_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool by name with given arguments.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Dictionary of arguments to pass to the tool.

        Returns:
            Dict with execution result in MCP-compatible format:
            {
                "content": <result string>,
                "tool": <tool_name>,
                "server": <server_name>,
                "status": "success" | "error"
            }
        """
        tool_instance = self._tools.get(tool_name)
        if not tool_instance:
            logger.error(f"Tool not found: {tool_name}")
            return {
                "content": f"Error: Tool '{tool_name}' not found",
                "tool": tool_name,
                "server": self.server_name,
                "status": "error",
            }

        logger.info(f"Executing tool: {tool_name}")
        logger.debug(f"Arguments: {arguments}")

        try:
            result = await tool_instance.ainvoke(arguments)

            # Convert result to string if needed
            if isinstance(result, str):
                content = result
            elif isinstance(result, (dict, list)):
                import json

                content = json.dumps(result, indent=2)
            else:
                content = str(result)

            logger.info(f"Tool {tool_name} executed successfully")
            return {
                "content": content,
                "tool": tool_name,
                "server": self.server_name,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name}: {e}", exc_info=True)
            return {
                "content": f"Error executing {tool_name}: {e}",
                "tool": tool_name,
                "server": self.server_name,
                "status": "error",
            }

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def __repr__(self) -> str:
        return f"ToolRegistry(server={self.server_name!r}, tools={list(self._tools.keys())})"
