"""Tool decorator for defining agent tools.

Provides a simple @tool decorator that automatically generates
MCP-compatible tool schemas from function signatures and docstrings.
"""

import inspect
import re
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type, get_type_hints

# Type mapping from Python types to JSON Schema types
TYPE_MAP: Dict[Type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _get_json_type(python_type: Type) -> str:
    """Convert Python type to JSON Schema type."""
    # Handle Optional types
    origin = getattr(python_type, "__origin__", None)
    if origin is not None:
        # For Optional[X], Union[X, None], etc., use the first non-None type
        args = getattr(python_type, "__args__", ())
        for arg in args:
            if arg is not type(None):
                return _get_json_type(arg)

    return TYPE_MAP.get(python_type, "string")


def _parse_docstring(docstring: str) -> Dict[str, Any]:
    """Parse docstring to extract description and argument descriptions.

    Supports Google-style docstrings:
        '''Short description.

        Longer description if needed.

        Args:
            param1: Description of param1
            param2: Description of param2

        Returns:
            Description of return value
        '''
    """
    if not docstring:
        return {"description": "", "args": {}}

    lines = docstring.strip().split("\n")
    description_lines = []
    arg_descriptions: Dict[str, str] = {}
    current_section = "description"
    current_arg = None
    current_arg_desc_lines: list = []

    for line in lines:
        stripped = line.strip()

        # Check for section headers
        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            current_section = "args"
            continue
        elif stripped.lower() in (
            "returns:",
            "return:",
            "yields:",
            "raises:",
            "examples:",
        ):
            # Save any pending arg description
            if current_arg and current_arg_desc_lines:
                arg_descriptions[current_arg] = " ".join(current_arg_desc_lines).strip()
            current_section = "other"
            continue

        if current_section == "description":
            description_lines.append(stripped)
        elif current_section == "args":
            # Check if this is a new argument (format: "arg_name: description")
            arg_match = re.match(r"^(\w+)\s*:\s*(.*)$", stripped)
            if arg_match:
                # Save previous arg if exists
                if current_arg and current_arg_desc_lines:
                    arg_descriptions[current_arg] = " ".join(
                        current_arg_desc_lines
                    ).strip()
                current_arg = arg_match.group(1)
                current_arg_desc_lines = (
                    [arg_match.group(2)] if arg_match.group(2) else []
                )
            elif current_arg and stripped:
                # Continuation of previous arg description
                current_arg_desc_lines.append(stripped)

    # Save last arg if exists
    if current_arg and current_arg_desc_lines:
        arg_descriptions[current_arg] = " ".join(current_arg_desc_lines).strip()

    # Clean up description
    description = " ".join(description_lines).strip()
    # Remove excessive whitespace
    description = re.sub(r"\s+", " ", description)

    return {"description": description, "args": arg_descriptions}


def _generate_input_schema(
    func: Callable, arg_descriptions: Dict[str, str]
) -> Dict[str, Any]:
    """Generate MCP-compatible input schema from function signature."""
    sig = inspect.signature(func)

    # Try to get type hints
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    properties: Dict[str, Any] = {}
    required: list = []

    for param_name, param in sig.parameters.items():
        # Skip self, cls, and **kwargs
        if param_name in ("self", "cls") or param.kind == inspect.Parameter.VAR_KEYWORD:
            continue

        # Get type from hints or default to string
        param_type = hints.get(param_name, str)
        json_type = _get_json_type(param_type)

        prop: Dict[str, Any] = {"type": json_type}

        # Add description if available
        if param_name in arg_descriptions:
            prop["description"] = arg_descriptions[param_name]

        properties[param_name] = prop

        # Check if required (no default value)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


class Tool:
    """Wrapper class for tool functions with MCP-compatible metadata."""

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.func = func
        self.name = name or func.__name__
        self._explicit_description = description

        # Parse docstring for descriptions
        parsed = _parse_docstring(func.__doc__ or "")
        self.description = (
            description or parsed["description"] or f"Execute {self.name}"
        )
        self.arg_descriptions = parsed["args"]

        # Generate input schema
        self.input_schema = _generate_input_schema(func, self.arg_descriptions)

        # Preserve function metadata
        wraps(func)(self)

    def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool function."""
        return self.func(*args, **kwargs)

    async def ainvoke(self, arguments: Dict[str, Any]) -> Any:
        """Async invocation with dict arguments (MCP-style)."""
        if inspect.iscoroutinefunction(self.func):
            return await self.func(**arguments)
        else:
            return self.func(**arguments)

    def invoke(self, arguments: Dict[str, Any]) -> Any:
        """Sync invocation with dict arguments (MCP-style)."""
        return self.func(**arguments)

    def to_mcp_format(self, server_name: str = "local") -> Dict[str, Any]:
        """Convert tool to MCP-compatible format.

        Returns:
            Dict with name, description, inputSchema, and server fields.
        """
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "server": server_name,
        }

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r}, description={self.description[:50]!r}...)"


def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable:
    """Decorator to create a tool from a function.

    Can be used with or without arguments:

        @tool
        def my_func(x: str) -> str:
            '''Do something.'''
            return x

        @tool(name="custom_name", description="Custom description")
        def another_func(x: str) -> str:
            return x

    Args:
        func: The function to wrap (when used without parentheses)
        name: Optional custom name for the tool (defaults to function name)
        description: Optional description (defaults to docstring)

    Returns:
        Tool instance wrapping the function
    """

    def decorator(f: Callable) -> Tool:
        return Tool(f, name=name, description=description)

    if func is not None:
        # Called without parentheses: @tool
        return decorator(func)
    else:
        # Called with parentheses: @tool(...)
        return decorator
