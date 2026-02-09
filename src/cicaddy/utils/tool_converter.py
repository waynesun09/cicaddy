"""Tool format conversion utilities for AI providers.

This module provides utilities to convert between different tool definition formats,
primarily from MCP (Model Context Protocol) format to provider-specific formats
like OpenAI, Claude, and Gemini.

Based on patterns from LlamaStack's OpenAI compatibility layer.
"""

from typing import Any, Dict, List, cast

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)


def convert_mcp_tools_to_openai(
    mcp_tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert MCP tool definitions to OpenAI function calling format.

    MCP tools have format:
    {
        "name": "tool_name",
        "description": "Tool description",
        "inputSchema": {
            "type": "object",
            "properties": {
                "param_name": {
                    "type": "string",
                    "description": "Parameter description"
                }
            },
            "required": ["param_name"]
        }
    }

    OpenAI format:
    {
        "type": "function",
        "function": {
            "name": "tool_name",
            "description": "Tool description",
            "parameters": {
                "type": "object",
                "properties": {
                    "param_name": {
                        "type": "string",
                        "description": "Parameter description"
                    }
                },
                "required": ["param_name"]
            }
        }
    }

    Args:
        mcp_tools: List of MCP tool definitions

    Returns:
        List of OpenAI-compatible tool definitions
    """
    openai_tools = []

    for tool in mcp_tools:
        try:
            # Extract basic tool information
            tool_name = tool.get("name")
            tool_description = tool.get("description", "")

            if not tool_name:
                logger.warning(f"Skipping tool with no name: {tool}")
                continue

            # Build OpenAI tool definition
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_description,
                },
            }

            # Convert input schema if present
            input_schema = tool.get("inputSchema")
            function_def = cast(Dict[str, Any], openai_tool["function"])
            if input_schema:
                # MCP inputSchema is already in JSON Schema format,
                # which is what OpenAI expects for parameters
                function_def["parameters"] = input_schema
            else:
                # Provide minimal schema if none exists
                function_def["parameters"] = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }

            openai_tools.append(openai_tool)
            logger.debug(f"Converted MCP tool '{tool_name}' to OpenAI format")

        except Exception as e:
            logger.error(
                f"Failed to convert MCP tool to OpenAI format: {tool}. Error: {e}"
            )
            continue

    logger.info(
        f"Converted {len(openai_tools)}/{len(mcp_tools)} MCP tools to OpenAI format"
    )
    return openai_tools


def convert_mcp_tools_to_claude(
    mcp_tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert MCP tool definitions to Claude tool format.

    Claude expects:
    {
        "name": "tool_name",
        "description": "Tool description",
        "input_schema": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }

    Args:
        mcp_tools: List of MCP tool definitions

    Returns:
        List of Claude-compatible tool definitions
    """
    claude_tools = []

    for tool in mcp_tools:
        try:
            tool_name = tool.get("name")
            tool_description = tool.get("description", "")

            if not tool_name:
                logger.warning(f"Skipping tool with no name: {tool}")
                continue

            claude_tool = {
                "name": tool_name,
                "description": tool_description,
            }

            # Claude uses input_schema (matches MCP inputSchema)
            input_schema = tool.get("inputSchema")
            if input_schema:
                claude_tool["input_schema"] = input_schema
            else:
                claude_tool["input_schema"] = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }

            claude_tools.append(claude_tool)
            logger.debug(f"Converted MCP tool '{tool_name}' to Claude format")

        except Exception as e:
            logger.error(
                f"Failed to convert MCP tool to Claude format: {tool}. Error: {e}"
            )
            continue

    logger.info(
        f"Converted {len(claude_tools)}/{len(mcp_tools)} MCP tools to Claude format"
    )
    return claude_tools


def _clean_schema_for_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean JSON Schema for Gemini compatibility by removing unsupported properties.

    Gemini doesn't support certain JSON Schema properties like:
    - additionalProperties
    - default
    - examples
    - $schema
    - definitions
    - enum (partially supported)
    - exclusiveMinimum/exclusiveMaximum (not supported)
    - Advanced validation keywords (multipleOf, const, etc.)
    """
    if not isinstance(schema, dict):
        return schema

    # Properties to remove for Gemini compatibility
    unsupported_props = {
        "additionalProperties",
        "default",
        "examples",
        "$schema",
        "definitions",
        "$ref",
        "$id",
        "title",
        "format",  # Gemini has limited format support
        "pattern",  # Gemini has limited pattern support
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
        "exclusiveMinimum",  # Not supported by Gemini
        "exclusiveMaximum",  # Not supported by Gemini
        "multipleOf",  # Advanced numeric validation
        "const",  # Constant value constraint
        "patternProperties",  # Object validation
        "dependencies",  # Schema dependencies
        "if",  # Conditional schemas
        "then",  # Conditional schemas
        "else",  # Conditional schemas
        "not",  # Negative schemas
        "contains",  # Array validation
    }

    cleaned = {}
    for key, value in schema.items():
        if key in unsupported_props:
            logger.debug(f"Removing unsupported Gemini schema property: {key}")
            continue

        if key == "properties" and isinstance(value, dict):
            # Recursively clean nested properties
            cleaned[key] = {
                prop_name: _clean_schema_for_gemini(prop_schema)
                for prop_name, prop_schema in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            # Clean array item schemas
            cleaned[key] = _clean_schema_for_gemini(value)
        elif isinstance(value, dict):
            # Recursively clean nested objects
            cleaned[key] = _clean_schema_for_gemini(value)
        elif isinstance(value, list):
            # Clean arrays of schemas
            cleaned[key] = [  # type: ignore[assignment]
                _clean_schema_for_gemini(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            # Keep primitive values
            cleaned[key] = value

    return cleaned


def convert_mcp_tools_to_gemini(
    mcp_tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert MCP tool definitions to Gemini function calling format.

    Gemini expects function declarations in this format:
    {
        "name": "tool_name",
        "description": "Tool description",
        "parameters": {
            "type": "object",
            "properties": {
                "param_name": {
                    "type": "string",
                    "description": "Parameter description"
                }
            },
            "required": ["param_name"]
        }
    }

    Args:
        mcp_tools: List of MCP tool definitions

    Returns:
        List of Gemini-compatible function declarations
    """
    gemini_tools = []

    for tool in mcp_tools:
        try:
            tool_name = tool.get("name")
            tool_description = tool.get("description", "")

            if not tool_name:
                logger.warning(f"Skipping tool with no name: {tool}")
                continue

            gemini_tool = {
                "name": tool_name,
                "description": tool_description,
            }

            # Gemini uses parameters (same as MCP inputSchema)
            input_schema = tool.get("inputSchema")
            if input_schema:
                # Clean the schema for Gemini compatibility
                cleaned_schema = _clean_schema_for_gemini(input_schema)
                gemini_tool["parameters"] = cleaned_schema
            else:
                gemini_tool["parameters"] = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }

            gemini_tools.append(gemini_tool)
            logger.debug(f"Converted MCP tool '{tool_name}' to Gemini format")

        except Exception as e:
            logger.error(
                f"Failed to convert MCP tool to Gemini format: {tool}. Error: {e}"
            )
            continue

    logger.info(
        f"Converted {len(gemini_tools)}/{len(mcp_tools)} MCP tools to Gemini format"
    )
    return gemini_tools


def convert_openai_tool_calls_to_mcp(
    openai_tool_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert OpenAI tool call responses back to MCP-compatible format.

    OpenAI tool calls:
    {
        "id": "call_123",
        "function": {
            "name": "tool_name",
            "arguments": '{"param": "value"}'  # JSON string
        },
        "type": "function"
    }

    MCP format:
    {
        "tool_name": "tool_name",
        "tool_server": "",  # Will be populated by executor
        "arguments": {"param": "value"}  # Parsed JSON object
    }

    Args:
        openai_tool_calls: List of OpenAI tool call objects

    Returns:
        List of MCP-compatible tool call objects
    """
    import json

    mcp_tool_calls = []

    for tool_call in openai_tool_calls:
        try:
            if "function" not in tool_call:
                logger.warning(f"Skipping tool call without function: {tool_call}")
                continue

            function = tool_call["function"]
            tool_name = function.get("name")
            arguments_str = function.get("arguments", "{}")

            if not tool_name:
                logger.warning(f"Skipping tool call without name: {tool_call}")
                continue

            # Parse arguments JSON string
            try:
                arguments = (
                    json.loads(arguments_str)
                    if isinstance(arguments_str, str)
                    else arguments_str
                )
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse tool call arguments for '{tool_name}': {e}"
                )
                arguments = {}

            # Ensure MCP compatibility: arguments must be an object (dict), not an array
            if isinstance(arguments, list):
                logger.warning(
                    f"Converting array arguments to object for MCP compatibility in tool '{tool_name}'"
                )
                if len(arguments) == 1 and isinstance(arguments[0], dict):
                    # Single dict item - use it directly
                    arguments = arguments[0]
                else:
                    # Multiple items or non-dict items - create indexed object
                    arguments = {f"arg_{i}": item for i, item in enumerate(arguments)}
            elif not isinstance(arguments, dict):
                logger.warning(
                    f"Converting non-dict arguments to object for MCP compatibility in tool '{tool_name}'"
                )
                arguments = {"value": arguments}

            mcp_tool_call = {
                "tool_name": tool_name,
                "tool_server": "",  # Will be filled by executor when matching to available tools
                "arguments": arguments,
            }

            mcp_tool_calls.append(mcp_tool_call)
            logger.debug(f"Converted OpenAI tool call '{tool_name}' to MCP format")

        except Exception as e:
            logger.error(
                f"Failed to convert OpenAI tool call to MCP format: {tool_call}. Error: {e}"
            )
            continue

    logger.info(
        f"Converted {len(mcp_tool_calls)}/{len(openai_tool_calls)} OpenAI tool calls to MCP format"
    )
    return mcp_tool_calls
