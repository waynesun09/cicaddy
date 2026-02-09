"""Unit tests for tool format conversion utilities."""

from unittest.mock import patch

from cicaddy.utils.tool_converter import (
    convert_mcp_tools_to_claude,
    convert_mcp_tools_to_gemini,
    convert_mcp_tools_to_openai,
    convert_openai_tool_calls_to_mcp,
)


class TestMCPToOpenAIConversion:
    """Test MCP to OpenAI tool format conversion."""

    def test_convert_basic_tool(self):
        """Test conversion of a basic MCP tool."""
        mcp_tools = [
            {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state",
                        }
                    },
                    "required": ["location"],
                },
            }
        ]

        openai_tools = convert_mcp_tools_to_openai(mcp_tools)

        assert len(openai_tools) == 1
        tool = openai_tools[0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["description"] == "Get current weather for a location"
        assert tool["function"]["parameters"]["type"] == "object"
        assert "location" in tool["function"]["parameters"]["properties"]
        assert tool["function"]["parameters"]["required"] == ["location"]

    def test_convert_tool_without_schema(self):
        """Test conversion of tool without input schema."""
        mcp_tools = [{"name": "simple_tool", "description": "A simple tool"}]

        openai_tools = convert_mcp_tools_to_openai(mcp_tools)

        assert len(openai_tools) == 1
        tool = openai_tools[0]
        assert tool["function"]["parameters"]["type"] == "object"
        assert tool["function"]["parameters"]["properties"] == {}
        assert tool["function"]["parameters"]["required"] == []

    def test_convert_tool_without_name(self):
        """Test that tools without names are skipped."""
        mcp_tools = [
            {"description": "Tool without name"},
            {"name": "valid_tool", "description": "Valid tool"},
        ]

        with patch("cicaddy.utils.tool_converter.logger") as mock_logger:
            openai_tools = convert_mcp_tools_to_openai(mcp_tools)

        assert len(openai_tools) == 1
        assert openai_tools[0]["function"]["name"] == "valid_tool"
        mock_logger.warning.assert_called_once()

    def test_convert_multiple_tools(self):
        """Test conversion of multiple tools."""
        mcp_tools = [
            {
                "name": "tool1",
                "description": "First tool",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "tool2",
                "description": "Second tool",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

        openai_tools = convert_mcp_tools_to_openai(mcp_tools)

        assert len(openai_tools) == 2
        assert openai_tools[0]["function"]["name"] == "tool1"
        assert openai_tools[1]["function"]["name"] == "tool2"


class TestMCPToClaudeConversion:
    """Test MCP to Claude tool format conversion."""

    def test_convert_basic_tool(self):
        """Test conversion of a basic MCP tool to Claude format."""
        mcp_tools = [
            {
                "name": "calculate",
                "description": "Perform calculation",
                "inputSchema": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            }
        ]

        claude_tools = convert_mcp_tools_to_claude(mcp_tools)

        assert len(claude_tools) == 1
        tool = claude_tools[0]
        assert tool["name"] == "calculate"
        assert tool["description"] == "Perform calculation"
        assert tool["input_schema"]["type"] == "object"
        assert "expression" in tool["input_schema"]["properties"]


class TestMCPToGeminiConversion:
    """Test MCP to Gemini tool format conversion."""

    def test_convert_basic_tool(self):
        """Test conversion of a basic MCP tool to Gemini format."""
        mcp_tools = [
            {
                "name": "search",
                "description": "Search for information",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]

        gemini_tools = convert_mcp_tools_to_gemini(mcp_tools)

        assert len(gemini_tools) == 1
        tool = gemini_tools[0]
        assert tool["name"] == "search"
        assert tool["description"] == "Search for information"
        assert tool["parameters"]["type"] == "object"
        assert "query" in tool["parameters"]["properties"]

    def test_convert_tool_with_unsupported_schema_properties(self):
        """Test that unsupported Gemini schema properties are removed."""
        mcp_tools = [
            {
                "name": "complex_tool",
                "description": "Tool with unsupported schema properties",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,  # Unsupported in Gemini
                    "properties": {
                        "param1": {
                            "type": "string",
                            "default": "default_value",  # Unsupported in Gemini
                            "description": "A string parameter",
                            "examples": [
                                "example1",
                                "example2",
                            ],  # Unsupported in Gemini
                            "pattern": "^[a-z]+$",  # Unsupported in Gemini
                            "minLength": 1,  # Unsupported in Gemini
                            "maxLength": 100,  # Unsupported in Gemini
                        },
                        "param2": {
                            "type": "integer",
                            "minimum": 0,  # Unsupported in Gemini
                            "maximum": 999,  # Unsupported in Gemini
                            "title": "Number Parameter",  # Unsupported in Gemini
                        },
                    },
                    "required": ["param1"],
                },
            }
        ]

        gemini_tools = convert_mcp_tools_to_gemini(mcp_tools)

        assert len(gemini_tools) == 1
        tool = gemini_tools[0]
        assert tool["name"] == "complex_tool"

        # Check that unsupported properties are removed from the main schema
        params = tool["parameters"]
        assert "additionalProperties" not in params
        assert params["type"] == "object"
        assert params["required"] == ["param1"]

        # Check that unsupported properties are removed from parameter definitions
        param1 = params["properties"]["param1"]
        assert param1["type"] == "string"
        assert param1["description"] == "A string parameter"  # This should be kept
        assert "default" not in param1
        assert "examples" not in param1
        assert "pattern" not in param1
        assert "minLength" not in param1
        assert "maxLength" not in param1

        param2 = params["properties"]["param2"]
        assert param2["type"] == "integer"
        assert "minimum" not in param2
        assert "maximum" not in param2
        assert "title" not in param2

    def test_convert_tool_with_exclusive_limits_and_advanced_keywords(self):
        """Test that exclusiveMinimum, exclusiveMaximum, and other advanced JSON Schema keywords are removed."""
        mcp_tools = [
            {
                "name": "advanced_tool",
                "description": "Tool with advanced schema validation keywords",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "number_param": {
                            "type": "number",
                            "exclusiveMinimum": 0,  # Not supported by Gemini
                            "exclusiveMaximum": 100,  # Not supported by Gemini
                            "multipleOf": 5,  # Not supported by Gemini
                        },
                        "const_param": {
                            "const": "fixed_value",  # Not supported by Gemini
                        },
                        "conditional_param": {
                            "type": "string",
                            "if": {"minLength": 5},  # Not supported by Gemini
                            "then": {"pattern": "^[A-Z]"},  # Not supported by Gemini
                            "else": {"pattern": "^[a-z]"},  # Not supported by Gemini
                        },
                    },
                    "patternProperties": {  # Not supported by Gemini
                        "^S_": {"type": "string"},
                    },
                    "dependencies": {  # Not supported by Gemini
                        "number_param": ["const_param"],
                    },
                    "not": {"required": ["number_param"]},  # Not supported by Gemini
                },
            }
        ]

        gemini_tools = convert_mcp_tools_to_gemini(mcp_tools)

        assert len(gemini_tools) == 1
        tool = gemini_tools[0]
        assert tool["name"] == "advanced_tool"

        params = tool["parameters"]

        # Check that advanced keywords are removed from main schema
        assert "patternProperties" not in params
        assert "dependencies" not in params
        assert "not" not in params

        # Check that exclusiveMinimum and exclusiveMaximum are removed
        number_param = params["properties"]["number_param"]
        assert number_param["type"] == "number"
        assert "exclusiveMinimum" not in number_param
        assert "exclusiveMaximum" not in number_param
        assert "multipleOf" not in number_param

        # Check that const is removed
        const_param = params["properties"]["const_param"]
        assert "const" not in const_param

        # Check that conditional keywords are removed
        conditional_param = params["properties"]["conditional_param"]
        assert conditional_param["type"] == "string"
        assert "if" not in conditional_param
        assert "then" not in conditional_param
        assert "else" not in conditional_param


class TestOpenAIToMCPConversion:
    """Test OpenAI tool call response to MCP format conversion."""

    def test_convert_basic_tool_call(self):
        """Test conversion of OpenAI tool call to MCP format."""
        openai_tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "San Francisco"}',
                },
            }
        ]

        mcp_tool_calls = convert_openai_tool_calls_to_mcp(openai_tool_calls)

        assert len(mcp_tool_calls) == 1
        call = mcp_tool_calls[0]
        assert call["tool_name"] == "get_weather"
        assert call["tool_server"] == ""  # Will be filled by executor
        assert call["arguments"] == {"location": "San Francisco"}

    def test_convert_tool_call_with_invalid_json(self):
        """Test handling of tool call with invalid JSON arguments."""
        openai_tool_calls = [
            {
                "id": "call_456",
                "type": "function",
                "function": {"name": "broken_tool", "arguments": '{"invalid": json}'},
            }  # Invalid JSON
        ]

        with patch("cicaddy.utils.tool_converter.logger") as mock_logger:
            mcp_tool_calls = convert_openai_tool_calls_to_mcp(openai_tool_calls)

        assert len(mcp_tool_calls) == 1
        call = mcp_tool_calls[0]
        assert call["tool_name"] == "broken_tool"
        assert call["arguments"] == {}  # Fallback to empty dict
        mock_logger.error.assert_called_once()

    def test_convert_tool_call_without_function(self):
        """Test that tool calls without function are skipped."""
        openai_tool_calls = [
            {
                "id": "call_789",
                "type": "function",
                # Missing "function" key
            },
            {
                "id": "call_999",
                "type": "function",
                "function": {"name": "valid_tool", "arguments": "{}"},
            },
        ]

        with patch("cicaddy.utils.tool_converter.logger") as mock_logger:
            mcp_tool_calls = convert_openai_tool_calls_to_mcp(openai_tool_calls)

        assert len(mcp_tool_calls) == 1
        assert mcp_tool_calls[0]["tool_name"] == "valid_tool"
        mock_logger.warning.assert_called_once()

    def test_convert_multiple_tool_calls(self):
        """Test conversion of multiple tool calls."""
        openai_tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "tool1", "arguments": '{"param1": "value1"}'},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "tool2", "arguments": '{"param2": "value2"}'},
            },
        ]

        mcp_tool_calls = convert_openai_tool_calls_to_mcp(openai_tool_calls)

        assert len(mcp_tool_calls) == 2
        assert mcp_tool_calls[0]["tool_name"] == "tool1"
        assert mcp_tool_calls[0]["arguments"] == {"param1": "value1"}
        assert mcp_tool_calls[1]["tool_name"] == "tool2"
        assert mcp_tool_calls[1]["arguments"] == {"param2": "value2"}


class TestErrorHandling:
    """Test error handling in tool conversion functions."""

    def test_convert_mcp_tools_with_malformed_tool(self):
        """Test handling of malformed MCP tools."""
        mcp_tools = [
            {"name": "good_tool", "description": "A good tool"},
            {
                # Malformed tool that might cause exceptions
                "name": None,  # Invalid name type
                "description": 123,  # Invalid description type
            },
        ]

        with patch("cicaddy.utils.tool_converter.logger") as mock_logger:
            openai_tools = convert_mcp_tools_to_openai(mcp_tools)

        # Should still convert the good tool
        assert len(openai_tools) == 1  # Good tool converted, malformed tool skipped
        assert openai_tools[0]["function"]["name"] == "good_tool"
        mock_logger.warning.assert_called()

    def test_convert_empty_tools_list(self):
        """Test conversion of empty tools list."""
        openai_tools = convert_mcp_tools_to_openai([])
        claude_tools = convert_mcp_tools_to_claude([])
        gemini_tools = convert_mcp_tools_to_gemini([])
        mcp_calls = convert_openai_tool_calls_to_mcp([])

        assert openai_tools == []
        assert claude_tools == []
        assert gemini_tools == []
        assert mcp_calls == []
