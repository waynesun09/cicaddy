"""Unit tests for Gemini provider tool calling functionality."""

import json
from unittest.mock import Mock, patch

import pytest

from cicaddy.ai_providers.base import ProviderMessage, ProviderResponse
from cicaddy.ai_providers.gemini import GeminiProvider


class TestGeminiToolCalling:
    """Test Gemini provider tool calling functionality."""

    @pytest.fixture
    def gemini_provider(self):
        """Create a Gemini provider for testing."""
        config = {
            "model_id": "gemini-2.5-flash",
            "api_key": "test-key",
            "temperature": 0.0,
        }
        return GeminiProvider(config)

    @pytest.fixture
    def sample_tools(self):
        """Sample MCP tools for testing."""
        return [
            {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state",
                        },
                        "units": {
                            "type": "string",
                            "description": "Temperature units",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
            {
                "name": "calculate",
                "description": "Perform mathematical calculation",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate",
                        }
                    },
                    "required": ["expression"],
                },
            },
        ]

    @pytest.fixture
    def sample_messages(self):
        """Sample messages for testing."""
        return [
            ProviderMessage(role="user", content="What's the weather in San Francisco?")
        ]

    def test_build_gemini_messages(self, gemini_provider, sample_messages):
        """Test conversion of ProviderMessage to Gemini format."""
        gemini_messages = gemini_provider._build_gemini_messages(sample_messages)

        assert len(gemini_messages) == 1
        assert gemini_messages[0]["role"] == "user"
        assert (
            gemini_messages[0]["parts"][0]["text"]
            == "What's the weather in San Francisco?"
        )

    def test_build_gemini_messages_with_system(self, gemini_provider):
        """Test handling of system messages in Gemini format."""
        messages = [
            ProviderMessage(role="system", content="You are a helpful assistant"),
            ProviderMessage(role="user", content="Hello"),
        ]

        gemini_messages = gemini_provider._build_gemini_messages(messages)

        assert len(gemini_messages) == 2
        assert gemini_messages[0]["role"] == "user"  # System converted to user
        assert "System:" in gemini_messages[0]["parts"][0]["text"]
        assert gemini_messages[1]["role"] == "user"

    def test_build_gemini_messages_with_assistant(self, gemini_provider):
        """Test handling of assistant messages in Gemini format."""
        messages = [
            ProviderMessage(role="user", content="Hello"),
            ProviderMessage(role="assistant", content="Hi there!"),
        ]

        gemini_messages = gemini_provider._build_gemini_messages(messages)

        assert len(gemini_messages) == 2
        assert gemini_messages[0]["role"] == "user"
        assert gemini_messages[1]["role"] == "model"  # Assistant -> model for Gemini

    @pytest.mark.asyncio
    @patch("cicaddy.ai_providers.gemini.convert_mcp_tools_to_gemini")
    async def test_chat_completion_with_tools_conversion(
        self, mock_converter, gemini_provider, sample_tools, sample_messages
    ):
        """Test that tools are properly converted for Gemini API."""
        # Mock the converter
        mock_converter.return_value = [
            {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
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

        # Mock Gemini API response
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[
            0
        ].text = "The weather in San Francisco is sunny, 72°F."
        mock_response.candidates[0].finish_reason = 1  # STOP
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 20
        mock_response.usage_metadata.candidates_token_count = 15
        mock_response.usage_metadata.total_token_count = 35

        # Mock the client and its models.generate_content method
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response

        gemini_provider.client = mock_client

        # Test chat completion with tools
        response = await gemini_provider.chat_completion(
            sample_messages, sample_tools
        )

        # Verify tools were converted
        mock_converter.assert_called_once_with(sample_tools)

        # Verify response
        assert isinstance(response, ProviderResponse)
        assert "sunny" in response.content
        assert response.model == "gemini-2.5-flash"
        assert len(response.tool_calls) == 0  # No tool calls in this response

    def test_extract_content_and_tool_calls_text_only(self, gemini_provider):
        """Test extraction of text-only response."""
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].text = "This is a text response."

        content, tool_calls = gemini_provider._extract_content_and_tool_calls(
            mock_response
        )

        assert content == "This is a text response."
        assert tool_calls == []

    def test_extract_content_and_tool_calls_with_function_call(self, gemini_provider):
        """Test extraction of response with function call."""
        # Mock response with function call
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = [Mock(), Mock()]

        # First part: text
        mock_response.candidates[0].content.parts[
            0
        ].text = "I'll check the weather for you."

        # Second part: function call
        mock_function_call = Mock()
        mock_function_call.name = "get_weather"
        mock_function_call.args = {"location": "San Francisco", "units": "fahrenheit"}
        mock_response.candidates[0].content.parts[1].function_call = mock_function_call
        # Remove text attribute from second part
        delattr(mock_response.candidates[0].content.parts[1], "text")

        content, tool_calls = gemini_provider._extract_content_and_tool_calls(
            mock_response
        )

        assert content == "I'll check the weather for you."
        assert len(tool_calls) == 1

        tool_call = tool_calls[0]
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_weather"

        # Parse the arguments JSON
        args = json.loads(tool_call["function"]["arguments"])
        assert args["location"] == "San Francisco"
        assert args["units"] == "fahrenheit"

    @pytest.mark.asyncio
    async def test_generate_content_with_tools_passes_tools_in_config(
        self, gemini_provider
    ):
        """Test that tools are passed to client.models.generate_content via config."""
        # Setup mock client
        mock_client = Mock()
        mock_response = Mock()
        # Provide non-empty candidates to avoid triggering empty response retry logic
        mock_candidate = Mock()
        mock_candidate.finish_reason = 1  # STOP
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response

        gemini_provider.client = mock_client

        # Test data
        contents = [{"role": "user", "parts": [{"text": "Test message"}]}]
        tools = [{"name": "test_tool", "description": "Test tool"}]

        # Call the method
        await gemini_provider._generate_content_with_tools_and_retry_check(
            contents, tools
        )

        # Verify client.models.generate_content was called with tools in config
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args
        assert call_kwargs.kwargs["model"] == "gemini-2.5-flash"
        assert call_kwargs.kwargs["contents"] == contents
        config = call_kwargs.kwargs["config"]
        assert "tools" in config
        assert config["tools"] == [{"function_declarations": tools}]

    @pytest.mark.asyncio
    async def test_generate_content_without_tools_uses_no_tools_in_config(
        self, gemini_provider
    ):
        """Test that no tools are passed in config when tools is None."""
        # Setup mock client
        mock_client = Mock()
        mock_response = Mock()
        # Provide non-empty candidates to avoid triggering empty response retry logic
        mock_candidate = Mock()
        mock_candidate.finish_reason = 1  # STOP
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response

        gemini_provider.client = mock_client

        # Test data
        contents = [{"role": "user", "parts": [{"text": "Test message"}]}]

        # Call the method without tools
        await gemini_provider._generate_content_with_tools_and_retry_check(
            contents, None
        )

        # Verify generate_content was called without tools in config
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        assert "tools" not in config

    @pytest.mark.asyncio
    async def test_generate_content_with_tools_and_retry_check_empty_response(
        self, gemini_provider
    ):
        """Test that empty Gemini response triggers TemporaryServiceError for retry."""
        from cicaddy.ai_providers.base import TemporaryServiceError

        mock_client = Mock()
        mock_response = Mock()
        mock_response.candidates = []  # Simulate an empty response
        mock_client.models.generate_content.return_value = mock_response

        gemini_provider.client = mock_client

        contents = [{"role": "user", "parts": [{"text": "Test message"}]}]

        with pytest.raises(
            TemporaryServiceError, match="Gemini returned empty response"
        ):
            await gemini_provider._generate_content_with_tools_and_retry_check(
                contents, None
            )

        mock_client.models.generate_content.assert_called_once()

    def test_extract_content_and_tool_calls_with_empty_response(self, gemini_provider):
        """Test handling of empty response."""
        # Mock response with no candidates
        mock_response = Mock()
        mock_response.candidates = []

        content, tool_calls = gemini_provider._extract_content_and_tool_calls(
            mock_response
        )

        assert content == ""
        assert tool_calls == []

    def test_extract_content_and_tool_calls_with_no_parts(self, gemini_provider):
        """Test handling of candidate with no parts."""
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = []

        content, tool_calls = gemini_provider._extract_content_and_tool_calls(
            mock_response
        )

        assert content == ""
        assert tool_calls == []

    def test_extract_content_and_tool_calls_handles_errors(self, gemini_provider):
        """Test handling of errors during content extraction."""
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = [Mock()]

        # Make accessing text attribute raise an exception to trigger error handling
        def text_side_effect():
            raise Exception("test error")

        type(mock_response.candidates[0].content.parts[0]).text = property(
            text_side_effect
        )

        with patch.object(
            gemini_provider,
            "_extract_content_gemini_specific",
            return_value="fallback content",
        ) as mock_fallback:
            content, tool_calls = gemini_provider._extract_content_and_tool_calls(
                mock_response
            )

        assert content == "fallback content"
        assert tool_calls == []
        mock_fallback.assert_called_once_with(mock_response)

    def test_extract_content_and_tool_calls_with_malformed_function_args(
        self, gemini_provider
    ):
        """Test handling of malformed function call arguments."""
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = [Mock()]

        # Mock a function call with problematic args
        mock_function_call = Mock()
        mock_function_call.name = "test_function"
        # Create args that can't be converted to dict
        mock_function_call.args = "not_a_dict_or_convertible"
        mock_response.candidates[0].content.parts[0].function_call = mock_function_call
        # Remove text attribute
        delattr(mock_response.candidates[0].content.parts[0], "text")

        content, tool_calls = gemini_provider._extract_content_and_tool_calls(
            mock_response
        )

        assert content == ""
        assert len(tool_calls) == 1
        # Should still create a tool call even with malformed args
        tool_call = tool_calls[0]
        assert tool_call["function"]["name"] == "test_function"
        # With improved conversion, string args are properly converted to JSON strings
        arguments_str = tool_call["function"]["arguments"]
        assert arguments_str == '"not_a_dict_or_convertible"'

    def test_convert_function_args_with_dict(self, gemini_provider):
        """Test argument conversion with dictionary args (normal case)."""
        mock_func_call = Mock()
        mock_func_call.name = "search_code"
        mock_func_call.args = {
            "query": "test search query",
            "includeCodeSnippets": True,
        }

        result = gemini_provider._convert_function_args_to_json(
            mock_func_call, "search_code"
        )

        parsed_args = json.loads(result)
        assert parsed_args["query"] == "test search query"
        assert parsed_args["includeCodeSnippets"] is True

    def test_convert_function_args_with_non_dict_type(self, gemini_provider):
        """Test argument conversion with non-dictionary type (improved behavior)."""
        mock_func_call = Mock()
        mock_func_call.name = "search_code"
        mock_func_call.args = "not_a_dict"

        result = gemini_provider._convert_function_args_to_json(
            mock_func_call, "search_code"
        )

        # With recursive conversion, strings are properly converted to JSON strings
        assert result == '"not_a_dict"'

    def test_convert_function_args_with_complex_dict(self, gemini_provider):
        """Test argument conversion with complex dictionary structure."""
        mock_func_call = Mock()
        mock_func_call.name = "complex_search"
        mock_func_call.args = {
            "query": "search term",
            "content": "additional content",
            "value": "some value",
            "nested": {"deep": "value"},
        }

        result = gemini_provider._convert_function_args_to_json(
            mock_func_call, "complex_search"
        )

        parsed_args = json.loads(result)
        assert parsed_args["query"] == "search term"
        assert parsed_args["content"] == "additional content"
        assert parsed_args["value"] == "some value"
        assert parsed_args["nested"]["deep"] == "value"

    def test_convert_function_args_with_empty_and_none_values(self, gemini_provider):
        """Test that empty/None values are preserved as provided."""
        mock_func_call = Mock()
        mock_func_call.name = "search_code"
        mock_func_call.args = {
            "query": "valid query",
            "text": "",
            "content": None,
        }  # Empty string is preserved  # None is preserved

        result = gemini_provider._convert_function_args_to_json(
            mock_func_call, "search_code"
        )

        parsed_args = json.loads(result)
        assert parsed_args["query"] == "valid query"
        assert parsed_args["text"] == ""
        assert parsed_args["content"] is None

    def test_convert_function_args_with_json_encode_error(self, gemini_provider):
        """Test improved handling of non-serializable objects."""
        mock_func_call = Mock()
        mock_func_call.name = "test_function"

        # Create a dict with an object that can't be JSON serialized
        class NonSerializable:
            pass

        mock_func_call.args = {"valid": "value", "invalid": NonSerializable()}

        result = gemini_provider._convert_function_args_to_json(
            mock_func_call, "test_function"
        )

        # With recursive conversion, valid data is preserved and non-serializable objects become empty dicts
        parsed_result = json.loads(result)
        assert parsed_result["valid"] == "value"
        assert (
            parsed_result["invalid"] == {}
        )  # Non-serializable object converted to empty dict

    def test_convert_function_args_sourcebot_search_code_scenario(
        self, gemini_provider
    ):
        """Test specific scenario matching the sourcebot search_code tool call that was failing."""
        mock_func_call = Mock()
        mock_func_call.name = "search_code"
        mock_func_call.args = {"query": "AI ML models", "includeCodeSnippets": True}

        result = gemini_provider._convert_function_args_to_json(
            mock_func_call, "search_code"
        )

        parsed_args = json.loads(result)
        assert "query" in parsed_args
        assert parsed_args["query"] == "AI ML models"
        assert isinstance(parsed_args["query"], str)
        assert parsed_args["includeCodeSnippets"] is True

    def test_convert_function_args_with_nested_structures(self, gemini_provider):
        """Test conversion with nested dictionary structures."""
        mock_func_call = Mock()
        mock_func_call.name = "complex_tool"
        mock_func_call.args = {
            "query": "main query",
            "nested": {
                "nested_value": "deep value",
                "array": [1, 2, 3],
                "boolean": False,
            },
            "simple": "value",
        }

        result = gemini_provider._convert_function_args_to_json(
            mock_func_call, "complex_tool"
        )

        parsed_args = json.loads(result)
        assert parsed_args["query"] == "main query"
        assert parsed_args["nested"]["nested_value"] == "deep value"
        assert parsed_args["nested"]["array"] == [1, 2, 3]
        assert parsed_args["nested"]["boolean"] is False
        assert parsed_args["simple"] == "value"

    def test_convert_function_args_with_dict_access(self, gemini_provider):
        """Test argument conversion with dict-like object that supports dict() conversion."""
        # Simple test using collections.UserDict which is a proper dict-like object
        from collections import UserDict

        mock_func_call = Mock()
        mock_func_call.name = "dict_like_test"
        mock_func_call.args = UserDict({"param1": "value1", "param2": 42})

        result = gemini_provider._convert_function_args_to_json(
            mock_func_call, "dict_like_test"
        )

        parsed_args = json.loads(result)
        assert parsed_args["param1"] == "value1"
        assert parsed_args["param2"] == 42

    def test_convert_function_args_with_empty_common_fields_ignored(
        self, gemini_provider
    ):
        """Test that empty string values in function args are preserved as specified."""
        mock_func_call = Mock()
        mock_func_call.name = "datarouter_test"
        mock_func_call.args = {"numOfDays": "7", "accountName": "", "status": "all"}

        result = gemini_provider._convert_function_args_to_json(
            mock_func_call, "datarouter_test"
        )

        parsed_args = json.loads(result)
        assert parsed_args["numOfDays"] == "7"
        assert parsed_args["accountName"] == ""  # Empty string preserved
        assert parsed_args["status"] == "all"

    def test_convert_function_args_with_map_composite(self, gemini_provider):
        """Test argument conversion with dict-like objects that support dict() conversion."""

        # Create a simple class that mimics MapComposite behavior - supports dict() conversion
        class MockMapComposite:
            def __init__(self, data):
                self._data = data

            def items(self):
                return self._data.items()

            def keys(self):
                return self._data.keys()

            def values(self):
                return self._data.values()

            def __iter__(self):
                return iter(
                    self._data.items()
                )  # Return key-value pairs for dict() constructor

            def __getitem__(self, key):
                return self._data[key]

        mock_func_call = Mock()
        mock_func_call.name = "getTotalRequestNumberFromPastDays"
        mock_func_call.args = MockMapComposite({"numOfDays": "7", "status": "all"})

        result = gemini_provider._convert_function_args_to_json(
            mock_func_call, "getTotalRequestNumberFromPastDays"
        )

        parsed_args = json.loads(result)
        assert parsed_args["numOfDays"] == "7"
        assert parsed_args["status"] == "all"

    def test_convert_protobuf_to_serializable_basic_types(self, gemini_provider):
        """Test recursive conversion handles basic types correctly."""
        # Test None
        assert gemini_provider._convert_protobuf_to_serializable(None) is None

        # Test primitives
        assert gemini_provider._convert_protobuf_to_serializable("test") == "test"
        assert gemini_provider._convert_protobuf_to_serializable(42) == 42
        assert gemini_provider._convert_protobuf_to_serializable(3.14) == 3.14
        assert gemini_provider._convert_protobuf_to_serializable(True) is True
        assert gemini_provider._convert_protobuf_to_serializable(False) is False

    def test_convert_protobuf_to_serializable_nested_dict(self, gemini_provider):
        """Test recursive conversion handles nested dictionaries with mixed types."""
        nested_data = {
            "level1": {
                "level2": {
                    "string": "value",
                    "number": 42,
                    "list": [1, 2, 3],
                    "boolean": True,
                }
            },
            "simple": "test",
        }

        result = gemini_provider._convert_protobuf_to_serializable(nested_data)
        assert result == nested_data  # Should be unchanged as it's already serializable

    def test_convert_protobuf_to_serializable_repeated_composite_mock(
        self, gemini_provider
    ):
        """Test recursive conversion handles RepeatedComposite-like objects."""

        # Create a mock RepeatedComposite-like object
        class MockRepeatedComposite:
            def __init__(self, items):
                self._items = items

            def __iter__(self):
                return iter(self._items)

            def __len__(self):
                return len(self._items)

            def __getitem__(self, index):
                return self._items[index]

        # Test with simple list data
        mock_repeated = MockRepeatedComposite(["item1", "item2", "item3"])
        result = gemini_provider._convert_protobuf_to_serializable(mock_repeated)
        assert result == ["item1", "item2", "item3"]

    def test_convert_protobuf_to_serializable_nested_protobuf_objects(
        self, gemini_provider
    ):
        """Test recursive conversion with nested protobuf-like objects."""
        from collections import UserDict

        # Create a complex nested structure with dict-like objects
        class MockRepeatedComposite:
            def __init__(self, items):
                self._items = items

            def __iter__(self):
                return iter(self._items)

        nested_structure = {
            "data": UserDict(
                {
                    "items": MockRepeatedComposite(
                        [{"name": "item1", "value": 10}, {"name": "item2", "value": 20}]
                    ),
                    "metadata": UserDict({"version": "1.0", "created": "2024-01-01"}),
                }
            ),
            "simple": "test",
        }

        result = gemini_provider._convert_protobuf_to_serializable(nested_structure)

        # Verify the structure was properly converted
        assert isinstance(result, dict)
        assert result["simple"] == "test"
        assert isinstance(result["data"], dict)
        assert isinstance(result["data"]["items"], list)
        assert len(result["data"]["items"]) == 2
        assert result["data"]["items"][0] == {"name": "item1", "value": 10}
        assert result["data"]["metadata"]["version"] == "1.0"

    def test_convert_function_args_with_repeated_composite_nested(
        self, gemini_provider
    ):
        """Test function argument conversion with RepeatedComposite nested in dict."""

        # Create a mock for testing the actual scenario that was failing
        class MockRepeatedComposite:
            def __init__(self, items):
                self._items = items

            def __iter__(self):
                return iter(self._items)

        mock_func_call = Mock()
        mock_func_call.name = "search_code"
        # Simulate the failing scenario: dict with RepeatedComposite value
        mock_func_call.args = {
            "query": "test search",
            "filters": MockRepeatedComposite(["filter1", "filter2"]),
            "includeCodeSnippets": True,
        }

        result = gemini_provider._convert_function_args_to_json(
            mock_func_call, "search_code"
        )

        parsed_args = json.loads(result)
        assert parsed_args["query"] == "test search"
        assert parsed_args["filters"] == ["filter1", "filter2"]
        assert parsed_args["includeCodeSnippets"] is True

    def test_convert_function_args_deeply_nested_protobuf(self, gemini_provider):
        """Test function argument conversion with deeply nested protobuf objects."""
        from collections import UserDict

        class MockRepeatedComposite:
            def __init__(self, items):
                self._items = items

            def __iter__(self):
                return iter(self._items)

        # Create a very complex nested structure
        mock_func_call = Mock()
        mock_func_call.name = "complex_tool"
        mock_func_call.args = {
            "config": UserDict(
                {
                    "search": {
                        "terms": MockRepeatedComposite(["term1", "term2"]),
                        "filters": UserDict(
                            {
                                "types": MockRepeatedComposite(["type1", "type2"]),
                                "enabled": True,
                            }
                        ),
                    },
                    "output": {"format": "json"},
                }
            ),
            "priority": 1,
        }

        result = gemini_provider._convert_function_args_to_json(
            mock_func_call, "complex_tool"
        )

        parsed_args = json.loads(result)
        assert parsed_args["priority"] == 1
        assert parsed_args["config"]["search"]["terms"] == ["term1", "term2"]
        assert parsed_args["config"]["search"]["filters"]["types"] == ["type1", "type2"]
        assert parsed_args["config"]["search"]["filters"]["enabled"] is True
        assert parsed_args["config"]["output"]["format"] == "json"

    def test_convert_function_args_with_unknown_object_fallback(self, gemini_provider):
        """Test function argument conversion with unknown object types."""

        class UnknownObject:
            def __init__(self, value):
                self.value = value

            def __str__(self):
                return f"UnknownObject({self.value})"

        mock_func_call = Mock()
        mock_func_call.name = "test_tool"
        mock_func_call.args = {"known": "value", "unknown": UnknownObject("test_value")}

        result = gemini_provider._convert_function_args_to_json(
            mock_func_call, "test_tool"
        )

        parsed_args = json.loads(result)
        assert parsed_args["known"] == "value"
        # Unknown object should be converted to dict of its attributes
        assert parsed_args["unknown"]["value"] == "test_value"

    def test_convert_function_args_with_json_serialization_error_recovery(
        self, gemini_provider
    ):
        """Test that serialization errors are handled gracefully."""

        class NonSerializableObject:
            def __init__(self):
                # Create circular reference to trigger serialization error
                self.circular = self

        mock_func_call = Mock()
        mock_func_call.name = "error_tool"
        mock_func_call.args = NonSerializableObject()

        # Should not raise exception, should handle circular reference gracefully
        result = gemini_provider._convert_function_args_to_json(
            mock_func_call, "error_tool"
        )

        # The recursive function should handle this case gracefully:
        # - Should not raise an exception
        # - Should return valid JSON with circular reference marker
        assert isinstance(result, str)

        # Should be valid JSON
        try:
            parsed_result = json.loads(result)
            json_valid = True

            # Should contain circular reference marker
            assert (
                parsed_result["circular"]
                == "<CIRCULAR_REFERENCE:NonSerializableObject>"
            )
        except json.JSONDecodeError:
            json_valid = False

        assert json_valid, f"Result should be valid JSON, got: {result[:100]}..."

    def test_convert_protobuf_to_serializable_circular_reference_prevention(
        self, gemini_provider
    ):
        """Test that circular references are handled proactively without RecursionError."""

        class CircularObject:
            def __init__(self, name):
                self.name = name
                self.circular = self  # Direct circular reference

        circular_obj = CircularObject("test")

        # This should not raise RecursionError due to proactive circular reference detection
        result = gemini_provider._convert_protobuf_to_serializable(circular_obj)

        # Verify the result is a dict with circular reference marker
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["circular"] == "<CIRCULAR_REFERENCE:CircularObject>"

    def test_convert_function_args_with_circular_reference_comprehensive(
        self, gemini_provider
    ):
        """Test function argument conversion with circular references is handled gracefully."""

        class Node:
            def __init__(self, value):
                self.value = value
                self.next = None

        # Create a circular linked list
        node1 = Node("first")
        node2 = Node("second")
        node1.next = node2
        node2.next = node1  # Circular reference

        mock_func_call = Mock()
        mock_func_call.name = "circular_test"
        mock_func_call.args = {"start_node": node1, "other_data": "valid"}

        # Should not raise RecursionError
        result = gemini_provider._convert_function_args_to_json(
            mock_func_call, "circular_test"
        )

        # Should be valid JSON
        parsed_args = json.loads(result)
        assert parsed_args["other_data"] == "valid"
        assert parsed_args["start_node"]["value"] == "first"
        assert parsed_args["start_node"]["next"]["value"] == "second"
        # The circular reference should be marked
        assert parsed_args["start_node"]["next"]["next"] == "<CIRCULAR_REFERENCE:Node>"

    def test_model_name_property(self, gemini_provider):
        """Test model_name property returns configured model."""
        assert gemini_provider.model_name == "gemini-2.5-flash"

    def test_model_name_property_default(self):
        """Test model_name property returns default when not configured."""
        provider = GeminiProvider({})
        assert provider.model_name == "gemini-3-flash-preview"

    @pytest.mark.asyncio
    async def test_initialize_creates_client(self, gemini_provider):
        """Test that initialize creates a genai.Client."""
        with patch("cicaddy.ai_providers.gemini.genai") as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client

            gemini_provider.config["api_key"] = "test-api-key"
            await gemini_provider.initialize()

            mock_genai.Client.assert_called_once_with(api_key="test-api-key")
            assert gemini_provider.client == mock_client

    @pytest.mark.asyncio
    async def test_initialize_without_api_key(self, gemini_provider):
        """Test that initialize sets client to None without API key."""
        gemini_provider.config.pop("api_key", None)
        with patch.dict("os.environ", {}, clear=True):
            await gemini_provider.initialize()
            assert gemini_provider.client is None

    @pytest.mark.asyncio
    async def test_shutdown_clears_client(self, gemini_provider):
        """Test that shutdown clears the client."""
        gemini_provider.client = Mock()
        await gemini_provider.shutdown()
        assert gemini_provider.client is None
