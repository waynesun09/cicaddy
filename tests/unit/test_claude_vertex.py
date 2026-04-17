"""Tests for Claude provider Vertex AI support."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cicaddy.ai_providers.base import ProviderMessage
from cicaddy.ai_providers.claude import ClaudeProvider


class TestClaudeVertexInitialization:
    """Tests for Vertex AI client initialization in ClaudeProvider."""

    @pytest.mark.asyncio
    async def test_vertex_initialize_creates_vertex_client(self):
        """Should create AsyncAnthropicVertex when vertex_project_id is set."""
        mock_vertex_client = MagicMock()

        with (
            patch("cicaddy.ai_providers.claude.VERTEX_AVAILABLE", True),
            patch(
                "cicaddy.ai_providers.claude.AsyncAnthropicVertex",
                return_value=mock_vertex_client,
            ) as mock_cls,
        ):
            provider = ClaudeProvider(
                {
                    "model_id": "claude-sonnet-4-6",
                    "vertex_project_id": "my-project",
                    "region": "us-east5",
                    "temperature": 0.0,
                }
            )
            await provider.initialize()

            mock_cls.assert_called_once_with(
                project_id="my-project",
                region="us-east5",
            )
            assert provider.client is mock_vertex_client

    @pytest.mark.asyncio
    async def test_vertex_initialize_custom_region(self):
        """Should pass custom region to AsyncAnthropicVertex."""
        with (
            patch("cicaddy.ai_providers.claude.VERTEX_AVAILABLE", True),
            patch(
                "cicaddy.ai_providers.claude.AsyncAnthropicVertex",
                return_value=MagicMock(),
            ) as mock_cls,
        ):
            provider = ClaudeProvider(
                {
                    "model_id": "claude-sonnet-4-6",
                    "vertex_project_id": "my-project",
                    "region": "europe-west4",
                    "temperature": 0.0,
                }
            )
            await provider.initialize()

            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["region"] == "europe-west4"

    @pytest.mark.asyncio
    async def test_vertex_initialize_default_region(self):
        """Should default to us-east5 when region not specified."""
        with (
            patch("cicaddy.ai_providers.claude.VERTEX_AVAILABLE", True),
            patch(
                "cicaddy.ai_providers.claude.AsyncAnthropicVertex",
                return_value=MagicMock(),
            ) as mock_cls,
        ):
            provider = ClaudeProvider(
                {
                    "model_id": "claude-sonnet-4-6",
                    "vertex_project_id": "my-project",
                    "temperature": 0.0,
                }
            )
            await provider.initialize()

            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["region"] == "us-east5"

    @pytest.mark.asyncio
    async def test_vertex_initialize_raises_when_extra_missing(self):
        """Should raise ImportError when anthropic[vertex] is not installed."""
        with patch("cicaddy.ai_providers.claude.VERTEX_AVAILABLE", False):
            provider = ClaudeProvider(
                {
                    "model_id": "claude-sonnet-4-6",
                    "vertex_project_id": "my-project",
                    "temperature": 0.0,
                }
            )
            with pytest.raises(ImportError, match="cicaddy\\[vertex\\]"):
                await provider.initialize()

    @pytest.mark.asyncio
    async def test_vertex_initialize_no_api_key_needed(self):
        """Vertex should not require ANTHROPIC_API_KEY."""
        with (
            patch("cicaddy.ai_providers.claude.VERTEX_AVAILABLE", True),
            patch(
                "cicaddy.ai_providers.claude.AsyncAnthropicVertex",
                return_value=MagicMock(),
            ),
            patch.dict("os.environ", {}, clear=True),
        ):
            provider = ClaudeProvider(
                {
                    "model_id": "claude-sonnet-4-6",
                    "vertex_project_id": "my-project",
                    "temperature": 0.0,
                }
            )
            # Should not raise — no API key required for Vertex
            await provider.initialize()
            assert provider.client is not None

    @pytest.mark.asyncio
    async def test_direct_api_still_works(self):
        """Without vertex_project_id, should use direct Anthropic API as before."""
        mock_client = MagicMock()

        with patch(
            "cicaddy.ai_providers.claude.AsyncAnthropic",
            return_value=mock_client,
        ) as mock_cls:
            provider = ClaudeProvider(
                {
                    "model_id": "claude-sonnet-4-6",
                    "api_key": "sk-ant-test",
                    "temperature": 0.0,
                }
            )
            await provider.initialize()

            mock_cls.assert_called_once_with(api_key="sk-ant-test")
            assert provider.client is mock_client


class TestClaudeVertexChatCompletion:
    """Verify chat_completion works identically with Vertex client."""

    @pytest.mark.asyncio
    async def test_vertex_chat_completion(self):
        """Vertex client should use the same messages.create interface."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Hello from Vertex")]
        mock_response.stop_reason = "end_turn"
        mock_response.model = "claude-sonnet-4-6"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with (
            patch("cicaddy.ai_providers.claude.VERTEX_AVAILABLE", True),
            patch(
                "cicaddy.ai_providers.claude.AsyncAnthropicVertex",
                return_value=mock_client,
            ),
        ):
            provider = ClaudeProvider(
                {
                    "model_id": "claude-sonnet-4-6",
                    "vertex_project_id": "my-project",
                    "region": "us-east5",
                    "temperature": 0.0,
                }
            )
            await provider.initialize()

            response = await provider.chat_completion(
                [ProviderMessage(role="user", content="Hello")]
            )

            assert response.content == "Hello from Vertex"
            assert response.model == "claude-sonnet-4-6"
            mock_client.messages.create.assert_called_once()
