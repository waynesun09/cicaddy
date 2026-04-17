"""Tests for Claude provider Vertex AI support."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cicaddy.ai_providers.base import ProviderMessage
from cicaddy.ai_providers.claude import ClaudeProvider
from cicaddy.utils.token_utils import TokenLimitManager


class TestClaudeVertexTokenLimits:
    """Verify anthropic-vertex resolves to Claude token limits."""

    def test_anthropic_vertex_resolves_claude_limits(self):
        """anthropic-vertex provider should use Claude's token limits, not fallback."""
        limits = TokenLimitManager.get_limits("anthropic-vertex", "claude-sonnet-4-6")
        assert limits["input"] == 200000
        assert limits["output"] == 65536

    def test_anthropic_alias_resolves_claude_limits(self):
        """anthropic provider should also resolve to Claude limits."""
        limits = TokenLimitManager.get_limits("anthropic", "claude-sonnet-4-6")
        assert limits["input"] == 200000
        assert limits["output"] == 65536

    def test_anthropic_vertex_default_limits(self):
        """anthropic-vertex with unknown model should use Claude defaults."""
        limits = TokenLimitManager.get_limits("anthropic-vertex")
        assert limits["input"] == 200000
        assert limits["output"] == 8192


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
    async def test_vertex_takes_precedence_over_api_key(self):
        """When both vertex_project_id and api_key are set, Vertex path wins."""
        mock_vertex_client = MagicMock()

        with (
            patch("cicaddy.ai_providers.claude.VERTEX_AVAILABLE", True),
            patch(
                "cicaddy.ai_providers.claude.AsyncAnthropicVertex",
                return_value=mock_vertex_client,
            ) as mock_vertex_cls,
            patch(
                "cicaddy.ai_providers.claude.AsyncAnthropic",
            ) as mock_direct_cls,
        ):
            provider = ClaudeProvider(
                {
                    "model_id": "claude-sonnet-4-6",
                    "vertex_project_id": "my-project",
                    "api_key": "not-a-real-key",  # noqa: S106
                    "region": "us-east5",
                    "temperature": 0.0,
                }
            )
            await provider.initialize()

            mock_vertex_cls.assert_called_once()
            mock_direct_cls.assert_not_called()
            assert provider.client is mock_vertex_client

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
                    "api_key": "not-a-real-key",  # noqa: S106
                    "temperature": 0.0,
                }
            )
            await provider.initialize()

            mock_cls.assert_called_once_with(api_key="not-a-real-key")
            assert provider.client is mock_client


class TestClaudeVertexChatCompletion:
    """Verify chat_completion works identically with Vertex client."""

    @pytest.mark.asyncio
    async def test_vertex_chat_completion(self):
        """Vertex client should use streaming messages interface."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Hello from Vertex")]
        mock_response.stop_reason = "end_turn"
        mock_response.model = "claude-sonnet-4-6"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

        # Mock the streaming context manager
        mock_stream = AsyncMock()
        mock_stream.get_final_message = AsyncMock(return_value=mock_response)
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)

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
            mock_client.messages.stream.assert_called_once()
