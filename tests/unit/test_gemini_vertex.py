"""Tests for Gemini provider Vertex AI support."""

from unittest.mock import MagicMock, patch

import pytest

from cicaddy.ai_providers.gemini import GeminiProvider
from cicaddy.utils.token_utils import TokenLimitManager


class TestGeminiVertexTokenLimits:
    """Verify gemini-vertex resolves to Gemini token limits."""

    def test_gemini_vertex_resolves_gemini_limits(self):
        limits = TokenLimitManager.get_limits("gemini-vertex", "gemini-3-flash-preview")
        gemini_limits = TokenLimitManager.get_limits("gemini", "gemini-3-flash-preview")
        assert limits == gemini_limits

    def test_gemini_vertex_default_limits(self):
        limits = TokenLimitManager.get_limits("gemini-vertex")
        gemini_limits = TokenLimitManager.get_limits("gemini")
        assert limits == gemini_limits


class TestGeminiVertexInitialization:
    """Tests for Vertex AI client initialization in GeminiProvider."""

    @pytest.mark.asyncio
    async def test_vertex_initialize_creates_vertex_client(self):
        """Should create genai.Client with vertexai=True when config has vertexai flag."""
        mock_client = MagicMock()

        with patch(
            "cicaddy.ai_providers.gemini.genai.Client",
            return_value=mock_client,
        ) as mock_cls:
            provider = GeminiProvider(
                {
                    "model_id": "gemini-3-flash-preview",
                    "vertexai": True,
                    "google_cloud_project": "my-project",
                    "google_cloud_location": "global",
                    "temperature": 0.0,
                }
            )
            await provider.initialize()

            mock_cls.assert_called_once_with(
                vertexai=True,
                project="my-project",
                location="global",
            )
            assert provider.client is mock_client

    @pytest.mark.asyncio
    async def test_vertex_initialize_custom_location(self):
        """Should pass custom location to genai.Client."""
        with patch(
            "cicaddy.ai_providers.gemini.genai.Client",
            return_value=MagicMock(),
        ) as mock_cls:
            provider = GeminiProvider(
                {
                    "model_id": "gemini-3-flash-preview",
                    "vertexai": True,
                    "google_cloud_project": "my-project",
                    "google_cloud_location": "europe-west4",
                    "temperature": 0.0,
                }
            )
            await provider.initialize()

            mock_cls.assert_called_once_with(
                vertexai=True,
                project="my-project",
                location="europe-west4",
            )

    @pytest.mark.asyncio
    async def test_vertex_initialize_default_location(self):
        """Should default to global when location is not set."""
        with patch(
            "cicaddy.ai_providers.gemini.genai.Client",
            return_value=MagicMock(),
        ) as mock_cls:
            provider = GeminiProvider(
                {
                    "model_id": "gemini-3-flash-preview",
                    "vertexai": True,
                    "google_cloud_project": "my-project",
                    "temperature": 0.0,
                }
            )
            await provider.initialize()

            mock_cls.assert_called_once_with(
                vertexai=True,
                project="my-project",
                location="global",
            )

    @pytest.mark.asyncio
    async def test_vertex_no_api_key_needed(self):
        """Vertex AI should not require GEMINI_API_KEY."""
        with patch(
            "cicaddy.ai_providers.gemini.genai.Client",
            return_value=MagicMock(),
        ):
            provider = GeminiProvider(
                {
                    "model_id": "gemini-3-flash-preview",
                    "vertexai": True,
                    "google_cloud_project": "my-project",
                    "temperature": 0.0,
                }
            )
            await provider.initialize()
            assert provider.client is not None

    @pytest.mark.asyncio
    async def test_api_key_mode_unchanged(self):
        """Without vertexai flag, should use API key mode as before."""
        mock_client = MagicMock()

        with patch(
            "cicaddy.ai_providers.gemini.genai.Client",
            return_value=mock_client,
        ) as mock_cls:
            provider = GeminiProvider(
                {
                    "model_id": "gemini-3-flash-preview",
                    "api_key": "test-api-key",
                    "temperature": 0.0,
                }
            )
            await provider.initialize()

            mock_cls.assert_called_once_with(api_key="test-api-key")
            assert provider.client is mock_client

    @pytest.mark.asyncio
    async def test_no_key_no_vertex_falls_back_to_mock(self):
        """Without vertexai flag and no API key, should fall back to mock mode."""
        with patch.dict("os.environ", {}, clear=True):
            provider = GeminiProvider(
                {
                    "model_id": "gemini-3-flash-preview",
                    "temperature": 0.0,
                }
            )
            await provider.initialize()
            assert provider.client is None

    @pytest.mark.asyncio
    async def test_vertex_raises_when_project_missing(self):
        """Should raise ValueError when vertexai=True but no project."""
        provider = GeminiProvider(
            {
                "model_id": "gemini-3-flash-preview",
                "vertexai": True,
                "temperature": 0.0,
            }
        )
        with pytest.raises(ValueError, match="google_cloud_project is required"):
            await provider.initialize()

    @pytest.mark.asyncio
    async def test_vertex_raises_when_google_auth_missing(self):
        """Should raise ImportError when google-auth is not installed."""
        with patch.dict("sys.modules", {"google.auth": None}):
            provider = GeminiProvider(
                {
                    "model_id": "gemini-3-flash-preview",
                    "vertexai": True,
                    "google_cloud_project": "my-project",
                    "temperature": 0.0,
                }
            )
            with pytest.raises(ImportError, match="google-auth"):
                await provider.initialize()
