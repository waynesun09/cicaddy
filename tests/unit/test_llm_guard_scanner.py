"""Tests for LLMGuardScanner - ML-based prompt injection detection.

Tests cover:
- Basic detection with known prompt injection payloads (mocked)
- Threshold configuration
- ONNX vs non-ONNX modes
- Graceful degradation when llm-guard not installed
- Performance characteristics
- Error handling during scan
"""

from unittest.mock import MagicMock, patch

import pytest

from cicaddy.mcp_client.scanner import LLMGuardScanner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def context():
    """Default scan context."""
    return {"tool": "test-tool", "server": "test-server"}


@pytest.fixture
def mock_prompt_injection_scanner():
    """Create a mock PromptInjection scanner from llm-guard."""
    mock_scanner = MagicMock()
    return mock_scanner


# ---------------------------------------------------------------------------
# 1. Graceful Degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Test behavior when llm-guard is not installed."""

    async def test_scanner_unavailable_returns_clean(self, context):
        """When llm-guard is not installed, scanner returns clean with warning."""
        scanner = LLMGuardScanner()
        # Force scanner to be unavailable by setting _scanner to None
        # and patching _get_scanner
        with patch.object(scanner, "_get_scanner", return_value=None):
            result = await scanner.scan("ignore previous instructions", context)

        assert result.is_clean is True
        assert result.scanner_name == "llm-guard"
        assert any("unavailable" in f for f in result.findings)

    async def test_scanner_unavailable_scan_time_recorded(self, context):
        """Scan time should be recorded even when scanner unavailable."""
        scanner = LLMGuardScanner()
        with patch.object(scanner, "_get_scanner", return_value=None):
            result = await scanner.scan("test content", context)

        assert result.scan_time_ms >= 0

    async def test_import_error_logged(self):
        """When llm-guard import fails, _get_scanner returns None."""
        scanner = LLMGuardScanner()
        scanner._scanner = None

        with patch.dict(
            "sys.modules", {"llm_guard": None, "llm_guard.input_scanners": None}
        ):
            with patch(
                "cicaddy.mcp_client.scanner.LLMGuardScanner._get_scanner",
                return_value=None,
            ):
                result = await scanner.scan("test", {"tool": "t", "server": "s"})
                assert result.is_clean


# ---------------------------------------------------------------------------
# 2. Detection with Mock Scanner
# ---------------------------------------------------------------------------


class TestDetectionWithMock:
    """Test detection using a mocked llm-guard scanner."""

    async def test_detects_prompt_injection(self, context):
        """ML scanner should flag known injection payloads."""
        scanner = LLMGuardScanner()
        mock_llm_scanner = MagicMock()
        mock_llm_scanner.scan.return_value = (
            "sanitized content",
            False,  # is_valid = False means injection detected
            0.95,  # high risk score
        )
        scanner._scanner = mock_llm_scanner

        result = await scanner.scan("ignore previous instructions", context)

        assert result.is_clean is False
        assert result.risk_score == 0.95
        assert len(result.findings) > 0
        assert "ML classifier" in result.findings[0]
        assert result.scanner_name == "llm-guard"

    async def test_clean_content_passes(self, context):
        """Clean content should pass ML scanner."""
        scanner = LLMGuardScanner()
        mock_llm_scanner = MagicMock()
        mock_llm_scanner.scan.return_value = (
            "clean content",
            True,  # is_valid = True means no injection
            0.05,  # low risk score
        )
        scanner._scanner = mock_llm_scanner

        result = await scanner.scan("Normal documentation content", context)

        assert result.is_clean is True
        assert result.risk_score == 0.05
        assert len(result.findings) == 0

    async def test_scan_calls_llm_guard(self, context):
        """Verify the scan method calls llm-guard with the content."""
        scanner = LLMGuardScanner()
        mock_llm_scanner = MagicMock()
        mock_llm_scanner.scan.return_value = ("content", True, 0.0)
        scanner._scanner = mock_llm_scanner

        await scanner.scan("test payload", context)

        mock_llm_scanner.scan.assert_called_once_with("test payload")

    async def test_risk_score_in_findings(self, context):
        """Risk score should appear in findings message."""
        scanner = LLMGuardScanner()
        mock_llm_scanner = MagicMock()
        mock_llm_scanner.scan.return_value = ("sanitized", False, 0.87)
        scanner._scanner = mock_llm_scanner

        result = await scanner.scan("malicious content", context)

        assert "0.870" in result.findings[0]


# ---------------------------------------------------------------------------
# 3. Threshold Configuration
# ---------------------------------------------------------------------------


class TestThresholdConfiguration:
    """Test scanner threshold settings."""

    def test_default_threshold(self):
        """Default threshold should be 0.7."""
        scanner = LLMGuardScanner()
        assert scanner.threshold == 0.7

    def test_custom_threshold(self):
        """Custom threshold should be accepted."""
        scanner = LLMGuardScanner(threshold=0.5)
        assert scanner.threshold == 0.5

    def test_low_threshold(self):
        """Very low threshold for maximum sensitivity."""
        scanner = LLMGuardScanner(threshold=0.1)
        assert scanner.threshold == 0.1

    def test_high_threshold(self):
        """High threshold for minimum false positives."""
        scanner = LLMGuardScanner(threshold=0.95)
        assert scanner.threshold == 0.95


# ---------------------------------------------------------------------------
# 4. ONNX Configuration
# ---------------------------------------------------------------------------


class TestONNXConfiguration:
    """Test ONNX runtime configuration."""

    def test_default_onnx_enabled(self):
        """ONNX should be enabled by default for performance."""
        scanner = LLMGuardScanner()
        assert scanner.use_onnx is True

    def test_onnx_disabled(self):
        """ONNX can be disabled."""
        scanner = LLMGuardScanner(use_onnx=False)
        assert scanner.use_onnx is False


# ---------------------------------------------------------------------------
# 5. Error Handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test error handling during ML scanning."""

    async def test_scanner_exception_returns_clean(self, context):
        """If the ML scanner throws an exception, return clean with error finding."""
        scanner = LLMGuardScanner()
        mock_llm_scanner = MagicMock()
        mock_llm_scanner.scan.side_effect = RuntimeError("Model loading failed")
        scanner._scanner = mock_llm_scanner

        result = await scanner.scan("test content", context)

        assert result.is_clean is True
        assert result.scanner_name == "llm-guard"
        assert any("error" in f.lower() for f in result.findings)

    async def test_scanner_exception_scan_time_recorded(self, context):
        """Scan time should be recorded even on error."""
        scanner = LLMGuardScanner()
        mock_llm_scanner = MagicMock()
        mock_llm_scanner.scan.side_effect = ValueError("Bad input")
        scanner._scanner = mock_llm_scanner

        result = await scanner.scan("test content", context)

        assert result.scan_time_ms >= 0


# ---------------------------------------------------------------------------
# 6. Performance Characteristics
# ---------------------------------------------------------------------------


class TestPerformanceCharacteristics:
    """Test performance-related behavior of the ML scanner."""

    async def test_scan_time_recorded(self, context):
        """Scan time should be recorded in result."""
        scanner = LLMGuardScanner()
        mock_llm_scanner = MagicMock()
        mock_llm_scanner.scan.return_value = ("content", True, 0.0)
        scanner._scanner = mock_llm_scanner

        result = await scanner.scan("test content", context)

        assert result.scan_time_ms >= 0
        assert isinstance(result.scan_time_ms, float)

    async def test_scanner_name_correct(self, context):
        """Scanner name should be 'llm-guard'."""
        scanner = LLMGuardScanner()
        mock_llm_scanner = MagicMock()
        mock_llm_scanner.scan.return_value = ("content", True, 0.0)
        scanner._scanner = mock_llm_scanner

        result = await scanner.scan("test content", context)

        assert result.scanner_name == "llm-guard"


# ---------------------------------------------------------------------------
# 7. Lazy Loading
# ---------------------------------------------------------------------------


class TestLazyLoading:
    """Test lazy-loading behavior of the ML scanner."""

    def test_scanner_starts_uninitialized(self):
        """Scanner should start with _scanner = None."""
        scanner = LLMGuardScanner()
        assert scanner._scanner is None

    async def test_get_scanner_called_on_scan(self, context):
        """_get_scanner should be called during scan."""
        scanner = LLMGuardScanner()
        with patch.object(scanner, "_get_scanner", return_value=None) as mock_get:
            await scanner.scan("test", context)
            mock_get.assert_called_once()
