"""Tests for quota and rate limit detection in MCP tool responses."""

import pytest

from cicaddy.mcp_client.quota_detector import detect_quota_error, is_temporary_rate_limit


class TestQuotaDetector:
    """Test quota/rate limit pattern detection."""

    def test_detect_monthly_quota_exceeded(self):
        """Test detection of monthly quota exhaustion."""
        content = "Monthly quota exceeded. Create a free API key at https://context7.com/dashboard for more requests."
        result = detect_quota_error(content)
        assert result is not None
        assert "quota exceeded" in result.lower()

    def test_detect_rate_limited(self):
        """Test detection of rate limiting."""
        content = "Rate limited or quota exceeded. Upgrade your plan at https://context7.com/plans for higher limits."
        result = detect_quota_error(content)
        assert result is not None
        # Detector may match either "rate limited" or "quota exceeded"
        assert "rate" in result.lower() or "quota" in result.lower()

    def test_detect_too_many_requests(self):
        """Test detection of too many requests error."""
        content = "Error: too many requests. Please try again later."
        result = detect_quota_error(content)
        assert result is not None
        assert "too many request" in result.lower()

    def test_detect_quota_limit(self):
        """Test detection of quota limit message."""
        content = "You have reached your quota limit for this month."
        result = detect_quota_error(content)
        assert result is not None
        assert "quota limit" in result.lower()

    def test_detect_request_limit_exceeded(self):
        """Test detection of request limit exceeded."""
        content = "Request limit exceeded. Please wait before making more requests."
        result = detect_quota_error(content)
        assert result is not None
        # Detector may match "request limit" or "limit exceeded"
        assert "limit" in result.lower()

    def test_no_quota_error_in_normal_response(self):
        """Test that normal responses are not flagged."""
        content = "Here is the documentation for the React library. It is a popular JavaScript framework..."
        result = detect_quota_error(content)
        assert result is None

    def test_no_quota_error_in_empty_content(self):
        """Test that empty content returns None."""
        assert detect_quota_error("") is None
        assert detect_quota_error(None) is None

    def test_detect_throttled(self):
        """Test detection of throttling message."""
        content = "Request throttled due to high volume. Please retry after a delay."
        result = detect_quota_error(content)
        assert result is not None
        assert "throttl" in result.lower()

    def test_case_insensitive_detection(self):
        """Test that detection is case-insensitive."""
        content = "RATE LIMIT EXCEEDED"
        result = detect_quota_error(content)
        assert result is not None

    def test_detect_in_longer_content(self):
        """Test detection when quota error is embedded in longer content."""
        content = """
        The library you are trying to access does not exist.

        Monthly quota exceeded. Create a free API key at https://context7.com/dashboard.

        Additional information about the error...
        """
        result = detect_quota_error(content)
        assert result is not None
        assert "quota exceeded" in result.lower()


class TestTemporaryRateLimitDetection:
    """Test differentiation between temporary and durable quota errors."""

    def test_monthly_quota_is_durable(self):
        """Test that monthly quota is identified as durable (not temporary)."""
        content = "Monthly quota exceeded. Upgrade your plan for more requests."
        assert not is_temporary_rate_limit(content)

    def test_upgrade_message_indicates_durable(self):
        """Test that upgrade/billing messages indicate durable quota."""
        content = "Quota exceeded. Please upgrade your billing plan."
        assert not is_temporary_rate_limit(content)

    def test_hourly_rate_limit_is_temporary(self):
        """Test that per-hour rate limits are temporary."""
        content = "Rate limit: 100 requests per hour. Try again later."
        assert is_temporary_rate_limit(content)

    def test_retry_after_is_temporary(self):
        """Test that 'retry after' indicates temporary limit."""
        content = "Too many requests. Retry after 60 seconds."
        assert is_temporary_rate_limit(content)

    def test_try_again_is_temporary(self):
        """Test that 'try again' suggests temporary limit."""
        content = "Request limit reached. Please try again in a few minutes."
        assert is_temporary_rate_limit(content)

    def test_generic_rate_limit_defaults_to_durable(self):
        """Test that ambiguous rate limits default to durable (safer)."""
        content = "Rate limited."
        assert not is_temporary_rate_limit(content)

    def test_empty_content_defaults_to_durable(self):
        """Test that empty content defaults to durable."""
        assert not is_temporary_rate_limit("")
        assert not is_temporary_rate_limit(None)
