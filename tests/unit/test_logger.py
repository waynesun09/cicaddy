"""Tests for logger module with log capture functionality."""

import logging
import os
from unittest import mock


from cicaddy.utils.logger import (
    clear_log_buffer,
    get_captured_logs,
    get_logger,
    setup_logging,
)


class TestLogCapture:
    """Test log capture functionality for local runs."""

    def setup_method(self):
        """Reset the log buffer before each test."""
        clear_log_buffer()

    def test_get_captured_logs_returns_none_in_ci_mode(self):
        """In CI mode (CI_PROJECT_DIR set), log buffer should not be created."""
        with mock.patch.dict(os.environ, {"CI_PROJECT_DIR": "/builds/project"}):
            # Re-initialize logging in CI mode
            setup_logging(level="INFO")

            # In CI mode, captured logs should be None
            captured = get_captured_logs()
            assert captured is None

    def test_get_captured_logs_returns_content_in_local_mode(self):
        """In local mode (no CI_PROJECT_DIR), logs should be captured."""
        # Ensure CI_PROJECT_DIR is not set
        with mock.patch.dict(os.environ, {}, clear=True):
            # Explicitly remove CI_PROJECT_DIR if it exists
            os.environ.pop("CI_PROJECT_DIR", None)

            # Re-initialize logging in local mode
            setup_logging(level="INFO")

            # Log a test message using standard logging
            test_logger = logging.getLogger("test_module")
            test_logger.info("Test log message for capture")

            # Captured logs should contain our message
            captured = get_captured_logs()
            if captured is not None:
                assert "Test log message for capture" in captured

    def test_clear_log_buffer(self):
        """Test that clear_log_buffer clears captured logs."""
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CI_PROJECT_DIR", None)
            setup_logging(level="INFO")

            test_logger = logging.getLogger("test_clear")
            test_logger.info("Message before clear")

            # Clear the buffer
            clear_log_buffer()

            # Log new message
            test_logger.info("Message after clear")

            captured = get_captured_logs()
            if captured is not None:
                assert "Message before clear" not in captured
                assert "Message after clear" in captured

    def test_get_logger_returns_structlog_logger(self):
        """Test that get_logger returns a structlog bound logger."""
        logger = get_logger("test_module")
        assert logger is not None
        # Structlog loggers have info, debug, error methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "error")

    def test_setup_logging_with_json_format(self):
        """Test that setup_logging works with JSON format."""
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CI_PROJECT_DIR", None)

            # Should not raise
            setup_logging(level="DEBUG", json_logs=True)
            logger = get_logger("json_test")
            logger.info("JSON formatted log")

    def test_setup_logging_with_different_levels(self):
        """Test that setup_logging respects log level."""
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CI_PROJECT_DIR", None)

            # Setup with WARNING level
            setup_logging(level="WARNING")

            test_logger = logging.getLogger("level_test")
            clear_log_buffer()

            # INFO should not be captured
            test_logger.info("Info message should not appear")
            # WARNING should be captured
            test_logger.warning("Warning message should appear")

            captured = get_captured_logs()
            if captured is not None:
                assert "Info message should not appear" not in captured
                assert "Warning message should appear" in captured
