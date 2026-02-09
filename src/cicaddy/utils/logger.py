"""Logging utilities for Cicaddy."""

import io
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import structlog


class JsonLogFormatter(logging.Formatter):
    """JSON formatter for log records to align with structlog JSON output."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


# Global string buffer to capture logs for local runs
_log_buffer: Optional[io.StringIO] = None
_file_handler: Optional[logging.StreamHandler] = None


def setup_logging(level: str = "INFO", json_logs: bool = False):
    """Setup structured logging with optional file capture for local runs."""
    global _log_buffer, _file_handler

    # Create buffer to capture logs when running locally (not in CI)
    is_ci = os.getenv("CI_PROJECT_DIR") is not None
    if not is_ci:
        _log_buffer = io.StringIO()

    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]

    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=False,
    )

    # Configure standard logging with handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler for stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.setLevel(getattr(logging, level.upper()))
    root_logger.addHandler(console_handler)

    # Buffer handler for local runs (captures logs for file output)
    if _log_buffer is not None:
        # Remove existing file handler if present to prevent duplicates on re-setup
        if _file_handler is not None and _file_handler in root_logger.handlers:
            root_logger.removeHandler(_file_handler)
        _file_handler = logging.StreamHandler(_log_buffer)
        # Use JSON formatter when json_logs is enabled for consistency with structlog
        if json_logs:
            _file_handler.setFormatter(JsonLogFormatter())
        else:
            _file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                )
            )
        _file_handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(_file_handler)


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def get_captured_logs() -> Optional[str]:
    """
    Get all captured log content for local runs.

    Returns:
        The captured log content as a string, or None if not capturing
        (e.g., running in CI mode).
    """
    global _log_buffer
    if _log_buffer is not None:
        return _log_buffer.getvalue()
    return None


def clear_log_buffer():
    """Clear the log buffer (useful for tests)."""
    global _log_buffer
    if _log_buffer is not None:
        _log_buffer.truncate(0)
        _log_buffer.seek(0)
