"""General quota and rate limit detection for MCP tool responses.

MCP tools (Context7, Browserbase, etc.) may return quota/rate limit errors
as successful responses with error text in the content field, rather than
as HTTP errors. This module provides general pattern detection to identify
such errors so they can be handled as durable failures that shouldn't be retried.
"""

import re
from typing import Optional

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)

# General patterns that indicate quota or rate limiting
# These patterns are intentionally broad to catch various MCP tools
QUOTA_PATTERNS = [
    # Explicit quota messages
    r"quota\s+exceeded",
    r"monthly\s+quota",
    r"quota\s+limit",
    # Rate limiting
    r"rate\s+limit(?:ed)?",
    r"too\s+many\s+requests?",
    r"request\s+limit",
    # Throttling
    r"throttled",
    r"throttling",
    # Service-specific but common patterns
    r"exceeded.*limit",
    r"limit.*exceeded",
]

# Compile patterns once for efficiency
COMPILED_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in QUOTA_PATTERNS]


def detect_quota_error(content: str) -> Optional[str]:
    """
    Detect if content contains quota or rate limit error messages.

    Args:
        content: Tool result content to check

    Returns:
        Matched error pattern if quota error detected, None otherwise
    """
    if not content or not isinstance(content, str):
        return None

    # Check first 1000 chars (error messages are typically at the start)
    check_content = content[:1000].lower()

    for pattern in COMPILED_PATTERNS:
        match = pattern.search(check_content)
        if match:
            matched_text = match.group(0)
            logger.info(
                f"Quota/rate limit detected in tool response: '{matched_text}'"
            )
            return matched_text

    return None


def is_temporary_rate_limit(content: str) -> bool:
    """
    Distinguish between temporary (hourly/per-IP) and durable (monthly quota) limits.

    Temporary limits might be worth retrying after delay.
    Durable limits (monthly quotas) should not be retried.

    Args:
        content: Tool result content

    Returns:
        True if this appears to be a temporary rate limit, False if durable quota
    """
    if not content:
        return False

    content_lower = content[:1000].lower()

    # Indicators of durable quota exhaustion
    durable_indicators = ["monthly quota", "quota exceeded", "upgrade", "billing"]

    # Indicators of temporary rate limiting
    temporary_indicators = ["per hour", "per minute", "try again", "retry after"]

    has_durable = any(indicator in content_lower for indicator in durable_indicators)
    has_temporary = any(
        indicator in content_lower for indicator in temporary_indicators
    )

    # If we see durable indicators without temporary ones, it's durable
    if has_durable and not has_temporary:
        return False

    # If we see temporary indicators, it's temporary
    if has_temporary:
        return True

    # Default: treat as durable (safer to not retry)
    return False
