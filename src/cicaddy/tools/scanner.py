"""Tool-level scanning wrapper for prompt injection detection.

This module provides a unified scanning interface that works with any tool type
(MCP servers, local file tools, etc.), extending the ContentScanner protocol
to the tool execution layer.

The ToolScanner wraps the core ContentScanner implementations (HeuristicScanner,
LLMGuardScanner, CompositeScanner) and provides tool-aware scanning logic that
respects per-source scan modes and blocking thresholds.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from cicaddy.utils.logger import get_logger

if TYPE_CHECKING:
    from cicaddy.mcp_client.scanner import ContentScanner

logger = get_logger(__name__)


class ToolScanner:
    """Tool-level wrapper for content scanning.

    Provides prompt injection scanning for tool results with configurable
    scan modes, blocking thresholds, and source-specific policies.

    This scanner works uniformly across all tool types (MCP, local, future)
    and shares the same scanner instances used by MCP clients.

    Example:
        >>> from cicaddy.mcp_client.scanner import HeuristicScanner
        >>> scanner = ToolScanner(
        ...     scanner=HeuristicScanner(),
        ...     scan_mode="enforce",
        ...     blocking_threshold=0.3,
        ... )
        >>> result = await scanner.scan_tool_result(
        ...     content="malicious content",
        ...     tool_name="read_file",
        ...     source="local",
        ... )
        >>> if result.blocked:
        ...     # Handle blocked content
    """

    def __init__(
        self,
        scanner: Optional["ContentScanner"] = None,
        scan_mode: str = "disabled",
        blocking_threshold: float = 0.3,
        detection_threshold: float = 0.0,
    ):
        """Initialize the tool scanner.

        Args:
            scanner: Content scanner instance (HeuristicScanner, LLMGuardScanner,
                or CompositeScanner). If None, scanning is disabled.
            scan_mode: Scanning mode - 'disabled' (no scanning), 'audit'
                (log warnings but pass content through), or 'enforce'
                (block malicious content).
            blocking_threshold: Risk score threshold for blocking content
                (0.0-1.0). Only applies in enforce mode. Content with
                risk_score >= blocking_threshold is blocked.
            detection_threshold: Risk score threshold for logging detections
                (0.0-1.0). Detections with risk_score >= detection_threshold
                are logged even if not blocked.
        """
        self.scanner = scanner
        self.scan_mode = scan_mode
        self.blocking_threshold = blocking_threshold
        self.detection_threshold = detection_threshold

    async def scan_tool_result(
        self,
        content: str,
        tool_name: str,
        source: str = "unknown",
    ) -> "ToolScanResult":
        """Scan a tool result for prompt injection attacks.

        Args:
            content: Tool result content to scan.
            tool_name: Name of the tool that produced the content.
            source: Source type of the tool (e.g., "mcp", "local", "external").

        Returns:
            ToolScanResult with scan details and blocking decision.
        """
        # If scanner not configured or disabled, pass through
        if not self.scanner or self.scan_mode == "disabled":
            return ToolScanResult(
                is_clean=True,
                risk_score=0.0,
                findings=[],
                blocked=False,
                scan_mode=self.scan_mode,
            )

        # Run the scan
        scan_result = await self.scanner.scan(
            content,
            {"tool": tool_name, "source": source},
        )

        # Determine if content should be blocked
        should_block = (
            not scan_result.is_clean
            and scan_result.risk_score >= self.blocking_threshold
            and self.scan_mode == "enforce"
        )

        # Log if detection threshold exceeded
        if (
            not scan_result.is_clean
            and scan_result.risk_score >= self.detection_threshold
        ):
            severity = "BLOCKED" if should_block else "DETECTED"
            logger.warning(
                f"[{severity}] Prompt injection in {source}/{tool_name}: "
                f"{scan_result.findings} (risk: {scan_result.risk_score:.2f}, "
                f"threshold: {self.blocking_threshold:.2f})"
            )

        return ToolScanResult(
            is_clean=scan_result.is_clean,
            risk_score=scan_result.risk_score,
            findings=scan_result.findings,
            blocked=should_block,
            scan_mode=self.scan_mode,
            scanner_name=scan_result.scanner_name,
            scan_time_ms=scan_result.scan_time_ms,
        )


class ToolScanResult:
    """Result of a tool-level scan.

    Extends ScanResult with blocking decision and scan mode metadata.
    """

    def __init__(
        self,
        is_clean: bool,
        risk_score: float,
        findings: list[str],
        blocked: bool,
        scan_mode: str,
        scanner_name: str = "",
        scan_time_ms: float = 0.0,
    ):
        """Initialize scan result.

        Args:
            is_clean: Whether content passed all scans (risk_score == 0.0).
            risk_score: Cumulative risk score (0.0-1.0).
            findings: List of detected patterns/issues.
            blocked: Whether content was blocked due to scan result.
            scan_mode: Scan mode used (disabled/audit/enforce).
            scanner_name: Name of the scanner that produced the result.
            scan_time_ms: Scan duration in milliseconds.
        """
        self.is_clean = is_clean
        self.risk_score = risk_score
        self.findings = findings
        self.blocked = blocked
        self.scan_mode = scan_mode
        self.scanner_name = scanner_name
        self.scan_time_ms = scan_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for logging/metadata."""
        return {
            "is_clean": self.is_clean,
            "risk_score": self.risk_score,
            "findings": self.findings,
            "blocked": self.blocked,
            "scan_mode": self.scan_mode,
            "scanner_name": self.scanner_name,
            "scan_time_ms": self.scan_time_ms,
        }

    def __repr__(self) -> str:
        status = (
            "BLOCKED" if self.blocked else ("CLEAN" if self.is_clean else "FLAGGED")
        )
        return (
            f"ToolScanResult(status={status}, risk={self.risk_score:.2f}, "
            f"findings={len(self.findings)})"
        )
