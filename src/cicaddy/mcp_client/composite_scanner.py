"""Composite scanner combining multiple detection strategies.

Runs multiple ContentScanner instances in sequence and aggregates their
results. Designed for a layered security approach where a fast heuristic
scanner runs first, and a slower ML-based scanner provides confirmation.

Aggregation Strategy:
    - Early exit: In standard mode, if the first scanner returns clean, skip
      remaining scanners (optimizes latency for clean content to <1ms).
    - Risk score: Uses the maximum risk_score from all scanners that ran.
    - Findings: Collects findings from all scanners that ran.
    - Clean status: Content is clean only if ALL scanners that ran agree
      it is clean. A single scanner flagging content blocks it.

Consensus Mode (require_consensus=True):
    Forces all scanners to run regardless of intermediate results. This is
    useful when you want ML confirmation of heuristic findings. Note that
    even in consensus mode, if any scanner flags the content, it is blocked
    (security-first approach).

Performance:
    - Clean content: <1ms (heuristic only, ML skipped via early exit)
    - Suspicious content: ~50-200ms (both heuristic and ML run)

Examples:
    Standard mode (any flag blocks):

    >>> composite = CompositeScanner(
    ...     scanners=[HeuristicScanner(), LLMGuardScanner()],
    ...     require_consensus=False,
    ... )
    >>> result = await composite.scan("normal docs", {})
    >>> result.is_clean  # True, only heuristic ran (early exit)
    True

    Consensus mode (all scanners run):

    >>> composite = CompositeScanner(
    ...     scanners=[HeuristicScanner(), LLMGuardScanner()],
    ...     require_consensus=True,
    ... )
    >>> result = await composite.scan("borderline content", {})
    >>> # Both scanners ran; blocked if either flagged it
"""

import time
from typing import Any, Dict, List

from cicaddy.utils.logger import get_logger

from .scanner import ContentScanner, ScanResult

logger = get_logger(__name__)


class CompositeScanner:
    """Runs multiple scanners and aggregates results.

    Fast heuristic scanner runs first. If it flags content,
    optionally runs ML scanner for confirmation.

    Two modes:
    1. require_consensus=False (default): Any scanner flagging means malicious
    2. require_consensus=True: All scanners must agree to flag

    Strategy:
    - Run heuristic scanner first (fast, <1ms)
    - If clean and require_consensus=False, skip ML scan (optimize latency)
    - If dirty or require_consensus=True, run ML scan for confirmation

    This provides adaptive latency: <1ms for clean content, ~50-200ms
    for suspicious content.
    """

    def __init__(
        self,
        scanners: List[ContentScanner],
        require_consensus: bool = False,
    ):
        """Initialize the composite scanner with a list of sub-scanners.

        Args:
            scanners: List of ContentScanner instances to run. Typically
                the fast HeuristicScanner is first, with LLMGuardScanner
                as an optional second pass.
            require_consensus: If False (default), any scanner flagging
                content marks it as malicious. If True, all scanners must
                agree to flag content.
        """
        self.scanners = scanners
        self.require_consensus = require_consensus

    async def scan(self, content: str, context: Dict[str, Any]) -> ScanResult:
        """Run all sub-scanners and aggregate their results.

        In default mode (require_consensus=False), stops early if the first
        scanner finds clean content. Aggregates findings and uses the maximum
        risk score across all scanners.

        Args:
            content: The text content to scan.
            context: Additional context dict passed to each sub-scanner.

        Returns:
            ScanResult with aggregated findings, max risk_score, and
            scanner_name set to 'composite'.
        """
        start = time.monotonic()

        results: List[ScanResult] = []
        for scanner in self.scanners:
            result = await scanner.scan(content, context)
            results.append(result)

            # Fast path: if heuristic scanner finds nothing, skip ML scanner
            if result.is_clean and not self.require_consensus:
                break

        all_findings: List[str] = []
        max_risk = 0.0
        for r in results:
            all_findings.extend(r.findings)
            max_risk = max(max_risk, r.risk_score)

        # Any scanner flagging issues means content is not clean.
        # In consensus mode all must agree it's clean; in standard mode same
        # behavior applies — a single flag blocks the content.
        is_clean = all(r.is_clean for r in results)

        return ScanResult(
            is_clean=is_clean,
            risk_score=max_risk,
            findings=all_findings,
            scanner_name="composite",
            scan_time_ms=(time.monotonic() - start) * 1000,
        )
