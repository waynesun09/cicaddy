"""LLM Guard-based semantic prompt injection scanner.

Uses transformer models (ProtectAI deberta-v3) to detect sophisticated prompt
injection attacks that regex heuristic patterns miss. Complements the
HeuristicScanner by catching semantically complex or novel attack vectors.

Requires the llm-guard optional dependency: ``pip install cicaddy[security]``

Performance:
    - ~50-200ms per scan depending on model size and hardware
    - ONNX runtime reduces latency by ~40% compared to PyTorch inference
    - Model is lazy-loaded on first scan call to avoid startup overhead

Integration:
    Typically used as the second scanner in a CompositeScanner pipeline:

    >>> from cicaddy.mcp_client.scanner import HeuristicScanner, CompositeScanner
    >>> from cicaddy.mcp_client.llm_guard_scanner import LLMGuardScanner
    >>> composite = CompositeScanner(
    ...     scanners=[HeuristicScanner(), LLMGuardScanner(threshold=0.7)],
    ...     require_consensus=False,
    ... )
    >>> result = await composite.scan("suspicious content", {})

Configuration:
    - threshold (float): Confidence threshold 0.0-1.0 (default: 0.7).
      Lower values increase sensitivity but may produce false positives.
    - use_onnx (bool): Use ONNX runtime for inference (default: True).
      Requires ``llm-guard[onnx]`` extra.
    - model_name (str): Hugging Face model ID (default:
      ``protectai/deberta-v3-base-prompt-injection-v2``).

Graceful Degradation:
    If llm-guard is not installed, the scanner returns is_clean=True with a
    'scanner unavailable' finding, allowing the system to operate with
    heuristic-only scanning.
"""

import time
from typing import Any, Dict, List

from cicaddy.utils.logger import get_logger

from .scanner import ScanResult

logger = get_logger(__name__)


class LLMGuardScanner:
    """ML-based scanner using llm-guard library.

    Higher accuracy but higher latency (~50-200ms). Optional dependency.
    Falls back gracefully if llm-guard is not installed.

    Uses the ProtectAI deberta-v3 model for prompt injection classification.
    Runs inference in a thread executor to avoid blocking the async event loop.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        use_onnx: bool = True,
        model_name: str = "protectai/deberta-v3-base-prompt-injection-v2",
    ):
        """Initialize the LLM Guard scanner.

        Args:
            threshold: Confidence threshold for injection detection (0.0-1.0).
                Scores above this threshold are flagged as injection.
            use_onnx: Whether to use ONNX runtime for faster inference.
                Requires llm-guard[onnx] to be installed.
            model_name: The Hugging Face model name for prompt injection detection.
        """
        self.threshold = threshold
        self.use_onnx = use_onnx
        self.model_name = model_name
        self._scanner = None

    def _get_scanner(self):
        """Lazy-load the llm-guard PromptInjection scanner on first use.

        Returns:
            PromptInjection scanner instance, or None if llm-guard is not
            installed.
        """
        if self._scanner is None:
            try:
                from llm_guard.input_scanners import PromptInjection
                from llm_guard.input_scanners.prompt_injection import MatchType

                self._scanner = PromptInjection(
                    threshold=self.threshold,
                    match_type=MatchType.FULL,
                    use_onnx=self.use_onnx,
                )
            except ImportError:
                logger.warning(
                    "llm-guard not installed. Install with: pip install cicaddy[security]"
                )
                return None
        return self._scanner

    async def scan(self, content: str, context: Dict[str, Any]) -> ScanResult:
        """Scan content for prompt injection using the ML-based classifier.

        Runs the llm-guard PromptInjection model in a thread executor to
        avoid blocking the async event loop. Returns a clean result if the
        scanner is unavailable (llm-guard not installed).

        Args:
            content: The text content to classify.
            context: Additional context dict (unused by ML scanner but
                required by ContentScanner protocol).

        Returns:
            ScanResult with ML classifier confidence as risk_score.
        """
        import asyncio

        start = time.monotonic()

        scanner = self._get_scanner()
        if scanner is None:
            return ScanResult(
                is_clean=True,
                scanner_name="llm-guard",
                findings=["scanner unavailable"],
                scan_time_ms=(time.monotonic() - start) * 1000,
            )

        # Run ML scanner in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        try:
            sanitized, is_valid, risk_score = await loop.run_in_executor(
                None, scanner.scan, content
            )
        except Exception as e:
            logger.error(f"Error in llm-guard scanning: {e}")
            return ScanResult(
                is_clean=True,
                scanner_name="llm-guard",
                findings=[f"scanner error: {str(e)}"],
                scan_time_ms=(time.monotonic() - start) * 1000,
            )

        scan_time = (time.monotonic() - start) * 1000

        findings: List[str] = []
        if not is_valid:
            findings.append(
                f"ML classifier detected injection (score: {risk_score:.3f})"
            )

        return ScanResult(
            is_clean=is_valid,
            risk_score=risk_score,
            findings=findings,
            scanner_name="llm-guard",
            scan_time_ms=scan_time,
        )
