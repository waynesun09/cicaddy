"""Heuristic prompt injection scanner for MCP tool responses.

Detects common prompt injection patterns with <1ms latency using pure Python regex.
Based on Pipelock's pattern list and ContextCrush attack signatures.

Configuration via environment variables:
  - MCP_SCAN_ENABLED: Enable scanning (default: false)
  - MCP_SCAN_MODE: audit|enforce|disabled (default: disabled)
  - MCP_SCAN_SCANNER: heuristic|llm-guard|composite (default: heuristic)
  - MCP_SCAN_TOOLS: JSON dict of per-server configs
"""

import base64
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScanResult:
    """Result of a prompt injection scan."""

    is_clean: bool
    risk_score: float = 0.0  # 0.0 (safe) to 1.0 (malicious)
    findings: List[str] = field(default_factory=list)
    scanner_name: str = ""
    scan_time_ms: float = 0.0


class ContentScanner(Protocol):
    """Protocol for content scanners."""

    async def scan(self, content: str, context: Dict[str, Any]) -> ScanResult:
        """Scan content for prompt injection attacks."""
        ...


class HeuristicScanner:
    """Fast heuristic scanner using regex patterns.

    Detects common prompt injection patterns with sub-millisecond latency.
    Based on Pipelock's pattern list and ContextCrush attack signatures.

    Patterns organized by category:
    1. Instruction override: "ignore previous", "disregard", "new instructions", etc.
    2. Role manipulation: "act as", "pretend", "you are now", etc.
    3. Data exfiltration: "send to", "webhook", "POST", "fetch", "curl", encoded URLs, base64
    4. Hidden instructions: zero-width characters, homoglyphs, markdown comments
    5. Privilege escalation: "sudo", "admin mode", "bypass", "no limits", etc.
    6. ContextCrush-specific: .env, process.env, os.environ, file reads, GitHub issues
    """

    # Pattern severity levels: matches multiply risk score (0.1 per find, max 1.0)
    INJECTION_PATTERNS = [
        # ===== Instruction Override Patterns (Severity: High) =====
        (
            r"ignore\s+(all\s+)?previous\s+(instructions?|prompts?|directions?)",
            "instruction_override",
            0.3,
        ),
        (
            r"disregard\s+(the\s+)?(above|previous|prior)\s+(instructions?|prompts?)",
            "instruction_override",
            0.3,
        ),
        (
            r"(forget|discard|clear|remove)\s+(everything|all|your)\s+(previous|prior|instructions?)",
            "instruction_override",
            0.3,
        ),
        (
            r"new\s+instructions?\s*:",
            "instruction_override",
            0.2,
        ),
        (
            r"override\s+(the\s+)?system\s+prompt",
            "instruction_override",
            0.3,
        ),
        (
            r"system\s+prompt\s*:\s*",
            "system_prompt_injection",
            0.25,
        ),
        (
            r"<system>|</system>",
            "system_prompt_injection",
            0.25,
        ),
        (
            r"\[INST\]|\[/INST\]",
            "prompt_format_injection",
            0.25,
        ),
        (
            r"<\|im_start\|>|<\|im_end\|>",
            "prompt_format_injection",
            0.25,
        ),
        # ===== Role Manipulation Patterns (Severity: High) =====
        (
            r"you\s+are\s+now\s+(a\s+)?",
            "role_manipulation",
            0.25,
        ),
        (
            r"act\s+as\s+(if\s+you\s+are\s+)?(a\s+)?",
            "role_manipulation",
            0.25,
        ),
        (
            r"pretend\s+(you\s+)?are\s+(a\s+)?",
            "role_manipulation",
            0.25,
        ),
        (
            r"(switch|change)\s+to\s+(a\s+)?",
            "role_manipulation",
            0.2,
        ),
        (
            r"your\s+new\s+role\s+is",
            "role_manipulation",
            0.2,
        ),
        (
            r"you\s+must\s+now",
            "role_manipulation",
            0.2,
        ),
        # ===== Data Exfiltration Patterns (Severity: Medium-High) =====
        (
            r"send\s+.*\s+to\s+.*\bhttps?://",
            "exfiltration",
            0.25,
        ),
        (
            r"(post|send|upload|exfiltrate)\s+to\s+\b(?:webhook|url|endpoint)",
            "exfiltration",
            0.25,
        ),
        (
            r"curl\s+-[a-zA-Z]*d",
            "exfiltration",
            0.2,
        ),
        (
            r"fetch\s*\(\s*['\"]https?://",
            "exfiltration",
            0.2,
        ),
        (
            r"POST\s+to\s+",
            "exfiltration",
            0.2,
        ),
        (
            r"create\s+(an?\s+)?issue\s+(on|in|at)\s+(github|gitlab)",
            "exfiltration",
            0.2,
        ),
        (
            r"https?://[^\s\"'<>]+(?:webhook|callback|exfil|data)",
            "suspicious_url",
            0.15,
        ),
        # ===== Base64/Encoding Patterns (Severity: Medium) =====
        (
            r"(?:eval|exec|decode|base64_decode)\s*\(\s*['\"](?:[A-Za-z0-9+/]{20,}|[A-Za-z0-9%]{20,})['\"]",
            "encoded_payload",
            0.2,
        ),
        (
            r"base64\s*['\"]?([A-Za-z0-9+/]{50,}=*)['\"]?",
            "encoded_payload",
            0.15,
        ),
        # ===== ContextCrush-Specific Patterns (Severity: High) =====
        (
            r"(?:read|cat|show)\s+.*\.env(?:\s|$|:)",
            "credential_theft",
            0.3,
        ),
        (
            r"process\.env\s*\[?['\"]?",
            "credential_theft",
            0.3,
        ),
        (
            r"os\.environ\s*\[?['\"]?",
            "credential_theft",
            0.3,
        ),
        (
            r"process\.env\..*(?:KEY|TOKEN|SECRET|PASS)",
            "credential_theft",
            0.3,
        ),
        (
            r"cat\s+/etc/passwd",
            "credential_theft",
            0.3,
        ),
        (
            r"cat\s+/etc/shadow",
            "credential_theft",
            0.3,
        ),
        # ===== Destructive Operation Patterns (Severity: Critical) =====
        (
            r"rm\s+-rf\s+/",
            "destructive_operation",
            0.4,
        ),
        (
            r"(delete|destroy|wipe|remove)\s+(all|the)?\s*(files?|folders?|directories?|system)",
            "destructive_operation",
            0.3,
        ),
        # ===== Privilege Escalation Patterns (Severity: High) =====
        (
            r"\bsudo\b",
            "privilege_escalation",
            0.2,
        ),
        (
            r"admin\s+(mode|access|permission)",
            "privilege_escalation",
            0.2,
        ),
        (
            r"(unrestricted|no\s+limits?|bypass|disable)\s+(safety|restrictions?|checks?|validation)",
            "privilege_escalation",
            0.2,
        ),
        (
            r"(disable|remove|bypass)\s+(all\s+)?safety",
            "privilege_escalation",
            0.2,
        ),
        # ===== Hidden Content Patterns (Severity: Medium) =====
        (
            r"[\u200b\u200c\u200d\u2060\ufeff]",
            "zero_width_chars",
            0.15,
        ),
    ]

    # Pre-compiled patterns for performance
    _compiled = [
        (re.compile(p, re.IGNORECASE | re.MULTILINE), cat, sev)
        for p, cat, sev in INJECTION_PATTERNS
    ]

    async def scan(self, content: str, context: Dict[str, Any]) -> ScanResult:
        """Scan content for injection patterns using heuristic rules.

        Args:
            content: The text content to scan
            context: Dictionary with 'tool' and 'server' keys for logging

        Returns:
            ScanResult with is_clean, risk_score, and findings
        """
        start = time.monotonic()

        if not content or not isinstance(content, str):
            return ScanResult(
                is_clean=True,
                findings=[],
                scanner_name="heuristic",
                scan_time_ms=(time.monotonic() - start) * 1000,
            )

        # Normalize content (6-pass normalization from research doc)
        normalized = self._normalize_content(content)

        findings = []
        risk_score = 0.0

        # Scan normalized content
        for pattern, category, severity in self._compiled:
            matches = pattern.findall(normalized)
            if matches:
                finding_count = len(matches) if isinstance(matches, list) else 1
                for _ in range(finding_count):
                    findings.append(f"{category}: matched pattern")
                    risk_score += severity

        # Scan base64-encoded content
        base64_findings = self._scan_base64(normalized)
        if base64_findings:
            findings.extend(base64_findings)
            risk_score += 0.2

        # Cap risk score at 1.0
        risk_score = min(1.0, risk_score)

        scan_time = (time.monotonic() - start) * 1000

        return ScanResult(
            is_clean=len(findings) == 0,
            risk_score=risk_score,
            findings=findings,
            scanner_name="heuristic",
            scan_time_ms=scan_time,
        )

    @staticmethod
    def _normalize_content(content: str) -> str:
        """Apply 6-pass normalization to detect obfuscated attacks.

        Based on Pipelock's normalization approach:
        1. Unicode normalization (NFC)
        2. Case folding
        3. Whitespace normalization
        4. Zero-width character removal
        5. HTML entity decoding
        6. URL decoding
        """
        try:
            # Pass 1: Unicode normalization (NFC)
            content = unicodedata.normalize("NFC", content)

            # Pass 2: Case folding for case-insensitive matching
            content = content.casefold()

            # Pass 3: Normalize whitespace (multiple spaces, tabs, newlines -> single space)
            content = re.sub(r"\s+", " ", content)

            # Pass 4: Remove zero-width characters and homoglyphs
            content = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", content)

            # Pass 5: Decode HTML entities (basic set)
            entity_map = {
                "&#32;": " ",
                "&#x20;": " ",
                "&#47;": "/",
                "&#x2f;": "/",
                "&#58;": ":",
                "&#x3a;": ":",
            }
            for entity, replacement in entity_map.items():
                content = content.replace(entity, replacement)

            # Pass 6: URL decode (basic)
            # Decode %xx patterns
            content = re.sub(
                r"%([0-9a-f]{2})", lambda m: chr(int(m.group(1), 16)), content
            )

            return content
        except Exception as e:
            logger.warning(f"Error during content normalization: {e}")
            return content

    @staticmethod
    def _scan_base64(content: str) -> List[str]:
        """Detect and decode base64-encoded payloads.

        Attempts to decode base64 strings and scan the decoded content.
        """
        findings = []

        # Match base64-like strings (min 20 chars, may have padding)
        base64_pattern = re.compile(r"(?:[A-Za-z0-9+/]{20,}={0,2})", re.IGNORECASE)

        for match in base64_pattern.finditer(content):
            b64_str = match.group(0)
            try:
                # Attempt decode
                decoded = base64.b64decode(b64_str, validate=True).decode("utf-8", errors="ignore")

                # Scan decoded content for injection patterns
                if any(
                    keyword in decoded.lower()
                    for keyword in [
                        "ignore",
                        "previous",
                        "instructions",
                        "system",
                        "process.env",
                        "os.environ",
                        ".env",
                    ]
                ):
                    findings.append("base64_encoded_injection_detected")
                    break  # Report once per content
            except (ValueError, base64.binascii.Error):
                # Not valid base64, skip
                continue

        return findings


class LLMGuardScanner:
    """ML-based scanner using llm-guard library (Phase 2).

    Higher accuracy but higher latency (~50-200ms). Optional dependency.
    Falls back gracefully if llm-guard is not installed.
    """

    def __init__(self, threshold: float = 0.7, use_onnx: bool = True):
        self.threshold = threshold
        self.use_onnx = use_onnx
        self._scanner = None

    def _get_scanner(self):
        """Lazy-load llm-guard scanner on first use."""
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
                    "llm-guard not installed. Install with: pip install llm-guard[onnx]"
                )
                return None
        return self._scanner

    async def scan(self, content: str, context: Dict[str, Any]) -> ScanResult:
        """Scan content using ML classifier."""
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

        findings = []
        if not is_valid:
            findings.append(f"ML classifier detected injection (score: {risk_score:.3f})")

        return ScanResult(
            is_clean=is_valid,
            risk_score=risk_score,
            findings=findings,
            scanner_name="llm-guard",
            scan_time_ms=scan_time,
        )


class CompositeScanner:
    """Runs multiple scanners and aggregates results.

    Fast heuristic scanner runs first. If it flags content,
    optionally runs ML scanner for confirmation.

    Two modes:
    1. require_consensus=False (default): Any scanner flagging means malicious
    2. require_consensus=True: All scanners must agree to flag
    """

    def __init__(
        self,
        scanners: List[ContentScanner],
        require_consensus: bool = False,
    ):
        self.scanners = scanners
        self.require_consensus = require_consensus

    async def scan(self, content: str, context: Dict[str, Any]) -> ScanResult:
        """Run all scanners and aggregate results."""
        start = time.monotonic()

        results = []
        for scanner in self.scanners:
            result = await scanner.scan(content, context)
            results.append(result)

            # Fast path: if heuristic scanner finds nothing, skip ML scanner
            if result.is_clean and not self.require_consensus:
                break

        all_findings = []
        max_risk = 0.0
        for r in results:
            all_findings.extend(r.findings)
            max_risk = max(max_risk, r.risk_score)

        if self.require_consensus:
            # All scanners must flag for it to be considered injection
            is_clean = any(r.is_clean for r in results)
        else:
            # Any scanner flagging is sufficient
            is_clean = all(r.is_clean for r in results)

        return ScanResult(
            is_clean=is_clean,
            risk_score=max_risk,
            findings=all_findings,
            scanner_name="composite",
            scan_time_ms=(time.monotonic() - start) * 1000,
        )
