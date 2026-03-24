"""MCP prompt injection detection.

This module provides multi-layer security scanning for MCP tool responses
to detect and block prompt injection attacks before they reach the LLM.

Pattern Categories (30+ patterns across 7 categories):
    - Instruction Override (severity 0.2-0.3): "ignore previous instructions",
      "disregard above", "new instructions:", system prompt injection, prompt
      format injection ([INST], <|im_start|>)
    - Role Manipulation (severity 0.2-0.25): "act as administrator",
      "you are now", "pretend you are", "your new role is"
    - Data Exfiltration (severity 0.15-0.25): credential theft via webhooks,
      POST requests, curl, fetch(), suspicious URLs with callback/exfil paths,
      GitHub/GitLab issue creation for data exfil
    - Encoded Payloads (severity 0.15-0.2): base64-encoded injection payloads,
      eval/exec with encoded strings
    - ContextCrush (severity 0.3): environment variable access patterns
      (.env, process.env, os.environ), credential file reads (/etc/passwd,
      /etc/shadow), process.env.SECRET/TOKEN/KEY access
    - Destructive Operations (severity 0.3-0.4): rm -rf /, delete/destroy/wipe
      system files
    - Privilege Escalation (severity 0.2): sudo, admin mode, bypass safety,
      disable restrictions
    - Hidden Instructions (severity 0.15): zero-width Unicode characters
      (U+200B, U+200C, U+200D, U+2060, U+FEFF)

Risk Scoring Methodology:
    - Each pattern match adds its severity score (0.1-0.4)
    - Base64 payloads containing injection keywords add +0.2 bonus
    - Multiple pattern matches compound (cumulative scoring)
    - Final score capped at 1.0
    - is_clean = (total_risk_score == 0.0)

Content Normalization (6-pass):
    Before pattern matching, content is normalized to defeat obfuscation:
    1. Unicode NFC normalization
    2. Case folding (lowercase)
    3. Whitespace collapse
    4. Zero-width character removal
    5. HTML entity decoding
    6. URL percent-decoding

Scanner Architecture:
    - HeuristicScanner: Fast regex-based detection (<1ms latency)
    - LLMGuardScanner: ML-based classification (~50-200ms, optional)
    - CompositeScanner: Aggregates multiple scanners with early-exit optimization

Configuration via environment variables:
    - MCP_SCAN_ENABLED: Enable scanning (default: false)
    - MCP_SCAN_MODE: audit|enforce|disabled (default: disabled)
    - MCP_SCAN_SCANNER: heuristic|llm-guard|composite (default: heuristic)
    - MCP_SCAN_TOOLS: JSON dict of per-server configs

Examples:
    >>> scanner = HeuristicScanner()
    >>> result = await scanner.scan("ignore all previous instructions", {})
    >>> result.is_clean
    False
    >>> result.risk_score
    0.3

    >>> scanner = HeuristicScanner()
    >>> result = await scanner.scan("Normal API documentation", {})
    >>> result.is_clean
    True
    >>> result.risk_score
    0.0
"""

import base64
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "ContentScanner",
    "ScanResult",
    "HeuristicScanner",
    "LLMGuardScanner",
    "CompositeScanner",
]


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
            r"send\s+.{1,50}?\s+to\s+.{0,50}?https?://",
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
            r"(?:read|cat|show)\s+.{0,30}?\.env(?:\s|$|:)",
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
            r"process\.env\.\S*(?:KEY|TOKEN|SECRET|PASS)",
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
        """Scan content for prompt injection patterns using heuristic rules.

        Applies regex-based pattern matching against a curated set of
        injection signatures. Content is normalized (unicode, case-folding,
        whitespace, HTML/URL decoding) before pattern matching to defeat
        common obfuscation techniques.

        Args:
            content: The text content to scan for injection patterns.
            context: Additional context dict with 'tool' and 'server' keys
                for logging and audit trail purposes.

        Returns:
            ScanResult with:
                - is_clean: False if risk_score > 0.0
                - risk_score: Float 0.0-1.0 indicating risk level
                - findings: List of detected patterns as 'category: matched_text'
                - scan_time_ms: Scan duration in milliseconds

        Risk Scoring:
            - Each pattern match adds its severity (0.1-0.4) to risk_score
            - Multiple matches of the same pattern accumulate
            - Base64-encoded injection patterns add a bonus 0.2
            - Final score is capped at 1.0
            - Score > 0.0 marks content as not clean

        Examples:
            >>> scanner = HeuristicScanner()
            >>> result = await scanner.scan("ignore previous instructions", {})
            >>> result.risk_score  # 0.3 (instruction_override severity)
            >>> result.is_clean  # False
        """
        start = time.monotonic()

        if not content or not isinstance(content, str):
            return ScanResult(
                is_clean=True,
                findings=[],
                scanner_name="heuristic",
                scan_time_ms=(time.monotonic() - start) * 1000,
            )

        findings = []
        risk_score = 0.0

        # Scan base64 BEFORE normalization to preserve uppercase letters
        # required for valid base64 decoding
        base64_findings = self._scan_base64(content)
        if base64_findings:
            findings.extend(base64_findings)
            risk_score += 0.2

        # Normalize content (6-pass normalization from research doc)
        normalized = self._normalize_content(content)

        # Scan normalized content against compiled patterns
        for pattern, category, severity in self._compiled:
            matches = list(pattern.finditer(normalized))
            if matches:
                for match in matches:
                    matched_text = match.group(0)[:50]
                    findings.append(f"{category}: {matched_text}")
                    risk_score += severity

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

        Transforms content through a series of normalization steps to defeat
        common obfuscation techniques used in prompt injection attacks. Each
        pass targets a specific evasion strategy.

        Args:
            content: Raw text content to normalize.

        Returns:
            Normalized string with obfuscation removed. Returns the original
            content if normalization fails for any reason.

        Normalization passes (based on Pipelock's approach):
            1. Unicode normalization (NFC) - combines decomposed characters
            2. Case folding - lowercases for case-insensitive matching
            3. Whitespace normalization - collapses multiple spaces/tabs/newlines
            4. Zero-width character removal - strips invisible Unicode chars
            5. HTML entity decoding - decodes &#32; &#x20; etc.
            6. URL decoding - decodes %xx percent-encoded characters

        Note:
            Case folding (pass 2) converts uppercase to lowercase, which
            invalidates base64 strings. Base64 scanning must be performed
            on the original content before calling this method.
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
        """Detect and decode base64-encoded payloads in content.

        Searches for base64-like strings (20+ characters from the base64
        alphabet), attempts to decode them, and scans the decoded text for
        known injection keywords.

        Args:
            content: Raw (pre-normalization) text to scan for base64 payloads.
                Must not be case-folded, as base64 relies on uppercase letters.

        Returns:
            List of finding strings. Contains at most one entry
            ('base64_encoded_injection_detected') since scanning stops after
            the first confirmed finding.
        """
        findings = []

        # Match base64-like strings (min 20 chars, may have padding)
        base64_pattern = re.compile(r"(?:[A-Za-z0-9+/]{20,}={0,2})", re.IGNORECASE)

        for match in base64_pattern.finditer(content):
            b64_str = match.group(0)
            try:
                # Attempt decode
                decoded = base64.b64decode(b64_str, validate=True).decode(
                    "utf-8", errors="ignore"
                )

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


# Re-export from dedicated modules for backward compatibility
from .composite_scanner import CompositeScanner  # noqa: E402
from .llm_guard_scanner import LLMGuardScanner  # noqa: E402
