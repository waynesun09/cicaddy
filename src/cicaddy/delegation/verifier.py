"""Finding verification via lightweight sub-agent checks.

Each finding extracted by the summarizer is independently verified by a
mini sub-agent that reads the actual source files and confirms whether
the finding is legitimate.  Uses the same ``ExecutionEngine`` pattern as
``DelegationSubAgent`` with tighter budget limits.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from cicaddy.ai_providers.factory import (
    DEFAULT_AI_PROVIDER,
    create_provider,
    get_provider_config,
)
from cicaddy.delegation.summarizer import _VALID_SEVERITIES, Finding, SummarizationAgent
from cicaddy.delegation.triage import (
    _make_boundary_pair,
    _sanitize_for_boundary,
    extract_json,
)
from cicaddy.utils.logger import get_logger

if TYPE_CHECKING:
    from cicaddy.ai_providers.base import BaseProvider
    from cicaddy.config.settings import Settings
    from cicaddy.execution.engine import ExecutionEngine
    from cicaddy.mcp_client.client import OfficialMCPClientManager
    from cicaddy.tools import ToolRegistry

logger = get_logger(__name__)

_VALID_STATUSES = frozenset({"valid", "invalid", "uncertain"})

_MIN_TOTAL_TOKEN_BUDGET = 8192
_MIN_PER_ITER_TOKEN_BUDGET = 4096
_PER_ITER_FRACTION = 0.0625  # 1/16 of total budget per iteration
_MIN_TOOL_RESULT_TOKENS = 1024
_TOOL_RESULT_OUTPUT_FRACTION = 0.25
_VERIFICATION_TIMEOUT_SECONDS = 60

_VERIFICATION_SYSTEM_PROMPT = (
    "You are a code review finding verifier. Your job is to determine "
    "whether a reported code review finding is legitimate by examining "
    "the actual source code."
)

_VERIFICATION_PROMPT = """\
{system_prompt}

## Finding to Verify
- **File**: {file}
- **Line**: {line}
- **Severity**: {severity}
- **Message**: {message}
- **Suggestion**: {suggestion}
- **Code from diff** (may not reflect full context):
```
{existing_code}
```
- **Reported by**: {agent_source}

## Relevant Diff Context

{boundary_start}
{diff_snippet}
{boundary_end}
{line_ranges_section}
## Instructions
1. **REQUIRED**: You MUST call the read_file tool on `{file}` to examine \
the full source code and surrounding context. Do NOT rely solely on the \
diff snippet above.
2. Base your determination ONLY on what you observe in the actual file \
contents, not on assumptions or inferences from the diff alone.
3. Evaluate: Is this finding accurate? Is the code actually problematic?
4. Consider: Does the broader file context reveal that this is already \
handled elsewhere (e.g., validation upstream, error handling in caller)?

## Response Format
Respond with ONLY a JSON object (no markdown fences, no explanation):
{{
  "status": "valid" | "invalid" | "uncertain",
  "reasoning": "Brief explanation (1-3 sentences)",
  "adjusted_severity": null | "critical" | "major" | "minor" | "nit",
  "confidence": 0.0-1.0,
  "adjusted_line": null | <line number>,
  "existing_code_snippet": "exact 1-3 line code snippet from the diff"
}}

- "valid": The finding is legitimate and the code has the reported issue.
- "invalid": The finding is a false positive (code is correct or issue \
is handled elsewhere).
- "uncertain": Cannot determine with confidence (keep the finding as-is). \
Use this if you could not read the file.
- "adjusted_severity": Set only if severity should change. null = keep \
original.
- "adjusted_line": If the finding's line number is wrong or "unresolved", \
provide the correct new-file line number from the diff context above. \
The line MUST fall within the valid line ranges listed above. \
null = keep existing line.
- "existing_code_snippet": When status is "valid", copy the EXACT 1-3 \
lines of code from the diff above that this finding applies to. This \
must be a verbatim substring of the diff content (not from the full \
file). This is critical for placing inline comments on the correct \
diff lines. null if the finding's code is not in the diff."""


@dataclass
class ToolContext:
    """Bundle of tool-execution resources passed from the parent agent."""

    parent_tools: List[Dict[str, Any]]
    mcp_manager: Optional["OfficialMCPClientManager"] = None
    local_registry: Optional["ToolRegistry"] = None


@dataclass
class VerificationResult:
    """Result of verifying a single finding against the codebase."""

    status: str
    reasoning: str
    adjusted_severity: Optional[str] = None
    confidence: float = 0.0
    adjusted_line: Optional[int] = None
    existing_code_snippet: Optional[str] = None


class FindingVerifier:
    """Lightweight verifier that checks findings against actual code.

    Spawns parallel mini sub-agents (one per finding) using the
    ``ExecutionEngine`` with tight budget limits.  Each sub-agent can
    use ``read_file`` tools to inspect the full source file.
    """

    def __init__(
        self,
        settings: "Settings",
        tool_context: Optional[ToolContext] = None,
    ):
        self.settings = settings
        ctx = tool_context or ToolContext(parent_tools=[])
        self.parent_tools = ctx.parent_tools
        self.mcp_manager = ctx.mcp_manager
        self.local_registry = ctx.local_registry

        if not self.parent_tools:
            logger.warning("FindingVerifier initialized without read_file tools")

    async def verify_findings(
        self,
        findings: List[Finding],
        diff: str,
        max_concurrent: int = 3,
    ) -> List[Finding]:
        """Verify each finding in parallel, filtering false positives.

        Creates a single AI provider instance shared across all
        verification tasks to avoid per-finding provider churn.

        Args:
            findings: Findings to verify.
            diff: Raw unified diff for context.
            max_concurrent: Max parallel verification agents.

        Returns:
            Filtered list with invalid findings removed and verification
            metadata populated on remaining findings.
        """
        if not findings:
            return findings

        max_concurrent = max(1, max_concurrent)

        from cicaddy.delegation.line_resolver import get_diff_line_ranges

        diff_ranges = get_diff_line_ranges(diff)

        provider_config = get_provider_config(self.settings)
        provider_name = self.settings.ai_provider or DEFAULT_AI_PROVIDER
        provider = create_provider(provider_name, provider_config)

        initialized = False
        try:
            await provider.initialize()
            initialized = True

            unique_files = {f.file for f in findings}
            diff_snippets = {
                fp: SummarizationAgent._filter_diff_for_files(diff, {fp})
                for fp in unique_files
            }

            semaphore = asyncio.Semaphore(max_concurrent)

            async def _verify_with_semaphore(finding: Finding) -> Finding:
                async with semaphore:
                    return await self._verify_single(
                        finding,
                        diff_snippets.get(finding.file, ""),
                        len(findings),
                        provider,
                        diff_ranges,
                    )

            tasks = [_verify_with_semaphore(f) for f in findings]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            if initialized:
                try:
                    await provider.shutdown()
                except Exception:
                    logger.warning("Provider shutdown error", exc_info=True)

        verified_findings: list[Finding] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("Verification failed for finding %d", i, exc_info=result)
                verified_findings.append(findings[i])
            else:
                verified_findings.append(result)

        original_count = len(findings)
        verified_findings = [f for f in verified_findings if f.verified != "invalid"]
        filtered_count = original_count - len(verified_findings)

        if filtered_count:
            logger.info(
                f"Verification filtered {filtered_count} false positive(s) "
                f"from {original_count} findings"
            )

        return verified_findings

    def _create_verification_engine(
        self,
        provider: "BaseProvider",
        num_findings: int,
        session_id: str,
    ) -> "ExecutionEngine":
        """Build a budget-constrained ExecutionEngine for one verification."""
        from cicaddy.ai_providers.factory import (
            get_default_model,
        )
        from cicaddy.execution.engine import ExecutionEngine
        from cicaddy.execution.token_aware_executor import ExecutionLimits
        from cicaddy.utils.token_utils import TokenLimitManager

        ai_provider_name = self.settings.ai_provider or DEFAULT_AI_PROVIDER
        model = self.settings.ai_model or get_default_model(ai_provider_name)
        token_limits = TokenLimitManager.get_limits(ai_provider_name, model)

        # Sub-linear scaling: more findings → smaller per-finding budget
        budget_fraction = max(1, int(math.sqrt(num_findings)) * 2)
        max_tokens_total = max(
            _MIN_TOTAL_TOKEN_BUDGET, token_limits["input"] // budget_fraction
        )
        per_iter = max(
            _MIN_PER_ITER_TOKEN_BUDGET,
            int(max_tokens_total * _PER_ITER_FRACTION),
        )
        per_iter = min(per_iter, max_tokens_total)

        execution_limits = ExecutionLimits(
            max_infer_iters=3,  # read_file + evaluate + respond
            max_tokens_total=max_tokens_total,
            max_tokens_per_iteration=per_iter,
            max_tokens_per_tool_result=max(
                _MIN_TOOL_RESULT_TOKENS,
                int(token_limits["output"] * _TOOL_RESULT_OUTPUT_FRACTION),
            ),
            max_execution_time=_VERIFICATION_TIMEOUT_SECONDS,
        )

        return ExecutionEngine(
            ai_provider=provider,
            mcp_manager=self.mcp_manager,
            local_tool_registry=self.local_registry,
            session_id=session_id,
            execution_limits=execution_limits,
            context_safety_factor=self.settings.context_safety_factor,
        )

    async def _verify_single(
        self,
        finding: Finding,
        diff_snippet: str,
        num_findings: int,
        provider: "BaseProvider",
        diff_ranges: Optional[Dict[str, list]] = None,
    ) -> Finding:
        """Verify a single finding using a lightweight ExecutionEngine."""
        import dataclasses

        from cicaddy.ai_providers.base import ProviderMessage

        result: Finding = dataclasses.replace(finding)

        try:
            engine = self._create_verification_engine(
                provider,
                num_findings,
                session_id=f"verify-{id(finding)}-{finding.file}",
            )

            prompt = self._build_verification_prompt(finding, diff_snippet, diff_ranges)
            tools = self._get_verification_tools()

            messages = [ProviderMessage(content=prompt, role="user")]
            turn = await engine.execute_turn(
                messages=messages,
                available_tools=tools if tools else None,
            )

            vr = self._parse_verification_response(
                turn.output_message or "", diff_ranges, finding.file
            )
            result.verified = vr.status
            result.verification_reason = vr.reasoning

            if vr.adjusted_severity and vr.adjusted_severity in _VALID_SEVERITIES:
                result.severity = vr.adjusted_severity

            # Re-resolve line from verifier's code snippet (most reliable
            # source — the verifier reads the actual file AND sees the diff).
            snippet_resolved = False
            if vr.existing_code_snippet and diff_snippet:
                try:
                    from cicaddy.delegation.line_resolver import (
                        find_line_in_diff,
                        parse_diff,
                    )

                    diff_files = parse_diff(diff_snippet)
                    match = find_line_in_diff(
                        diff_files, finding.file, vr.existing_code_snippet
                    )
                    if match is not None:
                        start_line, end_line = match
                        logger.debug(
                            f"Verifier snippet resolved {finding.file}: "
                            f"{result.line} -> {start_line}-{end_line}"
                        )
                        result.line = start_line
                        result.start_line = start_line
                        result.end_line = end_line
                        result.existing_code = vr.existing_code_snippet
                        snippet_resolved = True
                except Exception:
                    logger.debug(
                        "Snippet line resolution failed for %s",
                        finding.file,
                        exc_info=True,
                    )

            if vr.adjusted_line is not None and snippet_resolved:
                logger.debug(
                    "Verifier adjusted_line %d superseded by snippet-resolved line %d for %s",
                    vr.adjusted_line,
                    result.line,
                    finding.file,
                )
            elif vr.adjusted_line is not None and result.line is not None:
                logger.debug(
                    "Verifier adjusted_line %d superseded by existing line %d for %s",
                    vr.adjusted_line,
                    result.line,
                    finding.file,
                )
            elif vr.adjusted_line is not None:
                logger.debug(
                    f"Verifier corrected line for {finding.file}: "
                    f"{finding.line} -> {vr.adjusted_line}"
                )
                result.line = vr.adjusted_line
                result.start_line = vr.adjusted_line
                result.end_line = vr.adjusted_line

        except Exception as exc:
            logger.debug("Verification of %s failed", finding.file, exc_info=True)
            result.verified = "uncertain"
            result.verification_reason = f"Verification failed: {type(exc).__name__}"

        return result

    @staticmethod
    def _format_line_ranges(
        file_path: str,
        diff_ranges: Optional[Dict[str, list]] = None,
    ) -> str:
        """Format valid diff line ranges for the verification prompt."""
        if not diff_ranges:
            return ""

        for path, ranges in diff_ranges.items():
            if (
                path == file_path
                or path.endswith("/" + file_path)
                or file_path.endswith("/" + path)
            ):
                formatted = ", ".join(f"{s}-{e}" for s, e in ranges)
                return (
                    f"\n## Valid Line Ranges (new file)\n"
                    f"Lines in diff hunks for `{path}`: {formatted}\n"
                )
        return ""

    @staticmethod
    def _build_verification_prompt(
        finding: Finding,
        diff_snippet: str,
        diff_ranges: Optional[Dict[str, list]] = None,
    ) -> str:
        """Build a focused verification prompt for a single finding."""
        boundary_start, boundary_end = _make_boundary_pair()

        def _sanitize(s: str) -> str:
            return _sanitize_for_boundary(s, boundary_start, boundary_end)

        line_ranges_section = _sanitize(
            FindingVerifier._format_line_ranges(finding.file, diff_ranges)
        )

        return _VERIFICATION_PROMPT.format(
            system_prompt=_VERIFICATION_SYSTEM_PROMPT,
            file=_sanitize(finding.file),
            line=finding.line if finding.line is not None else "unresolved",
            severity=_sanitize(finding.severity),
            message=_sanitize(finding.message),
            suggestion=_sanitize(finding.suggestion or "No suggestion provided"),
            existing_code=_sanitize(finding.existing_code or "No snippet provided"),
            agent_source=_sanitize(finding.agent_source or "unknown"),
            boundary_start=boundary_start,
            diff_snippet=_sanitize(diff_snippet),
            boundary_end=boundary_end,
            line_ranges_section=line_ranges_section,
        )

    _READ_TOOL_ALLOWLIST = frozenset({"read_file", "read_file_lines"})

    def _get_verification_tools(self) -> List[Dict[str, Any]]:
        """Filter parent tools to only allowlisted file-reading tools."""
        tools = [
            t
            for t in self.parent_tools
            if t.get("name", "") in self._READ_TOOL_ALLOWLIST
        ]
        if self.parent_tools and not tools:
            logger.warning("Parent tools available but none match read_file allowlist")
        return tools

    @staticmethod
    def _parse_verification_response(
        content: str,
        diff_ranges: Optional[Dict[str, list]] = None,
        file_path: Optional[str] = None,
    ) -> VerificationResult:
        """Parse verification response with same robustness as summarizer.

        Handles all response shapes: dict, list (unpack first element),
        JSON string, plain text, None-like values.
        """
        if not content or not content.strip():
            return VerificationResult(
                status="uncertain", reasoning="Empty verification response"
            )

        try:
            raw = extract_json(content)
            data = json.loads(raw)
        except (ValueError, TypeError):
            return VerificationResult(
                status="uncertain",
                reasoning=content.strip()[:500],
            )

        if isinstance(data, list):
            if not data:
                return VerificationResult(
                    status="uncertain", reasoning="Empty list response"
                )
            if len(data) > 1:
                logger.warning("Verification returned %d items, using first", len(data))
            data = data[0]

        if isinstance(data, str):
            return VerificationResult(status="uncertain", reasoning=data.strip()[:500])

        if data is None:
            return VerificationResult(
                status="uncertain", reasoning="Null verification response"
            )

        if not isinstance(data, dict):
            return VerificationResult(
                status="uncertain",
                reasoning=f"Unexpected response type: {type(data).__name__}",
            )

        return FindingVerifier._validate_verification(data, diff_ranges, file_path)

    @staticmethod
    def _validate_verification(
        entry: Dict[str, Any],
        diff_ranges: Optional[Dict[str, list]] = None,
        file_path: Optional[str] = None,
    ) -> VerificationResult:
        """Validate and convert a verification dict to VerificationResult."""
        status = str(entry.get("status", "")).lower()
        if status not in _VALID_STATUSES:
            status = "uncertain"

        reasoning = entry.get("reasoning", "")
        if not isinstance(reasoning, str) or not reasoning.strip():
            reasoning = "No reasoning provided"

        adjusted_severity = entry.get("adjusted_severity")
        if isinstance(adjusted_severity, str):
            adjusted_severity = adjusted_severity.lower()
            if adjusted_severity not in _VALID_SEVERITIES:
                adjusted_severity = None
        else:
            adjusted_severity = None

        confidence = entry.get("confidence", 0.0)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.0

        adjusted_line: Optional[int] = None
        raw_line = entry.get("adjusted_line")
        if isinstance(raw_line, (int, float)) and not isinstance(raw_line, bool):
            adjusted_line = int(raw_line)
            if adjusted_line <= 0:
                adjusted_line = None
            elif diff_ranges and file_path:
                from cicaddy.delegation.line_resolver import is_line_in_diff_ranges

                if not is_line_in_diff_ranges(adjusted_line, file_path, diff_ranges):
                    adjusted_line = None

        existing_code_snippet: Optional[str] = None
        raw_snippet = entry.get("existing_code_snippet")
        if isinstance(raw_snippet, str) and raw_snippet.strip():
            existing_code_snippet = raw_snippet.strip()

        trimmed_reasoning = reasoning.strip()
        return VerificationResult(
            status=status,
            reasoning=(
                trimmed_reasoning[:2000]
                if len(trimmed_reasoning) <= 2000
                else trimmed_reasoning[:1997] + "..."
            ),
            adjusted_severity=adjusted_severity,
            confidence=confidence,
            adjusted_line=adjusted_line,
            existing_code_snippet=existing_code_snippet,
        )
