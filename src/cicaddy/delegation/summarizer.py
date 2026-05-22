"""AI-powered summarization of multi-agent delegation results.

Uses the parent agent's AI provider to condense multiple sub-agent
analyses into a concise consolidated review with structured findings
for inline comment support.  Employs a multi-turn schema-validated
conversation loop: the AI generates a response, the code validates
it against a JSON schema, and sends correction requests back until
the output is valid (up to ``max_turns``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from cicaddy.delegation.triage import (
    _make_boundary_pair,
    _sanitize_agent_name,
    _sanitize_for_boundary,
    extract_json,
)
from cicaddy.utils.logger import get_logger

if TYPE_CHECKING:
    from cicaddy.ai_providers.base import BaseProvider

logger = get_logger(__name__)

_VALID_SEVERITIES = frozenset({"critical", "major", "minor", "nit"})
_ERR_EMPTY = "AI response is empty"
_MAX_CORRECTION_TURNS = 8

_SCHEMA_PATH = Path(__file__).parent / "schemas" / "summarizer_response.schema.json"
_RESPONSE_SCHEMA: Dict[str, Any] = json.loads(_SCHEMA_PATH.read_text())

_SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a technical review summarizer. Condense the following "
    "multi-agent code review analyses into a single unified review."
)

_SUMMARY_RULES = """\
## Summary Rules
- Target 300-500 words for the summary
- Group findings by severity: Critical > Major > Minor > Nit
- De-duplicate: if multiple agents flagged the same issue, mention it once
- Preserve concrete, actionable suggestions — include code snippets when agents provided them
- Do NOT invent new findings — only summarize what agents reported
- Use markdown formatting for structure:
  - ### headings for each severity group with emoji and count (e.g. ### 🔴 Critical (2))
  - Emoji per severity: 🔴 Critical, 🟠 Major, 🟡 Minor, 🔵 Nit
  - Numbered list for findings within each severity group
  - Each finding on its own line: **`file_path`** — description
  - Keep suggestions inline and concise (one sentence), do NOT use fenced code blocks in the summary
  - Use backtick-wrapped file paths and inline code references
- Omit empty severity groups (if no Critical findings, skip that section)
- End with a brief **Overall Assessment** (1-2 sentences)"""

_RESPONSE_FORMAT = """\
## Response Format

Respond with ONLY a JSON object (no preamble, no explanation).
The JSON must match this schema exactly:

{schema_json}

Field rules:
- "summary" — required, non-empty PROSE markdown string with the consolidated review (NOT a JSON array — always human-readable text)
- "findings" — required array of structured findings extracted from the agent analyses
- Each finding requires: "file" (non-empty string), "severity" (critical|major|minor|nit), "message" (non-empty string)
- "existing_code" — include the EXACT code snippet from the diff (1-3 lines). Accurate snippets are critical for inline comment placement
- "line" — integer line number if the agent cited one, otherwise null
- "suggestion" — concrete fix when the agent provided one, otherwise null
- "agent_source" — name of the agent that reported the finding
- If no findings, use an empty array: []
- Do NOT invent findings — only extract what agents explicitly reported"""

_CORRECTION_PROMPT = """\
Your response did not match the required JSON schema. Errors:

{errors}

Fix these issues and respond with ONLY the corrected JSON object \
(no markdown fences, no explanation). The schema requires:
- Root: {{"summary": "<markdown string>", "findings": [...]}}
- Each finding: {{"file": "...", "severity": "critical|major|minor|nit", "message": "..."}}"""


@dataclass
class Finding:
    """A structured review finding that can be mapped to inline comments."""

    file: str
    line: Optional[int]
    severity: str
    message: str
    suggestion: Optional[str] = None
    agent_source: str = ""
    existing_code: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    verified: Optional[str] = None
    verification_reason: Optional[str] = None


@dataclass
class SummarizationResult:
    """Result of AI-powered review summarization."""

    summary: str
    individual_sections: str
    findings: List[Finding] = field(default_factory=list)
    footer: str = ""
    ai_summarized: bool = False


def _coerce_int(val: Any) -> Optional[int]:
    """Coerce a value to int, returning None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _format_finding_md(entry: dict) -> str:
    """Format a single finding dict as a rich markdown block."""
    msg = str(entry.get("message") or "").strip()
    file_path = entry.get("file", "")
    existing_code = entry.get("existing_code")
    suggestion = entry.get("suggestion")

    parts: list[str] = []
    if file_path:
        parts.append(f"In `{file_path}`:")
    if msg:
        parts.append(msg)

    if existing_code and isinstance(existing_code, str) and existing_code.strip():
        code = existing_code.strip()
        fence = "~~~" if "```" in code else "```"
        parts.append(f"\n{fence}\n{code}\n{fence}")

    if suggestion and isinstance(suggestion, str) and suggestion.strip():
        parts.append(f"\n**Suggestion:** {suggestion.strip()}")

    return "\n".join(parts) if parts else "No details provided."


def _validate_single_finding(
    i: int, entry: dict, allowed_keys: set, errors: List[str]
) -> None:
    """Validate a single finding entry, appending errors."""
    if (
        "file" not in entry
        or not isinstance(entry.get("file"), str)
        or not entry["file"].strip()
    ):
        errors.append(f"findings[{i}]: missing or empty required field 'file'")
    if (
        "message" not in entry
        or not isinstance(entry.get("message"), str)
        or not entry["message"].strip()
    ):
        errors.append(f"findings[{i}]: missing or empty required field 'message'")
    sev = entry.get("severity", "")
    if not isinstance(sev, str) or sev.lower() not in _VALID_SEVERITIES:
        errors.append(
            f"findings[{i}]: 'severity' must be one of "
            f"{sorted(_VALID_SEVERITIES)}, got {sev!r}"
        )

    for int_field in ("line", "start_line", "end_line"):
        int_val = entry.get(int_field)
        if int_val is not None and not isinstance(int_val, int):
            errors.append(
                f"findings[{i}]: '{int_field}' must be an integer "
                f"or null, got {type(int_val).__name__}"
            )
    sl = entry.get("start_line")
    el = entry.get("end_line")
    if isinstance(sl, int) and isinstance(el, int) and sl > el:
        errors.append(
            f"findings[{i}]: 'start_line' ({sl}) must be <= 'end_line' ({el})"
        )
    for str_field in ("existing_code", "suggestion"):
        val = entry.get(str_field)
        if val is not None and not isinstance(val, str):
            errors.append(
                f"findings[{i}]: '{str_field}' must be a string "
                f"or null, got {type(val).__name__}"
            )
    agent_src = entry.get("agent_source")
    if agent_src is not None and not isinstance(agent_src, str):
        errors.append(
            f"findings[{i}]: 'agent_source' must be a string "
            f"or null, got {type(agent_src).__name__}"
        )
    extra = set(entry.keys()) - allowed_keys
    if extra:
        errors.append(f"findings[{i}]: unexpected fields {extra}")


def _validate_findings_entries(findings: list[Any], errors: List[str]) -> None:
    """Validate each entry in the findings array, appending errors."""
    allowed_keys = set(
        _RESPONSE_SCHEMA["properties"]["findings"]["items"]["properties"].keys()
    )
    for i, entry in enumerate(findings):
        if not isinstance(entry, dict):
            errors.append(
                f"findings[{i}]: must be an object, got {type(entry).__name__}"
            )
            continue
        _validate_single_finding(i, entry, allowed_keys, errors)


class SummarizationAgent:
    """AI-powered summarization of multi-agent review results.

    Uses a multi-turn conversation with JSON schema validation to
    ensure the AI produces well-structured output.  On each turn the
    response is validated against the bundled JSON schema; validation
    errors are sent back as a correction prompt up to ``max_turns``.
    """

    def __init__(
        self,
        ai_provider: "BaseProvider",
        settings: Optional[Any] = None,
    ):
        self.ai_provider = ai_provider
        self.settings = settings

    async def summarize(
        self,
        agent_results: List[Dict[str, Any]],
        custom_instructions: str = "",
        diff: str = "",
    ) -> SummarizationResult:
        """Summarize multiple agent analyses into structured output.

        Uses a multi-turn schema-validated conversation:
        1. AI generates JSON response matching the schema
        2. Response is validated; errors trigger correction turns
        3. Deterministic + AI line resolution on extracted findings

        Args:
            agent_results: List of per-agent result dicts from the
                orchestrator (each with agent_name, status, analysis, etc.).
            custom_instructions: Optional user-provided summarization
                instructions.
            diff: Raw unified diff string for line number resolution.

        Returns:
            SummarizationResult with concise summary, individual agent
            sections, and structured findings.
        """
        successful = [r for r in agent_results if r.get("status") == "success"]
        footer = self._build_footer(agent_results)

        if not successful:
            return SummarizationResult(
                summary="No sub-agent results available.",
                individual_sections="",
                findings=[],
                footer=footer,
            )

        if len(successful) == 1:
            analysis = successful[0].get("analysis", "")
            return SummarizationResult(
                summary=analysis,
                individual_sections="",
                findings=[],
                footer=footer,
            )

        # 2+ successful agents — run AI summarization
        individual_sections = self._build_individual_sections(agent_results)

        try:
            prompt = self._build_summarization_prompt(successful, custom_instructions)

            from cicaddy.ai_providers.base import ProviderMessage

            messages = [ProviderMessage(content=prompt, role="user")]
            response = await self.ai_provider.chat_completion(messages)

            summary, findings = await self._validate_and_correct(
                messages, response.content
            )

            if diff and findings:
                findings = await self._resolve_lines(findings, diff)

            logger.info(
                f"Summarization complete: {len(findings)} findings extracted "
                f"from {len(successful)} agent analyses"
            )

            return SummarizationResult(
                summary=summary,
                individual_sections=individual_sections,
                findings=findings,
                footer=footer,
                ai_summarized=True,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Summarization failed, falling back to concatenation: {e}")
            return self._fallback_result(agent_results)
        except Exception as e:
            logger.warning(
                f"Summarization AI call failed, falling back: {type(e).__name__}"
            )
            return self._fallback_result(agent_results)

    def _build_summarization_prompt(
        self,
        successful_results: List[Dict[str, Any]],
        custom_instructions: str = "",
    ) -> str:
        """Build the summarization prompt for the AI."""
        boundary_start, boundary_end = _make_boundary_pair()

        analyses_parts = []
        for result in successful_results:
            agent_name = _sanitize_agent_name(result.get("agent_name", "unknown"))
            categories = ", ".join(
                _sanitize_agent_name(c) for c in result.get("categories", [])
            )
            analysis = _sanitize_for_boundary(
                result.get("analysis", ""), boundary_start, boundary_end
            )
            analyses_parts.append(
                f"### {agent_name} (categories: {categories})\n\n{analysis}"
            )
        analyses_section = "\n\n---\n\n".join(analyses_parts)

        custom_section = ""
        if custom_instructions:
            sanitized = _sanitize_for_boundary(
                custom_instructions, boundary_start, boundary_end
            )
            custom_section = f"\n## Additional Instructions\n{sanitized}\n"

        schema_json = json.dumps(_RESPONSE_SCHEMA, indent=2)
        response_format = _RESPONSE_FORMAT.format(schema_json=schema_json)

        return (
            f"{_SUMMARIZATION_SYSTEM_PROMPT}\n\n"
            f"{_SUMMARY_RULES}\n"
            f"{custom_section}\n"
            f"## Agent Analyses to Summarize\n\n"
            f"{boundary_start}\n{analyses_section}\n{boundary_end}\n\n"
            f"{response_format}"
        )

    async def _validate_and_correct(
        self,
        messages: list,
        response_content: str,
        max_turns: int = _MAX_CORRECTION_TURNS,
    ) -> tuple[str, List[Finding]]:
        """Multi-turn schema validation loop.

        Validates the AI response against the JSON schema.  If invalid,
        appends the response and a correction prompt to the conversation
        and re-calls the AI, up to ``max_turns`` total attempts.

        Note: mutates ``messages`` in-place with correction turns.

        Returns:
            Tuple of (summary_text, findings_list).
        """
        from cicaddy.ai_providers.base import ProviderMessage

        content = response_content
        for turn in range(max_turns):
            result, errors = self._attempt_validation(content, turn)
            if result is not None:
                return result

            if turn >= max_turns - 1:
                break

            logger.info(
                "Schema validation failed (turn %d/%d), requesting correction: %s",
                turn + 1,
                max_turns,
                "; ".join(errors),
            )

            messages.append(ProviderMessage(content=content, role="assistant"))
            correction = _CORRECTION_PROMPT.format(
                errors="\n".join(f"- {e}" for e in errors)
            )
            messages.append(ProviderMessage(content=correction, role="user"))

            response = await self.ai_provider.chat_completion(messages)
            content = response.content

        logger.warning(
            "Schema validation failed after %d turns, using raw text as summary",
            max_turns,
        )
        return self._fallback_extract(content, response_content)

    def _attempt_validation(
        self, content: str, turn: int
    ) -> tuple[Optional[tuple[str, List[Finding]]], List[str]]:
        """Attempt to validate a single response turn.

        Returns (result, errors) where result is (summary, findings) on
        success or None on failure.
        """
        text = content.strip()
        if not text:
            if turn == 0:
                raise ValueError(_ERR_EMPTY)
            return None, ["Response is empty."]

        parsed = self._parse_and_unpack(text)
        if parsed is not None:
            result = self._validate_parsed(parsed)
            if result is not None:
                summary, findings = result
                if turn > 0:
                    logger.info(
                        "Schema validation passed after %d correction turn(s)",
                        turn,
                    )
                return (summary, findings), []
            return None, self._validate_against_schema(parsed)
        return None, self._collect_validation_errors(text)

    def _fallback_extract(
        self, content: str, response_content: str
    ) -> tuple[str, List[Finding]]:
        """Extract summary from raw text when validation loop exhausts."""
        for candidate in (content.strip(), response_content.strip()):
            try:
                data = json.loads(extract_json(candidate))
                data = self._unpack_bare_array(data)
                if isinstance(data, dict) and isinstance(data.get("summary"), str):
                    return data["summary"], []
            except ValueError:
                continue
        return content.strip() or response_content.strip(), []

    def _parse_and_unpack(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON text, unpack bare arrays, return dict or None."""
        try:
            raw = extract_json(text)
            data = json.loads(raw)
        except ValueError:
            return None

        data = self._unpack_bare_array(data)
        if not isinstance(data, dict):
            return None
        return data

    def _validate_parsed(
        self, data: Dict[str, Any]
    ) -> Optional[tuple[str, List[Finding]]]:
        """Validate pre-parsed dict against schema, extract findings on success."""
        errors = self._validate_against_schema(data)
        if errors:
            return None

        summary = data["summary"]
        raw_findings = data.get("findings", [])
        findings = [
            f
            for entry in raw_findings
            if isinstance(entry, dict) and (f := self._validate_finding(entry))
        ]
        return summary, findings

    @staticmethod
    def _unpack_bare_array(data: Any) -> Any:
        """Wrap a bare JSON array into the expected object format.

        Gemini models often return ``[{...}, ...]`` instead of
        ``{"summary": "...", "findings": [...]}``.  When *data* is a
        list of dicts, wrap it as findings and synthesise a summary
        from the individual messages so the schema validator can
        accept it without an extra correction turn.
        """
        if not isinstance(data, list):
            return data

        findings = [e for e in data if isinstance(e, dict)]
        if not findings:
            return data
        dropped = len(data) - len(findings)
        if dropped:
            logger.debug("Dropped %d non-dict entries from bare array", dropped)

        _SEV_MAP = {
            "high": "major",
            "medium": "minor",
            "low": "nit",
            "warning": "minor",
            "info": "nit",
            "error": "critical",
        }
        severity_groups: dict[str, list[dict]] = {}
        for f in findings:
            sev = str(f.get("severity", "minor")).lower()
            sev = _SEV_MAP.get(sev, sev)
            if sev not in ("critical", "major", "minor", "nit"):
                sev = "minor"
            msg = f.get("message", "")
            if msg:
                severity_groups.setdefault(sev, []).append(f)

        parts: list[str] = []
        finding_num = 0
        for sev in ("critical", "major", "minor", "nit"):
            group = severity_groups.get(sev, [])
            if not group:
                continue
            parts.append(f"## {sev.title()}")
            for entry in group:
                finding_num += 1
                body = _format_finding_md(entry)
                parts.append(f"### {finding_num}. ({sev.title()})\n\n{body}")

        if parts:
            summary = "\n\n".join(parts)
        else:
            summary = "Review findings from sub-agent analyses."

        logger.info(
            "Unpacked bare JSON array (%d items) into schema-compliant object",
            len(findings),
        )
        return {"summary": summary, "findings": findings}

    @staticmethod
    def _validate_against_schema(data: Any) -> List[str]:
        """Validate parsed JSON against the response schema.

        Returns a list of error descriptions (empty if valid).
        """
        errors: List[str] = []

        if not isinstance(data, dict):
            errors.append(f"Root must be a JSON object, got {type(data).__name__}")
            return errors

        if "summary" not in data:
            errors.append("Missing required field: 'summary'")
        elif not isinstance(data["summary"], str):
            errors.append(
                f"'summary' must be a string, got {type(data['summary']).__name__}"
            )
        elif not data["summary"].strip():
            errors.append("'summary' must be non-empty")

        if "findings" not in data:
            errors.append("Missing required field: 'findings'")
        elif not isinstance(data["findings"], list):
            errors.append(
                f"'findings' must be an array, got {type(data['findings']).__name__}"
            )
        else:
            _validate_findings_entries(data["findings"], errors)

        allowed_root = {"summary", "findings"}
        extra_root = set(data.keys()) - allowed_root
        if extra_root:
            errors.append(f"Unexpected root fields: {extra_root}")

        return errors

    def _collect_validation_errors(self, text: str) -> List[str]:
        """Collect validation errors for a correction prompt."""
        try:
            raw = extract_json(text)
            data = json.loads(raw)
        except json.JSONDecodeError:
            return ["Response is not valid JSON. Respond with a JSON object only."]
        except ValueError:
            return [
                "Could not extract JSON from response. Respond with a JSON object only."
            ]

        data = self._unpack_bare_array(data)

        if not isinstance(data, dict):
            return [
                f"Response is a JSON {type(data).__name__}, not an object. "
                'Wrap in {{"summary": "...", "findings": [...]}}.'
            ]

        return self._validate_against_schema(data)

    @staticmethod
    def _validate_finding(entry: Dict[str, Any]) -> Optional[Finding]:
        """Validate and convert a single finding dict to Finding."""
        file_path = entry.get("file", "")
        if not isinstance(file_path, str) or not file_path.strip():
            return None

        severity = str(entry.get("severity", "")).lower()
        if severity not in _VALID_SEVERITIES:
            severity = "minor"

        message = entry.get("message", "")
        if not isinstance(message, str) or not message.strip():
            return None

        raw_line = entry.get("line")
        try:
            line = int(raw_line) if raw_line is not None else None
        except (TypeError, ValueError):
            line = None

        existing_code = entry.get("existing_code")
        if isinstance(existing_code, str) and not existing_code.strip():
            existing_code = None

        return Finding(
            file=file_path,
            line=line,
            severity=severity,
            message=message,
            suggestion=entry.get("suggestion"),
            agent_source=entry.get("agent_source") or "",
            existing_code=existing_code,
            start_line=_coerce_int(entry.get("start_line")),
            end_line=_coerce_int(entry.get("end_line")),
        )

    async def _resolve_lines(self, findings: List[Finding], diff: str) -> List[Finding]:
        """Two-step line resolution with hunk range validation."""
        from cicaddy.delegation.line_resolver import (
            resolve_findings,
            validate_findings_in_hunks,
        )

        resolved, unresolved = resolve_findings(findings, diff)

        if unresolved:
            ai_resolved = await self._ai_resolve_lines(unresolved, diff)
            resolved.extend(ai_resolved)

        return validate_findings_in_hunks(resolved, diff)

    @staticmethod
    def _filter_diff_for_files(diff: str, relevant_files: set[str]) -> str:
        """Filter a unified diff to only include hunks for relevant files."""
        filtered_lines: list[str] = []
        include_file = False
        pending_headers: list[str] = []
        for line in diff.splitlines():
            if line.startswith("diff --git"):
                include_file = False
                pending_headers = [line]
                continue
            if pending_headers:
                if line.startswith("+++ b/"):
                    path = line[6:]
                    include_file = any(
                        path == rf or path.endswith("/" + rf) or rf.endswith("/" + path)
                        for rf in relevant_files
                    )
                    if include_file:
                        filtered_lines.extend(pending_headers)
                        filtered_lines.append(line)
                    pending_headers = []
                else:
                    pending_headers.append(line)
                continue
            if include_file:
                filtered_lines.append(line)
        return "\n".join(filtered_lines)

    @staticmethod
    def _validate_line_mapping(
        mapping: dict,
        finding: Finding,
        diff_ranges: Optional[dict[str, list[tuple[int, int]]]],
    ) -> Optional[tuple[int, int]]:
        """Validate a single AI line mapping against diff ranges.

        Returns (start, end) if valid, or None to skip the mapping.
        """
        from cicaddy.delegation.line_resolver import is_line_in_diff_ranges

        start = mapping.get("start_line")
        if not isinstance(start, int) or start <= 0:
            return None

        if diff_ranges and not is_line_in_diff_ranges(start, finding.file, diff_ranges):
            logger.debug(
                f"Rejecting AI line {start} for {finding.file}: "
                f"not within any diff hunk"
            )
            return None

        end = mapping.get("end_line", start)
        resolved_end = int(end) if isinstance(end, int) else start
        if (
            diff_ranges
            and resolved_end != start
            and not is_line_in_diff_ranges(resolved_end, finding.file, diff_ranges)
        ):
            logger.debug(
                f"Clamping end_line {resolved_end} to {start} for "
                f"{finding.file}: end_line not in diff hunk"
            )
            resolved_end = start

        return (start, resolved_end)

    @staticmethod
    def _apply_line_mappings(
        mappings: list,
        unresolved: List[Finding],
        diff_ranges: Optional[dict[str, list[tuple[int, int]]]] = None,
    ) -> None:
        """Apply AI-resolved line mappings to unresolved findings in-place."""
        for mapping in mappings:
            if not isinstance(mapping, dict):
                continue
            idx = mapping.get("index")
            if not isinstance(idx, int) or idx < 0 or idx >= len(unresolved):
                continue
            result = SummarizationAgent._validate_line_mapping(
                mapping, unresolved[idx], diff_ranges
            )
            if result is not None:
                start, resolved_end = result
                unresolved[idx].line = start
                unresolved[idx].start_line = start
                unresolved[idx].end_line = resolved_end

    async def _ai_resolve_lines(
        self, unresolved: List[Finding], diff: str
    ) -> List[Finding]:
        """AI fallback for findings that deterministic resolution missed."""
        try:
            from cicaddy.ai_providers.base import ProviderMessage
            from cicaddy.delegation.line_resolver import (
                annotate_diff_with_line_numbers,
                get_diff_line_ranges,
            )

            relevant_files = {f.file for f in unresolved}
            filtered_diff = self._filter_diff_for_files(diff, relevant_files)
            annotated = annotate_diff_with_line_numbers(filtered_diff)
            diff_ranges = get_diff_line_ranges(filtered_diff)

            boundary_start, boundary_end = _make_boundary_pair()
            sanitized_diff = _sanitize_for_boundary(
                annotated, boundary_start, boundary_end
            )

            findings_for_prompt = []
            for i, f in enumerate(unresolved):
                entry = {
                    "index": i,
                    "file": f.file,
                    "message": _sanitize_for_boundary(
                        f.message[:200], boundary_start, boundary_end
                    ),
                }
                if f.existing_code:
                    entry["existing_code"] = _sanitize_for_boundary(
                        f.existing_code, boundary_start, boundary_end
                    )
                findings_for_prompt.append(entry)

            prompt = (
                "You are a code diff line number resolver. Given a unified diff "
                "with line numbers and a list of code findings, determine the "
                "exact line numbers where each finding occurs in the NEW version "
                "of the file.\n\n"
                "CRITICAL: You MUST only use line numbers that are visible in the "
                "annotated diff below. Each content line is prefixed with its line "
                "number (e.g., '  13 +    if result is None:'). Only use those "
                "numbered lines. If a finding refers to code that is NOT in any "
                "diff hunk, set start_line to null — do NOT guess a line number "
                "from the full file.\n\n"
                f"## Diff\n{boundary_start}\n{sanitized_diff}\n{boundary_end}\n\n"
                f"## Findings to resolve\n```json\n"
                f"{json.dumps(findings_for_prompt, indent=2)}\n```\n\n"
                "Respond with ONLY a JSON array (no markdown fences):\n"
                '[{"index": 0, "start_line": 42, "end_line": 44}, ...]\n'
                "Use null for start_line if the finding's code is not in the diff."
            )

            messages = [ProviderMessage(content=prompt, role="user")]
            response = await self.ai_provider.chat_completion(messages)

            content = extract_json(response.content)
            mappings = json.loads(content)
            if isinstance(mappings, list):
                self._apply_line_mappings(mappings, unresolved, diff_ranges)

            ai_resolved_count = sum(1 for f in unresolved if f.line is not None)
            logger.info(
                f"AI line mapping resolved {ai_resolved_count}/{len(unresolved)} "
                f"remaining findings"
            )

        except Exception as e:
            logger.warning(f"AI line mapping failed, findings remain unresolved: {e}")

        return unresolved

    @staticmethod
    def _build_individual_sections(
        agent_results: List[Dict[str, Any]],
    ) -> str:
        """Format full analyses into a collapsible <details> block."""
        sections = []
        for result in agent_results:
            status = result.get("status", "unknown")
            if status == "skipped":
                continue
            agent_name = result.get("agent_name", "Unknown")
            analysis = result.get("analysis", "")
            header = f"## {agent_name}"
            if status != "success":
                header += f" ({status})"
            sections.append(f"{header}\n\n{analysis}")

        if not sections:
            return ""

        body = "\n\n---\n\n".join(sections)
        return (
            "<details><summary>Individual Agent Analyses</summary>\n\n"
            f"{body}\n\n"
            "</details>"
        )

    @staticmethod
    def _build_footer(agent_results: List[Dict[str, Any]]) -> str:
        """Build the delegation summary footer line."""
        succeeded = sum(1 for r in agent_results if r.get("status") == "success")
        failed = sum(
            1 for r in agent_results if r.get("status") not in ("success", "skipped")
        )
        total_time = sum(r.get("execution_time", 0) for r in agent_results)
        agent_names = [
            r.get("agent_name", "unknown")
            for r in agent_results
            if r.get("status") != "skipped"
        ]

        footer = f"*Delegation summary: {succeeded} agent(s) succeeded"
        if failed:
            footer += f", {failed} failed"
        footer += (
            f" | Agents: {', '.join(agent_names)}"
            f" | Total sub-agent time: {total_time:.1f}s*"
        )
        return footer

    def _fallback_result(
        self, agent_results: List[Dict[str, Any]]
    ) -> SummarizationResult:
        """Build a SummarizationResult using deterministic concatenation."""
        sections = []
        for result in agent_results:
            status = result.get("status", "unknown")
            if status == "skipped":
                continue
            agent_name = result.get("agent_name", "Unknown")
            analysis = result.get("analysis", "")
            header = f"## {agent_name}"
            if status != "success":
                header += f" ({status})"
            sections.append(f"{header}\n\n{analysis}")

        summary = (
            "\n\n---\n\n".join(sections)
            if sections
            else "No sub-agent results available."
        )
        footer = self._build_footer(agent_results)

        return SummarizationResult(
            summary=summary,
            individual_sections="",
            findings=[],
            footer=footer,
        )
