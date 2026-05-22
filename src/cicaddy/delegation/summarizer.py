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
- Use markdown formatting with ## headings for severity groups
- Omit empty severity groups (if no Critical findings, skip that section)
- End with a brief overall assessment (1-2 sentences)"""

_RESPONSE_FORMAT = """\
## Response Format

Respond with ONLY a JSON object (no markdown fences, no preamble).
The JSON must match this schema exactly:

```json
{schema_json}
```

Field rules:
- "summary" — required, non-empty markdown string with the consolidated review
- "findings" — required array of structured findings
- Each finding requires: "file" (non-empty string), "severity" (critical|major|minor|nit), "message" (non-empty string)
- Optional finding fields: "existing_code" (string or null), "line" (integer or null), "suggestion" (string or null), "agent_source" (string)
- If no findings, use an empty array: []"""

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


@dataclass
class SummarizationResult:
    """Result of AI-powered review summarization."""

    summary: str
    individual_sections: str
    findings: List[Finding] = field(default_factory=list)
    footer: str = ""
    ai_summarized: bool = False


class SummarizationAgent:
    """AI-powered summarization of multi-agent review results.

    Uses a multi-turn conversation with JSON schema validation to
    ensure the AI produces well-structured output.  On each turn the
    response is validated against the bundled JSON schema; validation
    errors are sent back as a correction prompt up to ``max_turns``.
    """

    def __init__(self, ai_provider: "BaseProvider"):
        self.ai_provider = ai_provider

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

        Returns:
            Tuple of (summary_text, findings_list).
        """
        from cicaddy.ai_providers.base import ProviderMessage

        content = response_content
        for turn in range(max_turns):
            text = content.strip()
            if not text:
                if turn == 0:
                    raise ValueError(_ERR_EMPTY)
                errors = ["Response is empty."]
            else:
                result = self._try_parse_and_validate(text)
                if result is not None:
                    summary, findings = result
                    if turn > 0:
                        logger.info(
                            "Schema validation passed after %d correction turn(s)",
                            turn,
                        )
                    return summary, findings
                errors = self._collect_validation_errors(text)

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
        # Fallback: use raw text as summary, no structured findings
        fallback_text = response_content.strip()
        # Try to extract summary from JSON if possible
        try:
            data = json.loads(extract_json(fallback_text))
            if isinstance(data, dict) and isinstance(data.get("summary"), str):
                return data["summary"], []
        except (json.JSONDecodeError, ValueError):
            pass
        return fallback_text, []

    def _try_parse_and_validate(self, text: str) -> Optional[tuple[str, List[Finding]]]:
        """Try to parse JSON and validate against schema.

        Returns (summary, findings) on success, None on failure.
        """
        try:
            raw = extract_json(text)
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None

        if not isinstance(data, dict):
            return None

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
            for i, entry in enumerate(data["findings"]):
                if not isinstance(entry, dict):
                    errors.append(
                        f"findings[{i}]: must be an object, got {type(entry).__name__}"
                    )
                    continue
                if (
                    "file" not in entry
                    or not isinstance(entry.get("file"), str)
                    or not entry["file"].strip()
                ):
                    errors.append(
                        f"findings[{i}]: missing or empty required field 'file'"
                    )
                if (
                    "message" not in entry
                    or not isinstance(entry.get("message"), str)
                    or not entry["message"].strip()
                ):
                    errors.append(
                        f"findings[{i}]: missing or empty required field 'message'"
                    )
                sev = entry.get("severity", "")
                if not isinstance(sev, str) or sev.lower() not in _VALID_SEVERITIES:
                    errors.append(
                        f"findings[{i}]: 'severity' must be one of "
                        f"{sorted(_VALID_SEVERITIES)}, got {sev!r}"
                    )

                allowed_keys = {
                    "file",
                    "existing_code",
                    "line",
                    "severity",
                    "message",
                    "suggestion",
                    "agent_source",
                }
                extra = set(entry.keys()) - allowed_keys
                if extra:
                    errors.append(f"findings[{i}]: unexpected fields {extra}")

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
            agent_source=entry.get("agent_source", ""),
            existing_code=existing_code,
        )

    async def _resolve_lines(self, findings: List[Finding], diff: str) -> List[Finding]:
        """Two-step line resolution: deterministic first, AI fallback second."""
        from cicaddy.delegation.line_resolver import resolve_findings

        resolved, unresolved = resolve_findings(findings, diff)

        if unresolved:
            ai_resolved = await self._ai_resolve_lines(unresolved, diff)
            resolved.extend(ai_resolved)

        return resolved

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
                        path == rf or path.endswith(rf) or rf.endswith(path)
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
    def _apply_line_mappings(mappings: list, unresolved: List[Finding]) -> None:
        """Apply AI-resolved line mappings to unresolved findings in-place."""
        for mapping in mappings:
            if not isinstance(mapping, dict):
                continue
            idx = mapping.get("index")
            if not isinstance(idx, int) or idx < 0 or idx >= len(unresolved):
                continue
            start = mapping.get("start_line")
            end = mapping.get("end_line", start)
            if isinstance(start, int) and start > 0:
                unresolved[idx].line = start
                unresolved[idx].start_line = start
                unresolved[idx].end_line = int(end) if isinstance(end, int) else start

    async def _ai_resolve_lines(
        self, unresolved: List[Finding], diff: str
    ) -> List[Finding]:
        """AI fallback for findings that deterministic resolution missed."""
        try:
            from cicaddy.ai_providers.base import ProviderMessage
            from cicaddy.delegation.line_resolver import annotate_diff_with_line_numbers

            relevant_files = {f.file for f in unresolved}
            filtered_diff = self._filter_diff_for_files(diff, relevant_files)
            annotated = annotate_diff_with_line_numbers(filtered_diff)

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
                f"## Diff\n{boundary_start}\n{sanitized_diff}\n{boundary_end}\n\n"
                f"## Findings to resolve\n```json\n"
                f"{json.dumps(findings_for_prompt, indent=2)}\n```\n\n"
                "Respond with ONLY a JSON array (no markdown fences):\n"
                '[{"index": 0, "start_line": 42, "end_line": 44}, ...]'
            )

            messages = [ProviderMessage(content=prompt, role="user")]
            response = await self.ai_provider.chat_completion(messages)

            content = extract_json(response.content)
            mappings = json.loads(content)
            if isinstance(mappings, list):
                self._apply_line_mappings(mappings, unresolved)

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
