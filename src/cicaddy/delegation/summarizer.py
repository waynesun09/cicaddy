"""AI-powered summarization of multi-agent delegation results.

Uses the parent agent's AI provider (single lightweight call) to
condense multiple sub-agent analyses into a concise consolidated
review with structured findings for inline comment support.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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

_FINDINGS_RULES = """\
## Findings Extraction Rules
- Extract file path from agent analyses when referenced
- **Include the exact code snippet** the finding refers to in `existing_code`.
  Quote 1-3 lines of the relevant code from the diff. This is used for
  precise inline comment placement — accurate snippets are critical
- If an agent mentions a line number, include it in the `line` field as
  a best-effort hint. Otherwise set `line` to null (line resolution is
  handled separately)
- Map severity from agent output (Critical/Major/Minor/Nit)
- Include concrete suggestion/fix when the agent provided one
- Track which agent identified each finding via agent_source
- Do NOT invent findings — only extract what agents explicitly reported"""

_RESPONSE_FORMAT = """\
## Response Format

Respond with ONLY a JSON object in this exact format \
(no markdown code fences, no explanation):
{
  "summary": "Concise consolidated review in markdown...",
  "findings": [
    {
      "file": "path/to/file.py",
      "existing_code": "the relevant code snippet from the diff (1-3 lines)",
      "line": 42,
      "severity": "major",
      "message": "Description of the finding",
      "suggestion": "Concrete fix or null if none",
      "agent_source": "agent-name"
    }
  ]
}

"existing_code" is the exact code snippet the finding targets — quote it from \
the diff. "line" is a best-effort integer if the agent cited one, otherwise null."""


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

    Uses the parent agent's AI provider (single lightweight call, no new
    provider needed) to condense multiple sub-agent analyses into a
    concise summary with structured findings.
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

        Uses a two-step line resolution approach:
        1. AI generates findings with ``existing_code`` snippets
        2. Deterministic search maps snippets to line numbers, with
           AI fallback for unresolved findings

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

            summary, findings = self._parse_response(response.content)

            # Step 2: Resolve line numbers via deterministic diff search
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

        # Build analyses section
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

        return (
            f"{_SUMMARIZATION_SYSTEM_PROMPT}\n\n"
            f"{_SUMMARY_RULES}\n\n"
            f"{_FINDINGS_RULES}\n"
            f"{custom_section}\n"
            f"## Agent Analyses to Summarize\n\n"
            f"{boundary_start}\n{analyses_section}\n{boundary_end}\n\n"
            f"{_RESPONSE_FORMAT}"
        )

    def _parse_response(self, response_content: str) -> tuple[str, List[Finding]]:
        """Parse AI response into summary text and findings list."""
        content = extract_json(response_content)

        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("AI response is not a JSON object")

        summary = data.get("summary", "")
        if not summary or not summary.strip():
            raise ValueError("AI response missing 'summary' field")

        findings = []
        raw_findings = data.get("findings", [])
        if not isinstance(raw_findings, list):
            raw_findings = []
        for entry in raw_findings:
            if not isinstance(entry, dict):
                continue
            finding = self._validate_finding(entry)
            if finding:
                findings.append(finding)

        return summary, findings

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

    async def _ai_resolve_lines(
        self, unresolved: List[Finding], diff: str
    ) -> List[Finding]:
        """AI fallback for findings that deterministic resolution missed."""
        try:
            from cicaddy.ai_providers.base import ProviderMessage
            from cicaddy.delegation.line_resolver import annotate_diff_with_line_numbers

            # Filter diff to relevant files only
            relevant_files = {f.file for f in unresolved}
            filtered_lines: list[str] = []
            include_file = False
            for line in diff.splitlines():
                if line.startswith("diff --git"):
                    include_file = False
                if line.startswith("+++ b/"):
                    path = line[6:]
                    include_file = any(
                        path == rf or path.endswith(rf) or rf.endswith(path)
                        for rf in relevant_files
                    )
                if include_file:
                    filtered_lines.append(line)

            filtered_diff = "\n".join(filtered_lines)
            annotated = annotate_diff_with_line_numbers(filtered_diff)

            # Build compact findings list for the prompt
            findings_for_prompt = []
            for i, f in enumerate(unresolved):
                entry = {
                    "index": i,
                    "file": f.file,
                    "message": f.message[:200],
                }
                if f.existing_code:
                    entry["existing_code"] = f.existing_code
                findings_for_prompt.append(entry)

            prompt = (
                "You are a code diff line number resolver. Given a unified diff "
                "with line numbers and a list of code findings, determine the "
                "exact line numbers where each finding occurs in the NEW version "
                "of the file.\n\n"
                f"## Diff\n```\n{annotated}\n```\n\n"
                f"## Findings to resolve\n```json\n"
                f"{json.dumps(findings_for_prompt, indent=2)}\n```\n\n"
                "Respond with ONLY a JSON array (no markdown fences):\n"
                '[{"index": 0, "start_line": 42, "end_line": 44}, ...]'
            )

            messages = [ProviderMessage(content=prompt, role="user")]
            response = await self.ai_provider.chat_completion(messages)

            content = extract_json(response.content)
            mappings = json.loads(content)
            if not isinstance(mappings, list):
                return unresolved

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
                    unresolved[idx].end_line = (
                        int(end) if isinstance(end, int) else start
                    )

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
