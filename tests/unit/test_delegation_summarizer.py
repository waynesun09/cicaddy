"""Tests for cicaddy.delegation.summarizer module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from cicaddy.delegation.summarizer import (
    Finding,
    SummarizationAgent,
    SummarizationResult,
)


@pytest.fixture
def mock_ai_provider():
    provider = MagicMock()
    provider.chat_completion = AsyncMock()
    return provider


@pytest.fixture
def two_agent_results():
    return [
        {
            "agent_name": "general-reviewer",
            "status": "success",
            "analysis": "## Correctness\nFound issue in `src/foo.py` line 42: missing null check.\n\n## Summary\nLGTM with minor issues.",
            "categories": ["code_quality", "tests"],
            "rationale": "baseline review",
            "execution_time": 10.0,
            "tokens": 500,
        },
        {
            "agent_name": "security-reviewer",
            "status": "success",
            "analysis": "## Security\nNo SQL injection risks found.\nMinor: `src/bar.py` line 15 uses hardcoded secret.",
            "categories": ["security"],
            "rationale": "security changes detected",
            "execution_time": 8.0,
            "tokens": 400,
        },
    ]


@pytest.fixture
def ai_summary_response():
    return json.dumps(
        {
            "summary": "## Major\n- Missing null check in `src/foo.py:42`\n\n## Minor\n- Hardcoded secret in `src/bar.py:15`\n\n**Overall:** LGTM with minor issues.",
            "findings": [
                {
                    "file": "src/foo.py",
                    "existing_code": "value = get_data()",
                    "line": 42,
                    "severity": "major",
                    "message": "Missing null check",
                    "suggestion": "Add `if value is not None:` guard",
                    "agent_source": "general-reviewer",
                },
                {
                    "file": "src/bar.py",
                    "existing_code": 'SECRET = "hardcoded"',
                    "line": 15,
                    "severity": "minor",
                    "message": "Hardcoded secret",
                    "suggestion": "Use environment variable",
                    "agent_source": "security-reviewer",
                },
            ],
        }
    )


class TestSummarizationResult:
    """Tests for SummarizationResult dataclass."""

    def test_defaults(self):
        result = SummarizationResult(summary="test", individual_sections="")
        assert result.summary == "test"
        assert result.individual_sections == ""
        assert result.findings == []
        assert result.footer == ""
        assert result.ai_summarized is False

    def test_ai_summarized_flag(self):
        result = SummarizationResult(
            summary="test", individual_sections="", ai_summarized=True
        )
        assert result.ai_summarized is True


class TestFinding:
    """Tests for Finding dataclass."""

    def test_defaults(self):
        f = Finding(file="foo.py", line=10, severity="major", message="issue")
        assert f.suggestion is None
        assert f.agent_source == ""

    def test_full(self):
        f = Finding(
            file="foo.py",
            line=10,
            severity="critical",
            message="bug",
            suggestion="fix it",
            agent_source="general-reviewer",
        )
        assert f.suggestion == "fix it"
        assert f.agent_source == "general-reviewer"


class TestSummarizationAgent:
    """Tests for SummarizationAgent."""

    @pytest.mark.asyncio
    async def test_summarize_multiple_agents(
        self, mock_ai_provider, two_agent_results, ai_summary_response
    ):
        """2+ successful agents should trigger AI summarization."""
        mock_response = MagicMock()
        mock_response.content = ai_summary_response
        mock_ai_provider.chat_completion.return_value = mock_response

        agent = SummarizationAgent(mock_ai_provider)
        result = await agent.summarize(two_agent_results)

        assert isinstance(result, SummarizationResult)
        assert "Missing null check" in result.summary
        assert "Hardcoded secret" in result.summary
        assert len(result.findings) == 2
        assert result.findings[0].file == "src/foo.py"
        assert result.findings[0].line == 42
        assert result.findings[0].severity == "major"
        assert result.findings[0].existing_code == "value = get_data()"
        assert result.findings[1].agent_source == "security-reviewer"
        assert "<details>" in result.individual_sections
        assert "general-reviewer" in result.individual_sections
        assert "security-reviewer" in result.individual_sections
        assert "2 agent(s) succeeded" in result.footer
        assert result.ai_summarized is True
        mock_ai_provider.chat_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_single_agent_skips_ai_call(self, mock_ai_provider):
        """Single successful agent should skip AI call."""
        results = [
            {
                "agent_name": "general-reviewer",
                "status": "success",
                "analysis": "All good, no issues.",
                "categories": ["code_quality"],
                "execution_time": 5.0,
            }
        ]

        agent = SummarizationAgent(mock_ai_provider)
        result = await agent.summarize(results)

        assert result.summary == "All good, no issues."
        assert result.individual_sections == ""
        assert result.findings == []
        mock_ai_provider.chat_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_summarize_no_successful_results(self, mock_ai_provider):
        """No successful results should return empty result."""
        results = [
            {
                "agent_name": "broken",
                "status": "failed",
                "analysis": "Error",
                "execution_time": 0.1,
            }
        ]

        agent = SummarizationAgent(mock_ai_provider)
        result = await agent.summarize(results)

        assert result.summary == "No sub-agent results available."
        assert result.findings == []
        mock_ai_provider.chat_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_summarize_success_with_empty_findings(
        self, mock_ai_provider, two_agent_results
    ):
        """AI summarization succeeds but returns zero findings."""
        mock_response = MagicMock()
        mock_response.content = json.dumps(
            {"summary": "All looks good, no issues found.", "findings": []}
        )
        mock_ai_provider.chat_completion.return_value = mock_response

        agent = SummarizationAgent(mock_ai_provider)
        result = await agent.summarize(two_agent_results)

        assert result.ai_summarized is True
        assert result.findings == []
        assert "All looks good" in result.summary

    @pytest.mark.asyncio
    async def test_summarize_ai_failure_falls_back(
        self, mock_ai_provider, two_agent_results
    ):
        """AI failure should fall back to deterministic concatenation."""
        mock_ai_provider.chat_completion.side_effect = RuntimeError("API error")

        agent = SummarizationAgent(mock_ai_provider)
        result = await agent.summarize(two_agent_results)

        # Fallback: concatenated sections, no findings
        assert "## general-reviewer" in result.summary
        assert "## security-reviewer" in result.summary
        assert result.findings == []
        assert result.individual_sections == ""
        assert result.ai_summarized is False

    @pytest.mark.asyncio
    async def test_summarize_plain_text_response_used_as_summary(
        self, mock_ai_provider, two_agent_results
    ):
        """Plain text AI response should be used as summary directly."""
        mock_response = MagicMock()
        mock_response.content = "This is not JSON at all"
        mock_ai_provider.chat_completion.return_value = mock_response

        agent = SummarizationAgent(mock_ai_provider)
        result = await agent.summarize(two_agent_results)

        assert result.summary == "This is not JSON at all"
        assert result.findings == []
        assert result.ai_summarized is True

    @pytest.mark.asyncio
    async def test_custom_instructions_in_prompt(
        self, mock_ai_provider, two_agent_results, ai_summary_response
    ):
        """Custom instructions should appear in the prompt."""
        mock_response = MagicMock()
        mock_response.content = ai_summary_response
        mock_ai_provider.chat_completion.return_value = mock_response

        agent = SummarizationAgent(mock_ai_provider)
        await agent.summarize(
            two_agent_results, custom_instructions="Focus on security"
        )

        call_args = mock_ai_provider.chat_completion.call_args
        prompt = call_args[0][0][0].content
        assert "Focus on security" in prompt
        assert "Additional Instructions" in prompt

    @pytest.mark.asyncio
    async def test_boundary_markers_in_prompt(
        self, mock_ai_provider, two_agent_results, ai_summary_response
    ):
        """Prompt should use nonce-based boundary markers."""
        mock_response = MagicMock()
        mock_response.content = ai_summary_response
        mock_ai_provider.chat_completion.return_value = mock_response

        agent = SummarizationAgent(mock_ai_provider)
        await agent.summarize(two_agent_results)

        call_args = mock_ai_provider.chat_completion.call_args
        prompt = call_args[0][0][0].content
        assert "<<<BEGIN_CONTEXT_DATA_" in prompt
        assert "<<<END_CONTEXT_DATA_" in prompt

    @pytest.mark.asyncio
    async def test_findings_extraction(self, mock_ai_provider):
        """Findings should be correctly extracted from AI JSON."""
        results = [
            {
                "agent_name": "agent-a",
                "status": "success",
                "analysis": "A output",
                "categories": ["code"],
                "execution_time": 1,
            },
            {
                "agent_name": "agent-b",
                "status": "success",
                "analysis": "B output",
                "categories": ["arch"],
                "execution_time": 2,
            },
        ]

        response_data = {
            "summary": "Summary text",
            "findings": [
                {
                    "file": "main.py",
                    "line": 10,
                    "severity": "critical",
                    "message": "Buffer overflow",
                    "suggestion": None,
                    "agent_source": "agent-a",
                },
                {
                    "file": "utils.py",
                    "line": None,
                    "severity": "nit",
                    "message": "Naming convention",
                    "suggestion": "Rename to snake_case",
                    "agent_source": "agent-b",
                },
            ],
        }

        mock_response = MagicMock()
        mock_response.content = json.dumps(response_data)
        mock_ai_provider.chat_completion.return_value = mock_response

        agent = SummarizationAgent(mock_ai_provider)
        result = await agent.summarize(results)

        assert len(result.findings) == 2
        assert result.findings[0].severity == "critical"
        assert result.findings[0].suggestion is None
        assert result.findings[1].line is None
        assert result.findings[1].suggestion == "Rename to snake_case"

    @pytest.mark.asyncio
    async def test_findings_with_null_line(self, mock_ai_provider):
        """File-level findings (line=None) should be handled."""
        results = [
            {
                "agent_name": "a",
                "status": "success",
                "analysis": "x",
                "categories": [],
                "execution_time": 1,
            },
            {
                "agent_name": "b",
                "status": "success",
                "analysis": "y",
                "categories": [],
                "execution_time": 1,
            },
        ]

        response_data = {
            "summary": "OK",
            "findings": [
                {
                    "file": "setup.py",
                    "line": None,
                    "severity": "minor",
                    "message": "Missing license header",
                    "agent_source": "a",
                },
            ],
        }

        mock_response = MagicMock()
        mock_response.content = json.dumps(response_data)
        mock_ai_provider.chat_completion.return_value = mock_response

        agent = SummarizationAgent(mock_ai_provider)
        result = await agent.summarize(results)

        assert len(result.findings) == 1
        assert result.findings[0].line is None
        assert result.findings[0].file == "setup.py"


class TestParseResponseEdgeCases:
    """Tests for _parse_response edge cases (isinstance guards, whitespace)."""

    @pytest.mark.asyncio
    async def test_non_dict_json_used_as_summary(self, mock_ai_provider):
        """JSON array response should be used as summary text."""
        agent = SummarizationAgent(mock_ai_provider)
        summary, findings = agent._parse_response('["not", "a", "dict"]')
        assert summary == "['not', 'a', 'dict']"
        assert findings == []

    @pytest.mark.asyncio
    async def test_json_string_used_as_summary(self, mock_ai_provider):
        """JSON-encoded string response should be used as summary text."""
        agent = SummarizationAgent(mock_ai_provider)
        summary, findings = agent._parse_response(
            '"Here is a summary of the code review findings..."'
        )
        assert summary == "Here is a summary of the code review findings..."
        assert findings == []

    @pytest.mark.asyncio
    async def test_whitespace_only_summary_raises(self, mock_ai_provider):
        """Whitespace-only summary should raise ValueError."""
        agent = SummarizationAgent(mock_ai_provider)
        with pytest.raises(ValueError, match="missing 'summary' field"):
            agent._parse_response('{"summary": "   ", "findings": []}')

    @pytest.mark.asyncio
    async def test_non_dict_findings_skipped(self, mock_ai_provider):
        """Non-dict entries in findings array should be silently skipped."""
        response = json.dumps(
            {
                "summary": "Valid summary",
                "findings": [
                    "not a dict",
                    {"file": "a.py", "severity": "major", "message": "real finding"},
                    42,
                ],
            }
        )
        agent = SummarizationAgent(mock_ai_provider)
        summary, findings = agent._parse_response(response)
        assert summary == "Valid summary"
        assert len(findings) == 1
        assert findings[0].file == "a.py"


class TestValidateFinding:
    """Tests for SummarizationAgent._validate_finding."""

    def test_valid_finding(self):
        entry = {
            "file": "foo.py",
            "line": 10,
            "severity": "major",
            "message": "issue",
        }
        f = SummarizationAgent._validate_finding(entry)
        assert f is not None
        assert f.severity == "major"

    def test_missing_file_returns_none(self):
        entry = {"file": "", "severity": "major", "message": "issue"}
        assert SummarizationAgent._validate_finding(entry) is None

    def test_missing_message_returns_none(self):
        entry = {"file": "foo.py", "severity": "major", "message": ""}
        assert SummarizationAgent._validate_finding(entry) is None

    def test_invalid_severity_defaults_to_minor(self):
        entry = {"file": "foo.py", "severity": "URGENT", "message": "issue"}
        f = SummarizationAgent._validate_finding(entry)
        assert f is not None
        assert f.severity == "minor"

    def test_missing_severity_defaults_to_minor(self):
        entry = {"file": "foo.py", "message": "issue"}
        f = SummarizationAgent._validate_finding(entry)
        assert f is not None
        assert f.severity == "minor"

    def test_string_line_coerced_to_int(self):
        entry = {
            "file": "foo.py",
            "line": "42",
            "severity": "major",
            "message": "issue",
        }
        f = SummarizationAgent._validate_finding(entry)
        assert f is not None
        assert f.line == 42

    def test_non_numeric_line_becomes_none(self):
        entry = {
            "file": "foo.py",
            "line": "not-a-number",
            "severity": "major",
            "message": "issue",
        }
        f = SummarizationAgent._validate_finding(entry)
        assert f is not None
        assert f.line is None


class TestBuildIndividualSections:
    """Tests for SummarizationAgent._build_individual_sections."""

    def test_multiple_agents(self):
        results = [
            {"agent_name": "a", "status": "success", "analysis": "A output"},
            {"agent_name": "b", "status": "success", "analysis": "B output"},
        ]
        output = SummarizationAgent._build_individual_sections(results)
        assert "<details>" in output
        assert "## a" in output
        assert "## b" in output
        assert "A output" in output
        assert "B output" in output

    def test_skipped_excluded(self):
        results = [
            {"agent_name": "skip", "status": "skipped", "analysis": ""},
        ]
        output = SummarizationAgent._build_individual_sections(results)
        assert output == ""

    def test_failed_shown_with_status(self):
        results = [
            {"agent_name": "fail", "status": "error", "analysis": "err"},
        ]
        output = SummarizationAgent._build_individual_sections(results)
        assert "## fail (error)" in output


class TestBuildFooter:
    """Tests for SummarizationAgent._build_footer."""

    def test_all_succeeded(self):
        results = [
            {"agent_name": "a", "status": "success", "execution_time": 5.0},
            {"agent_name": "b", "status": "success", "execution_time": 3.0},
        ]
        footer = SummarizationAgent._build_footer(results)
        assert "2 agent(s) succeeded" in footer
        assert "Agents: a, b" in footer
        assert "8.0s" in footer

    def test_with_failures(self):
        results = [
            {"agent_name": "a", "status": "success", "execution_time": 1.0},
            {"agent_name": "b", "status": "failed", "execution_time": 0.5},
        ]
        footer = SummarizationAgent._build_footer(results)
        assert "1 agent(s) succeeded" in footer
        assert "1 failed" in footer


class TestTwoStepLineResolution:
    """Tests for the two-step line resolution flow in SummarizationAgent."""

    @pytest.mark.asyncio
    async def test_deterministic_resolution_with_diff(self, mock_ai_provider):
        """Findings with existing_code should be resolved via diff search."""
        diff = """\
diff --git a/src/foo.py b/src/foo.py
--- a/src/foo.py
+++ b/src/foo.py
@@ -10,4 +10,5 @@ def process():
     data = fetch()
     if data:
         result = transform(data)
+        validate(result)
         return result
"""
        response_data = {
            "summary": "Found issue",
            "findings": [
                {
                    "file": "src/foo.py",
                    "existing_code": "validate(result)",
                    "severity": "major",
                    "message": "Missing error handling",
                    "agent_source": "agent-a",
                },
            ],
        }
        mock_response = MagicMock()
        mock_response.content = json.dumps(response_data)
        mock_ai_provider.chat_completion.return_value = mock_response

        results = [
            {
                "agent_name": "a",
                "status": "success",
                "analysis": "A",
                "categories": ["code"],
                "execution_time": 1,
            },
            {
                "agent_name": "b",
                "status": "success",
                "analysis": "B",
                "categories": ["arch"],
                "execution_time": 2,
            },
        ]
        agent = SummarizationAgent(mock_ai_provider)
        result = await agent.summarize(results, diff=diff)

        assert len(result.findings) == 1
        assert result.findings[0].line == 13
        assert result.findings[0].start_line == 13
        assert result.findings[0].existing_code == "validate(result)"
        # Only the summarization call, no AI line mapping needed
        assert mock_ai_provider.chat_completion.call_count == 1

    @pytest.mark.asyncio
    async def test_ai_fallback_for_unresolved(self, mock_ai_provider):
        """Unresolved findings should trigger AI line-mapping call."""
        diff = """\
diff --git a/src/foo.py b/src/foo.py
--- a/src/foo.py
+++ b/src/foo.py
@@ -10,3 +10,4 @@ def process():
     data = fetch()
     if data:
+        validate(data)
         return data
"""
        # First call: summarization response with snippet that won't match
        summary_response = json.dumps(
            {
                "summary": "Found issue",
                "findings": [
                    {
                        "file": "src/foo.py",
                        "existing_code": "this snippet does not exist in diff",
                        "severity": "major",
                        "message": "Some issue",
                        "agent_source": "agent-a",
                    },
                ],
            }
        )
        # Second call: AI line mapping response
        mapping_response = '[{"index": 0, "start_line": 12, "end_line": 12}]'

        mock_ai_provider.chat_completion.side_effect = [
            MagicMock(content=summary_response),
            MagicMock(content=mapping_response),
        ]

        results = [
            {
                "agent_name": "a",
                "status": "success",
                "analysis": "A",
                "categories": ["code"],
                "execution_time": 1,
            },
            {
                "agent_name": "b",
                "status": "success",
                "analysis": "B",
                "categories": ["arch"],
                "execution_time": 2,
            },
        ]
        agent = SummarizationAgent(mock_ai_provider)
        result = await agent.summarize(results, diff=diff)

        assert len(result.findings) == 1
        assert result.findings[0].line == 12
        assert result.findings[0].start_line == 12
        # Two AI calls: summarization + line mapping
        assert mock_ai_provider.chat_completion.call_count == 2

    @pytest.mark.asyncio
    async def test_ai_fallback_failure_preserves_findings(self, mock_ai_provider):
        """If AI line mapping fails, findings remain with line=None."""
        diff = """\
diff --git a/src/foo.py b/src/foo.py
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,4 @@
 import os
+import sys
 x = 1
"""
        summary_response = json.dumps(
            {
                "summary": "Found issue",
                "findings": [
                    {
                        "file": "src/foo.py",
                        "existing_code": "no match here",
                        "severity": "minor",
                        "message": "Some issue",
                        "agent_source": "a",
                    },
                ],
            }
        )
        mock_ai_provider.chat_completion.side_effect = [
            MagicMock(content=summary_response),
            RuntimeError("API error"),
        ]

        results = [
            {
                "agent_name": "a",
                "status": "success",
                "analysis": "A",
                "categories": [],
                "execution_time": 1,
            },
            {
                "agent_name": "b",
                "status": "success",
                "analysis": "B",
                "categories": [],
                "execution_time": 1,
            },
        ]
        agent = SummarizationAgent(mock_ai_provider)
        result = await agent.summarize(results, diff=diff)

        assert len(result.findings) == 1
        assert result.findings[0].line is None
        assert result.ai_summarized is True

    @pytest.mark.asyncio
    async def test_no_diff_skips_resolution(self, mock_ai_provider):
        """Without a diff, findings are returned without line resolution."""
        response_data = {
            "summary": "Summary",
            "findings": [
                {
                    "file": "src/foo.py",
                    "existing_code": "some code",
                    "severity": "major",
                    "message": "Issue",
                    "agent_source": "a",
                },
            ],
        }
        mock_response = MagicMock()
        mock_response.content = json.dumps(response_data)
        mock_ai_provider.chat_completion.return_value = mock_response

        results = [
            {
                "agent_name": "a",
                "status": "success",
                "analysis": "A",
                "categories": [],
                "execution_time": 1,
            },
            {
                "agent_name": "b",
                "status": "success",
                "analysis": "B",
                "categories": [],
                "execution_time": 1,
            },
        ]
        agent = SummarizationAgent(mock_ai_provider)
        result = await agent.summarize(results, diff="")

        assert len(result.findings) == 1
        assert result.findings[0].line is None
        # Only one AI call (no line mapping without diff)
        assert mock_ai_provider.chat_completion.call_count == 1
