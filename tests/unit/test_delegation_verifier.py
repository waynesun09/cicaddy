"""Tests for cicaddy.delegation.verifier module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cicaddy.delegation.summarizer import Finding
from cicaddy.delegation.verifier import (
    _VALID_STATUSES,
    FindingVerifier,
    ToolContext,
    VerificationResult,
)


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.ai_provider = "gemini"
    settings.ai_model = "gemini-2.0-flash"
    settings.context_safety_factor = 0.85
    return settings


@pytest.fixture
def sample_findings():
    return [
        Finding(
            file="src/foo.py",
            line=42,
            severity="major",
            message="Missing null check",
            suggestion="Add `if value is not None:` guard",
            agent_source="general-reviewer",
            existing_code="value = get_data()",
        ),
        Finding(
            file="src/bar.py",
            line=15,
            severity="minor",
            message="Hardcoded secret",
            suggestion="Use environment variable",
            agent_source="security-reviewer",
            existing_code='SECRET = "hardcoded"',
        ),
    ]


@pytest.fixture
def parent_tools():
    return [
        {"name": "read_file", "description": "Read a file"},
        {"name": "glob_files", "description": "Glob for files"},
        {"name": "write_file", "description": "Write a file"},
    ]


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_defaults(self):
        r = VerificationResult(status="valid", reasoning="looks correct")
        assert r.status == "valid"
        assert r.reasoning == "looks correct"
        assert r.adjusted_severity is None
        assert r.confidence == pytest.approx(0.0)

    def test_full(self):
        r = VerificationResult(
            status="invalid",
            reasoning="already handled upstream",
            adjusted_severity="nit",
            confidence=0.95,
        )
        assert r.status == "invalid"
        assert r.adjusted_severity == "nit"
        assert r.confidence == pytest.approx(0.95)


class TestParseVerificationResponse:
    """Tests for FindingVerifier._parse_verification_response."""

    def test_valid_json_dict(self):
        content = json.dumps(
            {
                "status": "valid",
                "reasoning": "Code is indeed missing a null check",
                "adjusted_severity": None,
                "confidence": 0.9,
            }
        )
        result = FindingVerifier._parse_verification_response(content)
        assert result.status == "valid"
        assert "null check" in result.reasoning
        assert result.confidence == pytest.approx(0.9)

    def test_invalid_finding(self):
        content = json.dumps(
            {
                "status": "invalid",
                "reasoning": "Already handled by upstream validation",
                "adjusted_severity": None,
                "confidence": 0.85,
            }
        )
        result = FindingVerifier._parse_verification_response(content)
        assert result.status == "invalid"

    def test_uncertain_finding(self):
        content = json.dumps(
            {
                "status": "uncertain",
                "reasoning": "Cannot determine without runtime context",
                "confidence": 0.3,
            }
        )
        result = FindingVerifier._parse_verification_response(content)
        assert result.status == "uncertain"
        assert result.confidence == pytest.approx(0.3)

    def test_json_wrapped_in_list(self):
        content = json.dumps(
            [
                {
                    "status": "valid",
                    "reasoning": "Confirmed issue",
                    "confidence": 0.8,
                }
            ]
        )
        result = FindingVerifier._parse_verification_response(content)
        assert result.status == "valid"
        assert result.reasoning == "Confirmed issue"

    def test_json_list_multiple_items_uses_first(self):
        content = json.dumps(
            [
                {"status": "valid", "reasoning": "First", "confidence": 0.8},
                {"status": "invalid", "reasoning": "Second", "confidence": 0.5},
            ]
        )
        result = FindingVerifier._parse_verification_response(content)
        assert result.status == "valid"
        assert result.reasoning == "First"

    def test_empty_list_returns_uncertain(self):
        content = json.dumps([])
        result = FindingVerifier._parse_verification_response(content)
        assert result.status == "uncertain"
        assert "Empty list" in result.reasoning

    def test_json_string_returns_uncertain(self):
        content = json.dumps("some text response")
        result = FindingVerifier._parse_verification_response(content)
        assert result.status == "uncertain"
        assert "some text" in result.reasoning

    def test_null_response_returns_uncertain(self):
        content = json.dumps(None)
        result = FindingVerifier._parse_verification_response(content)
        assert result.status == "uncertain"
        assert "Null" in result.reasoning

    def test_empty_content_returns_uncertain(self):
        result = FindingVerifier._parse_verification_response("")
        assert result.status == "uncertain"
        assert "Empty" in result.reasoning

    def test_whitespace_only_returns_uncertain(self):
        result = FindingVerifier._parse_verification_response("   \n  ")
        assert result.status == "uncertain"
        assert "Empty" in result.reasoning

    def test_plain_text_not_json(self):
        result = FindingVerifier._parse_verification_response(
            "This is just a text response, not JSON."
        )
        assert result.status == "uncertain"

    def test_markdown_fenced_json(self):
        content = (
            '```json\n{"status": "valid", "reasoning": "ok", "confidence": 0.7}\n```'
        )
        result = FindingVerifier._parse_verification_response(content)
        assert result.status == "valid"
        assert result.reasoning == "ok"

    def test_severity_adjustment(self):
        content = json.dumps(
            {
                "status": "valid",
                "reasoning": "Issue exists but is minor, not major",
                "adjusted_severity": "minor",
                "confidence": 0.75,
            }
        )
        result = FindingVerifier._parse_verification_response(content)
        assert result.adjusted_severity == "minor"

    def test_invalid_severity_ignored(self):
        content = json.dumps(
            {
                "status": "valid",
                "reasoning": "Issue exists",
                "adjusted_severity": "extreme",
                "confidence": 0.8,
            }
        )
        result = FindingVerifier._parse_verification_response(content)
        assert result.adjusted_severity is None

    def test_confidence_clamped(self):
        content = json.dumps(
            {
                "status": "valid",
                "reasoning": "sure",
                "confidence": 1.5,
            }
        )
        result = FindingVerifier._parse_verification_response(content)
        assert result.confidence == pytest.approx(1.0)

    def test_confidence_floor(self):
        content = json.dumps(
            {
                "status": "valid",
                "reasoning": "sure",
                "confidence": -0.5,
            }
        )
        result = FindingVerifier._parse_verification_response(content)
        assert result.confidence == pytest.approx(0.0)

    def test_invalid_confidence_type(self):
        content = json.dumps(
            {
                "status": "valid",
                "reasoning": "sure",
                "confidence": "high",
            }
        )
        result = FindingVerifier._parse_verification_response(content)
        assert result.confidence == pytest.approx(0.0)


class TestValidateVerification:
    """Tests for FindingVerifier._validate_verification."""

    def test_valid_entry(self):
        entry = {
            "status": "valid",
            "reasoning": "Code is correct",
            "adjusted_severity": None,
            "confidence": 0.9,
        }
        result = FindingVerifier._validate_verification(entry)
        assert result.status == "valid"
        assert result.reasoning == "Code is correct"
        assert result.confidence == pytest.approx(0.9)

    def test_unknown_status_becomes_uncertain(self):
        entry = {"status": "maybe", "reasoning": "Not sure"}
        result = FindingVerifier._validate_verification(entry)
        assert result.status == "uncertain"

    def test_missing_status_becomes_uncertain(self):
        entry = {"reasoning": "No status field"}
        result = FindingVerifier._validate_verification(entry)
        assert result.status == "uncertain"

    def test_empty_reasoning_gets_default(self):
        entry = {"status": "valid", "reasoning": ""}
        result = FindingVerifier._validate_verification(entry)
        assert result.reasoning == "No reasoning provided"

    def test_missing_reasoning_gets_default(self):
        entry = {"status": "valid"}
        result = FindingVerifier._validate_verification(entry)
        assert result.reasoning == "No reasoning provided"

    def test_non_string_reasoning_gets_default(self):
        entry = {"status": "valid", "reasoning": 42}
        result = FindingVerifier._validate_verification(entry)
        assert result.reasoning == "No reasoning provided"

    def test_adjusted_severity_valid(self):
        entry = {
            "status": "valid",
            "reasoning": "exists",
            "adjusted_severity": "critical",
        }
        result = FindingVerifier._validate_verification(entry)
        assert result.adjusted_severity == "critical"

    def test_adjusted_severity_invalid_ignored(self):
        entry = {
            "status": "valid",
            "reasoning": "exists",
            "adjusted_severity": "blocker",
        }
        result = FindingVerifier._validate_verification(entry)
        assert result.adjusted_severity is None

    def test_adjusted_severity_non_string_ignored(self):
        entry = {
            "status": "valid",
            "reasoning": "exists",
            "adjusted_severity": 123,
        }
        result = FindingVerifier._validate_verification(entry)
        assert result.adjusted_severity is None


class TestGetVerificationTools:
    """Tests for FindingVerifier._get_verification_tools."""

    def test_filters_to_allowlisted_tools(self, mock_settings):
        verifier = FindingVerifier(
            settings=mock_settings,
            tool_context=ToolContext(
                parent_tools=[
                    {"name": "read_file", "description": "Read a file"},
                    {"name": "read_file_lines", "description": "Read file lines"},
                    {"name": "glob_files", "description": "Glob for files"},
                    {"name": "write_file", "description": "Write a file"},
                    {"name": "execute_command", "description": "Run command"},
                ],
            ),
        )
        tools = verifier._get_verification_tools()
        assert len(tools) == 2
        names = {t["name"] for t in tools}
        assert names == {"read_file", "read_file_lines"}

    def test_rejects_tools_with_read_and_file_in_name(self, mock_settings):
        """Ensure substring matching is NOT used — only exact allowlist."""
        verifier = FindingVerifier(
            settings=mock_settings,
            tool_context=ToolContext(
                parent_tools=[
                    {"name": "read_and_delete_file", "description": "Dangerous"},
                    {"name": "read_file_and_execute", "description": "Also dangerous"},
                    {"name": "file_reader", "description": "Not in allowlist"},
                ],
            ),
        )
        tools = verifier._get_verification_tools()
        assert tools == []

    def test_rejects_tools_with_only_read_in_name(self, mock_settings):
        verifier = FindingVerifier(
            settings=mock_settings,
            tool_context=ToolContext(
                parent_tools=[
                    {"name": "read_config", "description": "Read config"},
                    {"name": "read_url", "description": "Read URL"},
                ],
            ),
        )
        tools = verifier._get_verification_tools()
        assert tools == []

    def test_no_matching_tools(self, mock_settings):
        verifier = FindingVerifier(
            settings=mock_settings,
            tool_context=ToolContext(
                parent_tools=[
                    {"name": "glob_files", "description": "Glob for files"},
                    {"name": "write_file", "description": "Write a file"},
                ],
            ),
        )
        tools = verifier._get_verification_tools()
        assert tools == []

    def test_empty_parent_tools(self, mock_settings):
        verifier = FindingVerifier(
            settings=mock_settings,
            tool_context=ToolContext(parent_tools=[]),
        )
        tools = verifier._get_verification_tools()
        assert tools == []

    def test_malformed_tool_dict_without_name_key(self, mock_settings):
        verifier = FindingVerifier(
            settings=mock_settings,
            tool_context=ToolContext(
                parent_tools=[
                    {"description": "No name key"},
                    {"name": "read_file", "description": "Valid tool"},
                    {},
                ],
            ),
        )
        tools = verifier._get_verification_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "read_file"


class TestToolContextDefaults:
    """Tests for ToolContext default initialization paths."""

    def test_none_tool_context_defaults_to_empty(self, mock_settings):
        verifier = FindingVerifier(settings=mock_settings, tool_context=None)
        assert verifier.parent_tools == []
        assert verifier.mcp_manager is None
        assert verifier.local_registry is None

    def test_tool_context_with_all_fields(self, mock_settings):
        mcp = MagicMock()
        registry = MagicMock()
        ctx = ToolContext(
            parent_tools=[{"name": "read_file"}],
            mcp_manager=mcp,
            local_registry=registry,
        )
        verifier = FindingVerifier(settings=mock_settings, tool_context=ctx)
        assert len(verifier.parent_tools) == 1
        assert verifier.mcp_manager is mcp
        assert verifier.local_registry is registry

    def test_omitted_tool_context_defaults_to_empty(self, mock_settings):
        verifier = FindingVerifier(settings=mock_settings)
        assert verifier.parent_tools == []
        assert verifier.mcp_manager is None
        assert verifier.local_registry is None


class TestParseNonDictJsonTypes:
    """Tests for _parse_verification_response with non-dict JSON types."""

    def test_json_integer_returns_uncertain(self):
        result = FindingVerifier._parse_verification_response(json.dumps(42))
        assert result.status == "uncertain"
        assert "Unexpected response type" in result.reasoning

    def test_json_float_returns_uncertain(self):
        result = FindingVerifier._parse_verification_response(json.dumps(3.14))
        assert result.status == "uncertain"
        assert "Unexpected response type" in result.reasoning

    def test_json_boolean_returns_uncertain(self):
        result = FindingVerifier._parse_verification_response(json.dumps(True))
        assert result.status == "uncertain"
        assert "Unexpected response type" in result.reasoning


class TestCreateVerificationEngine:
    """Tests for _create_verification_engine budget calculations."""

    def test_budget_scales_with_num_findings(self, mock_settings):
        verifier = FindingVerifier(settings=mock_settings)
        provider = MagicMock()

        engine_1 = verifier._create_verification_engine(
            provider, num_findings=1, session_id="test-1"
        )
        engine_100 = verifier._create_verification_engine(
            provider, num_findings=100, session_id="test-100"
        )

        assert (
            engine_100.execution_limits.max_tokens_total
            <= engine_1.execution_limits.max_tokens_total
        )

    def test_budget_respects_minimums(self, mock_settings):
        from cicaddy.delegation.verifier import (
            _MIN_PER_ITER_TOKEN_BUDGET,
            _MIN_TOTAL_TOKEN_BUDGET,
        )

        verifier = FindingVerifier(settings=mock_settings)
        provider = MagicMock()

        engine = verifier._create_verification_engine(
            provider, num_findings=1000, session_id="test-min"
        )

        assert engine.execution_limits.max_tokens_total >= _MIN_TOTAL_TOKEN_BUDGET
        assert (
            engine.execution_limits.max_tokens_per_iteration
            >= _MIN_PER_ITER_TOKEN_BUDGET
        )

    def test_max_infer_iters_is_three(self, mock_settings):
        verifier = FindingVerifier(settings=mock_settings)
        provider = MagicMock()

        engine = verifier._create_verification_engine(
            provider, num_findings=5, session_id="test-iters"
        )

        assert engine.execution_limits.max_infer_iters == 3


class TestBuildVerificationPrompt:
    """Tests for FindingVerifier._build_verification_prompt."""

    def test_prompt_contains_finding_details(self):
        finding = Finding(
            file="src/auth.py",
            line=99,
            severity="critical",
            message="SQL injection vulnerability",
            suggestion="Use parameterized queries",
            agent_source="security-reviewer",
            existing_code='query = f"SELECT * FROM users WHERE id={user_id}"',
        )
        prompt = FindingVerifier._build_verification_prompt(
            finding, "diff --git a/src/auth.py"
        )
        assert "src/auth.py" in prompt
        assert "99" in prompt
        assert "critical" in prompt
        assert "SQL injection" in prompt
        assert "parameterized queries" in prompt
        assert "security-reviewer" in prompt
        assert "SELECT * FROM users" in prompt

    def test_prompt_handles_none_fields(self):
        finding = Finding(
            file="src/foo.py",
            line=None,
            severity="minor",
            message="Style issue",
        )
        prompt = FindingVerifier._build_verification_prompt(finding, "")
        assert "src/foo.py" in prompt
        assert "unresolved" in prompt
        assert "Style issue" in prompt
        assert "No suggestion provided" in prompt
        assert "No snippet provided" in prompt

    def test_prompt_contains_diff_context(self):
        finding = Finding(
            file="src/foo.py",
            line=10,
            severity="minor",
            message="test",
        )
        diff = "diff --git a/src/foo.py\n+new_code()"
        prompt = FindingVerifier._build_verification_prompt(finding, diff)
        assert "new_code()" in prompt


class TestVerifyFindings:
    """Integration tests for FindingVerifier.verify_findings."""

    @pytest.mark.asyncio
    async def test_empty_findings_noop(self, mock_settings):
        with patch("cicaddy.delegation.verifier.create_provider") as mock_create:
            verifier = FindingVerifier(settings=mock_settings)
            result = await verifier.verify_findings([], "diff content")

        assert result == []
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_valid_finding_kept(self, mock_settings):
        finding = Finding(
            file="src/foo.py",
            line=10,
            severity="major",
            message="Missing null check",
        )

        with patch.object(
            FindingVerifier,
            "_verify_single",
            new_callable=AsyncMock,
        ) as mock_verify:
            finding_copy = Finding(
                file="src/foo.py",
                line=10,
                severity="major",
                message="Missing null check",
                verified="valid",
                verification_reason="Confirmed: no null guard",
            )
            mock_verify.return_value = finding_copy

            verifier = FindingVerifier(settings=mock_settings)
            result = await verifier.verify_findings([finding], "diff")

        assert len(result) == 1
        assert result[0].verified == "valid"

    @pytest.mark.asyncio
    async def test_invalid_finding_filtered(self, mock_settings):
        finding = Finding(
            file="src/foo.py",
            line=10,
            severity="major",
            message="Missing null check",
        )

        with patch.object(
            FindingVerifier,
            "_verify_single",
            new_callable=AsyncMock,
        ) as mock_verify:
            finding_copy = Finding(
                file="src/foo.py",
                line=10,
                severity="major",
                message="Missing null check",
                verified="invalid",
                verification_reason="Already handled upstream",
            )
            mock_verify.return_value = finding_copy

            verifier = FindingVerifier(settings=mock_settings)
            result = await verifier.verify_findings([finding], "diff")

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_uncertain_finding_kept(self, mock_settings):
        finding = Finding(
            file="src/foo.py",
            line=10,
            severity="major",
            message="Potential issue",
        )

        with patch.object(
            FindingVerifier,
            "_verify_single",
            new_callable=AsyncMock,
        ) as mock_verify:
            finding_copy = Finding(
                file="src/foo.py",
                line=10,
                severity="major",
                message="Potential issue",
                verified="uncertain",
                verification_reason="Cannot determine",
            )
            mock_verify.return_value = finding_copy

            verifier = FindingVerifier(settings=mock_settings)
            result = await verifier.verify_findings([finding], "diff")

        assert len(result) == 1
        assert result[0].verified == "uncertain"

    @pytest.mark.asyncio
    async def test_severity_adjustment_applied(self, mock_settings):
        finding = Finding(
            file="src/foo.py",
            line=10,
            severity="major",
            message="Issue exists",
        )

        with patch.object(
            FindingVerifier,
            "_verify_single",
            new_callable=AsyncMock,
        ) as mock_verify:
            finding_copy = Finding(
                file="src/foo.py",
                line=10,
                severity="minor",
                message="Issue exists",
                verified="valid",
                verification_reason="Exists but minor",
            )
            mock_verify.return_value = finding_copy

            verifier = FindingVerifier(settings=mock_settings)
            result = await verifier.verify_findings([finding], "diff")

        assert len(result) == 1
        assert result[0].severity == "minor"

    @pytest.mark.asyncio
    async def test_verification_exception_keeps_finding(self, mock_settings):
        finding = Finding(
            file="src/foo.py",
            line=10,
            severity="major",
            message="Some issue",
        )

        with patch.object(
            FindingVerifier,
            "_verify_single",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Provider timeout"),
        ):
            verifier = FindingVerifier(settings=mock_settings)
            result = await verifier.verify_findings([finding], "diff")

        assert len(result) == 1
        assert result[0].file == "src/foo.py"

    @pytest.mark.asyncio
    async def test_mixed_results_filters_only_invalid(self, mock_settings):
        findings = [
            Finding(file="a.py", line=1, severity="major", message="Issue A"),
            Finding(file="b.py", line=2, severity="minor", message="Issue B"),
            Finding(file="c.py", line=3, severity="critical", message="Issue C"),
        ]

        def mock_verify(
            finding, diff_snippet, num_findings, provider, diff_ranges=None
        ):
            import dataclasses

            result = dataclasses.replace(finding)
            if finding.file == "b.py":
                result.verified = "invalid"
                result.verification_reason = "False positive"
            elif finding.file == "a.py":
                result.verified = "valid"
                result.verification_reason = "Confirmed"
            else:
                result.verified = "uncertain"
                result.verification_reason = "Unclear"
            return result

        with patch.object(
            FindingVerifier, "_verify_single", AsyncMock(side_effect=mock_verify)
        ):
            verifier = FindingVerifier(settings=mock_settings)
            result = await verifier.verify_findings(findings, "diff")

        assert len(result) == 2
        files = {f.file for f in result}
        assert files == {"a.py", "c.py"}


class TestProviderLifecycle:
    """Tests for provider creation and shutdown lifecycle."""

    @pytest.mark.asyncio
    async def test_provider_shutdown_called_on_success(self, mock_settings):
        finding = Finding(file="src/foo.py", line=10, severity="major", message="issue")
        mock_provider = MagicMock()
        mock_provider.initialize = AsyncMock()
        mock_provider.shutdown = AsyncMock()

        with (
            patch(
                "cicaddy.delegation.verifier.create_provider",
                return_value=mock_provider,
            ),
            patch("cicaddy.delegation.verifier.get_provider_config"),
            patch.object(
                FindingVerifier,
                "_verify_single",
                new_callable=AsyncMock,
                return_value=Finding(
                    file="src/foo.py",
                    line=10,
                    severity="major",
                    message="issue",
                    verified="valid",
                    verification_reason="ok",
                ),
            ),
        ):
            verifier = FindingVerifier(settings=mock_settings)
            await verifier.verify_findings([finding], "diff")

        mock_provider.initialize.assert_awaited_once()
        mock_provider.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_provider_shutdown_called_on_error(self, mock_settings):
        finding = Finding(file="src/foo.py", line=10, severity="major", message="issue")
        mock_provider = MagicMock()
        mock_provider.initialize = AsyncMock()
        mock_provider.shutdown = AsyncMock()

        with (
            patch(
                "cicaddy.delegation.verifier.create_provider",
                return_value=mock_provider,
            ),
            patch("cicaddy.delegation.verifier.get_provider_config"),
            patch.object(
                FindingVerifier,
                "_verify_single",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
        ):
            verifier = FindingVerifier(settings=mock_settings)
            result = await verifier.verify_findings([finding], "diff")

        mock_provider.shutdown.assert_awaited_once()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_provider_shutdown_exception_does_not_propagate(self, mock_settings):
        finding = Finding(file="src/foo.py", line=10, severity="major", message="issue")
        mock_provider = MagicMock()
        mock_provider.initialize = AsyncMock()
        mock_provider.shutdown = AsyncMock(side_effect=RuntimeError("shutdown failed"))

        with (
            patch(
                "cicaddy.delegation.verifier.create_provider",
                return_value=mock_provider,
            ),
            patch("cicaddy.delegation.verifier.get_provider_config"),
            patch.object(
                FindingVerifier,
                "_verify_single",
                new_callable=AsyncMock,
                return_value=Finding(
                    file="src/foo.py",
                    line=10,
                    severity="major",
                    message="issue",
                    verified="valid",
                    verification_reason="ok",
                ),
            ),
        ):
            verifier = FindingVerifier(settings=mock_settings)
            result = await verifier.verify_findings([finding], "diff")

        assert len(result) == 1
        assert result[0].verified == "valid"
        mock_provider.shutdown.assert_awaited_once()


class TestReasoningTruncation:
    """Tests for reasoning field length limits."""

    def test_long_reasoning_truncated_in_validate(self):
        long_reasoning = "x" * 5000
        entry = {"status": "valid", "reasoning": long_reasoning, "confidence": 0.9}
        result = FindingVerifier._validate_verification(entry)
        assert len(result.reasoning) == 2000
        assert result.reasoning.endswith("...")

    def test_short_reasoning_preserved(self):
        entry = {"status": "valid", "reasoning": "Short note", "confidence": 0.9}
        result = FindingVerifier._validate_verification(entry)
        assert result.reasoning == "Short note"


class TestValidStatuses:
    """Tests for valid status constants."""

    def test_expected_statuses(self):
        assert _VALID_STATUSES == {"valid", "invalid", "uncertain"}


class TestFindingVerificationFields:
    """Tests for Finding dataclass verification fields."""

    def test_verified_defaults_none(self):
        f = Finding(file="foo.py", line=1, severity="major", message="test")
        assert f.verified is None
        assert f.verification_reason is None

    def test_verified_can_be_set(self):
        f = Finding(
            file="foo.py",
            line=1,
            severity="major",
            message="test",
            verified="valid",
            verification_reason="Confirmed",
        )
        assert f.verified == "valid"
        assert f.verification_reason == "Confirmed"


class TestAdjustedLineValidation:
    """Tests for adjusted_line handling in _validate_verification."""

    def test_valid_int_within_diff_ranges(self):
        ranges = {"src/foo.py": [(10, 20), (30, 40)]}
        entry = {
            "status": "valid",
            "reasoning": "ok",
            "adjusted_line": 15,
        }
        result = FindingVerifier._validate_verification(
            entry, diff_ranges=ranges, file_path="src/foo.py"
        )
        assert result.adjusted_line == 15

    def test_outside_diff_ranges_becomes_none(self):
        ranges = {"src/foo.py": [(10, 20)]}
        entry = {
            "status": "valid",
            "reasoning": "ok",
            "adjusted_line": 50,
        }
        result = FindingVerifier._validate_verification(
            entry, diff_ranges=ranges, file_path="src/foo.py"
        )
        assert result.adjusted_line is None

    def test_negative_becomes_none(self):
        entry = {
            "status": "valid",
            "reasoning": "ok",
            "adjusted_line": -5,
        }
        result = FindingVerifier._validate_verification(entry)
        assert result.adjusted_line is None

    def test_zero_becomes_none(self):
        entry = {
            "status": "valid",
            "reasoning": "ok",
            "adjusted_line": 0,
        }
        result = FindingVerifier._validate_verification(entry)
        assert result.adjusted_line is None

    def test_non_int_becomes_none(self):
        entry = {
            "status": "valid",
            "reasoning": "ok",
            "adjusted_line": "line 15",
        }
        result = FindingVerifier._validate_verification(entry)
        assert result.adjusted_line is None

    def test_null_becomes_none(self):
        entry = {
            "status": "valid",
            "reasoning": "ok",
            "adjusted_line": None,
        }
        result = FindingVerifier._validate_verification(entry)
        assert result.adjusted_line is None

    def test_missing_field_becomes_none(self):
        entry = {
            "status": "valid",
            "reasoning": "ok",
        }
        result = FindingVerifier._validate_verification(entry)
        assert result.adjusted_line is None

    def test_no_diff_ranges_preserves_valid_int(self):
        entry = {
            "status": "valid",
            "reasoning": "ok",
            "adjusted_line": 42,
        }
        result = FindingVerifier._validate_verification(entry)
        assert result.adjusted_line == 42

    def test_float_coerced_to_int(self):
        entry = {
            "status": "valid",
            "reasoning": "ok",
            "adjusted_line": 15.0,
        }
        result = FindingVerifier._validate_verification(entry)
        assert result.adjusted_line == 15

    def test_boolean_rejected(self):
        entry = {
            "status": "valid",
            "reasoning": "ok",
            "adjusted_line": True,
        }
        result = FindingVerifier._validate_verification(entry)
        assert result.adjusted_line is None


class TestParseResponseAdjustedLine:
    """Tests for adjusted_line flowing through _parse_verification_response."""

    def test_adjusted_line_in_response(self):
        content = json.dumps(
            {
                "status": "valid",
                "reasoning": "Issue confirmed",
                "adjusted_line": 25,
                "confidence": 0.9,
            }
        )
        result = FindingVerifier._parse_verification_response(content)
        assert result.adjusted_line == 25

    def test_adjusted_line_validated_against_diff_ranges(self):
        ranges = {"src/foo.py": [(10, 20)]}
        content = json.dumps(
            {
                "status": "valid",
                "reasoning": "Issue confirmed",
                "adjusted_line": 50,
                "confidence": 0.9,
            }
        )
        result = FindingVerifier._parse_verification_response(
            content, diff_ranges=ranges, file_path="src/foo.py"
        )
        assert result.adjusted_line is None


class TestVerifyFindingsLineCorrection:
    """Tests for line correction in verify_findings flow."""

    @pytest.mark.asyncio
    async def test_line_corrected_for_ai_resolved_finding(self, mock_settings):
        finding = Finding(
            file="src/foo.py",
            line=1,
            severity="major",
            message="Missing null check",
            existing_code=None,
        )

        def mock_verify(
            finding, diff_snippet, num_findings, provider, diff_ranges=None
        ):
            import dataclasses

            result = dataclasses.replace(finding)
            result.verified = "valid"
            result.verification_reason = "Confirmed"
            result.line = 15
            result.start_line = 15
            result.end_line = 15
            return result

        with patch.object(
            FindingVerifier, "_verify_single", AsyncMock(side_effect=mock_verify)
        ):
            verifier = FindingVerifier(settings=mock_settings)
            result = await verifier.verify_findings([finding], "diff")

        assert len(result) == 1
        assert result[0].line == 15
        assert result[0].start_line == 15

    @pytest.mark.asyncio
    async def test_line_not_corrected_for_snippet_resolved_finding(self, mock_settings):
        finding = Finding(
            file="src/foo.py",
            line=42,
            severity="major",
            message="Missing null check",
            existing_code="value = get_data()",
        )

        def mock_verify(
            finding, diff_snippet, num_findings, provider, diff_ranges=None
        ):
            import dataclasses

            result = dataclasses.replace(finding)
            result.verified = "valid"
            result.verification_reason = "Confirmed"
            return result

        with patch.object(
            FindingVerifier, "_verify_single", AsyncMock(side_effect=mock_verify)
        ):
            verifier = FindingVerifier(settings=mock_settings)
            result = await verifier.verify_findings([finding], "diff")

        assert len(result) == 1
        assert result[0].line == 42


class TestBuildPromptLineRanges:
    """Tests for diff line ranges in verification prompt."""

    def test_prompt_includes_line_ranges(self):
        finding = Finding(
            file="src/foo.py",
            line=None,
            severity="major",
            message="Issue here",
        )
        diff_ranges = {"src/foo.py": [(10, 20), (30, 40)]}
        prompt = FindingVerifier._build_verification_prompt(
            finding, "diff content", diff_ranges
        )
        assert "Valid Line Ranges" in prompt
        assert "10-20" in prompt
        assert "30-40" in prompt

    def test_prompt_no_ranges_when_none(self):
        finding = Finding(
            file="src/foo.py",
            line=10,
            severity="major",
            message="Issue here",
        )
        prompt = FindingVerifier._build_verification_prompt(finding, "diff content")
        assert "Valid Line Ranges" not in prompt

    def test_prompt_no_ranges_for_unmatched_file(self):
        finding = Finding(
            file="src/bar.py",
            line=10,
            severity="major",
            message="Issue here",
        )
        diff_ranges = {"src/foo.py": [(10, 20)]}
        prompt = FindingVerifier._build_verification_prompt(
            finding, "diff content", diff_ranges
        )
        assert "Valid Line Ranges" not in prompt


class TestFormatLineRanges:
    """Tests for _format_line_ranges helper."""

    def test_formats_ranges(self):
        ranges = {"src/foo.py": [(10, 20), (30, 40)]}
        result = FindingVerifier._format_line_ranges("src/foo.py", ranges)
        assert "10-20, 30-40" in result

    def test_empty_when_no_ranges(self):
        result = FindingVerifier._format_line_ranges("src/foo.py", None)
        assert result == ""

    def test_empty_when_no_match(self):
        ranges = {"src/bar.py": [(10, 20)]}
        result = FindingVerifier._format_line_ranges("src/foo.py", ranges)
        assert result == ""

    def test_suffix_match(self):
        ranges = {"src/foo.py": [(10, 20)]}
        result = FindingVerifier._format_line_ranges("foo.py", ranges)
        assert "10-20" in result


class TestExistingCodeSnippetExtraction:
    """Tests for existing_code_snippet extraction in _validate_verification."""

    def test_snippet_extracted(self):
        entry = {
            "status": "valid",
            "reasoning": "Issue confirmed",
            "confidence": 0.9,
            "existing_code_snippet": "value = get_data()",
        }
        result = FindingVerifier._validate_verification(entry)
        assert result.existing_code_snippet == "value = get_data()"

    def test_snippet_stripped(self):
        entry = {
            "status": "valid",
            "reasoning": "ok",
            "existing_code_snippet": "  code()  \n",
        }
        result = FindingVerifier._validate_verification(entry)
        assert result.existing_code_snippet == "code()"

    def test_empty_snippet_becomes_none(self):
        entry = {
            "status": "valid",
            "reasoning": "ok",
            "existing_code_snippet": "",
        }
        result = FindingVerifier._validate_verification(entry)
        assert result.existing_code_snippet is None

    def test_whitespace_snippet_becomes_none(self):
        entry = {
            "status": "valid",
            "reasoning": "ok",
            "existing_code_snippet": "   \n  ",
        }
        result = FindingVerifier._validate_verification(entry)
        assert result.existing_code_snippet is None

    def test_null_snippet_becomes_none(self):
        entry = {
            "status": "valid",
            "reasoning": "ok",
            "existing_code_snippet": None,
        }
        result = FindingVerifier._validate_verification(entry)
        assert result.existing_code_snippet is None

    def test_missing_snippet_becomes_none(self):
        entry = {
            "status": "valid",
            "reasoning": "ok",
        }
        result = FindingVerifier._validate_verification(entry)
        assert result.existing_code_snippet is None

    def test_non_string_snippet_becomes_none(self):
        entry = {
            "status": "valid",
            "reasoning": "ok",
            "existing_code_snippet": 42,
        }
        result = FindingVerifier._validate_verification(entry)
        assert result.existing_code_snippet is None

    def test_snippet_flows_through_parse_response(self):
        content = json.dumps(
            {
                "status": "valid",
                "reasoning": "Confirmed",
                "confidence": 0.9,
                "existing_code_snippet": "if x is None:",
            }
        )
        result = FindingVerifier._parse_verification_response(content)
        assert result.existing_code_snippet == "if x is None:"


class TestVerifierSnippetLineResolution:
    """Tests for verifier snippet → find_line_in_diff line correction."""

    _DIFF = """\
diff --git a/src/foo.py b/src/foo.py
--- a/src/foo.py
+++ b/src/foo.py
@@ -10,6 +10,8 @@ def process():
     x = get_data()
     if x:
         handle(x)
+    value = get_data()
+    transform(value)
     return x
"""

    @pytest.mark.asyncio
    async def test_snippet_resolves_line_from_diff(self, mock_settings):
        """Verifier snippet found in diff → updates finding line."""
        finding = Finding(
            file="src/foo.py",
            line=None,
            severity="major",
            message="Missing null check on value",
        )

        mock_turn = MagicMock()
        mock_turn.output_message = json.dumps(
            {
                "status": "valid",
                "reasoning": "No null guard on value",
                "confidence": 0.9,
                "existing_code_snippet": "value = get_data()",
            }
        )

        mock_engine = MagicMock()
        mock_engine.execute_turn = AsyncMock(return_value=mock_turn)

        with patch.object(
            FindingVerifier, "_create_verification_engine", return_value=mock_engine
        ):
            verifier = FindingVerifier(settings=mock_settings)
            result = await verifier._verify_single(
                finding, self._DIFF, num_findings=1, provider=MagicMock()
            )

        assert result.verified == "valid"
        assert result.line == 13
        assert result.start_line == 13
        assert result.end_line == 13
        assert result.existing_code == "value = get_data()"

    @pytest.mark.asyncio
    async def test_snippet_not_in_diff_falls_back_to_adjusted_line(self, mock_settings):
        """Snippet not found in diff → falls back to adjusted_line."""
        finding = Finding(
            file="src/foo.py",
            line=None,
            severity="major",
            message="Issue in unrelated code",
        )

        diff_ranges = {"src/foo.py": [(10, 16)]}

        mock_turn = MagicMock()
        mock_turn.output_message = json.dumps(
            {
                "status": "valid",
                "reasoning": "Confirmed",
                "confidence": 0.8,
                "existing_code_snippet": "nonexistent_code()",
                "adjusted_line": 12,
            }
        )

        mock_engine = MagicMock()
        mock_engine.execute_turn = AsyncMock(return_value=mock_turn)

        with patch.object(
            FindingVerifier, "_create_verification_engine", return_value=mock_engine
        ):
            verifier = FindingVerifier(settings=mock_settings)
            result = await verifier._verify_single(
                finding,
                self._DIFF,
                num_findings=1,
                provider=MagicMock(),
                diff_ranges=diff_ranges,
            )

        assert result.line == 12

    @pytest.mark.asyncio
    async def test_snippet_resolves_multiline(self, mock_settings):
        """Multi-line snippet resolves start and end lines."""
        finding = Finding(
            file="src/foo.py",
            line=None,
            severity="major",
            message="Both lines need null checks",
        )

        mock_turn = MagicMock()
        mock_turn.output_message = json.dumps(
            {
                "status": "valid",
                "reasoning": "Confirmed",
                "confidence": 0.9,
                "existing_code_snippet": "value = get_data()\n    transform(value)",
            }
        )

        mock_engine = MagicMock()
        mock_engine.execute_turn = AsyncMock(return_value=mock_turn)

        with patch.object(
            FindingVerifier, "_create_verification_engine", return_value=mock_engine
        ):
            verifier = FindingVerifier(settings=mock_settings)
            result = await verifier._verify_single(
                finding, self._DIFF, num_findings=1, provider=MagicMock()
            )

        assert result.line == 13
        assert result.start_line == 13
        assert result.end_line == 14

    @pytest.mark.asyncio
    async def test_no_snippet_uses_adjusted_line_when_no_existing_code(
        self, mock_settings
    ):
        """No snippet returned, no existing_code → adjusted_line used."""
        finding = Finding(
            file="src/foo.py",
            line=None,
            severity="major",
            message="Issue",
            existing_code=None,
        )

        diff_ranges = {"src/foo.py": [(10, 16)]}

        mock_turn = MagicMock()
        mock_turn.output_message = json.dumps(
            {
                "status": "valid",
                "reasoning": "Confirmed",
                "confidence": 0.8,
                "adjusted_line": 12,
            }
        )

        mock_engine = MagicMock()
        mock_engine.execute_turn = AsyncMock(return_value=mock_turn)

        with patch.object(
            FindingVerifier, "_create_verification_engine", return_value=mock_engine
        ):
            verifier = FindingVerifier(settings=mock_settings)
            result = await verifier._verify_single(
                finding,
                self._DIFF,
                num_findings=1,
                provider=MagicMock(),
                diff_ranges=diff_ranges,
            )

        assert result.line == 12

    @pytest.mark.asyncio
    async def test_snippet_sets_existing_code_on_finding(self, mock_settings):
        """When snippet resolves, existing_code is set on the finding."""
        finding = Finding(
            file="src/foo.py",
            line=None,
            severity="major",
            message="Issue",
            existing_code=None,
        )

        mock_turn = MagicMock()
        mock_turn.output_message = json.dumps(
            {
                "status": "valid",
                "reasoning": "Confirmed",
                "confidence": 0.9,
                "existing_code_snippet": "value = get_data()",
            }
        )

        mock_engine = MagicMock()
        mock_engine.execute_turn = AsyncMock(return_value=mock_turn)

        with patch.object(
            FindingVerifier, "_create_verification_engine", return_value=mock_engine
        ):
            verifier = FindingVerifier(settings=mock_settings)
            result = await verifier._verify_single(
                finding, self._DIFF, num_findings=1, provider=MagicMock()
            )

        assert result.existing_code == "value = get_data()"
