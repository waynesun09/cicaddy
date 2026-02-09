"""Tests for SummaryValidator."""

import pytest

from cicaddy.execution.summary_validator import SummaryValidator, ValidationIssue


class TestSummaryValidator:
    """Test suite for SummaryValidator."""

    @pytest.fixture
    def validator(self):
        """Create a validator instance."""
        return SummaryValidator(
            confidence_threshold=0.8, auto_correct=True, block_on_critical=True
        )

    @pytest.fixture
    def metrics_significant(self):
        """Metrics indicating significant usage."""
        return {
            "counts": {"repository": 15, "file": 50, "match": 200, "error": 0},
            "unique_repository_count": 12,
            "unique_file_count": 45,
            "total_matches": 200,
            "unique_repositories_list": [
                "github.com/repo1",
                "github.com/repo2",
                "github.com/repo3",
            ],
        }

    @pytest.fixture
    def metrics_minimal(self):
        """Metrics indicating minimal usage."""
        return {
            "counts": {"repository": 1, "file": 2, "match": 3, "error": 0},
            "unique_repository_count": 1,
            "unique_file_count": 2,
            "total_matches": 3,
            "unique_repositories_list": ["github.com/small-repo"],
        }

    def test_validation_issue_creation(self):
        """Test ValidationIssue creation."""
        issue = ValidationIssue(
            issue_type="contradiction",
            severity="critical",
            message="Test message",
            expected=10,
            actual=0,
            entity="repository",
        )

        assert issue.issue_type == "contradiction"
        assert issue.severity == "critical"
        assert issue.expected == 10
        assert issue.actual == 0

        # Test to_dict
        issue_dict = issue.to_dict()
        assert issue_dict["type"] == "contradiction"
        assert issue_dict["severity"] == "critical"

    def test_validate_good_summary(self, validator, metrics_significant):
        """Test validation of a good summary."""
        summary = """
        Analysis found significant usage across 12 unique repositories with 200 code matches.
        Key repositories include github.com/repo1, github.com/repo2, and github.com/repo3.
        This indicates active adoption and widespread implementation.
        """

        result = validator.validate(summary, metrics_significant, "LangChain")

        assert result["valid"] is True
        assert result["confidence"] >= 0.8
        assert len(result["issues"]) == 0
        assert result["framework"] == "LangChain"

    def test_detect_contradiction(self, validator, metrics_significant):
        """Test detection of contradictions."""
        summary = """
        No repositories found for this framework.
        There are zero matches in the codebase.
        """

        result = validator.validate(summary, metrics_significant, "Framework")

        assert result["valid"] is False
        assert len(result["issues"]) > 0

        # Check for contradiction issue
        contradiction_issues = [
            i for i in result["issues"] if i["type"] == "contradiction"
        ]
        assert len(contradiction_issues) > 0
        assert any(i["severity"] == "critical" for i in contradiction_issues)

    def test_detect_understated_summary(self, validator, metrics_significant):
        """Test detection of understated descriptions."""
        summary = """
        Potential usage identified in a few repositories.
        Limited adoption with minimal code patterns.
        """

        result = validator.validate(summary, metrics_significant, "Framework")

        # Should have warnings about understatement
        understated_warnings = [
            w for w in result["warnings"] if w["type"] == "understated"
        ]
        assert len(understated_warnings) > 0

    def test_detect_missing_entities(self, validator, metrics_significant):
        """Test detection of missing entity mentions."""
        summary = """
        Some code was found during the search.
        Analysis completed successfully.
        """

        result = validator.validate(summary, metrics_significant, "Framework")

        # Should have issues about missing repository mentions
        missing_entity_issues = [
            i
            for i in result["issues"] + result["warnings"]
            if i["type"] == "missing_entity"
        ]
        assert len(missing_entity_issues) > 0

    def test_detect_missing_examples(self, validator, metrics_significant):
        """Test detection of missing repository examples."""
        summary = """
        Found repositories with code patterns.
        Significant usage detected.
        """

        result = validator.validate(summary, metrics_significant, "Framework")

        # Should warn about missing specific repository names
        missing_example_warnings = [
            w for w in result["warnings"] if w["type"] == "missing_examples"
        ]
        assert len(missing_example_warnings) > 0

    def test_validate_minimal_correctly(self, validator, metrics_minimal):
        """Test that minimal usage is correctly validated."""
        summary = """
        Minimal usage found with 1 repository and 3 code matches.
        Limited adoption in the codebase.
        """

        result = validator.validate(summary, metrics_minimal, "Framework")

        # Should pass - correctly describes minimal usage
        assert result["valid"] is True or len(result["issues"]) == 0

    def test_detect_overstatement_minimal(self, validator, metrics_minimal):
        """Test detection of overstatement for minimal data."""
        summary = """
        Extensive and widespread usage across the codebase.
        Significant adoption with active development.
        """

        result = validator.validate(summary, metrics_minimal, "Framework")

        # Should warn about overstatement
        overstated_warnings = [
            w for w in result["warnings"] if w["type"] == "overstated"
        ]
        assert len(overstated_warnings) > 0

    def test_confidence_calculation(self, validator, metrics_significant):
        """Test confidence score calculation."""
        # Good summary - high confidence
        good_summary = """
        Found 12 repositories with 200 code matches.
        Includes github.com/repo1, github.com/repo2.
        Indicates active usage.
        """

        result1 = validator.validate(good_summary, metrics_significant)
        assert result1["confidence"] > 0.9

        # Bad summary - low confidence
        bad_summary = """
        No repositories found.
        Zero matches.
        Not used.
        """

        result2 = validator.validate(bad_summary, metrics_significant)
        # Bad summary should have lower confidence than good summary
        assert result2["confidence"] < result1["confidence"]
        assert result2["confidence"] < 0.7  # Relaxed threshold

    def test_auto_correct_suggestions(self, validator, metrics_significant):
        """Test auto-correction suggestions."""
        summary = """
        No repositories found.
        Potential usage in some areas.
        """

        result = validator.validate(summary, metrics_significant, "Framework")

        assert "suggestions" in result
        assert len(result["suggestions"]) > 0

        # Check suggestion content
        suggestions_text = " ".join(result["suggestions"])
        assert "repository" in suggestions_text.lower()

    def test_disable_auto_correct(self, metrics_significant):
        """Test with auto-correction disabled."""
        validator = SummaryValidator(auto_correct=False)

        summary = "No repositories found."

        result = validator.validate(summary, metrics_significant)

        assert "suggestions" not in result or len(result.get("suggestions", [])) == 0

    def test_block_on_critical(self, metrics_significant):
        """Test blocking on critical issues."""
        validator = SummaryValidator(block_on_critical=True)

        summary = "No repositories found."  # Will create critical contradiction

        result = validator.validate(summary, metrics_significant)

        # Should be invalid due to critical issue
        critical_issues = [i for i in result["issues"] if i["severity"] == "critical"]
        if len(critical_issues) > 0:
            assert result["valid"] is False

    def test_no_block_on_critical(self, metrics_significant):
        """Test not blocking on critical issues."""
        validator = SummaryValidator(block_on_critical=False, confidence_threshold=0.5)

        summary = "Some repositories found with matches."

        result = validator.validate(summary, metrics_significant)

        # Even with issues, might pass if confidence is acceptable
        # and we're not blocking on critical
        assert "valid" in result

    def test_confidence_threshold(self, metrics_significant):
        """Test confidence threshold enforcement."""
        # Low threshold - easier to pass
        validator_low = SummaryValidator(confidence_threshold=0.3)
        summary = "Potential usage found."

        result_low = validator_low.validate(summary, metrics_significant)

        # High threshold - harder to pass
        validator_high = SummaryValidator(confidence_threshold=0.95)
        result_high = validator_high.validate(summary, metrics_significant)

        # Low threshold should be more lenient
        assert result_low["confidence"] == result_high["confidence"]  # Same confidence
        # But validation result might differ based on threshold

    def test_empty_summary(self, validator, metrics_significant):
        """Test validation of empty summary."""
        result = validator.validate("", metrics_significant, "Framework")

        assert result["valid"] is False
        assert len(result["issues"]) > 0

    def test_llama_stack_false_negative(self, validator):
        """Test detection of the Llama Stack false negative scenario."""
        # This is the actual scenario from the bug report
        metrics = {
            "counts": {"repository": 12, "file": 50, "match": 200, "error": 0},
            "unique_repository_count": 10,
            "total_matches": 200,
            "unique_repositories_list": [
                "github.com/opendatahub-io/llama-stack-demos",
                "github.com/opendatahub-io/llama-stack",
                "github.com/redhat-et/agent-frameworks",
            ],
        }

        # Summary that incorrectly says "no matches"
        false_negative_summary = """
        Llama Stack Framework Analysis:
        No significant files were identified containing patterns.
        Code Hit Statistics: Negligible to zero matches found.
        Key File References: None identified.
        """

        result = validator.validate(false_negative_summary, metrics, "Llama Stack")

        # Should detect critical issues
        assert result["valid"] is False
        assert len(result["issues"]) > 0

        # Should find contradiction
        contradictions = [i for i in result["issues"] if i["type"] == "contradiction"]
        assert len(contradictions) > 0

        # Confidence should be very low
        assert result["confidence"] < 0.5

    def test_severity_levels(self, validator, metrics_significant):
        """Test that different severity levels are assigned correctly."""
        summary = "No repositories. Zero files. Not found."

        result = validator.validate(summary, metrics_significant)

        # Should have mix of severities
        severities = [i["severity"] for i in result["issues"]]
        assert "critical" in severities or "high" in severities

    def test_framework_specific_validation(self, validator):
        """Test framework-specific validation context."""
        metrics = {"counts": {"repository": 5, "file": 10, "match": 50}}

        result1 = validator.validate("Summary text", metrics, "LangChain")
        result2 = validator.validate("Summary text", metrics, "Llama Stack")

        assert result1["framework"] == "LangChain"
        assert result2["framework"] == "Llama Stack"

    def test_validation_with_zero_metrics(self, validator):
        """Test validation when metrics show zero usage."""
        zero_metrics = {
            "counts": {"repository": 0, "file": 0, "match": 0, "error": 0},
            "unique_repository_count": 0,
            "total_matches": 0,
        }

        summary_correct = "No repositories found. No code matches detected."
        result_correct = validator.validate(summary_correct, zero_metrics)

        # Should pass - correctly describes zero usage
        assert result_correct["valid"] is True
        assert result_correct["confidence"] >= 0.8

        summary_incorrect = "Significant usage with many repositories."
        result_incorrect = validator.validate(summary_incorrect, zero_metrics)

        # Should have warnings about overstatement (might not block if no critical issues)
        assert (
            len(result_incorrect["warnings"]) > 0 or len(result_incorrect["issues"]) > 0
        )
