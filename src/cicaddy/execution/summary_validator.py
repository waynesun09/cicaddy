"""Summary validator for checking AI summaries against ground truth metrics."""

import re
from typing import Any, Dict, List, Optional

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationIssue:
    """Represents a validation issue found during summary validation."""

    def __init__(
        self,
        issue_type: str,
        severity: str,
        message: str,
        expected: Any = None,
        actual: Any = None,
        entity: Optional[str] = None,
    ):
        """
        Initialize a validation issue.

        Args:
            issue_type: Type of issue (missing_entity, contradiction, understated, etc)
            severity: Severity level (critical, high, medium, low)
            message: Human-readable message
            expected: Expected value based on metrics
            actual: Actual value found in summary
            entity: Related entity (if applicable)
        """
        self.issue_type = issue_type
        self.severity = severity
        self.message = message
        self.expected = expected
        self.actual = actual
        self.entity = entity

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": self.issue_type,
            "severity": self.severity,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
            "entity": self.entity,
        }


class SummaryValidator:
    """
    Validates AI-generated summaries against ground truth metrics.

    Performs multiple validation checks:
    - Quantitative consistency (counts match)
    - Contradiction detection (saying "none" when count > 0)
    - Entity preservation (important entities mentioned)
    - Severity alignment (quantity matches description)
    """

    # Severity thresholds
    SIGNIFICANT_THRESHOLD = 10  # >10 = significant, not "potential"
    MINIMAL_THRESHOLD = 3  # <3 = minimal or negligible

    # Negative indicators in text
    NEGATIVE_WORDS = [
        "no",
        "zero",
        "none",
        "negligible",
        "minimal",
        "not found",
        "absent",
    ]
    UNDERSTATED_WORDS = ["potential", "possible", "maybe", "might", "limited"]

    def __init__(
        self,
        confidence_threshold: float = 0.8,
        auto_correct: bool = True,
        block_on_critical: bool = True,
    ):
        """
        Initialize the validator.

        Args:
            confidence_threshold: Minimum confidence score to pass validation
            auto_correct: Whether to suggest auto-corrections
            block_on_critical: Whether to block on critical issues
        """
        self.confidence_threshold = confidence_threshold
        self.auto_correct = auto_correct
        self.block_on_critical = block_on_critical

        logger.info(
            f"SummaryValidator initialized: confidence_threshold={confidence_threshold}, "
            f"auto_correct={auto_correct}, block_on_critical={block_on_critical}"
        )

    def validate(
        self,
        summary: str,
        metrics: Dict[str, Any],
        framework_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate an AI summary against ground truth metrics.

        Args:
            summary: AI-generated summary text
            metrics: Ground truth metrics dictionary
            framework_name: Optional framework name for context

        Returns:
            Dictionary with validation results:
                - valid: Boolean indicating if summary passes validation
                - issues: List of critical issues found
                - warnings: List of warnings
                - confidence: Confidence score (0.0-1.0)
                - suggestions: Auto-correction suggestions (if enabled)
        """
        logger.debug(f"Validating summary for framework: {framework_name or 'unknown'}")

        issues: List[ValidationIssue] = []
        warnings: List[ValidationIssue] = []

        # Normalize summary for analysis
        summary_lower = summary.lower()

        # Check 1: Quantitative consistency
        self._check_quantitative_consistency(
            summary, summary_lower, metrics, framework_name, issues, warnings
        )

        # Check 2: Contradiction detection
        self._check_contradictions(
            summary, summary_lower, metrics, framework_name, issues, warnings
        )

        # Check 3: Entity preservation
        self._check_entity_preservation(
            summary, summary_lower, metrics, framework_name, issues, warnings
        )

        # Check 4: Severity alignment
        self._check_severity_alignment(
            summary, summary_lower, metrics, framework_name, issues, warnings
        )

        # Calculate confidence score
        confidence = self._calculate_confidence(summary, metrics, issues, warnings)

        # Determine if validation passed
        valid = (
            len(issues) == 0
            and confidence >= self.confidence_threshold
            and (
                not self.block_on_critical
                or len([i for i in issues if i.severity == "critical"]) == 0
            )
        )

        result = {
            "valid": valid,
            "issues": [issue.to_dict() for issue in issues],
            "warnings": [warning.to_dict() for warning in warnings],
            "confidence": confidence,
            "framework": framework_name,
        }

        # Add suggestions if auto_correct is enabled
        if self.auto_correct and (issues or warnings):
            result["suggestions"] = self._generate_suggestions(
                summary, metrics, issues, warnings
            )

        log_level = logger.warning if not valid else logger.info
        log_level(
            f"Validation result for {framework_name or 'summary'}: "
            f"valid={valid}, confidence={confidence:.2f}, "
            f"issues={len(issues)}, warnings={len(warnings)}"
        )

        return result

    def _check_quantitative_consistency(
        self,
        summary: str,
        summary_lower: str,
        metrics: Dict[str, Any],
        framework_name: Optional[str],
        issues: List[ValidationIssue],
        warnings: List[ValidationIssue],
    ):
        """Check that quantitative metrics are mentioned when significant."""
        counts = metrics.get("counts", {})

        # Check repositories
        repo_count = counts.get("repository", 0)
        if repo_count > 0:
            if "repositor" not in summary_lower and "repo" not in summary_lower:
                severity = (
                    "high" if repo_count > self.SIGNIFICANT_THRESHOLD else "medium"
                )
                issues.append(
                    ValidationIssue(
                        issue_type="missing_entity",
                        severity=severity,
                        message=f"{repo_count} repositories found but not mentioned in summary",
                        expected=repo_count,
                        actual=0,
                        entity="repository",
                    )
                )

        # Check files
        file_count = counts.get("file", 0)
        if file_count > self.SIGNIFICANT_THRESHOLD:
            if "file" not in summary_lower:
                warnings.append(
                    ValidationIssue(
                        issue_type="missing_entity",
                        severity="medium",
                        message=f"{file_count} files found but not mentioned in summary",
                        expected=file_count,
                        actual=0,
                        entity="file",
                    )
                )

        # Check matches
        match_count = counts.get("match", 0)
        if match_count > self.SIGNIFICANT_THRESHOLD:
            if "match" not in summary_lower and "found" not in summary_lower:
                warnings.append(
                    ValidationIssue(
                        issue_type="missing_metric",
                        severity="low",
                        message=f"{match_count} code matches found but not mentioned",
                        expected=match_count,
                        actual=0,
                        entity="match",
                    )
                )

    def _check_contradictions(
        self,
        summary: str,
        summary_lower: str,
        metrics: Dict[str, Any],
        framework_name: Optional[str],
        issues: List[ValidationIssue],
        warnings: List[ValidationIssue],
    ):
        """Check for contradictions between summary and metrics."""
        counts = metrics.get("counts", {})

        for entity_type, count in counts.items():
            if count > self.SIGNIFICANT_THRESHOLD:
                # Check if summary contains negative words
                for negative_word in self.NEGATIVE_WORDS:
                    pattern = rf"\b{negative_word}\b.*\b{entity_type}"
                    if re.search(pattern, summary_lower):
                        issues.append(
                            ValidationIssue(
                                issue_type="contradiction",
                                severity="critical",
                                message=f"Summary says '{negative_word}' for {entity_type} but {count} found",
                                expected=count,
                                actual=negative_word,
                                entity=entity_type,
                            )
                        )

    def _check_entity_preservation(
        self,
        summary: str,
        summary_lower: str,
        metrics: Dict[str, Any],
        framework_name: Optional[str],
        issues: List[ValidationIssue],
        warnings: List[ValidationIssue],
    ):
        """Check that important unique entities are mentioned."""
        # Extract unique counts from metrics
        unique_repos = metrics.get("unique_repository_count", 0) or len(
            metrics.get("unique_repositories", [])
        )

        # If we have many unique entities, summary should reflect this
        if unique_repos > self.SIGNIFICANT_THRESHOLD:
            # Look for actual repository names in summary
            repo_list = metrics.get("unique_repositories_list", [])
            if repo_list:
                # Check if at least one repo is mentioned
                repos_mentioned = sum(
                    1 for repo in repo_list if repo.lower() in summary_lower
                )
                if repos_mentioned == 0:
                    warnings.append(
                        ValidationIssue(
                            issue_type="missing_examples",
                            severity="medium",
                            message=f"No specific repository names mentioned despite {unique_repos} unique repos found",
                            expected=1,  # At least one
                            actual=0,
                            entity="repository_name",
                        )
                    )

    def _check_severity_alignment(
        self,
        summary: str,
        summary_lower: str,
        metrics: Dict[str, Any],
        framework_name: Optional[str],
        issues: List[ValidationIssue],
        warnings: List[ValidationIssue],
    ):
        """Check that severity/importance descriptions align with quantities."""
        unique_repos = metrics.get("unique_repository_count", 0) or len(
            metrics.get("unique_repositories", [])
        )
        total_matches = metrics.get(
            "total_matches", metrics.get("counts", {}).get("match", 0)
        )

        # Check for understatement
        if unique_repos > self.SIGNIFICANT_THRESHOLD or total_matches > 50:
            for understated_word in self.UNDERSTATED_WORDS:
                if understated_word in summary_lower:
                    warnings.append(
                        ValidationIssue(
                            issue_type="understated",
                            severity="medium",
                            message=f"Summary uses '{understated_word}' but {unique_repos} repos "
                            f"and {total_matches} matches suggest significant/active usage",
                            expected="significant/active",
                            actual=understated_word,
                            entity="severity_descriptor",
                        )
                    )
                    break

        # Check for overstatement
        if unique_repos < self.MINIMAL_THRESHOLD and total_matches < 10:
            if any(
                word in summary_lower
                for word in ["significant", "extensive", "widespread", "active"]
            ):
                warnings.append(
                    ValidationIssue(
                        issue_type="overstated",
                        severity="low",
                        message=f"Summary suggests significant usage but only {unique_repos} repos "
                        f"and {total_matches} matches found",
                        expected="minimal/limited",
                        actual="significant/active",
                        entity="severity_descriptor",
                    )
                )

    def _calculate_confidence(
        self,
        summary: str,
        metrics: Dict[str, Any],
        issues: List[ValidationIssue],
        warnings: List[ValidationIssue],
    ) -> float:
        """
        Calculate confidence score for the summary.

        Args:
            summary: Summary text
            metrics: Ground truth metrics
            issues: List of issues found
            warnings: List of warnings found

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Start with perfect score
        confidence = 1.0

        # Deduct for issues based on severity
        severity_penalties = {"critical": 0.3, "high": 0.2, "medium": 0.1, "low": 0.05}

        for issue in issues:
            penalty = severity_penalties.get(issue.severity, 0.1)
            confidence -= penalty

        for warning in warnings:
            penalty = (
                severity_penalties.get(warning.severity, 0.05) / 2
            )  # Warnings count less
            confidence -= penalty

        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))

    def _generate_suggestions(
        self,
        summary: str,
        metrics: Dict[str, Any],
        issues: List[ValidationIssue],
        warnings: List[ValidationIssue],
    ) -> List[str]:
        """
        Generate auto-correction suggestions.

        Args:
            summary: Original summary
            metrics: Ground truth metrics
            issues: Validation issues
            warnings: Validation warnings

        Returns:
            List of suggestion strings
        """
        suggestions = []

        # Suggest adding missing entities
        for issue in issues + warnings:
            if issue.issue_type == "missing_entity":
                suggestions.append(
                    f"Add mention of {issue.entity}: {issue.expected} {issue.entity}(s) found"
                )

            elif issue.issue_type == "contradiction":
                suggestions.append(
                    f"Remove '{issue.actual}' for {issue.entity} - actually found {issue.expected}"
                )

            elif issue.issue_type == "understated":
                suggestions.append(
                    f"Change '{issue.actual}' to '{issue.expected}' to reflect quantity"
                )

            elif issue.issue_type == "missing_examples":
                # Suggest adding top repository names
                repo_list = metrics.get("unique_repositories_list", [])
                if repo_list:
                    top_repos = repo_list[:3]
                    suggestions.append(
                        f"Add specific repository examples: {', '.join(top_repos)}"
                    )

        return suggestions
