"""Tests for cicaddy.delegation.line_resolver module."""

from __future__ import annotations

from cicaddy.delegation.line_resolver import (
    annotate_diff_with_line_numbers,
    find_line_in_diff,
    parse_diff,
    resolve_findings,
    validate_findings_in_hunks,
)
from cicaddy.delegation.summarizer import Finding

# Sample unified diff for testing
SAMPLE_DIFF = """\
diff --git a/src/foo.py b/src/foo.py
--- a/src/foo.py
+++ b/src/foo.py
@@ -10,7 +10,9 @@ def process():
     data = fetch()
     if data:
         result = transform(data)
-        return result
+        if result is None:
+            raise ValueError("transform failed")
+        return result
     return None


@@ -25,3 +27,6 @@ def helper():
     x = 1
     y = 2
     return x + y
+
+def new_function():
+    return 42
"""

MULTI_FILE_DIFF = """\
diff --git a/src/foo.py b/src/foo.py
--- a/src/foo.py
+++ b/src/foo.py
@@ -10,4 +10,5 @@ def process():
     data = fetch()
     if data:
         result = transform(data)
+        validate(result)
         return result
diff --git a/src/bar.py b/src/bar.py
--- a/src/bar.py
+++ b/src/bar.py
@@ -1,3 +1,4 @@
 import os
+import sys

 SECRET = "hardcoded"
"""


class TestParseDiff:
    """Tests for parse_diff()."""

    def test_parse_single_file(self):
        files = parse_diff(SAMPLE_DIFF)
        assert len(files) == 1
        assert files[0].path == "src/foo.py"
        assert len(files[0].hunks) == 2

    def test_parse_multi_file(self):
        files = parse_diff(MULTI_FILE_DIFF)
        assert len(files) == 2
        assert files[0].path == "src/foo.py"
        assert files[1].path == "src/bar.py"

    def test_hunk_line_numbers(self):
        files = parse_diff(SAMPLE_DIFF)
        hunk = files[0].hunks[0]
        assert hunk.old_start == 10
        assert hunk.old_count == 7
        assert hunk.new_start == 10
        assert hunk.new_count == 9

    def test_line_types(self):
        files = parse_diff(SAMPLE_DIFF)
        hunk = files[0].hunks[0]
        types = [dl.type for dl in hunk.lines]
        assert "context" in types
        assert "add" in types
        assert "remove" in types

    def test_new_line_numbers(self):
        files = parse_diff(SAMPLE_DIFF)
        hunk = files[0].hunks[0]
        add_lines = [dl for dl in hunk.lines if dl.type == "add"]
        # First added line should be at new line 13
        assert add_lines[0].new_lineno == 13
        assert "if result is None:" in add_lines[0].content

    def test_empty_diff(self):
        assert parse_diff("") == []
        assert parse_diff("   ") == []

    def test_parse_content_resembling_headers(self):
        """Content lines that look like diff headers should be parsed as content."""
        diff = """\
diff --git a/x.py b/x.py
--- a/x.py
+++ b/x.py
@@ -1,4 +1,4 @@
 normal line
--- some old delimiter
+++ b/not-a-header
 end
"""
        files = parse_diff(diff)
        assert len(files) == 1
        hunk = files[0].hunks[0]
        # "-- some old delimiter" should be parsed as a removal, not skipped
        remove_lines = [dl for dl in hunk.lines if dl.type == "remove"]
        assert len(remove_lines) == 1
        assert "some old delimiter" in remove_lines[0].content
        # "++ b/not-a-header" should be parsed as an addition
        add_lines = [dl for dl in hunk.lines if dl.type == "add"]
        assert len(add_lines) == 1
        assert "b/not-a-header" in add_lines[0].content

    def test_parse_no_count_hunk(self):
        """Handle @@ -1 +1 @@ (no comma count)."""
        diff = """\
diff --git a/x.py b/x.py
--- a/x.py
+++ b/x.py
@@ -1 +1 @@
-old
+new
"""
        files = parse_diff(diff)
        assert len(files) == 1
        hunk = files[0].hunks[0]
        assert hunk.old_count == 1
        assert hunk.new_count == 1


class TestFindLineInDiff:
    """Tests for find_line_in_diff()."""

    def test_exact_match(self):
        files = parse_diff(SAMPLE_DIFF)
        result = find_line_in_diff(files, "src/foo.py", "if result is None:")
        assert result is not None
        assert result[0] == 13  # start_line

    def test_exact_match_added_function(self):
        files = parse_diff(SAMPLE_DIFF)
        result = find_line_in_diff(files, "src/foo.py", "def new_function():")
        assert result is not None
        assert result[0] == 31

    def test_multi_line_snippet(self):
        files = parse_diff(SAMPLE_DIFF)
        snippet = "if result is None:\n            raise ValueError"
        result = find_line_in_diff(files, "src/foo.py", snippet)
        assert result is not None
        assert result[0] == 13
        assert result[1] == 14  # end_line

    def test_no_match(self):
        files = parse_diff(SAMPLE_DIFF)
        result = find_line_in_diff(files, "src/foo.py", "nonexistent code")
        assert result is None

    def test_wrong_file(self):
        files = parse_diff(SAMPLE_DIFF)
        result = find_line_in_diff(files, "src/other.py", "if result is None:")
        assert result is None

    def test_basename_match(self):
        """Should find file by suffix match."""
        files = parse_diff(MULTI_FILE_DIFF)
        result = find_line_in_diff(files, "bar.py", "import sys")
        assert result is not None
        assert result[0] == 2

    def test_context_line_match(self):
        """Should match context lines (unchanged code)."""
        files = parse_diff(SAMPLE_DIFF)
        result = find_line_in_diff(files, "src/foo.py", "data = fetch()")
        assert result is not None
        assert result[0] == 10

    def test_whitespace_normalization(self):
        """Should match despite whitespace differences."""
        files = parse_diff(SAMPLE_DIFF)
        result = find_line_in_diff(files, "src/foo.py", "  if result is None:  ")
        assert result is not None

    def test_empty_snippet(self):
        files = parse_diff(SAMPLE_DIFF)
        assert find_line_in_diff(files, "src/foo.py", "") is None
        assert find_line_in_diff(files, "src/foo.py", "   ") is None

    def test_empty_files(self):
        assert find_line_in_diff([], "src/foo.py", "code") is None

    def test_exact_equality_preferred_over_substring(self):
        """Short snippet should match the exact line, not a substring of a longer line."""
        diff = """\
diff --git a/src/foo.py b/src/foo.py
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,4 +1,4 @@
 x = 10
 x = 1
-old_line
+new_line
 other
"""
        files = parse_diff(diff)
        result = find_line_in_diff(files, "src/foo.py", "x = 1")
        assert result is not None
        # Should match line 2 (exact "x = 1"), not line 1 ("x = 10" which contains "x = 1")
        assert result[0] == 2

    def test_multiline_snippet_verifies_subsequent_lines(self):
        """Multi-line snippet must verify all lines, not just the first."""
        diff = """\
diff --git a/src/foo.py b/src/foo.py
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,7 +1,7 @@
 if data:
     wrong_body()
 if data:
-    old_body()
+    correct_body()
 if data:
     another_body()
 end
"""
        files = parse_diff(diff)
        # Should match the second "if data:" (line 3) because "correct_body()" follows it
        result = find_line_in_diff(files, "src/foo.py", "if data:\n    correct_body()")
        assert result is not None
        assert result[0] == 3
        assert result[1] == 4

    def test_multiline_skips_fuzzy_fallback(self):
        """Multi-line snippet skips fuzzy fallback to avoid false positives."""
        diff = """\
diff --git a/src/foo.py b/src/foo.py
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,3 @@
 if data:
-    old()
+    new()
 end
"""
        files = parse_diff(diff)
        # Multi-line snippet with mismatched subsequent lines: exact/normalized
        # won't match, and fuzzy fallback is skipped for multi-line snippets
        # so the AI fallback path can handle it instead of false-resolving
        result = find_line_in_diff(files, "src/foo.py", "if data:\n    nonexistent()")
        assert result is None

    def test_multiline_rejects_cross_hunk_gap(self):
        """Multi-line snippet should not match across non-consecutive hunks."""
        diff = """\
diff --git a/src/foo.py b/src/foo.py
--- a/src/foo.py
+++ b/src/foo.py
@@ -10,2 +10,2 @@
     if data:
-        old()
+        new()
@@ -50,2 +50,2 @@
         validate(result)
-        old_validate()
+        new_validate()
"""
        files = parse_diff(diff)
        # "new()" is at new line 11 (end of hunk 1)
        # "validate(result)" is at new line 50 (start of hunk 2)
        # These are not consecutive (11 -> 50), so multi-line should not match
        result = find_line_in_diff(files, "src/foo.py", "new()\n    validate(result)")
        assert result is None

    def test_multiline_no_match_at_all(self):
        """Multi-line snippet with no matching first line returns None."""
        diff = """\
diff --git a/src/foo.py b/src/foo.py
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,3 @@
 if data:
-    old()
+    new()
 end
"""
        files = parse_diff(diff)
        result = find_line_in_diff(
            files, "src/foo.py", "totally_different:\n    nonexistent()"
        )
        assert result is None


class TestResolveFindings:
    """Tests for resolve_findings()."""

    def test_resolve_with_existing_code(self):
        findings = [
            Finding(
                file="src/foo.py",
                line=None,
                severity="major",
                message="Missing null check",
                existing_code="if result is None:",
            ),
        ]
        resolved, unresolved = resolve_findings(findings, SAMPLE_DIFF)
        assert len(resolved) == 1
        assert len(unresolved) == 0
        assert resolved[0].line == 13
        assert resolved[0].start_line == 13

    def test_already_has_line(self):
        """Findings with line numbers should be kept as resolved."""
        findings = [
            Finding(
                file="src/foo.py",
                line=42,
                severity="minor",
                message="Style issue",
            ),
        ]
        resolved, _ = resolve_findings(findings, SAMPLE_DIFF)
        assert len(resolved) == 1
        assert resolved[0].line == 42

    def test_no_existing_code(self):
        """Findings without existing_code go to unresolved."""
        findings = [
            Finding(
                file="src/foo.py",
                line=None,
                severity="minor",
                message="File-level issue",
            ),
        ]
        resolved, unresolved = resolve_findings(findings, SAMPLE_DIFF)
        assert len(resolved) == 0
        assert len(unresolved) == 1

    def test_mixed_findings(self):
        """Mix of resolvable, pre-resolved, and unresolvable findings."""
        findings = [
            Finding(
                file="src/foo.py",
                line=None,
                severity="major",
                message="Found it",
                existing_code="if result is None:",
            ),
            Finding(
                file="src/foo.py",
                line=42,
                severity="minor",
                message="Already resolved",
            ),
            Finding(
                file="src/foo.py",
                line=None,
                severity="nit",
                message="Cannot find",
                existing_code="this code does not exist in diff",
            ),
        ]
        resolved, unresolved = resolve_findings(findings, SAMPLE_DIFF)
        assert len(resolved) == 2  # first (matched) + second (pre-resolved)
        assert len(unresolved) == 1  # third (no match)

    def test_empty_diff(self):
        findings = [
            Finding(
                file="src/foo.py",
                line=None,
                severity="major",
                message="test",
                existing_code="code",
            ),
        ]
        resolved, unresolved = resolve_findings(findings, "")
        assert len(resolved) == 0
        assert len(unresolved) == 1

    def test_empty_findings(self):
        resolved, unresolved = resolve_findings([], SAMPLE_DIFF)
        assert resolved == []
        assert unresolved == []


class TestResolveOverridesAILine:
    """Verify that snippet-based resolution overrides AI-guessed line numbers."""

    def test_overrides_line_1_with_snippet_match(self):
        """AI defaulted to line=1, but existing_code matches line 13."""
        finding = Finding(
            file="src/foo.py",
            line=1,
            severity="major",
            message="result is None check",
            existing_code="if result is None:",
        )
        resolved, unresolved = resolve_findings([finding], SAMPLE_DIFF)
        assert len(resolved) == 1
        assert len(unresolved) == 0
        assert resolved[0].line == 13
        assert resolved[0].start_line == 13
        assert resolved[0].end_line == 13

    def test_overrides_wrong_line_with_snippet_match(self):
        """AI guessed line=99, but snippet matches line 31."""
        finding = Finding(
            file="src/foo.py",
            line=99,
            severity="minor",
            message="new function",
            existing_code="def new_function():",
        )
        resolved, unresolved = resolve_findings([finding], SAMPLE_DIFF)
        assert len(resolved) == 1
        assert resolved[0].line == 31
        assert resolved[0].start_line == 31
        assert resolved[0].end_line == 31

    def test_keeps_ai_line_when_no_existing_code(self):
        """No existing_code — keep the AI's line number as-is."""
        finding = Finding(
            file="src/foo.py",
            line=5,
            severity="minor",
            message="general issue",
        )
        resolved, unresolved = resolve_findings([finding], SAMPLE_DIFF)
        assert len(resolved) == 1
        assert resolved[0].line == 5

    def test_keeps_ai_line_when_snippet_not_found(self):
        """existing_code doesn't match anything — keep AI's line."""
        finding = Finding(
            file="src/foo.py",
            line=3,
            severity="minor",
            message="nonexistent code",
            existing_code="this_code_does_not_exist()",
        )
        resolved, unresolved = resolve_findings([finding], SAMPLE_DIFF)
        assert len(resolved) == 1
        assert resolved[0].line == 3

    def test_unresolved_when_no_line_and_no_snippet(self):
        """No line, no existing_code — truly unresolved."""
        finding = Finding(
            file="src/foo.py",
            line=None,
            severity="minor",
            message="vague finding",
        )
        resolved, unresolved = resolve_findings([finding], SAMPLE_DIFF)
        assert len(resolved) == 0
        assert len(unresolved) == 1

    def test_unresolved_when_snippet_fails_and_no_ai_line(self):
        """existing_code present but not in diff, no AI line — unresolved."""
        finding = Finding(
            file="src/foo.py",
            line=None,
            severity="minor",
            message="unresolvable",
            existing_code="nonexistent_code()",
        )
        resolved, unresolved = resolve_findings([finding], SAMPLE_DIFF)
        assert len(resolved) == 0
        assert len(unresolved) == 1

    def test_overrides_with_multiline_snippet(self):
        """AI line is wrong, multi-line existing_code resolves correctly."""
        finding = Finding(
            file="src/foo.py",
            line=99,
            severity="major",
            message="multi-line check",
            existing_code="if result is None:\n            raise ValueError",
        )
        resolved, unresolved = resolve_findings([finding], SAMPLE_DIFF)
        assert len(resolved) == 1
        assert resolved[0].line == 13
        assert resolved[0].start_line == 13
        assert resolved[0].end_line == 14


class TestAnnotateDiffWithLineNumbers:
    """Tests for annotate_diff_with_line_numbers()."""

    def test_adds_line_numbers_to_additions(self):
        annotated = annotate_diff_with_line_numbers(SAMPLE_DIFF)
        lines = annotated.splitlines()
        # Find the first added line
        add_lines = [ln for ln in lines if "+" in ln and "if result is None" in ln]
        assert len(add_lines) == 1
        assert add_lines[0].strip().startswith("13")

    def test_preserves_headers(self):
        annotated = annotate_diff_with_line_numbers(SAMPLE_DIFF)
        assert "diff --git" in annotated
        assert "+++ b/src/foo.py" in annotated

    def test_annotate_content_resembling_headers(self):
        """Content lines resembling headers inside hunks should be annotated."""
        diff = """\
diff --git a/x.py b/x.py
--- a/x.py
+++ b/x.py
@@ -1,4 +1,4 @@
 normal line
--- old delimiter
+++ new delimiter
 end
"""
        annotated = annotate_diff_with_line_numbers(diff)
        lines = annotated.splitlines()
        # The "--- old delimiter" is a removal — should be blank-padded
        removal_lines = [ln for ln in lines if "old delimiter" in ln]
        assert len(removal_lines) == 1
        assert removal_lines[0].startswith("     ")  # blank-padded removal
        # The "+++ new delimiter" is an addition — should be line-numbered
        addition_lines = [ln for ln in lines if "new delimiter" in ln]
        assert len(addition_lines) == 1
        assert addition_lines[0].strip()[0].isdigit()

    def test_empty_diff(self):
        assert annotate_diff_with_line_numbers("") == ""
        assert annotate_diff_with_line_numbers("  ") == "  "


# Diff with two hunks for validation tests
VALIDATION_DIFF = """\
diff --git a/src/foo.py b/src/foo.py
--- a/src/foo.py
+++ b/src/foo.py
@@ -10,6 +10,7 @@ def process():
     data = fetch()
     if data:
         result = transform(data)
+        validate(result)
         return result
     return None

@@ -30,3 +31,5 @@ def helper():
     x = 1
     y = 2
     return x + y
+
+def new_func():
"""


class TestValidateFindingsInHunks:
    """Tests for validate_findings_in_hunks()."""

    def test_line_inside_hunk_kept(self):
        """Finding with line inside a hunk is kept as-is."""
        finding = Finding(file="src/foo.py", line=12, severity="major", message="issue")
        finding.start_line = 12
        finding.end_line = 12
        result = validate_findings_in_hunks([finding], VALIDATION_DIFF)
        assert result[0].line == 12
        assert result[0].start_line == 12
        assert result[0].end_line == 12

    def test_line_outside_all_hunks_cleared(self):
        """Finding with line outside all hunks gets line cleared to None."""
        finding = Finding(file="src/foo.py", line=50, severity="major", message="issue")
        finding.start_line = 50
        finding.end_line = 50
        result = validate_findings_in_hunks([finding], VALIDATION_DIFF)
        assert result[0].line is None
        assert result[0].start_line is None
        assert result[0].end_line is None

    def test_start_end_clamped_to_hunk(self):
        """Finding partially overlapping a hunk gets clamped to hunk boundaries."""
        finding = Finding(file="src/foo.py", line=8, severity="major", message="issue")
        finding.start_line = 8
        finding.end_line = 12
        result = validate_findings_in_hunks([finding], VALIDATION_DIFF)
        assert result[0].line == 10
        assert result[0].start_line == 10
        assert result[0].end_line == 12

    def test_end_clamped_past_hunk(self):
        """Finding extending past hunk end gets end_line clamped."""
        finding = Finding(file="src/foo.py", line=14, severity="major", message="issue")
        finding.start_line = 14
        finding.end_line = 20
        result = validate_findings_in_hunks([finding], VALIDATION_DIFF)
        assert result[0].start_line == 14
        assert result[0].end_line == 16

    def test_no_line_passthrough(self):
        """Finding with line=None passes through unchanged."""
        finding = Finding(
            file="src/foo.py", line=None, severity="minor", message="file-level"
        )
        result = validate_findings_in_hunks([finding], VALIDATION_DIFF)
        assert result[0].line is None

    def test_file_not_in_diff_passthrough(self):
        """Finding for a file not in the diff passes through unchanged."""
        finding = Finding(
            file="src/other.py", line=42, severity="major", message="issue"
        )
        result = validate_findings_in_hunks([finding], VALIDATION_DIFF)
        assert result[0].line == 42

    def test_multiple_hunks_second_hunk(self):
        """Finding in the second hunk is kept."""
        finding = Finding(file="src/foo.py", line=33, severity="major", message="issue")
        finding.start_line = 33
        finding.end_line = 33
        result = validate_findings_in_hunks([finding], VALIDATION_DIFF)
        assert result[0].line == 33

    def test_range_spanning_two_hunks_clamped(self):
        """Finding spanning from hunk 1 to hunk 2 should be clamped, not validated."""
        finding = Finding(file="src/foo.py", line=14, severity="major", message="issue")
        finding.start_line = 14
        finding.end_line = 33
        result = validate_findings_in_hunks([finding], VALIDATION_DIFF)
        assert result[0].start_line == 14
        assert result[0].end_line == 16

    def test_completely_outside_between_hunks_cleared(self):
        """Finding between two hunks (gap) gets cleared."""
        finding = Finding(file="src/foo.py", line=22, severity="major", message="issue")
        finding.start_line = 22
        finding.end_line = 22
        result = validate_findings_in_hunks([finding], VALIDATION_DIFF)
        assert result[0].line is None

    def test_suffix_path_matching(self):
        """Finding with short path matches diff path by suffix."""
        finding = Finding(file="foo.py", line=12, severity="major", message="issue")
        finding.start_line = 12
        finding.end_line = 12
        result = validate_findings_in_hunks([finding], VALIDATION_DIFF)
        assert result[0].line == 12

    def test_empty_diff_returns_findings_unchanged(self):
        """Empty diff returns findings unchanged."""
        finding = Finding(file="src/foo.py", line=42, severity="major", message="issue")
        result = validate_findings_in_hunks([finding], "")
        assert result[0].line == 42

    def test_empty_findings_returns_empty(self):
        """Empty findings list returns empty list."""
        result = validate_findings_in_hunks([], VALIDATION_DIFF)
        assert result == []

    def test_line_only_no_start_end(self):
        """Finding with line but no start_line/end_line uses line for both."""
        finding = Finding(file="src/foo.py", line=12, severity="major", message="issue")
        result = validate_findings_in_hunks([finding], VALIDATION_DIFF)
        assert result[0].line == 12

    def test_line_only_outside_cleared(self):
        """Finding with line only (no start/end) outside hunk gets cleared."""
        finding = Finding(file="src/foo.py", line=50, severity="major", message="issue")
        result = validate_findings_in_hunks([finding], VALIDATION_DIFF)
        assert result[0].line is None

    def test_stats_logging(self, capsys):
        """Verify logging of validation stats."""
        findings = [
            Finding(file="src/foo.py", line=12, severity="major", message="valid"),
            Finding(file="src/foo.py", line=50, severity="major", message="outside"),
        ]
        findings[0].start_line = 12
        findings[0].end_line = 12
        findings[1].start_line = 50
        findings[1].end_line = 50
        validate_findings_in_hunks(findings, VALIDATION_DIFF)
        captured = capsys.readouterr()
        assert "1 valid" in captured.out
        assert "1 cleared" in captured.out


class TestGetDiffLineRanges:
    """Tests for get_diff_line_ranges()."""

    def test_single_hunk(self):
        from cicaddy.delegation.line_resolver import get_diff_line_ranges

        ranges = get_diff_line_ranges(SAMPLE_DIFF)
        assert "src/foo.py" in ranges
        # Two hunks: first covers lines 10-18, second covers lines 27-32
        assert len(ranges["src/foo.py"]) == 2

    def test_multi_file(self):
        from cicaddy.delegation.line_resolver import get_diff_line_ranges

        ranges = get_diff_line_ranges(MULTI_FILE_DIFF)
        assert "src/foo.py" in ranges
        assert "src/bar.py" in ranges

    def test_empty_diff(self):
        from cicaddy.delegation.line_resolver import get_diff_line_ranges

        assert get_diff_line_ranges("") == {}
        assert get_diff_line_ranges("   ") == {}

    def test_hunk_range_boundaries(self):
        from cicaddy.delegation.line_resolver import get_diff_line_ranges

        ranges = get_diff_line_ranges(SAMPLE_DIFF)
        foo_ranges = ranges["src/foo.py"]
        # First hunk: new lines 10-18 (context + adds)
        first_start, first_end = foo_ranges[0]
        assert first_start == 10
        assert first_end == 18
        # Second hunk: new lines 27-32
        second_start, second_end = foo_ranges[1]
        assert second_start == 27
        assert second_end == 32

    def test_removal_only_hunk(self):
        from cicaddy.delegation.line_resolver import get_diff_line_ranges

        diff = """\
diff --git a/src/old.py b/src/old.py
--- a/src/old.py
+++ b/src/old.py
@@ -10,3 +10,0 @@
-removed1
-removed2
-removed3
"""
        ranges = get_diff_line_ranges(diff)
        assert ranges == {}


class TestIsLineInDiffRanges:
    """Tests for is_line_in_diff_ranges()."""

    def test_line_in_range(self):
        from cicaddy.delegation.line_resolver import (
            get_diff_line_ranges,
            is_line_in_diff_ranges,
        )

        ranges = get_diff_line_ranges(SAMPLE_DIFF)
        assert is_line_in_diff_ranges(13, "src/foo.py", ranges) is True

    def test_line_outside_range(self):
        from cicaddy.delegation.line_resolver import (
            get_diff_line_ranges,
            is_line_in_diff_ranges,
        )

        ranges = get_diff_line_ranges(SAMPLE_DIFF)
        # Line 5 is before any hunk
        assert is_line_in_diff_ranges(5, "src/foo.py", ranges) is False
        # Line 20 is between hunks
        assert is_line_in_diff_ranges(20, "src/foo.py", ranges) is False

    def test_line_at_boundaries(self):
        from cicaddy.delegation.line_resolver import (
            get_diff_line_ranges,
            is_line_in_diff_ranges,
        )

        ranges = get_diff_line_ranges(SAMPLE_DIFF)
        # Exact boundary of first hunk
        assert is_line_in_diff_ranges(10, "src/foo.py", ranges) is True
        assert is_line_in_diff_ranges(18, "src/foo.py", ranges) is True
        # Just outside
        assert is_line_in_diff_ranges(9, "src/foo.py", ranges) is False
        assert is_line_in_diff_ranges(19, "src/foo.py", ranges) is False

    def test_suffix_file_match(self):
        from cicaddy.delegation.line_resolver import (
            get_diff_line_ranges,
            is_line_in_diff_ranges,
        )

        ranges = get_diff_line_ranges(MULTI_FILE_DIFF)
        # "bar.py" should match "src/bar.py"
        assert is_line_in_diff_ranges(2, "bar.py", ranges) is True

    def test_unknown_file(self):
        from cicaddy.delegation.line_resolver import (
            get_diff_line_ranges,
            is_line_in_diff_ranges,
        )

        ranges = get_diff_line_ranges(SAMPLE_DIFF)
        assert is_line_in_diff_ranges(10, "unknown.py", ranges) is False

    def test_empty_ranges(self):
        from cicaddy.delegation.line_resolver import is_line_in_diff_ranges

        assert is_line_in_diff_ranges(10, "foo.py", {}) is False
