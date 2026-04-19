"""Tests for cicaddy.delegation.line_resolver module."""

from __future__ import annotations

from cicaddy.delegation.line_resolver import (
    annotate_diff_with_line_numbers,
    find_line_in_diff,
    parse_diff,
    resolve_findings,
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

    def test_multiline_falls_back_to_fuzzy_single_line(self):
        """Multi-line snippet with mismatched subsequent lines falls to fuzzy first-line match."""
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
        # Exact/normalized won't match (subsequent line differs),
        # but fuzzy fallback matches "if data:" as single-line
        result = find_line_in_diff(files, "src/foo.py", "if data:\n    nonexistent()")
        assert result is not None
        assert result[0] == result[1]  # single-line match from fuzzy

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

    def test_empty_diff(self):
        assert annotate_diff_with_line_numbers("") == ""
        assert annotate_diff_with_line_numbers("  ") == "  "
