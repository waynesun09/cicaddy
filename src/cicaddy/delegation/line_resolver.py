"""Deterministic line number resolution for delegation findings.

Parses unified diffs and maps code snippets to precise line numbers,
enabling accurate inline comment placement on PRs/MRs.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

from cicaddy.utils.logger import get_logger

if TYPE_CHECKING:
    from cicaddy.delegation.summarizer import Finding

logger = get_logger(__name__)

_RE_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class DiffLine:
    """A single line from a unified diff hunk."""

    type: str  # "context", "add", "remove"
    content: str  # line content without the leading +/-/space
    old_lineno: Optional[int] = None  # line number in old file
    new_lineno: Optional[int] = None  # line number in new file


@dataclass
class DiffHunk:
    """A parsed hunk from a unified diff."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[DiffLine] = field(default_factory=list)


@dataclass
class DiffFile:
    """A parsed file from a unified diff."""

    path: str
    hunks: List[DiffHunk] = field(default_factory=list)


def parse_diff(diff: str) -> List[DiffFile]:
    """Parse a unified diff string into structured DiffFile objects.

    Handles standard unified diff format with ---/+++ file headers
    and @@ hunk headers.
    """
    if not diff or not diff.strip():
        return []

    files: List[DiffFile] = []
    current_file: Optional[DiffFile] = None
    current_hunk: Optional[DiffHunk] = None
    old_lineno = 0
    new_lineno = 0

    for raw_line in diff.splitlines():
        # New file header (---/+++ pair or diff --git)
        if raw_line.startswith("+++ b/"):
            path = raw_line[6:]
            current_file = DiffFile(path=path)
            files.append(current_file)
            current_hunk = None
            continue

        if raw_line.startswith("--- ") or raw_line.startswith("diff --git"):
            continue

        # Hunk header
        match = _RE_HUNK_HEADER.match(raw_line)
        if match and current_file is not None:
            old_start = int(match.group(1))
            old_count = int(match.group(2)) if match.group(2) else 1
            new_start = int(match.group(3))
            new_count = int(match.group(4)) if match.group(4) else 1
            current_hunk = DiffHunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
            )
            current_file.hunks.append(current_hunk)
            old_lineno = old_start
            new_lineno = new_start
            continue

        if current_hunk is None:
            continue

        # Diff content lines
        if raw_line.startswith("+"):
            content = raw_line[1:]
            current_hunk.lines.append(
                DiffLine(type="add", content=content, new_lineno=new_lineno)
            )
            new_lineno += 1
        elif raw_line.startswith("-"):
            content = raw_line[1:]
            current_hunk.lines.append(
                DiffLine(type="remove", content=content, old_lineno=old_lineno)
            )
            old_lineno += 1
        elif raw_line.startswith(" ") or raw_line == "":
            content = raw_line[1:] if raw_line.startswith(" ") else ""
            current_hunk.lines.append(
                DiffLine(
                    type="context",
                    content=content,
                    old_lineno=old_lineno,
                    new_lineno=new_lineno,
                )
            )
            old_lineno += 1
            new_lineno += 1
        # Skip \ No newline at end of file and other metadata

    return files


def _normalize(text: str) -> str:
    """Normalize whitespace for fuzzy matching."""
    return " ".join(text.split())


def find_line_in_diff(
    diff_files: List[DiffFile],
    file_path: str,
    code_snippet: str,
) -> Optional[Tuple[int, int]]:
    """Find the line range for a code snippet in a parsed diff.

    Searches the new-side lines (context + additions) for the snippet.
    Returns (start_line, end_line) in the new file, or None if not found.

    Matching strategy:
    1. Exact substring match on new-side lines
    2. Normalized whitespace match
    3. Fuzzy match via difflib (93% cutoff)
    """
    if not code_snippet or not code_snippet.strip():
        return None

    # Find matching file (try exact path, then basename match)
    target_file = None
    for df in diff_files:
        if df.path == file_path:
            target_file = df
            break
    if target_file is None:
        # Try basename/suffix match
        for df in diff_files:
            if df.path.endswith(file_path) or file_path.endswith(df.path):
                target_file = df
                break
    if target_file is None:
        return None

    # Collect new-side lines with their line numbers
    new_lines: List[Tuple[int, str]] = []
    for hunk in target_file.hunks:
        for dl in hunk.lines:
            if dl.type in ("context", "add") and dl.new_lineno is not None:
                new_lines.append((dl.new_lineno, dl.content))

    if not new_lines:
        return None

    snippet_lines = [ln for ln in code_snippet.splitlines() if ln.strip()]
    if not snippet_lines:
        return None

    # Strategy 1: Exact substring match on first snippet line
    first_line = snippet_lines[0].strip()
    matches = _find_exact_matches(new_lines, first_line, snippet_lines)
    if matches:
        return matches

    # Strategy 2: Normalized whitespace match
    first_norm = _normalize(first_line)
    matches = _find_normalized_matches(new_lines, first_norm, snippet_lines)
    if matches:
        return matches

    # Strategy 3: Fuzzy match (single line)
    return _find_fuzzy_match(new_lines, first_line)


def _verify_subsequent_lines(
    new_lines: List[Tuple[int, str]],
    start_idx: int,
    snippet_lines: List[str],
) -> bool:
    """Verify that subsequent snippet lines match subsequent new-side lines."""
    if len(snippet_lines) <= 1:
        return True
    for offset in range(1, len(snippet_lines)):
        next_idx = start_idx + offset
        if next_idx >= len(new_lines):
            return False
        _, next_content = new_lines[next_idx]
        expected = snippet_lines[offset].strip()
        if expected not in next_content and next_content.strip() != expected:
            return False
    return True


def _find_exact_matches(
    new_lines: List[Tuple[int, str]],
    first_line: str,
    snippet_lines: List[str],
) -> Optional[Tuple[int, int]]:
    """Find exact match for snippet in new-side lines.

    Prefers exact equality (stripped) over substring containment.
    Verifies subsequent lines for multi-line snippets.
    """
    # Pass 1: exact equality (stripped)
    for idx, (lineno, content) in enumerate(new_lines):
        if content.strip() == first_line:
            if _verify_subsequent_lines(new_lines, idx, snippet_lines):
                end = lineno + len(snippet_lines) - 1
                return (lineno, end)

    # Pass 2: substring containment
    for idx, (lineno, content) in enumerate(new_lines):
        if first_line in content:
            if _verify_subsequent_lines(new_lines, idx, snippet_lines):
                end = lineno + len(snippet_lines) - 1
                return (lineno, end)

    return None


def _find_normalized_matches(
    new_lines: List[Tuple[int, str]],
    first_norm: str,
    snippet_lines: List[str],
) -> Optional[Tuple[int, int]]:
    """Find normalized whitespace match for snippet in new-side lines.

    Verifies subsequent lines for multi-line snippets.
    """
    for idx, (lineno, content) in enumerate(new_lines):
        if _normalize(content) == first_norm:
            if _verify_subsequent_lines(new_lines, idx, snippet_lines):
                end = lineno + len(snippet_lines) - 1
                return (lineno, end)
    return None


def _find_fuzzy_match(
    new_lines: List[Tuple[int, str]],
    target: str,
) -> Optional[Tuple[int, int]]:
    """Find fuzzy match using difflib with 93% cutoff."""
    contents = [content.strip() for _, content in new_lines]
    matches = difflib.get_close_matches(target, contents, n=1, cutoff=0.93)
    if matches:
        for lineno, content in new_lines:
            if content.strip() == matches[0]:
                return (lineno, lineno)
    return None


def resolve_findings(
    findings: List["Finding"],
    diff: str,
) -> Tuple[List["Finding"], List["Finding"]]:
    """Resolve line numbers for findings using deterministic diff search.

    For each finding with ``existing_code``, searches the diff for a
    matching code snippet and sets ``line`` to the resolved start line.

    Args:
        findings: List of Finding objects, some with ``existing_code``.
        diff: Raw unified diff string.

    Returns:
        Tuple of (resolved, unresolved) Finding lists.
    """
    if not diff or not findings:
        return [], list(findings)

    diff_files = parse_diff(diff)
    if not diff_files:
        return [], list(findings)

    resolved: List["Finding"] = []
    unresolved: List["Finding"] = []

    for finding in findings:
        # Already has a line number — keep it
        if finding.line is not None:
            resolved.append(finding)
            continue

        existing_code = getattr(finding, "existing_code", None)
        if not existing_code:
            unresolved.append(finding)
            continue

        result = find_line_in_diff(diff_files, finding.file, existing_code)
        if result is not None:
            start_line, end_line = result
            finding.line = start_line
            finding.start_line = start_line
            finding.end_line = end_line
            resolved.append(finding)
            logger.debug(
                f"Resolved finding in {finding.file} to lines {start_line}-{end_line}"
            )
        else:
            unresolved.append(finding)
            logger.debug(
                f"Could not resolve finding in {finding.file}: "
                f"snippet not found in diff"
            )

    logger.info(
        f"Line resolution: {len(resolved)} resolved, "
        f"{len(unresolved)} unresolved out of {len(findings)} findings"
    )
    return resolved, unresolved


def annotate_diff_with_line_numbers(diff: str) -> str:
    """Add line number annotations to diff for AI line-mapping fallback.

    Prefixes each diff content line with its new-file line number
    (for + and context lines) or old-file line number (for - lines).
    """
    if not diff or not diff.strip():
        return diff

    output_lines: List[str] = []
    new_lineno = 0
    old_lineno = 0
    in_hunk = False

    for raw_line in diff.splitlines():
        if raw_line.startswith("diff --git") or raw_line.startswith("---"):
            output_lines.append(raw_line)
            in_hunk = False
            continue

        if raw_line.startswith("+++ "):
            output_lines.append(raw_line)
            in_hunk = False
            continue

        match = _RE_HUNK_HEADER.match(raw_line)
        if match:
            old_lineno = int(match.group(1))
            new_lineno = int(match.group(3))
            output_lines.append(raw_line)
            in_hunk = True
            continue

        if not in_hunk:
            output_lines.append(raw_line)
            continue

        if raw_line.startswith("+"):
            output_lines.append(f"{new_lineno:>4} {raw_line}")
            new_lineno += 1
        elif raw_line.startswith("-"):
            output_lines.append(f"     {raw_line}")
            old_lineno += 1
        elif raw_line.startswith(" ") or raw_line == "":
            output_lines.append(f"{new_lineno:>4} {raw_line}")
            old_lineno += 1
            new_lineno += 1
        else:
            output_lines.append(raw_line)

    return "\n".join(output_lines)
