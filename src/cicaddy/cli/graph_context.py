"""CLI command for extracting code knowledge graph review context."""

import argparse
import json
import subprocess  # nosec B404
import sys
from pathlib import Path


def get_changed_files(base_ref: str, repo_path: str) -> list[str]:
    """Get changed files from git diff against base ref."""
    result = subprocess.run(  # nosec B603 B607
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        capture_output=True,
        text=True,
        cwd=repo_path,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    return [f for f in result.stdout.strip().split("\n") if f]


def cmd_graph_context(args: argparse.Namespace) -> int:
    """Handle the 'graph-context' command."""
    base_ref = args.base_ref
    max_depth = args.max_depth
    max_lines = args.max_lines
    output_file = args.output
    repo_path = args.repo

    try:
        changed_files = get_changed_files(base_ref, repo_path)
    except RuntimeError as e:
        print(f"Git diff failed: {e}", file=sys.stderr)
        _write_output(
            {"status": "error", "error": str(e), "changed_files": []}, output_file
        )
        return 1

    if not changed_files:
        result = {"status": "no_changes", "changed_files": []}
        _write_output(result, output_file)
        return 0

    try:
        from code_review_graph.graph import GraphStore
    except ImportError:
        print(
            "code-review-graph not installed. Install with: pip install cicaddy[graph]",
            file=sys.stderr,
        )
        result = {"status": "no_graph", "changed_files": changed_files}
        _write_output(result, output_file)
        return 0

    try:
        store = GraphStore(repo_path)
        context = store.get_review_context(
            changed_files,
            max_depth=max_depth,
            include_source=True,
            max_lines_per_file=max_lines,
        )
        _write_output(context, output_file)
        return 0
    except Exception as e:
        print(f"Graph analysis failed ({type(e).__name__}): {e}", file=sys.stderr)
        result = {"status": "error", "error": str(e), "changed_files": changed_files}
        _write_output(result, output_file)
        return 1


def _write_output(data: dict, output_file: str | None) -> None:
    """Write JSON output to file or stdout."""
    if output_file:
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Graph context written to {output_file}", file=sys.stderr)
    else:
        json.dump(data, sys.stdout, indent=2)
        print()
