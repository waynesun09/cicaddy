"""Provenance detection for agent content sources.

This module determines whether files are from trusted local sources
or external/untrusted sources (submodules, dependencies, downloads).

Used by scanning system to apply appropriate security policies:
- Local project files: Skip or audit scan (trust code review)
- External files: Enforce scan (supply chain risk)
"""

import subprocess  # nosec B404 - needed for git operations
from pathlib import Path
from typing import Optional

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)


def is_external_source(
    path: Path,
    workspace_root: Optional[Path] = None,
) -> bool:
    """Check if a file is from an external/untrusted source.

    A file is considered external if it:
    1. Is in a git submodule
    2. Is not tracked by git (downloaded, generated)
    3. Is outside the workspace root

    Args:
        path: Path to the file to check.
        workspace_root: Root of the workspace. If None, attempts to find
            git root from path.

    Returns:
        True if file is from external source, False if local/trusted.

    Examples:
        >>> # Local project file
        >>> is_external_source(Path("/repo/AGENT.md"))
        False

        >>> # Submodule file
        >>> is_external_source(Path("/repo/vendor/lib/AGENT.md"))
        True

        >>> # Downloaded file (not in git)
        >>> is_external_source(Path("/repo/downloads/skill.md"))
        True
    """
    if not path.exists():
        return False  # Non-existent files are safe (can't be read)

    # Find workspace root if not provided
    if workspace_root is None:
        workspace_root = _find_git_root(path)
        if workspace_root is None:
            # Not in a git repository - treat as external
            logger.debug(f"File not in git repository: {path}")
            return True

    # Check if path is outside workspace
    try:
        path.resolve().relative_to(workspace_root.resolve())
    except ValueError:
        logger.debug(f"File outside workspace: {path}")
        return True

    # Check if file is in a git submodule
    if _is_in_submodule(path, workspace_root):
        logger.debug(f"File in git submodule: {path}")
        return True

    # Check if file is tracked by git
    if not _is_git_tracked(path, workspace_root):
        logger.debug(f"File not tracked by git: {path}")
        return True

    # File is local and tracked
    return False


def _find_git_root(path: Path) -> Optional[Path]:
    """Find the git repository root containing the given path.

    Args:
        path: Path to start search from.

    Returns:
        Path to git root, or None if not in a git repository.
    """
    try:
        result = subprocess.run(  # nosec B603, B607 - git command with hardcoded args
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path if path.is_dir() else path.parent,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def _is_in_submodule(path: Path, workspace_root: Path) -> bool:
    """Check if path is in a git submodule.

    Uses .gitmodules file parsing for performance (no subprocess).

    Args:
        path: Path to check.
        workspace_root: Root of the git repository.

    Returns:
        True if path is in a submodule.
    """
    gitmodules_path = workspace_root / ".gitmodules"
    if not gitmodules_path.is_file():
        return False

    try:
        content = gitmodules_path.read_text(encoding="utf-8")
    except OSError:
        return False

    # Parse .gitmodules for submodule paths
    submodule_paths = []
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("path ="):
            submodule_path = line.split("=", 1)[1].strip()
            submodule_paths.append(workspace_root / submodule_path)

    # Check if path is under any submodule
    path_resolved = path.resolve()
    for submodule_path in submodule_paths:
        try:
            submodule_resolved = submodule_path.resolve()
            path_resolved.relative_to(submodule_resolved)
            return True
        except (ValueError, OSError):
            continue

    return False


def _is_git_tracked(path: Path, workspace_root: Path) -> bool:
    """Check if a file is tracked by git.

    Args:
        path: Path to check.
        workspace_root: Root of the git repository.

    Returns:
        True if file is tracked by git.
    """
    try:
        # Make path relative to workspace for git ls-files
        rel_path = path.resolve().relative_to(workspace_root.resolve())

        result = subprocess.run(  # nosec B603, B607 - git command with hardcoded args
            ["git", "ls-files", "--error-unmatch", str(rel_path)],
            cwd=workspace_root,
            capture_output=True,
            timeout=2,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return False


def get_provenance_label(path: Path, workspace_root: Optional[Path] = None) -> str:
    """Get a human-readable provenance label for a file.

    Args:
        path: Path to check.
        workspace_root: Root of the workspace.

    Returns:
        One of: "local", "submodule", "untracked", "external", "unknown"
    """
    if not path.exists():
        return "unknown"

    if workspace_root is None:
        workspace_root = _find_git_root(path)
        if workspace_root is None:
            return "external"

    # Check order matters (most specific first)
    if _is_in_submodule(path, workspace_root):
        return "submodule"

    if not _is_git_tracked(path, workspace_root):
        return "untracked"

    try:
        path.resolve().relative_to(workspace_root.resolve())
        return "local"
    except ValueError:
        return "external"
