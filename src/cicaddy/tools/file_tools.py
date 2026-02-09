"""Local file system tools for AI agents.

Provides glob file search and file reading capabilities
constrained to a working directory for security.
"""

import os
from pathlib import Path
from typing import Optional

from cicaddy.utils.logger import get_logger

from .decorator import tool
from .registry import ToolRegistry

logger = get_logger(__name__)

# Configuration constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_GLOB_RESULTS = 1000
DEFAULT_LINE_LIMIT = 2000
MAX_LINE_LENGTH = 2000

# Module-level working directory (can be configured)
_working_directory: Optional[Path] = None


def set_working_directory(path: Optional[str] = None) -> Path:
    """Set the working directory for file operations.

    Args:
        path: Path to use as working directory. If None, uses current
              working directory.

    Returns:
        The resolved working directory path.

    Raises:
        ValueError: If path doesn't exist or is not a directory.
    """
    global _working_directory

    if path:
        work_dir = Path(path).resolve()
    else:
        work_dir = Path.cwd()

    if not work_dir.exists():
        raise ValueError(f"Working directory does not exist: {work_dir}")
    if not work_dir.is_dir():
        raise ValueError(f"Working directory is not a directory: {work_dir}")

    _working_directory = work_dir
    logger.info(f"Local file tools working directory set to: {work_dir}")
    return work_dir


def get_working_directory() -> Path:
    """Get the current working directory for file operations.

    Returns:
        The working directory path. Defaults to cwd if not set.
    """
    global _working_directory
    if _working_directory is None:
        _working_directory = Path.cwd()
    return _working_directory


def _validate_path(file_path: str) -> Path:
    """Validate that a path is within the working directory.

    Args:
        file_path: Path to validate (can be relative or absolute).

    Returns:
        Resolved absolute path.

    Raises:
        ValueError: If path attempts to escape working directory.
    """
    work_dir = get_working_directory()

    # Handle both relative and absolute paths
    if os.path.isabs(file_path):
        full_path = Path(file_path).resolve()
    else:
        full_path = (work_dir / file_path).resolve()

    # Security check: ensure path is within working directory
    try:
        full_path.relative_to(work_dir)
    except ValueError:
        raise ValueError(
            f"Path traversal not allowed: '{file_path}' escapes working directory"
        )

    return full_path


@tool
def glob_files(
    pattern: str, path: str = "", max_results: int = MAX_GLOB_RESULTS
) -> str:
    """Search for files matching a glob pattern in the working directory.

    Supports patterns like '*.py', '**/*.json', 'src/**/*.ts'.
    Returns a list of matching file paths sorted by modification time (newest first).

    Args:
        pattern: Glob pattern to match files. Examples: '*.py' (Python files
            in current dir), '**/*.json' (all JSON files recursively),
            'src/**/*.ts' (TypeScript files under src/).
        path: Optional subdirectory to search in, relative to working directory.
            If not provided, searches from the working directory root.
        max_results: Maximum number of results to return. Default: 1000.
            Results are sorted by modification time (newest first).

    Returns:
        List of matching file paths, one per line.
    """
    try:
        # Validate and resolve search path
        if path:
            base_dir = _validate_path(path)
            if not base_dir.is_dir():
                return f"Error: path is not a directory: {path}"
        else:
            base_dir = get_working_directory()

        # Find matching files
        matches = []

        logger.debug(f"Glob searching in {base_dir} with pattern: {pattern}")

        # Use pathlib glob
        work_dir = get_working_directory()
        for file_path in base_dir.glob(pattern):
            if file_path.is_file():
                # Store as relative path with modification time for sorting
                try:
                    rel_path = file_path.relative_to(work_dir)
                    mtime = file_path.stat().st_mtime
                except (OSError, ValueError):
                    continue
                matches.append((str(rel_path), mtime))

        # Sort by modification time (newest first) and limit results
        matches.sort(key=lambda x: x[1], reverse=True)
        matches = matches[:max_results]

        # Extract just the paths
        result_paths = [m[0] for m in matches]

        if not result_paths:
            return f"No files found matching pattern: {pattern}"
        else:
            header = f"Found {len(result_paths)} file(s):"
            return header + "\n" + "\n".join(result_paths)

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Glob files error: {e}", exc_info=True)
        return f"Error: {e}"


@tool
def read_file(file_path: str, offset: int = 1, limit: int = DEFAULT_LINE_LIMIT) -> str:
    """Read the contents of a file in the working directory.

    Returns file contents with line numbers (similar to 'cat -n' output).
    Supports reading specific sections of large files using offset and limit.

    Args:
        file_path: Path to the file to read, relative to the working directory.
            Must be within the working directory (no path traversal allowed).
        offset: Line number to start reading from (1-indexed). Default: 1.
        limit: Maximum number of lines to read. Default: 2000.
            Use this for reading specific sections of large files.

    Returns:
        File contents with line numbers, or an error message.
    """
    try:
        # Validate and resolve path
        full_path = _validate_path(file_path)

        if not full_path.exists():
            return f"Error: file not found: {file_path}"

        if not full_path.is_file():
            return f"Error: not a file: {file_path}"

        # Check file size
        file_size = full_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            return (
                f"Error: file too large ({file_size:,} bytes, "
                f"max {MAX_FILE_SIZE:,} bytes): {file_path}"
            )

        # Ensure valid offset
        offset = max(1, offset)

        # Read file line-by-line to avoid loading the entire file into memory
        try:
            numbered_lines = []
            total_lines = 0
            start_idx = offset - 1  # Convert to 0-indexed
            end_idx = start_idx + limit

            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f):
                    total_lines = line_num + 1
                    if line_num < start_idx:
                        continue
                    if line_num >= end_idx:
                        # Keep counting total lines but stop collecting
                        continue
                    # Truncate long lines
                    line_content = line.rstrip()
                    if len(line_content) > MAX_LINE_LENGTH:
                        line_content = line_content[:MAX_LINE_LENGTH] + "..."
                    # Right-align line numbers, followed by tab (1-indexed)
                    numbered_lines.append(f"{line_num + 1:6d}\t{line_content}")

        except UnicodeDecodeError:
            return f"Error: unable to read file content: {file_path}"

        if total_lines == 0:
            return f"File is empty: {file_path}"

        if not numbered_lines:
            return (
                f"No lines in range (offset={offset}, limit={limit}). "
                f"File has {total_lines} total lines: {file_path}"
            )

        result_text = "\n".join(numbered_lines)

        # Add metadata header
        actual_end = min(offset + len(numbered_lines) - 1, total_lines)
        if offset > 1 or end_idx < total_lines:
            header = (
                f"# File: {file_path} (lines {offset}-{actual_end} of {total_lines})\n"
            )
        else:
            header = f"# File: {file_path} ({total_lines} lines)\n"

        return header + result_text

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Read file error: {e}", exc_info=True)
        return f"Error: {e}"


def create_local_file_registry(working_directory: Optional[str] = None) -> ToolRegistry:
    """Create a tool registry with local file tools.

    Args:
        working_directory: Optional path to use as working directory.
            If None, uses current working directory.

    Returns:
        ToolRegistry with glob_files and read_file tools registered.
    """
    # Set working directory
    set_working_directory(working_directory)

    # Create and populate registry
    registry = ToolRegistry(server_name="local")
    registry.register(glob_files)
    registry.register(read_file)

    logger.info(f"Created local file registry with {len(registry)} tools")
    return registry
