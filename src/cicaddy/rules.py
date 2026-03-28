"""Agent rules discovery and loading from workspace rule files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)

# Rule files in priority order (generic first, then provider-specific)
GENERIC_RULE_FILES = ("AGENT.md", "AGENTS.md")

# Provider name -> rule file name mapping
PROVIDER_RULE_FILES = {
    "gemini": "GEMINI.md",
    "claude": "CLAUDE.md",
    "openai": "COPILOT.md",
}

# All known rule file names for discovery
ALL_RULE_FILES = (*GENERIC_RULE_FILES, *PROVIDER_RULE_FILES.values())


def load_agent_rules(
    workspace_path: Path,
    provider: Optional[str] = None,
) -> str:
    """Load agent rules from workspace rule files.

    Discovers and reads agent rule files from the workspace directory.
    Generic rule files (AGENT.md, AGENTS.md) are always loaded.
    Provider-specific files are loaded based on the provider parameter.

    Args:
        workspace_path: Path to the project workspace directory.
        provider: AI provider name (e.g., "gemini", "claude") for
            loading provider-specific rule files.

    Returns:
        Concatenated rule content wrapped in XML tags, or empty string if none found.
    """
    if not workspace_path.is_dir():
        return ""

    sections: list[str] = []

    # Load generic rule files (always)
    for filename in GENERIC_RULE_FILES:
        content = _read_rule_file(workspace_path / filename)
        if content:
            sections.append(
                f'<agent_rules source="{filename}">\n{content}\n</agent_rules>'
            )

    # Load provider-specific rule file
    if provider:
        provider_file = PROVIDER_RULE_FILES.get(provider.lower())
        if provider_file:
            content = _read_rule_file(workspace_path / provider_file)
            if content:
                sections.append(
                    f'<agent_rules source="{provider_file}" provider="{provider}">\n{content}\n</agent_rules>'
                )

    if not sections:
        return ""

    result = "\n\n".join(sections)
    logger.info(f"Loaded agent rules from {len(sections)} file(s) in {workspace_path}")
    return result


def discover_rule_files(workspace_path: Path) -> list[Path]:
    """Discover all agent rule files in the workspace.

    Args:
        workspace_path: Path to the project workspace directory.

    Returns:
        List of paths to discovered rule files.
    """
    if not workspace_path.is_dir():
        return []

    found = []
    for filename in ALL_RULE_FILES:
        path = workspace_path / filename
        if path.is_file():
            found.append(path)
    return found


def _read_rule_file(path: Path) -> str:
    """Read a rule file and return its content.

    Args:
        path: Path to the rule file.

    Returns:
        File content stripped of leading/trailing whitespace, or empty string.
    """
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        logger.warning(f"Failed to read rule file: {path}")
        return ""
