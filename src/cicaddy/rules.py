"""Agent rules discovery and loading from workspace rule files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from cicaddy.utils.logger import get_logger

if TYPE_CHECKING:
    from cicaddy.tools.scanner import ToolScanner

logger = get_logger(__name__)

# Rule files in priority order (generic first, then provider-specific)
GENERIC_RULE_FILES = ("AGENT.md", "AGENTS.md")

# Provider name -> rule file name mapping
PROVIDER_RULE_FILES = {
    "gemini": "GEMINI.md",
    "gemini-vertex": "GEMINI.md",
    "claude": "CLAUDE.md",
    "anthropic-vertex": "CLAUDE.md",
    "openai": "COPILOT.md",
}

# All known rule file names for discovery
ALL_RULE_FILES = (*GENERIC_RULE_FILES, *PROVIDER_RULE_FILES.values())


def load_agent_rules(
    workspace_path: Path,
    provider: Optional[str] = None,
    scanner: Optional["ToolScanner"] = None,
    scan_mode: str = "disabled",
) -> str:
    """Load agent rules from workspace rule files.

    Discovers and reads agent rule files from the workspace directory.
    Generic rule files (AGENT.md, AGENTS.md) are always loaded.
    Provider-specific files are loaded based on the provider parameter.

    Args:
        workspace_path: Path to the project workspace directory.
        provider: AI provider name (e.g., "gemini", "claude") for
            loading provider-specific rule files.
        scanner: Optional ToolScanner for scanning external rule files.
        scan_mode: Scanning mode (disabled/audit/enforce) if scanner provided.

    Returns:
        Concatenated rule content wrapped in XML tags, or empty string if none found.

    Raises:
        ValueError: If external rule file fails security scan in enforce mode.
    """
    if not workspace_path.is_dir():
        return ""

    sections: list[str] = []

    # Load generic rule files (always)
    for filename in GENERIC_RULE_FILES:
        content = _read_rule_file(
            workspace_path / filename,
            workspace_root=workspace_path,
            scanner=scanner,
            scan_mode=scan_mode,
        )
        if content:
            sections.append(
                f'<agent_rules source="{filename}">\n{content}\n</agent_rules>'
            )

    # Load provider-specific rule file
    if provider:
        provider_file = PROVIDER_RULE_FILES.get(provider.lower())
        if provider_file:
            content = _read_rule_file(
                workspace_path / provider_file,
                workspace_root=workspace_path,
                scanner=scanner,
                scan_mode=scan_mode,
            )
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


def _read_rule_file(
    path: Path,
    workspace_root: Optional[Path] = None,
    scanner: Optional["ToolScanner"] = None,
    scan_mode: str = "disabled",
) -> str:
    """Read and optionally scan a rule file.

    Args:
        path: Path to the rule file.
        workspace_root: Root of the workspace for provenance detection.
        scanner: Optional ToolScanner for prompt injection detection.
        scan_mode: Scanning mode (disabled/audit/enforce).

    Returns:
        File content stripped of leading/trailing whitespace, or empty string.

    Raises:
        ValueError: If file fails security scan in enforce mode.
    """
    if not path.is_file():
        return ""

    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError:
        logger.warning(f"Failed to read rule file: {path}")
        return ""

    # Scan if scanner provided and mode not disabled
    if scanner and scan_mode != "disabled":
        import asyncio

        from cicaddy.security.provenance import (
            get_provenance_label,
            is_external_source,
        )

        # Determine provenance
        is_external = is_external_source(path, workspace_root)
        provenance = get_provenance_label(path, workspace_root)

        # Only scan external files (local files trusted via code review)
        if is_external:
            logger.debug(f"Scanning external rule file ({provenance}): {path}")

            # Run scan
            scan_result = asyncio.run(
                scanner.scan_tool_result(
                    content=content,
                    tool_name=f"rule:{path.name}",
                    source=provenance,
                )
            )

            # Handle scan result
            if scan_result.blocked:
                # Blocked in enforce mode
                error_msg = (
                    f"Rule file {path} blocked by security scanner "
                    f"(risk: {scan_result.risk_score:.2f}): "
                    f"{', '.join(scan_result.findings)}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            elif not scan_result.is_clean:
                # Flagged in audit mode
                logger.warning(
                    f"Rule file {path} flagged by security scanner "
                    f"(risk: {scan_result.risk_score:.2f}): "
                    f"{', '.join(scan_result.findings)}"
                )
        else:
            logger.debug(f"Skipping scan for local rule file ({provenance}): {path}")

    return content
