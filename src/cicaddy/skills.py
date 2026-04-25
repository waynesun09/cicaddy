"""Skill discovery and loading from workspace and global skill directories."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import yaml

from cicaddy.utils.logger import get_logger

if TYPE_CHECKING:
    from cicaddy.tools.scanner import ToolScanner

logger = get_logger(__name__)

# Cross-tool standard skill directory (agentskills.io)
CROSS_TOOL_SKILLS_DIR = ".agents/skills"

# Provider-specific skill directories
PROVIDER_SKILLS_DIRS: dict[str, str] = {
    "claude": ".claude/skills",
    "gemini": ".gemini/skills",
    "openai": ".github/skills",
}

SKILL_FILE_NAME = "SKILL.md"
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class SkillMetadata:
    """Discovered skill metadata."""

    name: str
    description: str
    location: Path
    source: str  # "project", "global", or "bundled"
    metadata: dict[str, Any] = field(default_factory=dict)

    def body(self) -> str:
        """Read the skill body content (everything after frontmatter)."""
        try:
            content = self.location.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
        return _strip_frontmatter(content)


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from content, returning the body.

    Uses line-by-line scanning instead of regex to avoid ReDoS risk.
    """
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return content.strip()
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "".join(lines[idx + 1 :]).strip()
    return content.strip()


def discover_skills(
    workspace_path: Path,
    provider: Optional[str] = None,
    scanner: Optional["ToolScanner"] = None,
    scan_mode: str = "disabled",
) -> list[SkillMetadata]:
    """Discover skills from project, provider-specific, and global directories.

    Precedence order (first match wins for same skill name):
    1. Provider-specific project dir (e.g., .claude/skills/, .gemini/skills/)
    2. Cross-tool project dir (.agents/skills/)
    3. Global user dir (~/.agents/skills/)

    Args:
        workspace_path: Path to the project workspace directory.
        provider: AI provider name (e.g., "gemini", "claude") for
            scanning provider-specific skill directories.
        scanner: Optional ToolScanner for scanning external skill files.
        scan_mode: Scanning mode (disabled/audit/enforce) if scanner provided.

    Returns:
        Sorted list of discovered skills. Skills that fail security scan
        (in enforce mode) are excluded.
    """
    skills_by_name: dict[str, SkillMetadata] = {}
    for root, source in _iter_skill_roots(workspace_path, provider=provider):
        if not root.is_dir():
            continue
        for skill_dir in sorted(root.iterdir()):
            if not skill_dir.is_dir():
                continue
            metadata = _read_skill(
                skill_dir,
                source=source,
                workspace_root=workspace_path,
                scanner=scanner,
                scan_mode=scan_mode,
            )
            if metadata is None:
                continue
            key = metadata.name.casefold()
            if key not in skills_by_name:
                skills_by_name[key] = metadata

    return sorted(skills_by_name.values(), key=lambda item: item.name.casefold())


def render_skills_prompt(skills: list[SkillMetadata]) -> str:
    """Render skills as a prompt section for AI context.

    Args:
        skills: List of discovered skills.

    Returns:
        Formatted skills prompt section.
    """
    if not skills:
        return ""
    lines = ["## Available Skills\n"]
    for skill in skills:
        lines.append(f"- **{skill.name}**: {skill.description}")
        body = skill.body()
        if body:
            # Indent body content
            indented = "\n".join(f"  {line}" for line in body.splitlines())
            lines.append(indented)
    return "\n".join(lines)


def _scan_skill_content(
    skill_dir: Path,
    skill_file: Path,
    content: str,
    source: str,
    workspace_root: Optional[Path],
    scanner: "ToolScanner",
) -> bool:
    """Scan skill content for prompt injection.

    Args:
        skill_dir: Path to the skill directory.
        skill_file: Path to the skill file.
        content: Full skill content.
        source: Source type ("project", "global", or "bundled").
        workspace_root: Root of the workspace for provenance detection.
        scanner: ToolScanner for prompt injection detection.

    Returns:
        True if skill should be included, False if blocked.
    """
    import asyncio

    from cicaddy.security.provenance import get_provenance_label, is_external_source

    skill_body = _strip_frontmatter(content)

    # Bundled skills are first-party trusted content — skip scanning
    if source == "bundled":
        logger.debug(f"Skipping scan for bundled skill: {skill_file}")
        return True

    # Determine if skill is from external source
    is_external = source == "global" or (
        source == "project"
        and workspace_root
        and is_external_source(skill_file, workspace_root)
    )

    # Get provenance label
    if source == "project" and workspace_root:
        provenance = get_provenance_label(skill_file, workspace_root)
    else:
        provenance = "global"

    # Scan external skills
    if is_external:
        logger.debug(f"Scanning external skill ({provenance}): {skill_file}")

        scan_result = asyncio.run(
            scanner.scan_tool_result(
                content=skill_body,
                tool_name=f"skill:{skill_dir.name}",
                source=provenance,
            )
        )

        if scan_result.blocked:
            logger.error(
                f"Skill {skill_dir.name} blocked by security scanner "
                f"(risk: {scan_result.risk_score:.2f}): "
                f"{', '.join(scan_result.findings)}"
            )
            return False
        elif not scan_result.is_clean:
            logger.warning(
                f"Skill {skill_dir.name} flagged by security scanner "
                f"(risk: {scan_result.risk_score:.2f}): "
                f"{', '.join(scan_result.findings)}"
            )
    else:
        logger.debug(f"Skipping scan for local skill ({provenance}): {skill_file}")

    return True


def _read_skill(
    skill_dir: Path,
    *,
    source: str,
    workspace_root: Optional[Path] = None,
    scanner: Optional["ToolScanner"] = None,
    scan_mode: str = "disabled",
) -> Optional[SkillMetadata]:
    """Read, validate, and optionally scan a single skill directory.

    Args:
        skill_dir: Path to the skill directory.
        source: Source type ("project", "global", or "bundled").
        workspace_root: Root of the workspace for provenance detection.
        scanner: Optional ToolScanner for prompt injection detection.
        scan_mode: Scanning mode (disabled/audit/enforce).

    Returns:
        SkillMetadata if skill is valid and passes security scan, None otherwise.
    """
    skill_file = skill_dir / SKILL_FILE_NAME
    if not skill_file.is_file():
        return None

    try:
        content = skill_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None

    frontmatter = _parse_frontmatter(content)
    if not _is_valid_frontmatter(skill_dir=skill_dir, frontmatter=frontmatter):
        return None

    # Scan skill body if scanner provided
    if scanner and scan_mode != "disabled":
        if not _scan_skill_content(
            skill_dir, skill_file, content, source, workspace_root, scanner
        ):
            return None

    name = str(frontmatter["name"]).strip()
    description = str(frontmatter["description"]).strip()
    extra_metadata = frontmatter.get("metadata") or {}

    # Log when cross-tool skills contain execution-oriented subdirectories
    # that cicaddy cannot use (no bash tool, no on-demand loading)
    if (skill_dir / "scripts").is_dir():
        logger.info(
            f"Skill '{name}' contains scripts/ directory "
            "(ignored — cicaddy has no execution engine)"
        )
    if (skill_dir / "references").is_dir():
        logger.info(
            f"Skill '{name}' contains references/ directory "
            "(ignored — no on-demand loading)"
        )

    return SkillMetadata(
        name=name,
        description=description,
        location=skill_file.resolve(),
        source=source,
        metadata={
            str(k).casefold(): v for k, v in extra_metadata.items() if k is not None
        },
    )


def _parse_frontmatter(content: str) -> dict[str, Any]:
    """Parse YAML frontmatter from skill file content."""
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            payload = "\n".join(lines[1:idx])
            try:
                parsed = yaml.safe_load(payload)
            except yaml.YAMLError:
                return {}
            if isinstance(parsed, dict):
                return {str(key).lower(): value for key, value in parsed.items()}
    return {}


def _is_valid_frontmatter(*, skill_dir: Path, frontmatter: dict[str, object]) -> bool:
    """Validate skill frontmatter fields."""
    name = frontmatter.get("name")
    description = frontmatter.get("description")

    if not isinstance(name, str) or not name.strip():
        return False
    if len(name.strip()) > 64:
        return False
    if name.strip() != skill_dir.name:
        return False
    if SKILL_NAME_PATTERN.fullmatch(name.strip()) is None:
        return False

    if not isinstance(description, str) or not description.strip():
        return False
    if len(description.strip()) > 1024:
        return False

    # Validate metadata field if present
    meta = frontmatter.get("metadata")
    if meta is not None:
        if not isinstance(meta, dict):
            return False
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items()):
            return False

    return True


def _iter_skill_roots(
    workspace_path: Path,
    provider: Optional[str] = None,
) -> list[tuple[Path, str]]:
    """Iterate over skill root directories in precedence order.

    Order: provider-specific project > cross-tool project > global > bundled.
    """
    roots: list[tuple[Path, str]] = []

    # 1. Provider-specific project dir (highest precedence)
    if provider:
        provider_dir = PROVIDER_SKILLS_DIRS.get(provider.lower())
        if provider_dir:
            roots.append((workspace_path / provider_dir, "project"))

    # 2. Cross-tool project dir (.agents/skills/)
    roots.append((workspace_path / CROSS_TOOL_SKILLS_DIR, "project"))

    # 3. Global user dir
    roots.append((Path.home() / CROSS_TOOL_SKILLS_DIR, "global"))

    # 4. Bundled package skills (lowest precedence)
    bundled_dir = Path(__file__).parent / "bundled_skills"
    roots.append((bundled_dir, "bundled"))

    return roots
