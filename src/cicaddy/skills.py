"""Skill discovery and loading from workspace and global skill directories."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from cicaddy.utils.logger import get_logger

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
    source: str  # "project" or "global"
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

    Returns:
        Sorted list of discovered skills.
    """
    skills_by_name: dict[str, SkillMetadata] = {}
    for root, source in _iter_skill_roots(workspace_path, provider=provider):
        if not root.is_dir():
            continue
        for skill_dir in sorted(root.iterdir()):
            if not skill_dir.is_dir():
                continue
            metadata = _read_skill(skill_dir, source=source)
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


def _read_skill(skill_dir: Path, *, source: str) -> Optional[SkillMetadata]:
    """Read and validate a single skill directory."""
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

    name = str(frontmatter["name"]).strip()
    description = str(frontmatter["description"]).strip()
    extra_metadata = frontmatter.get("metadata") or {}

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

    Order: provider-specific project > cross-tool project > global.
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

    return roots
