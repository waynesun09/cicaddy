"""Skill discovery and loading from workspace and global skill directories."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_SKILLS_DIR = ".agents/skills"
SKILL_FILE_NAME = "SKILL.md"
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SKILL_SOURCES = ("project", "global")


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
        front_matter_pattern = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
        try:
            content = self.location.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
        return front_matter_pattern.sub("", content, count=1).strip()


def discover_skills(workspace_path: Path) -> list[SkillMetadata]:
    """Discover skills from project and global directories.

    Project skills take precedence over global skills with the same name.

    Args:
        workspace_path: Path to the project workspace directory.

    Returns:
        Sorted list of discovered skills.
    """
    skills_by_name: dict[str, SkillMetadata] = {}
    for root, source in _iter_skill_roots(workspace_path):
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


def _iter_skill_roots(workspace_path: Path) -> list[tuple[Path, str]]:
    """Iterate over skill root directories in precedence order."""
    roots: list[tuple[Path, str]] = []
    for source in SKILL_SOURCES:
        if source == "project":
            roots.append((workspace_path / PROJECT_SKILLS_DIR, source))
        elif source == "global":
            roots.append((Path.home() / PROJECT_SKILLS_DIR, source))
    return roots
