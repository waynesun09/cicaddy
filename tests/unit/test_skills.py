"""Tests for skills discovery and loading."""

from pathlib import Path

import pytest

from cicaddy.skills import (
    PROJECT_SKILLS_DIR,
    SKILL_FILE_NAME,
    SkillMetadata,
    _is_valid_frontmatter,
    _parse_frontmatter,
    discover_skills,
    render_skills_prompt,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace directory."""
    return tmp_path


def _create_skill(base: Path, name: str, description: str, body: str = "") -> Path:
    """Helper to create a skill directory with SKILL.md."""
    skill_dir = base / PROJECT_SKILLS_DIR / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = f"---\nname: {name}\ndescription: {description}\n---\n{body}"
    (skill_dir / SKILL_FILE_NAME).write_text(content, encoding="utf-8")
    return skill_dir


def test_discover_skills_empty_workspace(workspace: Path):
    """Empty workspace yields no skills."""
    result = discover_skills(workspace)
    assert result == []


def test_discover_skills_with_project_skills(workspace: Path):
    """Skills in .agents/skills are discovered."""
    _create_skill(workspace, "my-skill", "A test skill")
    result = discover_skills(workspace)
    assert len(result) == 1
    assert result[0].name == "my-skill"
    assert result[0].description == "A test skill"
    assert result[0].source == "project"


def test_discover_skills_precedence(workspace: Path, tmp_path: Path, monkeypatch):
    """Project skills take precedence over global skills with the same name."""
    # Create project skill
    _create_skill(workspace, "my-skill", "Project version")

    # Create global skill at home dir
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    global_skills_dir = fake_home / PROJECT_SKILLS_DIR / "my-skill"
    global_skills_dir.mkdir(parents=True)
    (global_skills_dir / SKILL_FILE_NAME).write_text(
        "---\nname: my-skill\ndescription: Global version\n---\n", encoding="utf-8"
    )

    result = discover_skills(workspace)
    assert len(result) == 1
    assert result[0].description == "Project version"
    assert result[0].source == "project"


def test_skill_body_content(workspace: Path):
    """Skill body() returns content after frontmatter."""
    _create_skill(
        workspace, "code-review", "Review code", body="Check for bugs.\nFix issues."
    )
    skills = discover_skills(workspace)
    assert len(skills) == 1
    body = skills[0].body()
    assert "Check for bugs." in body
    assert "Fix issues." in body


def test_render_skills_prompt(workspace: Path):
    """render_skills_prompt produces formatted output."""
    _create_skill(workspace, "lint-check", "Run linting", body="Use ruff.")
    skills = discover_skills(workspace)
    prompt = render_skills_prompt(skills)
    assert "## Available Skills" in prompt
    assert "**lint-check**" in prompt
    assert "Run linting" in prompt
    assert "Use ruff." in prompt


def test_render_skills_prompt_empty():
    """render_skills_prompt with no skills returns empty string."""
    assert render_skills_prompt([]) == ""


def test_parse_frontmatter_valid():
    """Valid YAML frontmatter is parsed correctly."""
    content = "---\nname: my-skill\ndescription: A skill\n---\nBody content"
    fm = _parse_frontmatter(content)
    assert fm["name"] == "my-skill"
    assert fm["description"] == "A skill"


def test_parse_frontmatter_invalid():
    """Missing frontmatter returns empty dict."""
    assert _parse_frontmatter("No frontmatter here") == {}
    assert _parse_frontmatter("") == {}


def test_parse_frontmatter_no_closing():
    """Frontmatter without closing --- returns empty dict."""
    content = "---\nname: test\ndescription: broken"
    assert _parse_frontmatter(content) == {}


def test_is_valid_frontmatter_name_mismatch(tmp_path: Path):
    """Frontmatter name must match directory name."""
    skill_dir = tmp_path / "actual-name"
    skill_dir.mkdir()
    fm = {"name": "different-name", "description": "test"}
    assert not _is_valid_frontmatter(skill_dir=skill_dir, frontmatter=fm)


def test_is_valid_frontmatter_name_too_long(tmp_path: Path):
    """Name over 64 characters is rejected."""
    long_name = "a" * 65
    skill_dir = tmp_path / long_name
    skill_dir.mkdir()
    fm = {"name": long_name, "description": "test"}
    assert not _is_valid_frontmatter(skill_dir=skill_dir, frontmatter=fm)


def test_is_valid_frontmatter_invalid_name_pattern(tmp_path: Path):
    """Names with uppercase or special chars are rejected."""
    skill_dir = tmp_path / "Invalid_Name"
    skill_dir.mkdir()
    fm = {"name": "Invalid_Name", "description": "test"}
    assert not _is_valid_frontmatter(skill_dir=skill_dir, frontmatter=fm)


def test_is_valid_frontmatter_missing_description(tmp_path: Path):
    """Missing description is rejected."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    fm = {"name": "my-skill"}
    assert not _is_valid_frontmatter(skill_dir=skill_dir, frontmatter=fm)


def test_is_valid_frontmatter_description_too_long(tmp_path: Path):
    """Description over 1024 characters is rejected."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    fm = {"name": "my-skill", "description": "x" * 1025}
    assert not _is_valid_frontmatter(skill_dir=skill_dir, frontmatter=fm)


def test_skill_metadata_fields(workspace: Path):
    """SkillMetadata stores all expected fields."""
    _create_skill(workspace, "test-skill", "Test description")
    skills = discover_skills(workspace)
    assert len(skills) == 1
    skill = skills[0]
    assert skill.name == "test-skill"
    assert skill.description == "Test description"
    assert skill.source == "project"
    assert skill.location.name == SKILL_FILE_NAME
    assert skill.metadata == {}


def test_skill_metadata_with_extra_metadata(workspace: Path):
    """Skills with metadata field in frontmatter are parsed."""
    skill_dir = workspace / PROJECT_SKILLS_DIR / "my-tool"
    skill_dir.mkdir(parents=True)
    content = (
        "---\n"
        "name: my-tool\n"
        "description: A tool skill\n"
        "metadata:\n"
        "  version: '1.0'\n"
        "  author: test\n"
        "---\n"
        "Body here"
    )
    (skill_dir / SKILL_FILE_NAME).write_text(content, encoding="utf-8")
    skills = discover_skills(workspace)
    assert len(skills) == 1
    assert skills[0].metadata == {"version": "1.0", "author": "test"}


def test_discover_skills_sorted(workspace: Path):
    """Discovered skills are sorted alphabetically by name."""
    _create_skill(workspace, "zebra", "Last skill")
    _create_skill(workspace, "alpha", "First skill")
    _create_skill(workspace, "middle", "Middle skill")
    skills = discover_skills(workspace)
    names = [s.name for s in skills]
    assert names == ["alpha", "middle", "zebra"]


def test_discover_skills_skips_non_directory(workspace: Path):
    """Files (not directories) in skills dir are ignored."""
    skills_dir = workspace / PROJECT_SKILLS_DIR
    skills_dir.mkdir(parents=True)
    (skills_dir / "not-a-dir.txt").write_text("junk", encoding="utf-8")
    result = discover_skills(workspace)
    assert result == []


def test_discover_skills_skips_missing_skill_file(workspace: Path):
    """Directories without SKILL.md are ignored."""
    skill_dir = workspace / PROJECT_SKILLS_DIR / "no-skill"
    skill_dir.mkdir(parents=True)
    result = discover_skills(workspace)
    assert result == []


def test_skill_body_no_frontmatter():
    """body() on a skill with no frontmatter returns full content."""
    skill = SkillMetadata(
        name="test",
        description="test",
        location=Path("/nonexistent"),
        source="project",
    )
    # When file doesn't exist, body returns ""
    assert skill.body() == ""


def test_is_valid_frontmatter_invalid_metadata_type(tmp_path: Path):
    """Non-dict metadata field is rejected."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    fm = {"name": "my-skill", "description": "test", "metadata": "not-a-dict"}
    assert not _is_valid_frontmatter(skill_dir=skill_dir, frontmatter=fm)


def test_is_valid_frontmatter_non_string_metadata_values(tmp_path: Path):
    """Metadata values must be strings."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    fm = {"name": "my-skill", "description": "test", "metadata": {"key": 123}}
    assert not _is_valid_frontmatter(skill_dir=skill_dir, frontmatter=fm)
