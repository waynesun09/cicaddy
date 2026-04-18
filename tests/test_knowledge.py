"""Tests for bundled knowledge and skills loading."""

from pathlib import Path

from cicaddy.knowledge import CICADDY_CONTEXT, get_bundled_context
from cicaddy.skills import _iter_skill_roots, discover_skills


class TestBundledKnowledge:
    """Test bundled knowledge context."""

    def test_cicaddy_context_not_empty(self):
        """Bundled context string must contain content."""
        assert len(CICADDY_CONTEXT) > 100

    def test_cicaddy_context_has_xml_tags(self):
        """Bundled context must be wrapped in XML tags."""
        assert "<cicaddy_reference" in CICADDY_CONTEXT
        assert "</cicaddy_reference>" in CICADDY_CONTEXT

    def test_cicaddy_context_has_model_info(self):
        """Bundled context must contain model reference information."""
        assert "gemini-3" in CICADDY_CONTEXT
        assert "claude-" in CICADDY_CONTEXT

    def test_cicaddy_context_has_config_info(self):
        """Bundled context must contain config reference information."""
        assert "AI_PROVIDER" in CICADDY_CONTEXT
        assert "AI_MODEL" in CICADDY_CONTEXT
        assert "MCP_SERVERS_CONFIG" in CICADDY_CONTEXT

    def test_get_bundled_context_returns_content(self):
        """get_bundled_context() must return the context string."""
        result = get_bundled_context()
        assert result == CICADDY_CONTEXT


class TestBundledSkills:
    """Test bundled skills discovery."""

    def test_bundled_skills_dir_exists(self):
        """Bundled skills directory must exist in the package."""
        bundled_dir = (
            Path(__file__).parent.parent / "src" / "cicaddy" / "bundled_skills"
        )
        assert bundled_dir.is_dir()

    def test_bundled_skills_in_iter_roots(self):
        """_iter_skill_roots must include bundled dir as last entry."""
        roots = _iter_skill_roots(Path("/nonexistent"))
        sources = [source for _, source in roots]
        assert "bundled" in sources
        # Bundled must be last (lowest precedence)
        assert roots[-1][1] == "bundled"

    def test_bundled_skills_last_with_provider(self):
        """Bundled skills must remain last even with provider-specific roots."""
        roots = _iter_skill_roots(Path("/nonexistent"), provider="gemini")
        assert roots[-1][1] == "bundled"

    def test_bundled_skill_files_valid(self):
        """Each bundled skill must have valid SKILL.md with required frontmatter."""
        bundled_dir = (
            Path(__file__).parent.parent / "src" / "cicaddy" / "bundled_skills"
        )
        skill_dirs = [d for d in bundled_dir.iterdir() if d.is_dir()]
        assert len(skill_dirs) >= 2, (
            "Expected at least model-reference and cicaddy-config"
        )

        for skill_dir in skill_dirs:
            skill_file = skill_dir / "SKILL.md"
            assert skill_file.is_file(), f"Missing SKILL.md in {skill_dir.name}"

            content = skill_file.read_text(encoding="utf-8")
            assert "---" in content, f"Missing frontmatter in {skill_dir.name}"
            assert f"name: {skill_dir.name}" in content, (
                f"Skill name must match directory name: {skill_dir.name}"
            )
            assert "description:" in content, f"Missing description in {skill_dir.name}"

    def test_discover_bundled_skills(self, tmp_path):
        """discover_skills must find bundled skills when workspace has none."""
        # Use empty workspace — only bundled skills should be found
        skills = discover_skills(tmp_path)
        skill_names = {s.name for s in skills}
        assert "model-reference" in skill_names
        assert "cicaddy-config" in skill_names

    def test_bundled_skills_have_source_bundled(self, tmp_path):
        """Bundled skills must have source='bundled'."""
        skills = discover_skills(tmp_path)
        bundled = [s for s in skills if s.source == "bundled"]
        assert len(bundled) >= 2

    def test_workspace_skill_overrides_bundled(self, tmp_path):
        """A workspace skill with same name must override bundled skill."""
        # Create a workspace skill that shadows bundled 'model-reference'
        skill_dir = tmp_path / ".agents" / "skills" / "model-reference"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: model-reference\ndescription: Custom override\n---\nCustom content"
        )

        skills = discover_skills(tmp_path)
        model_ref = [s for s in skills if s.name == "model-reference"]
        assert len(model_ref) == 1
        assert model_ref[0].source == "project"
        assert model_ref[0].description == "Custom override"

    def test_bundled_skills_body_readable(self, tmp_path):
        """Bundled skill body() must return non-empty content."""
        skills = discover_skills(tmp_path)
        for skill in skills:
            if skill.source == "bundled":
                body = skill.body()
                assert len(body) > 50, f"Skill {skill.name} body too short"
