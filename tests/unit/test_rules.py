"""Tests for agent rules discovery and loading."""

from pathlib import Path

import pytest

from cicaddy.rules import (
    ALL_RULE_FILES,
    GENERIC_RULE_FILES,
    PROVIDER_RULE_FILES,
    _read_rule_file,
    discover_rule_files,
    load_agent_rules,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace directory."""
    return tmp_path


def test_load_agent_rules_empty_workspace(workspace: Path):
    """Loading rules from an empty workspace returns empty string."""
    result = load_agent_rules(workspace)
    assert result == ""


def test_load_agent_rules_with_agent_md(workspace: Path):
    """Loading rules picks up AGENT.md."""
    (workspace / "AGENT.md").write_text(
        "# Agent Rules\nDo good things.", encoding="utf-8"
    )
    result = load_agent_rules(workspace)
    assert "Do good things." in result
    assert '<agent_rules source="AGENT.md">' in result


def test_load_agent_rules_with_multiple_files(workspace: Path):
    """Loading rules concatenates multiple generic rule files."""
    (workspace / "AGENT.md").write_text("Rule A", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("Rule B", encoding="utf-8")
    result = load_agent_rules(workspace)
    assert "Rule A" in result
    assert "Rule B" in result
    assert result.count("<agent_rules") == 2


def test_load_agent_rules_provider_specific(workspace: Path):
    """Provider-specific files are loaded when provider is given."""
    (workspace / "GEMINI.md").write_text("Gemini rules", encoding="utf-8")
    result = load_agent_rules(workspace, provider="gemini")
    assert "Gemini rules" in result
    assert 'provider="gemini"' in result


def test_load_agent_rules_provider_gemini(workspace: Path):
    """Gemini provider maps to GEMINI.md."""
    (workspace / "GEMINI.md").write_text("gemini specific", encoding="utf-8")
    result = load_agent_rules(workspace, provider="Gemini")  # case-insensitive
    assert "gemini specific" in result
    assert 'source="GEMINI.md"' in result


def test_load_agent_rules_provider_claude(workspace: Path):
    """Claude provider maps to CLAUDE.md."""
    (workspace / "CLAUDE.md").write_text("claude specific", encoding="utf-8")
    result = load_agent_rules(workspace, provider="claude")
    assert "claude specific" in result
    assert 'source="CLAUDE.md"' in result


def test_load_agent_rules_nonexistent_workspace():
    """Non-existent workspace returns empty string."""
    result = load_agent_rules(Path("/nonexistent/path/abc123"))
    assert result == ""


def test_discover_rule_files(workspace: Path):
    """Discover returns paths to all existing rule files."""
    (workspace / "AGENT.md").write_text("a", encoding="utf-8")
    (workspace / "CLAUDE.md").write_text("b", encoding="utf-8")
    found = discover_rule_files(workspace)
    assert len(found) == 2
    names = {p.name for p in found}
    assert names == {"AGENT.md", "CLAUDE.md"}


def test_discover_rule_files_nonexistent():
    """Discover on non-existent path returns empty list."""
    found = discover_rule_files(Path("/nonexistent/path/abc123"))
    assert found == []


def test_read_rule_file_not_found(workspace: Path):
    """Reading a non-existent file returns empty string."""
    result = _read_rule_file(workspace / "MISSING.md")
    assert result == ""


def test_read_rule_file_success(workspace: Path):
    """Reading an existing file returns stripped content."""
    path = workspace / "AGENT.md"
    path.write_text("  hello world  \n", encoding="utf-8")
    result = _read_rule_file(path)
    assert result == "hello world"


def test_load_agent_rules_generic_and_provider(workspace: Path):
    """Both generic and provider-specific rules are loaded together."""
    (workspace / "AGENT.md").write_text("generic rules", encoding="utf-8")
    (workspace / "CLAUDE.md").write_text("claude rules", encoding="utf-8")
    result = load_agent_rules(workspace, provider="claude")
    assert "generic rules" in result
    assert "claude rules" in result
    assert result.count("<agent_rules") == 2


def test_load_agent_rules_unknown_provider(workspace: Path):
    """Unknown provider name does not cause errors."""
    (workspace / "AGENT.md").write_text("rules", encoding="utf-8")
    result = load_agent_rules(workspace, provider="unknown_provider")
    assert "rules" in result
    # Only the generic file should be loaded
    assert result.count("<agent_rules") == 1


def test_provider_rule_files_mapping():
    """Verify provider rule file mapping is complete."""
    assert PROVIDER_RULE_FILES["gemini"] == "GEMINI.md"
    assert PROVIDER_RULE_FILES["claude"] == "CLAUDE.md"
    assert PROVIDER_RULE_FILES["openai"] == "COPILOT.md"


def test_all_rule_files_includes_all():
    """ALL_RULE_FILES should include both generic and provider-specific files."""
    for f in GENERIC_RULE_FILES:
        assert f in ALL_RULE_FILES
    for f in PROVIDER_RULE_FILES.values():
        assert f in ALL_RULE_FILES
