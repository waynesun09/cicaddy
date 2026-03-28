"""Tests for provenance detection."""

import tempfile
from pathlib import Path

import pytest

from cicaddy.security.provenance import (
    _find_git_root,
    _is_in_submodule,
    get_provenance_label,
    is_external_source,
)


class TestProvenanceDetection:
    """Test provenance detection for files."""

    def test_non_existent_file_is_safe(self):
        """Test that non-existent files are considered safe."""
        path = Path("/nonexistent/file.md")
        assert not is_external_source(path)

    def test_file_outside_workspace_is_external(self):
        """Test that files outside workspace are external."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()

            external_file = Path(tmpdir) / "external" / "file.md"
            external_file.parent.mkdir()
            external_file.write_text("content")

            # File outside workspace is external
            assert is_external_source(external_file, workspace)

    def test_git_submodule_detection(self, tmp_path):
        """Test detection of files in git submodules."""
        # Create workspace with .gitmodules
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        gitmodules = workspace / ".gitmodules"
        gitmodules.write_text(
            """
[submodule "vendor/lib"]
    path = vendor/lib
    url = https://example.com/lib.git
"""
        )

        # Create submodule directory and file
        submodule_dir = workspace / "vendor" / "lib"
        submodule_dir.mkdir(parents=True)
        submodule_file = submodule_dir / "AGENT.md"
        submodule_file.write_text("malicious content")

        # Should detect as submodule
        assert _is_in_submodule(submodule_file, workspace)

    def test_file_not_in_submodule(self, tmp_path):
        """Test detection of files not in submodules."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # No .gitmodules file
        local_file = workspace / "AGENT.md"
        local_file.write_text("content")

        assert not _is_in_submodule(local_file, workspace)

    def test_provenance_labels(self):
        """Test provenance label generation."""
        # Non-existent file
        assert get_provenance_label(Path("/nonexistent.md")) == "unknown"


class TestGitIntegration:
    """Test git integration (requires git to be installed)."""

    @pytest.mark.skipif(
        not Path(".git").exists(),
        reason="Not in a git repository",
    )
    def test_find_git_root_in_repo(self):
        """Test finding git root when in a repository."""
        # This test file should be in a git repo
        root = _find_git_root(Path(__file__))
        assert root is not None
        assert (root / ".git").exists()

    def test_find_git_root_outside_repo(self, tmp_path):
        """Test finding git root outside a repository."""
        temp_file = tmp_path / "file.md"
        temp_file.write_text("content")

        root = _find_git_root(temp_file)
        # In temp directory, should not find a git root
        # (unless system temp is in a git repo, which would be unusual)
        # Just check it doesn't error
        assert root is None or root.exists()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_workspace_root_none(self, tmp_path):
        """Test behavior when workspace_root is None."""
        file_path = tmp_path / "file.md"
        file_path.write_text("content")

        # Should try to find git root, or treat as external
        result = is_external_source(file_path, workspace_root=None)
        # Result depends on whether tmp_path is in a git repo
        assert isinstance(result, bool)

    def test_symlink_handling(self, tmp_path):
        """Test handling of symlinks."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Create real file
        real_file = workspace / "real.md"
        real_file.write_text("content")

        # Create symlink to real file
        link_file = workspace / "link.md"
        link_file.symlink_to(real_file)

        # Both should resolve to same provenance
        # (symlinks are resolved in provenance check)
        real_external = is_external_source(real_file, workspace)
        link_external = is_external_source(link_file, workspace)

        # In tmp_path (not a git repo), files are untracked = external
        # Both should have same external status
        assert real_external == link_external
