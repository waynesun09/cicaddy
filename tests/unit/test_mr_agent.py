"""Unit tests for the base MergeRequestAgent."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cicaddy.agent.mr_agent import MergeRequestAgent
from cicaddy.config.settings import CoreSettings

# Keys to strip from env to get clean test settings
_CLEAN_KEYS = (
    "CI_MERGE_REQUEST_IID",
    "CI_MERGE_REQUEST_SOURCE_BRANCH_NAME",
    "CI_MERGE_REQUEST_TARGET_BRANCH_NAME",
    "CI_COMMIT_REF_NAME",
    "CI_DEFAULT_BRANCH",
    "CI_MERGE_REQUEST_TITLE",
    "CI_PIPELINE_SOURCE",
    "CI_COMMIT_SHA",
    "AI_TASK_FILE",
    "LOG_LEVEL",
    "AI_PROVIDER",
    "MAX_INFER_ITERS",
    "MAX_EXECUTION_TIME",
    "CONTEXT_SAFETY_FACTOR",
    "SSL_VERIFY",
)


def _clean_env():
    """Return env dict with test-sensitive keys removed."""
    return {k: v for k, v in os.environ.items() if k not in _CLEAN_KEYS}


def _make_settings(**env_overrides):
    """Create CoreSettings with specific env vars set."""
    env = _clean_env()
    env.update(env_overrides)
    with patch.dict(os.environ, env, clear=True):
        return CoreSettings()


class TestMergeRequestAgentInit:
    """Test MergeRequestAgent initialization and field resolution."""

    def test_resolve_mr_iid_from_settings(self):
        """MR IID should be read from settings (via env var alias)."""
        settings = _make_settings(CI_MERGE_REQUEST_IID="42")
        agent = MergeRequestAgent(settings)
        assert agent.merge_request_iid == "42"

    def test_resolve_mr_iid_from_env(self):
        """MR IID should fall back to CI_MERGE_REQUEST_IID env var at runtime."""
        settings = _make_settings()
        with patch.dict(os.environ, {"CI_MERGE_REQUEST_IID": "99"}):
            agent = MergeRequestAgent(settings)
        assert agent.merge_request_iid == "99"

    def test_resolve_mr_iid_none(self):
        """MR IID should be None when not available."""
        settings = _make_settings()
        env = _clean_env()
        with patch.dict(os.environ, env, clear=True):
            agent = MergeRequestAgent(settings)
        assert agent.merge_request_iid is None

    def test_resolve_source_branch_from_settings(self):
        """Source branch should come from settings first."""
        settings = _make_settings(CI_MERGE_REQUEST_SOURCE_BRANCH_NAME="feature/foo")
        agent = MergeRequestAgent(settings)
        assert agent.source_branch == "feature/foo"

    def test_resolve_source_branch_from_env(self):
        """Source branch should fall back to CI env vars."""
        settings = _make_settings()
        with patch.dict(
            os.environ, {"CI_MERGE_REQUEST_SOURCE_BRANCH_NAME": "feat/bar"}
        ):
            agent = MergeRequestAgent(settings)
        assert agent.source_branch == "feat/bar"

    def test_resolve_source_branch_from_commit_ref(self):
        """Source branch should fall back to CI_COMMIT_REF_NAME."""
        settings = _make_settings()
        env = _clean_env()
        env["CI_COMMIT_REF_NAME"] = "my-branch"
        with patch.dict(os.environ, env, clear=True):
            agent = MergeRequestAgent(settings)
        assert agent.source_branch == "my-branch"

    def test_resolve_source_branch_default(self):
        """Source branch should default to HEAD."""
        settings = _make_settings()
        env = _clean_env()
        with patch.dict(os.environ, env, clear=True):
            agent = MergeRequestAgent(settings)
        assert agent.source_branch == "HEAD"

    def test_resolve_target_branch_from_settings(self):
        """Target branch should come from settings first."""
        settings = _make_settings(CI_MERGE_REQUEST_TARGET_BRANCH_NAME="release/1.0")
        agent = MergeRequestAgent(settings)
        assert agent.target_branch == "release/1.0"

    def test_resolve_target_branch_from_env(self):
        """Target branch should fall back to CI env vars."""
        settings = _make_settings()
        with patch.dict(os.environ, {"CI_MERGE_REQUEST_TARGET_BRANCH_NAME": "develop"}):
            agent = MergeRequestAgent(settings)
        assert agent.target_branch == "develop"

    def test_resolve_target_branch_default(self):
        """Target branch should default to main."""
        settings = _make_settings()
        env = _clean_env()
        with patch.dict(os.environ, env, clear=True):
            agent = MergeRequestAgent(settings)
        assert agent.target_branch == "main"


class TestMergeRequestAgentSessionId:
    """Test session ID generation."""

    def test_session_id_with_iid(self):
        settings = _make_settings(CI_MERGE_REQUEST_IID="123")
        agent = MergeRequestAgent(settings)
        assert agent.get_session_id() == "mr_123"

    def test_session_id_without_iid(self):
        settings = _make_settings()
        env = _clean_env()
        with patch.dict(os.environ, env, clear=True):
            agent = MergeRequestAgent(settings)
        assert agent.get_session_id() == "mr_unknown"


class TestMergeRequestAgentPrompt:
    """Test prompt building."""

    def test_build_prompt_contains_branches(self):
        """Prompt should include source and target branch info."""
        settings = _make_settings(
            CI_MERGE_REQUEST_IID="10",
            CI_MERGE_REQUEST_SOURCE_BRANCH_NAME="feat/x",
            CI_MERGE_REQUEST_TARGET_BRANCH_NAME="main",
        )
        agent = MergeRequestAgent(settings)

        context = {
            "project": {"name": "test-project"},
            "diff": "diff --git a/foo.py b/foo.py\n+added line",
            "diff_summary": {
                "modified_files": 1,
                "added_lines": 1,
                "removed_lines": 0,
                "total_lines": 2,
            },
            "merge_request": {
                "title": "Add feature X",
                "source_branch": "feat/x",
                "target_branch": "main",
            },
            "mcp_tools": [],
        }

        prompt = agent.build_analysis_prompt(context)

        assert "feat/x" in prompt
        assert "main" in prompt
        assert "Add feature X" in prompt
        assert "diff" in prompt
        assert "code_review" in prompt.lower() or "Code Review" in prompt

    def test_build_prompt_with_mcp_tools(self):
        """Prompt should include MCP tool information."""
        settings = _make_settings(CI_MERGE_REQUEST_IID="10")
        agent = MergeRequestAgent(settings)

        context = {
            "project": {},
            "diff": "",
            "diff_summary": {},
            "merge_request": {
                "title": "Test MR",
                "source_branch": "feat",
                "target_branch": "main",
            },
            "mcp_tools": [
                {
                    "name": "search_code",
                    "description": "Search code in repo",
                    "server": "sourcebot",
                    "inputSchema": {
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            }
                        },
                        "required": ["query"],
                    },
                }
            ],
        }

        prompt = agent.build_analysis_prompt(context)
        assert "search_code" in prompt
        assert "sourcebot" in prompt
        assert "query" in prompt


class TestMergeRequestAgentFactory:
    """Test factory registration."""

    def test_merge_request_registered(self):
        from cicaddy.agent.factory import AgentFactory

        available = AgentFactory.get_available_agent_types()
        assert "merge_request" in available

    def test_factory_registry_points_to_mr_agent(self):
        from cicaddy.agent.factory import AgentFactory

        registered_class = AgentFactory._registry.get("merge_request")
        assert registered_class is MergeRequestAgent


class TestMergeRequestAgentReviewContext:
    """Test review context generation."""

    @pytest.mark.asyncio
    async def test_get_review_context_from_env(self):
        """Review context should use env vars when no platform analyzer."""
        settings = _make_settings(
            CI_MERGE_REQUEST_IID="55",
            CI_MERGE_REQUEST_SOURCE_BRANCH_NAME="feat/test",
            CI_MERGE_REQUEST_TARGET_BRANCH_NAME="main",
        )
        agent = MergeRequestAgent(settings)

        # Mock diff_analyzer to avoid git calls
        agent.diff_analyzer = MagicMock()
        mock_diff_summary = AsyncMock(
            return_value={
                "total_lines": 10,
                "added_lines": 5,
                "removed_lines": 2,
                "modified_files": 1,
                "has_changes": True,
            }
        )
        agent.get_diff_summary = mock_diff_summary

        with patch.dict(
            os.environ,
            {"CI_MERGE_REQUEST_TITLE": "My MR Title"},
        ):
            context = await agent.get_review_context()

        assert context["analysis_type"] == "merge_request"
        assert context["mr_iid"] == "55"
        assert context["source_branch"] == "feat/test"
        assert context["target_branch"] == "main"
        assert context["merge_request"]["title"] == "My MR Title"


class TestMergeRequestAgentReport:
    """Test report generation."""

    @pytest.mark.asyncio
    async def test_generate_report_adds_mr_metadata(self):
        """Report should include MR-specific metadata."""
        settings = _make_settings(
            CI_MERGE_REQUEST_IID="77",
            CI_MERGE_REQUEST_SOURCE_BRANCH_NAME="feat/report",
            CI_MERGE_REQUEST_TARGET_BRANCH_NAME="main",
        )
        agent = MergeRequestAgent(settings)
        agent.start_time = MagicMock()
        agent.start_time.isoformat.return_value = "2025-01-01T00:00:00"

        analysis_result = {
            "status": "success",
            "ai_analysis": "Everything looks good.",
        }
        context = {
            "merge_request": {"title": "Test MR", "source_branch": "feat/report"},
            "source_branch": "feat/report",
            "target_branch": "main",
        }

        report = await agent.generate_report(analysis_result, context)

        assert report["merge_request"]["title"] == "Test MR"
        assert report["source_branch"] == "feat/report"
        assert report["target_branch"] == "main"
