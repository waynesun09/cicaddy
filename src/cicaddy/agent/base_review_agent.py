"""Base review agent with shared functionality for code review agents."""

from abc import abstractmethod
from typing import Any, Dict, Optional

from cicaddy.agent.base import BaseAIAgent
from cicaddy.config.settings import Settings
from cicaddy.git.diff_analyzer import DiffAnalyzer
from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)


class BaseReviewAgent(BaseAIAgent):
    """Base class for code review agents with shared review functionality."""

    def __init__(self, settings: Optional[Settings] = None):
        super().__init__(settings)
        self.diff_analyzer: Optional[DiffAnalyzer] = None

    async def initialize(self):
        """Initialize the review agent with diff analyzer."""
        await super().initialize()

        # Initialize diff analyzer with git working directory
        working_dir = getattr(self.settings, "git_working_directory", None)
        self.diff_analyzer = DiffAnalyzer(working_directory=working_dir)

        logger.info(
            "Review agent initialized",
            git_working_directory=working_dir,
            diff_analyzer_available=self.diff_analyzer is not None,
        )

    @abstractmethod
    async def get_diff_content(self) -> str:
        """
        Get the diff content for this review.

        Returns:
            Diff content as string

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass

    @abstractmethod
    async def get_review_context(self) -> Dict[str, Any]:
        """
        Get review-specific context information.

        Returns:
            Dictionary containing review context

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass

    async def get_analysis_context(self) -> Dict[str, Any]:
        """
        Gather analysis context including diff and review-specific information.

        Returns:
            Dictionary containing complete analysis context
        """
        if not self.diff_analyzer:
            raise ValueError("Diff analyzer not initialized - call initialize() first")

        logger.info("Gathering analysis context for review")

        # Get diff content
        try:
            diff_content = await self.get_diff_content()
        except Exception as e:
            logger.error(f"Failed to get diff content: {e}")
            diff_content = f"Error retrieving diff: {str(e)}"

        # Get review-specific context
        try:
            review_context = await self.get_review_context()
        except Exception as e:
            logger.error(f"Failed to get review context: {e}")
            review_context = {"error": f"Failed to get review context: {str(e)}"}

        # Get project information if GitLab analyzer is available
        project_info = {}
        if self.gitlab_analyzer:
            try:
                project_info = await self.gitlab_analyzer.get_project_info()
            except Exception as e:
                logger.warning(f"Could not get project info: {e}")
                project_info = {"name": "Unknown Project", "error": str(e)}

        # Combine all context
        context = {
            "project": project_info,
            "diff": diff_content,
            "gitlab_available": self.gitlab_analyzer is not None,
            "timestamp": self.start_time.isoformat(),
            "diff_lines": len(diff_content.splitlines()) if diff_content else 0,
            **review_context,  # Merge review-specific context
        }

        logger.info(
            "Analysis context gathered",
            diff_lines=context["diff_lines"],
            analysis_type=context.get("analysis_type", "unknown"),
            gitlab_available=context["gitlab_available"],
        )

        return context

    async def get_diff_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the diff changes.

        Returns:
            Dictionary containing diff statistics and summary
        """
        try:
            diff_content = await self.get_diff_content()

            lines = diff_content.splitlines()
            added_lines = len([line for line in lines if line.startswith("+")])
            removed_lines = len([line for line in lines if line.startswith("-")])
            modified_files = len(
                [line for line in lines if line.startswith("diff --git")]
            )

            return {
                "total_lines": len(lines),
                "added_lines": added_lines,
                "removed_lines": removed_lines,
                "modified_files": modified_files,
                "has_changes": len(lines) > 0,
            }

        except Exception as e:
            logger.error(f"Failed to generate diff summary: {e}")
            return {
                "total_lines": 0,
                "added_lines": 0,
                "removed_lines": 0,
                "modified_files": 0,
                "has_changes": False,
                "error": str(e),
            }

    def _validate_initialized(self):
        """Validate that the agent is properly initialized."""
        if not self.diff_analyzer:
            raise ValueError(
                "Review agent not properly initialized - diff analyzer missing"
            )
