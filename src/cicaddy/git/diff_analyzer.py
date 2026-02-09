"""Shared diff analysis functionality using git CLI."""

import subprocess  # nosec B404
from typing import Optional

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)


class DiffAnalyzer:
    """Centralized diff analysis functionality using git CLI (platform-agnostic)."""

    def __init__(self, working_directory: Optional[str] = None):
        """
        Initialize diff analyzer.

        Args:
            working_directory: Git repository directory (defaults to current directory)
        """
        self.working_directory = working_directory

    async def get_branch_diff(
        self, source_branch: str, target_branch: str = "main", context_lines: int = 10
    ) -> str:
        """
        Get diff between two branches.

        Args:
            source_branch: Source branch to compare from
            target_branch: Target branch to compare against (defaults to main)
            context_lines: Number of context lines for diff

        Returns:
            Diff content as string
        """
        logger.info(f"Generating diff between {source_branch} and {target_branch}")

        try:
            # Ensure we have the latest refs
            fetch_cmd = ["git", "fetch", "origin", target_branch, source_branch]
            subprocess.run(  # nosec B603
                fetch_cmd, cwd=self.working_directory, check=True, capture_output=True
            )

            # Get merge base to find common ancestor
            merge_base_cmd = [
                "git",
                "merge-base",
                f"origin/{target_branch}",
                f"origin/{source_branch}",
            ]
            result = subprocess.run(  # nosec B603
                merge_base_cmd,
                cwd=self.working_directory,
                capture_output=True,
                text=True,
                check=True,
            )
            merge_base = result.stdout.strip()

            # Generate diff from merge base to source branch
            diff_cmd = [
                "git",
                "diff",
                f"-U{context_lines}",
                f"{merge_base}..origin/{source_branch}",
            ]
            result = subprocess.run(  # nosec B603
                diff_cmd,
                cwd=self.working_directory,
                capture_output=True,
                text=True,
                check=True,
            )

            diff_content = result.stdout
            logger.info(f"Generated diff with {len(diff_content.splitlines())} lines")

            return diff_content

        except subprocess.CalledProcessError as e:
            error_msg = f"Git command failed: {e.cmd} - {e.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to generate branch diff: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    async def get_current_branch_diff(
        self, target_branch: str = "main", context_lines: int = 10
    ) -> str:
        """
        Get diff between current branch/HEAD and target branch.

        Args:
            target_branch: Target branch to compare against (defaults to main)
            context_lines: Number of context lines for diff

        Returns:
            Diff content as string
        """
        logger.info(f"Generating diff between HEAD and {target_branch}")

        try:
            # Ensure we have the latest target branch
            fetch_cmd = ["git", "fetch", "origin", target_branch]
            subprocess.run(  # nosec B603
                fetch_cmd, cwd=self.working_directory, check=True, capture_output=True
            )

            # Get merge base between HEAD and target branch
            merge_base_cmd = ["git", "merge-base", "HEAD", f"origin/{target_branch}"]
            result = subprocess.run(  # nosec B603
                merge_base_cmd,
                cwd=self.working_directory,
                capture_output=True,
                text=True,
                check=True,
            )
            merge_base = result.stdout.strip()

            # Generate diff from merge base to HEAD
            diff_cmd = ["git", "diff", f"-U{context_lines}", f"{merge_base}...HEAD"]
            result = subprocess.run(  # nosec B603
                diff_cmd,
                cwd=self.working_directory,
                capture_output=True,
                text=True,
                check=True,
            )

            diff_content = result.stdout
            logger.info(
                f"Generated current branch diff with {len(diff_content.splitlines())} lines"
            )

            return diff_content

        except subprocess.CalledProcessError as e:
            error_msg = f"Git command failed: {e.cmd} - {e.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to generate current branch diff: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    async def get_merge_request_diff_by_branches(
        self, source_branch: str, target_branch: str, context_lines: int = 10
    ) -> str:
        """
        Get merge request style diff between branches.

        Args:
            source_branch: Source branch of the MR
            target_branch: Target branch of the MR
            context_lines: Number of context lines for diff

        Returns:
            Diff content as string
        """
        # This is essentially the same as branch_diff but with MR semantics
        return await self.get_branch_diff(source_branch, target_branch, context_lines)

    def get_current_branch_name(self) -> str:
        """
        Get the current branch name.

        Returns:
            Current branch name
        """
        try:
            cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
            result = subprocess.run(  # nosec B603
                cmd,
                cwd=self.working_directory,
                capture_output=True,
                text=True,
                check=True,
            )
            branch_name = result.stdout.strip()
            logger.debug(f"Current branch: {branch_name}")
            return branch_name

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to get current branch: {e.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_default_branch(self) -> str:
        """
        Get the default branch name (usually main or master).

        Returns:
            Default branch name
        """
        try:
            # Try to get the default branch from remote
            cmd = ["git", "symbolic-ref", "refs/remotes/origin/HEAD"]
            result = subprocess.run(  # nosec B603
                cmd, cwd=self.working_directory, capture_output=True, text=True
            )

            if result.returncode == 0:
                # Extract branch name from refs/remotes/origin/main
                default_branch = result.stdout.strip().split("/")[-1]
                logger.debug(f"Default branch from remote: {default_branch}")
                return default_branch

            # Fallback to common defaults
            for branch in ["main", "master"]:
                cmd = ["git", "rev-parse", "--verify", f"origin/{branch}"]
                result = subprocess.run(  # nosec B603
                    cmd, cwd=self.working_directory, capture_output=True, text=True
                )
                if result.returncode == 0:
                    logger.debug(f"Found default branch: {branch}")
                    return branch

            # Final fallback
            logger.warning("Could not determine default branch, defaulting to 'main'")
            return "main"

        except Exception as e:
            logger.warning(
                f"Error determining default branch: {e}, defaulting to 'main'"
            )
            return "main"
