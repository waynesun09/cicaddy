"""Branch Review AI Agent for analyzing branch changes against target branch."""

import os
from typing import Any, Dict, Optional

from cicaddy.config.settings import Settings
from cicaddy.utils.logger import get_logger

from .base_review_agent import BaseReviewAgent

logger = get_logger(__name__)


class BranchReviewAgent(BaseReviewAgent):
    """AI Agent specialized for branch analysis and code review."""

    def __init__(self, settings: Optional[Settings] = None):
        super().__init__(settings)
        self.source_branch = self._get_source_branch()
        self.target_branch = self._get_target_branch()

    def _get_source_branch(self) -> str:
        """Get source branch from CI environment or current branch."""
        # Try CI environment variables first
        ci_branch = os.getenv("CI_COMMIT_REF_NAME")
        if ci_branch:
            logger.debug(f"Source branch from CI: {ci_branch}")
            return ci_branch

        # Try custom environment variable
        custom_branch = os.getenv("BRANCH_REVIEW_SOURCE_BRANCH")
        if custom_branch:
            logger.debug(f"Source branch from config: {custom_branch}")
            return custom_branch

        # Default to current branch (will be detected by diff analyzer)
        logger.debug("Source branch will be detected as current branch")
        return "HEAD"

    def _get_target_branch(self) -> str:
        """Get target branch from CI environment or configuration."""
        # Try custom environment variable first
        custom_target = os.getenv("BRANCH_REVIEW_TARGET_BRANCH")
        if custom_target:
            logger.debug(f"Target branch from config: {custom_target}")
            return custom_target

        # Try CI default branch
        ci_default = os.getenv("CI_DEFAULT_BRANCH")
        if ci_default:
            logger.debug(f"Target branch from CI default: {ci_default}")
            return ci_default

        # Default to main
        logger.debug("Target branch defaulting to main")
        return "main"

    async def get_diff_content(self) -> str:
        """Get branch diff content comparing source to target branch."""
        self._validate_initialized()

        logger.info(
            f"Getting diff between {self.source_branch} and {self.target_branch}"
        )

        # If source is HEAD, use current branch diff
        if self.source_branch == "HEAD":
            diff_content = await self.diff_analyzer.get_current_branch_diff(
                target_branch=self.target_branch,
                context_lines=self.settings.git_diff_context_lines,
            )
        else:
            # Use specific branch comparison
            diff_content = await self.diff_analyzer.get_branch_diff(
                source_branch=self.source_branch,
                target_branch=self.target_branch,
                context_lines=self.settings.git_diff_context_lines,
            )

        return diff_content

    async def get_review_context(self) -> Dict[str, Any]:
        """Get branch review specific context."""
        logger.info(
            f"Getting review context for branch comparison: {self.source_branch} -> {self.target_branch}"
        )

        # Get current branch name if using HEAD
        actual_source_branch = self.source_branch
        if self.source_branch == "HEAD":
            try:
                actual_source_branch = self.diff_analyzer.get_current_branch_name()
            except Exception as e:
                logger.warning(f"Could not get current branch name: {e}")
                actual_source_branch = "current"

        # Get diff summary for additional context
        diff_summary = await self.get_diff_summary()

        context = {
            "analysis_type": "branch_review",
            "source_branch": actual_source_branch,
            "target_branch": self.target_branch,
            "diff_summary": diff_summary,
            "comparison_type": "branch_to_branch",
        }

        # Add CI context if available
        ci_pipeline_source = os.getenv("CI_PIPELINE_SOURCE")
        ci_commit_sha = os.getenv("CI_COMMIT_SHA")
        ci_commit_ref_name = os.getenv("CI_COMMIT_REF_NAME")

        if ci_pipeline_source:
            context["ci_context"] = {
                "pipeline_source": ci_pipeline_source,
                "commit_sha": ci_commit_sha,
                "commit_ref_name": ci_commit_ref_name,
            }

        return context

    def build_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """
        Build analysis prompt for branch code review.

        Args:
            context: Analysis context from get_analysis_context()

        Returns:
            String prompt for AI analysis
        """
        # Get enabled tasks for this branch analysis
        enabled_tasks = self.settings.get_enabled_tasks()

        # Get available tools information with schemas
        tools_info = []
        for tool in context.get("mcp_tools", []):
            tool_info = f"- {tool['name']}: {tool.get('description', 'No description available')}"
            if "server" in tool:
                tool_info += f" (from {tool['server']} server)"

            # Add schema information if available
            if "inputSchema" in tool and tool["inputSchema"]:
                schema = tool["inputSchema"]
                if "properties" in schema:
                    params = []
                    required_params = schema.get("required", [])
                    for param_name, param_info in schema["properties"].items():
                        param_type = param_info.get("type", "unknown")
                        param_desc = param_info.get("description", "")
                        is_required = param_name in required_params
                        required_text = " (required)" if is_required else " (optional)"
                        params.append(
                            f"    {param_name} ({param_type}){required_text}: {param_desc}"
                        )

                    if params:
                        tool_info += "\n  Parameters:\n" + "\n".join(params)
                else:
                    tool_info += "\n  Parameters: No parameters required"

            tools_info.append(tool_info)

        tools_list = "\n".join(tools_info) if tools_info else "No MCP tools available"

        # Get branch context
        source_branch = context.get("source_branch", "unknown")
        target_branch = context.get("target_branch", "unknown")
        diff_content = context.get("diff", "")
        diff_summary = context.get("diff_summary", {})

        # Build comprehensive prompt for branch analysis
        prompt = f"""
You are an AI agent performing branch analysis on a code project.

Project: {context.get("project", {}).get("name", "Unknown")}
Branch Comparison: {source_branch} → {target_branch}
Analysis Type: Branch Review

Change Summary:
- Modified Files: {diff_summary.get("modified_files", 0)}
- Lines Added: {diff_summary.get("added_lines", 0)}
- Lines Removed: {diff_summary.get("removed_lines", 0)}
- Total Diff Lines: {diff_summary.get("total_lines", 0)}

Code Changes:
```diff
{diff_content}
```

Available MCP Tools:
{tools_list}

Enabled Analysis Tasks: {", ".join(enabled_tasks)}

Instructions:
1. Analyze the branch changes compared to {target_branch}
2. For each enabled task, provide detailed analysis:
   - code_review: Focus on code quality, best practices, potential bugs
   - security_scan: Identify security vulnerabilities and risks
3. Use available MCP tools when they can provide additional insights
4. Pay attention to parameter types and requirements when calling tools
5. Provide actionable recommendations and specific code suggestions
6. Focus on changes that impact deployment readiness and code quality
7. Consider the branch context - this is a proposed set of changes

{self._get_task_specific_instructions(enabled_tasks)}

Please provide your comprehensive branch analysis.
"""

        return prompt

    def _get_task_specific_instructions(self, enabled_tasks: list) -> str:
        """Get task-specific instructions based on enabled tasks."""
        instructions = []

        if "code_review" in enabled_tasks:
            custom_prompt = self.settings.review_prompt
            if custom_prompt:
                instructions.append(f"Code Review Instructions: {custom_prompt}")
            else:
                instructions.append(
                    """
Branch Review Focus:
- **Summary**: Summarize differences between current branch and target
- **Impact Analysis**: Assess potential impact of branch changes
- **Security Review**: Identify security implications of changes
- **Quality Assessment**: Evaluate code quality improvements or regressions
- **Recommendations**: Provide specific recommendations for the branch
- **Deployment Readiness**: Assess if changes are ready for merge
"""
                )

        if "security_scan" in enabled_tasks:
            instructions.append(
                """
Security Analysis Focus:
- **Vulnerabilities**: Identify security issues in branch changes
- **Attack Vectors**: Describe potential attack scenarios introduced
- **Risk Assessment**: Rate severity (Critical/High/Medium/Low)
- **Mitigation**: Provide specific security fixes for branch
"""
            )

        return "\n".join(instructions)

    async def send_notifications(
        self, report: Dict[str, Any], analysis_result: Dict[str, Any]
    ):
        """Send notifications for branch reviews.

        Platform-specific comment posting (e.g., GitLab commit comments)
        is handled by platform plugin overrides.
        """
        logger.info(
            f"Sending notifications for branch review: {self.source_branch} -> {self.target_branch}"
        )
        await super().send_notifications(report, analysis_result)

    def get_session_id(self) -> str:
        """
        Get unique session ID for this branch analysis session.

        Returns:
            String session identifier based on branch names
        """
        safe_source = self.source_branch.replace("/", "_").replace(":", "_")
        safe_target = self.target_branch.replace("/", "_").replace(":", "_")
        return f"branch_{safe_source}_to_{safe_target}"

    async def process_branch_review(self) -> Dict[str, Any]:
        """
        Main entry point for processing branch reviews.

        This is a convenience method that calls the base analyze() method
        and returns the results in the expected format.

        Returns:
            Dict containing analysis results and execution metadata
        """
        logger.info(
            f"Processing branch review: {self.source_branch} -> {self.target_branch}"
        )

        # Initialize all components (GitLab analyzer, MCP, etc.)
        await self.initialize()

        # Use the base class analyze method which implements the full pipeline
        result = await self.analyze()

        # Return in expected format for backwards compatibility
        return result["analysis_result"]

    async def generate_report(
        self, analysis_result: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate formatted report with branch-specific information."""
        # Use base class report generation
        report = await super().generate_report(analysis_result, context)

        # Add branch-specific metadata for notifications
        report["source_branch"] = context.get("source_branch", "unknown")
        report["target_branch"] = context.get("target_branch", "unknown")

        return report
