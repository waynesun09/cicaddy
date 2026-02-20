"""Base Merge Request AI Agent for platform-agnostic MR/PR analysis."""

import os
from typing import Any, Dict, Optional

from cicaddy.config.settings import Settings
from cicaddy.utils.logger import get_logger

from .base_review_agent import BaseReviewAgent

logger = get_logger(__name__)


class MergeRequestAgent(BaseReviewAgent):
    """Base MR/PR agent using git CLI for platform-agnostic diffs.

    Provides merge request analysis using git CLI (via DiffAnalyzer) for diffs
    and CI environment variables for context. Platform-specific extensions
    (e.g., GitLab, GitHub) can override methods to use API-based diffs and
    richer metadata.
    """

    def __init__(self, settings: Optional[Settings] = None):
        super().__init__(settings)
        self.merge_request_iid = self._resolve_mr_iid()
        self.source_branch = self._resolve_source_branch()
        self.target_branch = self._resolve_target_branch()

    def _resolve_mr_iid(self) -> Optional[str]:
        """Resolve merge request IID from settings or environment."""
        mr_iid = getattr(self.settings, "merge_request_iid", None)
        if mr_iid:
            logger.debug(f"MR IID from settings: {mr_iid}")
            return mr_iid

        env_iid = os.getenv("CI_MERGE_REQUEST_IID")
        if env_iid:
            logger.debug(f"MR IID from environment: {env_iid}")
            return env_iid

        return None

    def _resolve_source_branch(self) -> str:
        """Resolve source branch from settings, environment, or default."""
        # Settings field first
        branch = getattr(self.settings, "merge_request_source_branch", None)
        if branch:
            logger.debug(f"Source branch from settings: {branch}")
            return branch

        # CI environment variables
        branch = os.getenv("CI_MERGE_REQUEST_SOURCE_BRANCH_NAME")
        if branch:
            logger.debug(
                f"Source branch from CI_MERGE_REQUEST_SOURCE_BRANCH_NAME: {branch}"
            )
            return branch

        branch = os.getenv("CI_COMMIT_REF_NAME")
        if branch:
            logger.debug(f"Source branch from CI_COMMIT_REF_NAME: {branch}")
            return branch

        logger.debug("Source branch defaulting to HEAD")
        return "HEAD"

    def _resolve_target_branch(self) -> str:
        """Resolve target branch from settings, environment, or default."""
        # Settings field first
        branch = getattr(self.settings, "merge_request_target_branch", None)
        if branch:
            logger.debug(f"Target branch from settings: {branch}")
            return branch

        # CI environment variables
        branch = os.getenv("CI_MERGE_REQUEST_TARGET_BRANCH_NAME")
        if branch:
            logger.debug(
                f"Target branch from CI_MERGE_REQUEST_TARGET_BRANCH_NAME: {branch}"
            )
            return branch

        branch = os.getenv("CI_DEFAULT_BRANCH")
        if branch:
            logger.debug(f"Target branch from CI_DEFAULT_BRANCH: {branch}")
            return branch

        logger.debug("Target branch defaulting to main")
        return "main"

    async def get_diff_content(self) -> str:
        """Get merge request diff content using git CLI."""
        self._validate_initialized()
        assert self.diff_analyzer is not None

        logger.info(
            f"Getting MR diff between {self.source_branch} and {self.target_branch}"
        )

        if self.source_branch == "HEAD":
            diff_content = await self.diff_analyzer.get_current_branch_diff(
                target_branch=self.target_branch,
                context_lines=self.settings.git_diff_context_lines,
            )
        else:
            diff_content = await self.diff_analyzer.get_merge_request_diff_by_branches(
                source_branch=self.source_branch,
                target_branch=self.target_branch,
                context_lines=self.settings.git_diff_context_lines,
            )

        return diff_content

    async def get_review_context(self) -> Dict[str, Any]:
        """Get merge request review context from environment variables."""
        logger.info(
            f"Getting review context for MR !{self.merge_request_iid}: "
            f"{self.source_branch} -> {self.target_branch}"
        )

        # Get diff summary for additional context
        diff_summary = await self.get_diff_summary()

        context: Dict[str, Any] = {
            "analysis_type": "merge_request",
            "mr_iid": self.merge_request_iid,
            "source_branch": self.source_branch,
            "target_branch": self.target_branch,
            "diff_summary": diff_summary,
        }

        # Build merge_request metadata from CI env vars
        mr_title = os.getenv("CI_MERGE_REQUEST_TITLE", "")
        context["merge_request"] = {
            "title": mr_title or f"MR !{self.merge_request_iid}",
            "source_branch": self.source_branch,
            "target_branch": self.target_branch,
        }

        # Add CI context if available
        ci_pipeline_source = os.getenv("CI_PIPELINE_SOURCE")
        ci_commit_sha = os.getenv("CI_COMMIT_SHA")
        if ci_pipeline_source:
            context["ci_context"] = {
                "pipeline_source": ci_pipeline_source,
                "commit_sha": ci_commit_sha,
            }

        return context

    def build_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Build analysis prompt for merge request code review.

        Supports DSPy YAML task definitions via AI_TASK_FILE environment variable.
        Falls back to built-in prompt if AI_TASK_FILE is not set.
        """
        # Check for DSPy task file first
        task_file = os.getenv("AI_TASK_FILE")
        if task_file:
            mr_context = self._prepare_dspy_context(context)
            dspy_prompt = self.build_dspy_prompt(task_file, mr_context)
            if dspy_prompt:
                return dspy_prompt

        # Get enabled tasks
        enabled_tasks = self.settings.get_enabled_tasks()

        # Format tools list
        tools_list = self._format_tools_list(context.get("mcp_tools", []))

        # Get MR context
        mr_data = context.get("merge_request", {})
        diff_content = context.get("diff", "")
        diff_summary = context.get("diff_summary", {})

        prompt = f"""
You are an AI agent performing merge request analysis.

Project: {context.get("project", {}).get("name", "Unknown")}
Merge Request: {mr_data.get("title", "Unknown")}
Source Branch: {mr_data.get("source_branch", self.source_branch)}
Target Branch: {mr_data.get("target_branch", self.target_branch)}

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
1. Analyze the merge request and code changes thoroughly
2. For each enabled task, provide detailed analysis:
   - code_review: Focus on code quality, best practices, potential bugs
   - security_scan: Identify security vulnerabilities and risks
3. Use available MCP tools when they can provide additional insights
4. Pay attention to parameter types and requirements when calling tools
5. Provide actionable recommendations and specific code suggestions
6. Consider the merge request context - this is a proposed set of changes

{self._get_task_specific_instructions(enabled_tasks)}

Please provide your comprehensive analysis.
"""

        return prompt

    def _prepare_dspy_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context dict with MR-specific data for DSPy prompt building."""
        mr_context = context.copy()

        mr_data = context.get("merge_request", {})
        mr_context["mr_title"] = mr_data.get("title", "Unknown")
        mr_context["mr_description"] = mr_data.get("description", "")
        mr_context["mr_author"] = mr_data.get("author", {}).get("name", "Unknown")
        mr_context["target_branch"] = mr_data.get("target_branch", self.target_branch)
        mr_context["source_branch"] = mr_data.get("source_branch", self.source_branch)
        mr_context["mr_iid"] = self.merge_request_iid
        mr_context["diff_content"] = context.get("diff", "")
        mr_context["enabled_tasks"] = self.settings.get_enabled_tasks()

        return mr_context

    def _format_tools_list(self, mcp_tools: list) -> str:
        """Format MCP tool schemas for inclusion in prompts."""
        tools_info = []
        for tool in mcp_tools:
            tool_name = tool.get("name", "unnamed")
            tool_info = (
                f"- {tool_name}: {tool.get('description', 'No description available')}"
            )
            if "server" in tool:
                tool_info += f" (from {tool['server']} server)"

            if tool.get("inputSchema"):
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

        return "\n".join(tools_info) if tools_info else "No MCP tools available"

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
Code Review Focus:
- **Summary**: Briefly summarize the changes
- **Issues**: Identify bugs, logic errors, edge cases, security vulnerabilities
- **Recommendations**: Suggest concrete fixes and improvements
- **Code Quality**: Assess readability, maintainability, best practices
"""
                )

        if "security_scan" in enabled_tasks:
            instructions.append(
                """
Security Analysis Focus:
- **Vulnerabilities**: Identify security issues (injection, auth, crypto, etc.)
- **Attack Vectors**: Describe potential attack scenarios
- **Risk Assessment**: Rate severity (Critical/High/Medium/Low)
- **Mitigation**: Provide specific security fixes
"""
            )

        return "\n".join(instructions)

    def get_session_id(self) -> str:
        """Get unique session ID for this MR analysis session."""
        return f"mr_{self.merge_request_iid or 'unknown'}"

    async def process_merge_request(self) -> Dict[str, Any]:
        """Main entry point for processing merge requests.

        Convenience method that calls analyze() and returns results.
        Note: Caller is expected to call initialize() before this method.
        """
        logger.info("Processing merge request %s", self.merge_request_iid)

        result = await self.analyze()

        return result["analysis_result"]

    async def generate_report(
        self, analysis_result: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate formatted report with MR-specific information."""
        report = await super().generate_report(analysis_result, context)

        if "merge_request" in context:
            report["merge_request"] = context["merge_request"]
        report["source_branch"] = context.get("source_branch", self.source_branch)
        report["target_branch"] = context.get("target_branch", self.target_branch)

        return report
