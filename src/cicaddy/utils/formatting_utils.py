"""Shared formatting utilities for comments and notifications."""

from typing import Any, Dict, Optional


class CommentFormatter:
    """Utility class for formatting comments and notifications."""

    DEFAULT_PROJECT_URL = "https://github.com/cicaddy/cicaddy"

    @staticmethod
    def format_footer(link_url: Optional[str] = None) -> str:
        """
        Format standard comment footer.

        Args:
            link_url: Optional URL to link to (defaults to project repository)

        Returns:
            Formatted footer string
        """
        url = link_url or CommentFormatter.DEFAULT_PROJECT_URL
        return f"🤖 Generated with [Cicaddy]({url})"

    @staticmethod
    def format_gitlab_footer(link_url: Optional[str] = None) -> str:
        """Format standard GitLab comment footer.

        Backward-compatible alias for format_footer().
        """
        return CommentFormatter.format_footer(link_url)

    @staticmethod
    def format_analysis_sections(analysis_result: Dict[str, Any]) -> str:
        """
        Format analysis results into sections for GitLab comments.

        Args:
            analysis_result: Analysis result containing tasks and content

        Returns:
            Formatted sections string
        """
        sections = ""

        for task, result in analysis_result.items():
            if isinstance(result, dict):
                if result.get("status") == "success":
                    sections += f"## {task.replace('_', ' ').title()}\n\n"
                    sections += result.get("content", "No content available") + "\n\n"
                else:
                    sections += f"## {task.replace('_', ' ').title()} (Error)\n\n"
                    sections += f"❌ Error: {result.get('error', 'Unknown error')}\n\n"

        return sections

    @staticmethod
    def format_new_execution_details(analysis_result: Dict[str, Any]) -> str:
        """
        Format execution details for new execution engine format.

        Args:
            analysis_result: Analysis result with execution_steps and tool_calls

        Returns:
            Formatted details string
        """
        execution_steps = analysis_result.get("execution_steps", [])
        tool_calls = analysis_result.get("tool_calls", [])

        if not tool_calls:
            return ""

        details = "## 🔧 Analysis Details\n\n"
        details += f"- **Execution Steps**: {len(execution_steps)}\n"
        details += f"- **Tool Calls**: {len(tool_calls)}\n"
        details += f"- **Model Used**: {analysis_result.get('model_used', 'Unknown')}\n"
        details += (
            f"- **Execution Time**: {analysis_result.get('execution_time', 0):.1f}s\n\n"
        )

        return details


class SlackFormatter:
    """Utility class for formatting Slack messages."""

    @staticmethod
    def format_simple_analysis_summary(
        results: Dict[str, Any], mr_data: Dict[str, Any]
    ) -> str:
        """
        Format simple analysis summary for Slack.

        Args:
            results: Analysis results
            mr_data: Merge request data

        Returns:
            Formatted Slack message
        """
        message = "🤖 *AI Analysis Complete*\n"
        message += f"📋 MR: {mr_data['title']}\n"
        message += f"🔗 {mr_data['web_url']}\n\n"

        for task, result in results.items():
            status_emoji = "✅" if result["status"] == "success" else "❌"
            message += f"{status_emoji} {task.replace('_', ' ').title()}\n"

        return message
