"""Slack webhook notifications with markdown conversion."""

from typing import Any, Dict, List, Optional

from cicaddy.utils.logger import get_logger

from .base import SlackBaseNotifier
from .formatters import MarkdownToSlackConverter, MessageSectionBuilder

logger = get_logger(__name__)


class SlackNotifier(SlackBaseNotifier):
    """Slack webhook notifier with markdown support."""

    def __init__(self, webhook_urls: str | List[str], ssl_verify: bool = True):
        # Handle backward compatibility for single URL string
        if isinstance(webhook_urls, str):
            webhook_urls = [webhook_urls]

        super().__init__(webhook_urls, ssl_verify)
        self.converter = MarkdownToSlackConverter()

    async def send_notification(self, message: str, **kwargs) -> Dict[str, Any]:
        """Send notification to Slack webhook."""
        # Convert markdown to Slack format
        slack_message = self.converter.convert(message)

        # Use base class implementation
        return await super().send_notification(slack_message, **kwargs)

    async def send_merge_request_notification(
        self, mr_data: Dict[str, Any], analysis_results: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Send formatted merge request analysis notification."""

        # Build rich notification
        message = self._format_mr_notification(mr_data, analysis_results)

        # Send with default bot settings
        return await self.send_notification(
            message,
            username=kwargs.get("username", "Cicaddy"),
            icon_emoji=kwargs.get("icon_emoji", ":robot_face:"),
            **kwargs,
        )

    def _format_mr_notification(
        self, mr_data: Dict[str, Any], analysis_results: Dict[str, Any]
    ) -> str:
        """Format merge request notification message."""

        # Use formatters for consistent styling
        message = MessageSectionBuilder.build_header("AI Analysis Complete")

        # Project info if available
        project_name = self._get_project_name()
        if project_name:
            message += f"🏗️ *Project:* {project_name}\n"

        # MR Info using formatter
        message += MessageSectionBuilder.build_mr_info(mr_data)

        # Add CI job link
        message = self._add_ci_job_link(message)
        message += "\n"

        # Analysis Results Summary using formatter
        message += MessageSectionBuilder.build_analysis_results_summary(
            analysis_results
        )

        # Add detailed results if available
        if any(r.get("status") == "success" for r in analysis_results.values()):
            message += MessageSectionBuilder.build_key_findings(analysis_results)

        # Footer with AI usage info
        report_id = "mr_notification"  # Simple ID for MR notifications
        message = self._add_footer(
            message, report_id, "Cicaddy", analysis_results
        )

        return message

    async def send_error_notification(
        self, error: str, context: Optional[Dict[str, Any]] = None
    ):
        """Send error notification to Slack."""
        message = MessageSectionBuilder.build_header("Cicaddy Error", "🚨")
        message += f"❌ *Error:* {error}\n"

        if context:
            message += "\n*Context:*\n"
            # Add project name prominently if available
            if context.get("project_name"):
                message += f"• *Project:* {context['project_name']}\n"

            for key, value in context.items():
                if key != "project_name":  # Already shown above
                    message += f"• *{key}:* {value}\n"

        # Add CI build URL if available
        message = self._add_ci_job_link(message)

        message += "\n────────────────────\n"
        message += "Please check the CI pipeline logs for more details."

        return await self.send_notification(
            message, username="Cicaddy", icon_emoji=":warning:"
        )
