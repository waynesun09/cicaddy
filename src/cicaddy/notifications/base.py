"""Base notification classes with common utilities."""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx

from cicaddy.utils.logger import get_logger
from cicaddy.utils.token_utils import TokenUsageExtractor

logger = get_logger(__name__)


class BaseNotifier(ABC):
    """Abstract base class for all notifiers with common utilities."""

    def __init__(self, webhook_url: str, ssl_verify: bool = True):
        self.webhook_url = webhook_url
        self.ssl_verify = ssl_verify

    @abstractmethod
    async def send_notification(self, message: str, **kwargs) -> Dict[str, Any]:
        """Send notification. Must be implemented by subclasses."""
        pass

    def _get_ci_job_url(self) -> str:
        """Get CI job URL from environment variables with enhanced parsing."""
        # CI environment provides these environment variables
        ci_pipeline_url = os.getenv("CI_PIPELINE_URL")
        ci_job_id = os.getenv("CI_JOB_ID")

        if ci_pipeline_url and ci_job_id:
            # Construct job URL from pipeline URL
            # CI_PIPELINE_URL format: https://gitlab.com/project/path/-/pipelines/12345
            base_url = (
                ci_pipeline_url.rsplit("/-/pipelines/", 1)[0]
                if "/-/pipelines/" in ci_pipeline_url
                else ""
            )
            if base_url:
                return f"{base_url}/-/jobs/{ci_job_id}"

        # Fallback to job URL if available
        return os.getenv("CI_JOB_URL", "")

    def _get_project_name(self) -> Optional[str]:
        """Get project name from environment variables."""
        # Try project name first, fallback to project path
        project_name = os.getenv("CI_PROJECT_NAME")
        if not project_name:
            project_path = os.getenv("CI_PROJECT_PATH")
            if project_path:
                # Extract project name from path (group/project -> project)
                project_name = project_path.split("/")[-1]

        return project_name

    def _get_branch_url(self, branch_name: str) -> str:
        """Get GitLab branch URL for creating clickable links."""
        project_url = os.getenv("CI_PROJECT_URL")
        if project_url and branch_name and branch_name != "unknown":
            return f"{project_url}/-/tree/{branch_name}"
        return ""

    def _get_commit_url(self, commit_sha: str = None) -> str:
        """Get GitLab commit URL for creating clickable links."""
        project_url = os.getenv("CI_PROJECT_URL")
        commit_sha = commit_sha or os.getenv("CI_COMMIT_SHA")
        if project_url and commit_sha:
            return f"{project_url}/-/commit/{commit_sha}"
        return ""

    def _normalize_agent_type_for_artifacts(self, agent_type: str) -> str:
        """Normalize agent type for artifact naming (matches BaseAIAgent logic)."""
        normalized = agent_type.lower()
        # Handle specific agent types first, then generic patterns
        if "branchreview" in normalized:
            return "branchreview"
        elif "mergerequest" in normalized or "mr" in normalized:
            return "mr"
        elif "cron" in normalized or "taskagent" in normalized:
            return "task"
        else:
            # Generic cleanup for other agent types
            return (
                normalized.replace("aiagent", "")
                .replace("agent", "")
                .replace("review", "")
                or "analysis"
            )

    def _get_html_artifact_url(
        self, report_id: str, agent_type: str = "analysis"
    ) -> str:
        """Generate CI artifact URL for HTML report."""
        ci_server_url = os.getenv("CI_SERVER_URL")
        ci_project_path = os.getenv("CI_PROJECT_PATH")
        ci_job_id = os.getenv("CI_JOB_ID")

        if ci_server_url and ci_project_path and ci_job_id:
            # GitLab artifact URL format:
            # https://gitlab.com/project/path/-/jobs/12345/artifacts/file/branchreview_20250827_213801.html
            # Use report_id directly since it already contains normalized agent type
            html_filename = f"{report_id}.html"
            return f"{ci_server_url}/{ci_project_path}/-/jobs/{ci_job_id}/artifacts/file/{html_filename}"

        return ""

    def _get_gitlab_auth_headers(self) -> dict:
        """Get GitLab authentication headers for API requests."""
        headers = {}

        # Try CI_JOB_TOKEN first (available in CI environments)
        ci_job_token = os.getenv("CI_JOB_TOKEN")
        if ci_job_token:
            headers["JOB-TOKEN"] = ci_job_token
            logger.debug("Using CI_JOB_TOKEN for GitLab authentication")
            return headers

        # Fall back to custom GITLAB_TOKEN if available
        gitlab_token = os.getenv("GITLAB_TOKEN")
        if gitlab_token:
            headers["PRIVATE-TOKEN"] = gitlab_token
            logger.debug("Using GITLAB_TOKEN for GitLab authentication")
            return headers

        logger.debug("No GitLab authentication tokens available")
        return headers

    def _extract_summary(self, text: str, max_length: int = 200) -> str:
        """Extract meaningful summary from analysis text."""
        if not text:
            return ""

        # Clean up the text
        text = text.strip()

        # Split into sentences
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]

        if not sentences:
            return text[:max_length] + "..." if len(text) > max_length else text

        # Try to get first complete sentence(s) within limit
        summary = ""
        for sentence in sentences:
            if len(summary + sentence + ".") <= max_length:
                summary += sentence + ". "
            else:
                break

        if not summary:
            # If no complete sentence fits, truncate the first sentence
            summary = sentences[0][: max_length - 3] + "..."

        return summary.strip()

    def _count_by_severity(self, findings: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count findings by severity level."""
        counts: Dict[str, int] = {}
        for finding in findings:
            result = finding.get("result", {})
            severity = result.get("severity", "unknown").lower()
            counts[severity] = counts.get(severity, 0) + 1
        return counts

    def _get_top_vulnerable_packages(
        self, vulnerabilities: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Get top vulnerable packages sorted by severity."""
        packages = []
        for vuln in vulnerabilities:
            packages.append(
                {
                    "name": vuln.get("package", "unknown"),
                    "severity": vuln.get("severity", "unknown"),
                }
            )

        # Sort by severity (critical > high > medium > low)
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        packages.sort(key=lambda x: severity_order.get(x["severity"].lower(), 4))

        return packages

    def _extract_insights(self, result: Dict[str, Any]) -> List[str]:
        """Extract key insights from tool result."""
        insights = []

        if isinstance(result, dict):
            # Look for common insight fields
            for key in ["insights", "recommendations", "findings", "summary"]:
                if key in result:
                    value = result[key]
                    if isinstance(value, list):
                        insights.extend([str(item) for item in value[:3]])
                    elif isinstance(value, str):
                        insights.append(value)

        return insights[:3]  # Return top 3 insights


class SlackBaseNotifier(BaseNotifier):
    """Base Slack notifier with common webhook functionality."""

    def __init__(self, webhook_urls: List[str], ssl_verify: bool = True):
        """Initialize with list of webhook URLs."""
        # Convert single URL to list for backward compatibility
        if isinstance(webhook_urls, str):
            webhook_urls = [webhook_urls]
        elif not webhook_urls:
            webhook_urls = []

        self.webhook_urls = webhook_urls
        self.ssl_verify = ssl_verify

        # Set first URL as primary for backward compatibility with parent class
        if webhook_urls:
            super().__init__(webhook_urls[0], ssl_verify)
        else:
            super().__init__("", ssl_verify)

    async def send_notification(self, message: str, **kwargs) -> Dict[str, Any]:
        """Send notification to all configured Slack webhooks."""
        if not self.webhook_urls:
            logger.warning("No Slack webhook URLs configured")
            return {"status": "skipped", "reason": "no_webhooks_configured"}

        logger.info(
            f"Sending Slack notification to {len(self.webhook_urls)} webhook(s)"
        )

        # Build Slack payload
        payload = self._build_payload(message, **kwargs)

        # Send to all webhooks
        results = []
        successful_sends = 0

        async with httpx.AsyncClient(verify=self.ssl_verify) as client:
            for i, webhook_url in enumerate(self.webhook_urls):
                try:
                    logger.debug(f"Sending to webhook {i + 1}/{len(self.webhook_urls)}")
                    response = await client.post(
                        webhook_url, json=payload, timeout=30.0
                    )
                    response.raise_for_status()

                    results.append(
                        {
                            "webhook_index": i,
                            "status": "success",
                            "webhook_url": webhook_url[:50] + "..."
                            if len(webhook_url) > 50
                            else webhook_url,
                        }
                    )
                    successful_sends += 1
                    logger.debug(f"Successfully sent to webhook {i + 1}")

                except Exception as e:
                    logger.error(f"Failed to send to webhook {i + 1}: {e}")
                    results.append(
                        {
                            "webhook_index": i,
                            "status": "failed",
                            "error": str(e),
                            "webhook_url": webhook_url[:50] + "..."
                            if len(webhook_url) > 50
                            else webhook_url,
                        }
                    )

        logger.info(
            f"Slack notification completed: {successful_sends}/{len(self.webhook_urls)} webhooks successful"
        )

        return {
            "status": "sent" if successful_sends > 0 else "failed",
            "message_length": len(message),
            "webhooks_total": len(self.webhook_urls),
            "webhooks_successful": successful_sends,
            "webhooks_failed": len(self.webhook_urls) - successful_sends,
            "results": results,
        }

    def _build_payload(self, message: str, **kwargs) -> Dict[str, Any]:
        """Build Slack webhook payload."""
        payload = {
            "text": message,
            "mrkdwn": True,
            "unfurl_links": False,
            "unfurl_media": False,
        }

        # Add optional fields
        if kwargs.get("username"):
            payload["username"] = kwargs["username"]

        if kwargs.get("icon_emoji"):
            payload["icon_emoji"] = kwargs["icon_emoji"]

        return payload

    def _add_ci_job_link(self, message: str) -> str:
        """Add CI job link to message if available."""
        ci_job_url = self._get_ci_job_url()
        if ci_job_url:
            message += f"🔗 *CI Job:* <{ci_job_url}|View Build>\n"
        return message

    def _add_html_report_link(
        self, message: str, report_id: str, agent_type: str = "analysis"
    ) -> str:
        """Add HTML report link to message if available."""
        html_artifact_url = self._get_html_artifact_url(report_id, agent_type)
        if html_artifact_url:
            message += f"📊 *Full Report:* <{html_artifact_url}|View HTML Report>\n"
        return message

    @staticmethod
    def _extract_ai_usage_info(
        analysis_result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Extract AI usage information from analysis result."""
        # Determine AI provider from analysis_result.model_used, defaulting to "unknown"
        default_provider = analysis_result.get("model_used", "unknown")

        token_data = TokenUsageExtractor.extract_token_usage(
            analysis_result, default_provider=default_provider, include_model_info=False
        )

        # Return None if no meaningful AI usage data
        if not TokenUsageExtractor.has_meaningful_usage(token_data):
            return None

        return token_data

    def _add_footer(
        self,
        message: str,
        report_id: str,
        agent_name: str = "Cicaddy",
        analysis_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add standard footer to message."""
        message += f"\n📋 *Report ID:* `{report_id}`\n"
        message += "────────────────────\n"

        # Try to extract AI usage info if analysis_result is provided
        if analysis_result:
            from cicaddy.utils.token_utils import TokenUsageExtractor

            ai_usage = self._extract_ai_usage_info(analysis_result)
            if ai_usage:
                compact_usage = TokenUsageExtractor.format_compact_usage(ai_usage)
                if compact_usage:
                    provider_display_name = ai_usage.get("provider", "Cicaddy").title()
                    message += (
                        f"🤖 *Powered by {provider_display_name}* | {compact_usage}"
                    )
                    return message

        # Fall back to default footer
        message += f"🤖 *Powered by {agent_name}*"
        return message
