"""Rich Slack notifications with enhanced formatting for all report types."""

import json
import os
from typing import Any, Dict, List, Optional, Union, cast

from cicaddy.ai_providers.base import ProviderMessage
from cicaddy.ai_providers.factory import create_provider, get_provider_config
from cicaddy.config.settings import load_settings
from cicaddy.utils.logger import get_logger

from .base import SlackBaseNotifier
from .formatters import MessageSectionBuilder, SeverityFormatter

logger = get_logger(__name__)


class RichSlackNotifier(SlackBaseNotifier):
    """Rich Slack notifier with enhanced formatting for all report types."""

    # Slack webhook message size limit (official limit from Slack API docs)
    # Reference: https://docs.slack.dev/changelog/2018-truncating-really-long-messages/
    SLACK_MESSAGE_LIMIT = 40000  # characters

    # AI-generated summary section limit (allow ~25% of total for summary sections)
    SLACK_SUMMARY_SECTION_LIMIT = 10000  # characters

    # Constants for truncation detection
    TRUNCATION_INDICATORS = [
        "[Content truncated due to token limits",  # Matches PromptTruncator.TRUNCATION_NOTICE
        "[... truncated]",
        "Analysis was truncated",
        "due to maximum token limit",
        "Response was truncated",
    ]

    def __init__(self, webhook_urls: str | List[str], ssl_verify: bool = True):
        # Handle backward compatibility for single URL string
        if isinstance(webhook_urls, str):
            webhook_urls = [webhook_urls]

        super().__init__(webhook_urls, ssl_verify)

    async def send_notification(self, message: str, **kwargs) -> Dict[str, Any]:
        """Send basic notification to Slack (compatibility method)."""
        return await super().send_notification(message, **kwargs)

    async def send_cron_report(
        self, report: Dict[str, Any], analysis_result: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Backward compatibility method - delegates to send_formatted_report."""
        return await self.send_formatted_report(report, analysis_result, **kwargs)

    async def send_formatted_report(
        self, report: Dict[str, Any], analysis_result: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Send formatted report to Slack with appropriate styling based on report type."""

        # Check agent type for specialized handling
        agent_type = report.get("agent_type", "").lower()
        task_type = report.get("task_type", "unknown")

        if "branch" in agent_type or "mergerequest" in agent_type or "mr" in agent_type:
            return await self.send_code_review_report(report, analysis_result, **kwargs)
        elif task_type == "security_audit":
            return await self.send_security_report(report, analysis_result, **kwargs)
        elif task_type == "quality_report":
            return await self.send_quality_report(report, analysis_result, **kwargs)
        elif task_type == "dependency_check":
            return await self.send_dependency_report(report, analysis_result, **kwargs)
        elif task_type == "custom":
            return await self.send_custom_report(report, analysis_result, **kwargs)
        else:
            return await self.send_general_report(report, analysis_result, **kwargs)

    async def send_code_review_report(
        self, report: Dict[str, Any], analysis_result: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Send unified code review report that adapts to branch or MR context."""

        # Detect context: MR data vs branch data
        mr_data = report.get("merge_request", {})
        is_mr_review = bool(mr_data)

        # Get branch information
        if is_mr_review:
            # MR context: get from MR data
            source_branch = mr_data.get("source_branch", "unknown")
            target_branch = mr_data.get("target_branch", "unknown")
        else:
            # Branch context: get from report or CI environment
            source_branch = report.get("source_branch", "unknown")
            target_branch = report.get("target_branch", "unknown")

            # Fallback to CI environment variables if not in report
            if source_branch == "unknown":
                source_branch = os.getenv("CI_COMMIT_BRANCH", "unknown")
            if target_branch == "unknown":
                target_branch = os.getenv("CI_DEFAULT_BRANCH", "main")

        # Build adaptive message
        status = analysis_result.get("status", "unknown")
        execution_time = analysis_result.get("execution_time", 0)

        # Adaptive header and emoji
        if is_mr_review:
            status_emoji = "✅" if status == "success" else "❌"
            header = f"Merge Request Analysis Complete {status_emoji}"
        else:
            status_emoji = "🔍" if status == "success" else "⚠️"
            header = f"Branch Review Complete {status_emoji}"

        message = MessageSectionBuilder.build_header(header)

        # Project info
        project_name = self._get_project_name() or report.get("project", "Unknown")
        message += f"🏗️ *Project:* {project_name}\n"

        # MR-specific information (only when available)
        if is_mr_review:
            message += f"📋 *MR:* {mr_data.get('title', 'Unknown')}\n"
            message += (
                f"👤 *Author:* {mr_data.get('author', {}).get('name', 'Unknown')}\n"
            )

        # Clickable branch links (core functionality)
        source_branch_link = self._format_branch_link(source_branch)
        target_branch_link = self._format_branch_link(target_branch)

        if is_mr_review:
            message += f"🌿 *Branches:* {source_branch_link} → {target_branch_link}\n"
        else:
            message += (
                f"🌿 *Branch Analysis:* {source_branch_link} → {target_branch_link}\n"
            )

        message += f"⏱️ *Analysis Time:* {execution_time:.1f}s\n"

        # MR URL link (only for MR reviews)
        if is_mr_review:
            mr_url = mr_data.get("web_url")
            if mr_url:
                mr_iid = mr_data.get("iid", "N/A")
                message += f"🔗 *Merge Request:* <{mr_url}|#{mr_iid}>\n"

        # Add CI job link
        message = self._add_ci_job_link(message)
        message += "\n"

        # Analysis results (shared logic)
        ai_analysis = analysis_result.get("ai_analysis", "")
        if ai_analysis:
            # Extract key insights from analysis
            summary = self._extract_branch_summary(ai_analysis)
            if summary:
                message += "📊 *Analysis Summary:*\n"
                message += f"{summary}\n\n"

        # Tool execution summary (shared logic)
        tool_calls = analysis_result.get("tool_calls", [])
        if tool_calls:
            message += f"🔧 *Tools Used:* {len(tool_calls)} tools executed\n"

        # Status and recommendations (shared logic)
        if status == "success":
            message += "✅ *Status:* Analysis completed successfully\n"

            # Extract recommendations if available
            recommendations = self._extract_recommendations(ai_analysis)
            if recommendations:
                message += f"\n💡 *Key Recommendations:*\n{recommendations}\n"
        else:
            error = analysis_result.get("error", "Unknown error")
            message += f"❌ *Status:* Analysis failed - {error}\n"

        # Clickable commit link (shared logic)
        commit_sha = os.getenv("CI_COMMIT_SHA")
        if commit_sha:
            commit_link = self._format_commit_link(commit_sha)
            message += f"\n🔗 *Commit:* {commit_link}"

        # Footer (adaptive)
        report_id = report.get("report_id", "unknown")
        message = self._add_html_report_link(
            message, report_id, report.get("agent_type", "analysis")
        )

        if is_mr_review:
            footer_agent = "GitLab MR Review Agent"
            username = kwargs.get("username", "GitLab MR Review")
            icon_emoji = kwargs.get("icon_emoji", ":merge:")
        else:
            footer_agent = "GitLab Branch Review Agent"
            username = kwargs.get("username", "GitLab Branch Review")
            icon_emoji = kwargs.get("icon_emoji", ":mag:")

        message = self._add_footer(message, report_id, footer_agent, analysis_result)

        return await self.send_notification(
            message, username=username, icon_emoji=icon_emoji, **kwargs
        )

    def _extract_branch_summary(self, ai_analysis: str) -> str:
        """Extract concise summary from branch analysis with flexible fallbacks."""
        if not ai_analysis:
            return ""

        # Try structured extraction first
        structured_summary = self._extract_structured_summary(ai_analysis)
        if structured_summary:
            return structured_summary

        # Try flexible extraction
        flexible_summary = self._extract_flexible_summary(ai_analysis)
        if flexible_summary:
            return flexible_summary

        # Final fallback: take first meaningful paragraph
        return self._extract_simple_summary(ai_analysis)

    def _extract_structured_summary(self, ai_analysis: str) -> str:
        """Extract summary from structured markdown sections."""
        # Look for summary sections in common formats
        summary_indicators = [
            "## Summary",
            "## Analysis Summary",
            "### Summary",
            "**Summary:**",
            "Summary:",
            "## Key Findings",
            "### Key Findings",
            "**Key Findings:**",
        ]

        lines = ai_analysis.split("\n")
        summary_lines = []
        in_summary = False

        for line in lines:
            line = line.strip()

            # Check if we hit a summary section
            if any(indicator in line for indicator in summary_indicators):
                in_summary = True
                continue

            # If in summary and hit another section, stop
            if in_summary and (
                line.startswith("##") or (line.startswith("**") and line.endswith("**"))
            ):
                break

            # Collect summary lines
            if in_summary and line:
                # Clean up markdown and limit length
                clean_line = line.replace("*", "").replace("`", "").strip()
                if clean_line and not clean_line.startswith("#"):
                    summary_lines.append(f"• {clean_line}")

                # Limit to 3-4 key points
                if len(summary_lines) >= 4:
                    break

        return "\n".join(summary_lines) if summary_lines else ""

    def _extract_flexible_summary(self, ai_analysis: str) -> str:
        """Extract summary using flexible approach when no structured sections found."""
        lines = ai_analysis.split("\n")
        summary_lines = []

        # Look for meaningful sentences with business value
        for line in lines:
            line = line.strip()

            # Skip empty lines, markdown headers, and tool execution details
            if (
                not line
                or line.startswith("#")
                or any(
                    skip_term in line.lower()
                    for skip_term in [
                        "mcp tool",
                        "tool was",
                        "executed",
                        "following tools",
                        "tools were utilized",
                        "tool call",
                        "```",
                    ]
                )
            ):
                continue

            # Look for lines with analytical content
            if any(
                content_term in line.lower()
                for content_term in [
                    "analysis",
                    "review",
                    "shows",
                    "indicates",
                    "reveals",
                    "finds",
                    "suggests",
                    "recommends",
                    "identified",
                    "observed",
                    "detected",
                    "changes",
                    "added",
                    "removed",
                    "modified",
                    "updated",
                    "improved",
                    "issue",
                    "problem",
                    "concern",
                    "warning",
                    "error",
                    "success",
                    "quality",
                    "security",
                    "performance",
                    "maintainability",
                ]
            ):
                # Clean up and format the line
                clean_line = line.replace("*", "").replace("`", "").strip()
                if len(clean_line) > 20 and len(clean_line) < 200:  # Reasonable length
                    summary_lines.append(f"• {clean_line}")

                    # Limit to 3 key points for Slack readability
                    if len(summary_lines) >= 3:
                        break

        # If still no summary, try to get first meaningful sentences
        if not summary_lines:
            sentences = [
                s.strip()
                for s in ai_analysis.replace("\n", " ").split(".")
                if s.strip()
            ]
            for sentence in sentences[:2]:  # First 2 sentences
                if len(sentence) > 30 and not any(
                    skip in sentence.lower() for skip in ["tool", "executed", "mcp"]
                ):
                    clean_sentence = sentence.replace("*", "").replace("`", "").strip()
                    summary_lines.append(f"• {clean_sentence}.")
                    break

        return "\n".join(summary_lines) if summary_lines else ""

    def _extract_simple_summary(self, ai_analysis: str) -> str:
        """Simple fallback: extract first meaningful paragraph from AI analysis."""
        if not ai_analysis:
            return ""

        # Split into paragraphs and find the first substantial one
        paragraphs = ai_analysis.split("\n\n")

        for paragraph in paragraphs:
            lines = [line.strip() for line in paragraph.split("\n") if line.strip()]

            # Skip headers, code blocks, and tool execution details
            content_lines = []
            for line in lines:
                if (
                    not line.startswith("#")
                    and not line.startswith("```")
                    and not any(
                        skip in line.lower()
                        for skip in ["mcp tool", "tool was", "executed"]
                    )
                    and len(line) > 10
                ):  # Meaningful content
                    content_lines.append(line)

            if content_lines:
                # Take first 2-3 sentences for concise summary
                content = " ".join(content_lines)
                sentences = content.split(". ")

                summary_sentences = []
                for sentence in sentences[:3]:  # Max 3 sentences
                    if len(sentence.strip()) > 15:  # Meaningful sentence
                        summary_sentences.append(sentence.strip())

                if summary_sentences:
                    summary = ". ".join(summary_sentences)
                    if not summary.endswith("."):
                        summary += "."

                    # Limit length for Slack readability
                    if len(summary) > 300:
                        summary = summary[:297] + "..."

                    return summary

        # Last resort: just take first 200 characters
        clean_text = ai_analysis.replace("#", "").replace("*", "").strip()
        if len(clean_text) > 200:
            return clean_text[:197] + "..."
        return clean_text if len(clean_text) > 10 else ""

    def _format_branch_link(self, branch_name: str) -> str:
        """Format branch name as clickable link or plain text fallback."""
        if not branch_name or branch_name == "unknown":
            return "`unknown`"

        branch_url = self._get_branch_url(branch_name)
        if branch_url:
            # Slack format for clickable links: <URL|text>
            return f"<{branch_url}|`{branch_name}`>"
        else:
            # Fallback to plain text if URL can't be constructed
            return f"`{branch_name}`"

    def _format_commit_link(self, commit_sha: Optional[str] = None) -> str:
        """Format commit SHA as clickable link or plain text fallback."""
        commit_sha = commit_sha or os.getenv("CI_COMMIT_SHA")
        if not commit_sha:
            return "`unknown`"

        commit_url = self._get_commit_url(commit_sha)
        short_sha = commit_sha[:8]

        if commit_url:
            # Slack format for clickable links: <URL|text>
            return f"<{commit_url}|`{short_sha}...`>"
        else:
            # Fallback to plain text if URL can't be constructed
            return f"`{short_sha}...`"

    def _extract_recommendations(self, ai_analysis: str) -> str:
        """Extract recommendations from branch analysis."""
        if not ai_analysis:
            return ""

        # Look for recommendation sections
        rec_indicators = [
            "## Recommendations",
            "### Recommendations",
            "**Recommendations:**",
            "Recommendations:",
            "## Next Steps",
            "### Next Steps",
        ]

        lines = ai_analysis.split("\n")
        rec_lines = []
        in_recommendations = False

        for line in lines:
            line = line.strip()

            # Check if we hit a recommendations section
            if any(indicator in line for indicator in rec_indicators):
                in_recommendations = True
                continue

            # If in recommendations and hit another section, stop
            if in_recommendations and (
                line.startswith("##") or (line.startswith("**") and line.endswith("**"))
            ):
                break

            # Collect recommendation lines
            if in_recommendations and line:
                # Clean up markdown and limit length
                clean_line = line.replace("*", "").replace("`", "").strip()
                if clean_line and not clean_line.startswith("#"):
                    if clean_line.startswith("-") or clean_line.startswith("•"):
                        rec_lines.append(f"• {clean_line[1:].strip()}")
                    else:
                        rec_lines.append(f"• {clean_line}")

                # Limit to 3 key recommendations
                if len(rec_lines) >= 3:
                    break

        return "\n".join(rec_lines) if rec_lines else ""

    async def send_security_report(
        self, report: Dict[str, Any], analysis_result: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Send security audit report with threat indicators."""

        # Count security findings
        findings = analysis_result.get("security_findings", [])
        total_findings = len(findings)

        # Use formatters for consistent styling
        severity_emoji = SeverityFormatter.get_severity_emoji(total_findings)
        message = MessageSectionBuilder.build_header(
            "Security Audit Complete", severity_emoji
        )
        message += MessageSectionBuilder.build_project_info(
            report["project"],
            report["execution_time"],
            report["scope"].replace("_", " ").title(),
        )

        # Add CI job link
        message = self._add_ci_job_link(message)

        message += "\n"

        # Security findings summary
        severity_counts = self._count_by_severity(findings) if findings else {}
        message += MessageSectionBuilder.build_security_findings_summary(
            findings, severity_counts
        )

        # AI analysis summary
        if "ai_analysis" in analysis_result:
            ai_summary = self._extract_summary(analysis_result["ai_analysis"])
            if ai_summary:
                message += f"\n🤖 *AI Analysis:*\n{ai_summary}\n"

        # Add HTML report link and footer
        message = self._add_html_report_link(
            message, report["report_id"], report.get("agent_type", "analysis")
        )
        message = self._add_footer(
            message, report["report_id"], "Cicaddy", analysis_result
        )

        return await self.send_notification(
            message,
            username="Security Agent",
            icon_emoji=":shield:",
        )

    async def send_quality_report(
        self, report: Dict[str, Any], analysis_result: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Send code quality report with metrics."""

        metrics = analysis_result.get("quality_metrics", [])

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(metrics)
        quality_emoji = SeverityFormatter.get_severity_emoji(
            100 - quality_score
        )  # Invert for quality

        message = MessageSectionBuilder.build_header(
            "Code Quality Report", quality_emoji
        )
        message += MessageSectionBuilder.build_project_info(
            report["project"], report["execution_time"]
        )
        message += f"🎯 *Overall Score:* {quality_score}/100\n"

        # Add CI job link
        message = self._add_ci_job_link(message)

        message += "\n"

        # Quality metrics
        if metrics:
            quality_metrics = {}
            for metric in metrics[:5]:  # Top 5 metrics
                tool_name = metric.get("tool", "Unknown")
                result = metric.get("result", {})
                if "score" in result:
                    quality_metrics[tool_name] = f"{result['score']}%"
                elif "value" in result:
                    quality_metrics[tool_name] = result["value"]

            if quality_metrics:
                message += MessageSectionBuilder.build_quality_metrics(quality_metrics)

        # Recommendations
        ai_analysis_text = analysis_result.get("ai_analysis", "")
        recommendations = self._extract_recommendations(ai_analysis_text)
        if recommendations:
            message += "\n💡 *Top Recommendations:*\n"
            for rec in recommendations[:3]:
                message += f"• {rec}\n"

        # Trends (if available)
        trends = kwargs.get("trends", {})
        if trends:
            message += MessageSectionBuilder.build_trends_section(trends)

        # Add HTML report link and footer
        message = self._add_html_report_link(
            message, report["report_id"], report.get("agent_type", "analysis")
        )
        message = self._add_footer(
            message, report["report_id"], "Cicaddy", analysis_result
        )

        return await self.send_notification(
            message,
            username="Quality Agent",
            icon_emoji=":bar_chart:",
        )

    async def send_dependency_report(
        self, report: Dict[str, Any], analysis_result: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Send dependency audit report."""

        vulnerabilities = analysis_result.get("vulnerabilities", [])
        updates_available = analysis_result.get("updates_available", [])

        # Determine alert level
        vuln_count = len(vulnerabilities)
        alert_emoji = SeverityFormatter.get_severity_emoji(vuln_count)

        message = MessageSectionBuilder.build_header(
            "Dependency Audit Report", alert_emoji
        )
        message += MessageSectionBuilder.build_project_info(
            report["project"], report["execution_time"]
        )

        # Add CI job link
        message = self._add_ci_job_link(message)

        message += "\n"

        # Dependency summary
        security_updates = [u for u in updates_available if u.get("security", False)]
        message += MessageSectionBuilder.build_dependency_summary(
            vuln_count, len(updates_available), security_updates
        )

        # Top vulnerable packages
        if vulnerabilities:
            top_packages = self._get_top_vulnerable_packages(vulnerabilities)
            if top_packages:
                message += MessageSectionBuilder.build_top_packages_section(
                    top_packages
                )

        # Action items
        if vuln_count > 0 or security_updates:
            message += MessageSectionBuilder.build_action_items(
                vuln_count, security_updates
            )

        # Add HTML report link and footer
        message = self._add_html_report_link(
            message, report["report_id"], report.get("agent_type", "analysis")
        )
        message = self._add_footer(
            message, report["report_id"], "Cicaddy", analysis_result
        )

        return await self.send_notification(
            message,
            username="Dependency Agent",
            icon_emoji=":package:",
        )

    async def send_custom_report(
        self, report: Dict[str, Any], analysis_result: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Send custom analysis report with AI-powered summarization."""

        # Build simple header
        message = MessageSectionBuilder.build_header("AI Agent Analysis Report")

        # Add CI job link
        message = self._add_ci_job_link(message)
        message += "\n"

        # Calculate remaining space for content (Slack limit 40,000 chars)
        header_length = len(message)
        footer_length = 200  # Estimate for HTML link + footer
        # Use 90% of limit as conservative buffer for encoding overhead
        available_space = (
            int(self.SLACK_MESSAGE_LIMIT * 0.9) - header_length - footer_length
        )

        # Process analysis results with AI summarization
        content = ""

        # Check if we have analysis results
        if analysis_result.get("status") == "success" and analysis_result.get(
            "ai_analysis"
        ):
            # Get both potential data sources
            knowledge = analysis_result.get("accumulated_knowledge", {})
            ai_analysis_text = analysis_result.get("ai_analysis", "")
            summary_generated = False

            # PRIORITY 1: Use BOTH tool data AND AI analysis together (richest summary)
            if (
                knowledge
                and knowledge.get("tool_results")
                and ai_analysis_text
                and len(ai_analysis_text.strip()) > 0
            ):
                try:
                    slack_summary = await self._generate_summary_from_combined_sources(
                        tool_results=knowledge["tool_results"],
                        ai_analysis=ai_analysis_text,
                        servers_used=knowledge.get("servers_used", []),
                        report_context={
                            "task_type": report.get("task_type", "monitoring"),
                            "project": report.get("project", "System"),
                            "agent_type": report.get("agent_type", "analysis"),
                        },
                    )
                    content += f"{slack_summary}\n\n"
                    logger.info(
                        "Generated Slack summary from combined sources (tool data + AI analysis)"
                    )
                    summary_generated = True
                except Exception as e:
                    logger.warning(f"Failed to generate combined summary: {e}")
                    # Let subsequent fallbacks handle it

            # FALLBACK 1: Only accumulated_knowledge available (no AI analysis) OR combined failed
            if not summary_generated and knowledge and knowledge.get("tool_results"):
                # Use accumulated knowledge for rich, data-driven summary
                try:
                    slack_summary = await self._generate_summary_from_tool_data(
                        tool_results=knowledge["tool_results"],
                        servers_used=knowledge.get("servers_used", []),
                        report_context={
                            "task_type": report.get("task_type", "monitoring"),
                            "project": report.get("project", "System"),
                            "agent_type": report.get("agent_type", "analysis"),
                        },
                    )
                    content += f"{slack_summary}\n\n"
                    logger.info(
                        "Generated Slack summary from accumulated knowledge (full tool data)"
                    )
                    summary_generated = True
                except Exception as e:
                    logger.warning(
                        f"Failed to generate summary from accumulated knowledge: {e}"
                    )

            # FALLBACK 2: Only ai_analysis available (no tool data) OR all previous failed
            if (
                not summary_generated
                and ai_analysis_text
                and len(ai_analysis_text.strip()) > 0
            ):
                # Check if analysis was truncated due to token limits
                was_truncated = self._check_if_truncated(
                    ai_analysis_text, analysis_result
                )

                # Try AI-powered Slack formatting (old approach with compacted text)
                try:
                    slack_summary = await self._generate_slack_summary(
                        ai_analysis_text,
                        {
                            "task_type": report.get("task_type", "monitoring"),
                            "project": report.get("project", "System"),
                            "agent_type": report.get("agent_type", "analysis"),
                        },
                    )

                    if slack_summary:
                        # Use AI-generated Slack-native format
                        content += f"{slack_summary}\n\n"
                        summary_generated = True

                        # Add truncation notice if applicable
                        if was_truncated:
                            truncation_notice = self._create_truncation_notice(
                                analysis_result
                            )
                            content += f"{truncation_notice}\n\n"
                    else:
                        # Fallback to existing method if AI formatting fails
                        raise Exception("AI Slack formatting returned None")

                except Exception as e:
                    logger.warning(f"AI Slack formatting failed: {e}")
                    # No summary_generated flag set here - will fall through to template if needed

        # Handle failed analysis
        elif analysis_result.get("status") == "failed":
            error = analysis_result.get("error", "Unknown error")

            # Check if failure was due to token limits
            if "token" in error.lower() and (
                "limit" in error.lower() or "exceed" in error.lower()
            ):
                content += "⚠️ *Analysis limited by token constraints*\n"
                content += "The analysis scope exceeded AI model limits. "
                # Try to show what was completed
                tool_calls = analysis_result.get("tool_calls", [])
                if tool_calls:
                    successful_calls = [
                        tc for tc in tool_calls if tc.get("status") == "success"
                    ]
                    if successful_calls:
                        content += f"Successfully completed {len(successful_calls)} tool executions before limit.\n\n"
                    else:
                        content += "Consider reducing the analysis scope.\n\n"
                else:
                    content += "Consider reducing the analysis scope.\n\n"
            else:
                # Truncate error message if too long
                if len(error) > available_space - 100:
                    error = error[: available_space - 103] + "..."
                content += f"❌ *Analysis failed:* {error}\n\n"

        # Fallback for unexpected structure
        else:
            content += "❌ *No analysis results available*\n"
            content += "The analysis may have encountered errors or no tools were executed.\n\n"

        message += content

        # Add HTML report link and footer
        message = self._add_html_report_link(
            message, report["report_id"], report.get("agent_type", "analysis")
        )
        message = self._add_footer(
            message, report["report_id"], "Cicaddy", analysis_result
        )

        return await self.send_notification(
            message,
            username="AI Agent",
            icon_emoji=":robot_face:",
        )

    async def send_general_report(
        self, report: Dict[str, Any], analysis_result: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Send general analysis report."""

        message = MessageSectionBuilder.build_header("Scheduled Analysis Report")
        message += MessageSectionBuilder.build_project_info(
            report["project"],
            report["execution_time"],
            report.get("task_type", "analysis").replace("_", " ").title(),
        )

        # Add CI job link
        message = self._add_ci_job_link(message)

        message += "\n"

        # General analysis summary
        if "ai_insights" in analysis_result:
            insights = self._extract_summary(analysis_result["ai_insights"])
            if insights:
                message += MessageSectionBuilder.build_ai_analysis_section(insights)

        # Add HTML report link and footer
        message = self._add_html_report_link(
            message, report["report_id"], report.get("agent_type", "analysis")
        )
        message = self._add_footer(
            message, report["report_id"], "Cicaddy", analysis_result
        )

        return await self.send_notification(
            message,
            username="Cicaddy",
            icon_emoji=":robot_face:",
        )

    # Helper methods

    async def _generate_slack_summary(
        self, ai_analysis: str, report_context: Dict[str, Any]
    ) -> Optional[str]:
        """Use AI to transform technical analysis into Slack-compatible format."""
        try:
            settings = load_settings()
            provider_config = get_provider_config(settings)
            provider_name = settings.ai_provider or "gemini"

            # Create AI provider
            ai_provider = create_provider(provider_name, provider_config)
            await ai_provider.initialize()

            try:
                # Extract context information
                report_type = report_context.get("task_type", "monitoring")
                project_name = report_context.get("project", "System")

                # Extract system name from analysis (e.g., DataRouter, Kubernetes)
                system_name = self._extract_system_name(ai_analysis, project_name)

                # Build context-aware prompt
                prompt = f"""Transform this technical monitoring analysis into a concise Slack notification.

ANALYSIS TO TRANSFORM:
{ai_analysis}

SLACK FORMAT REQUIREMENTS:
- Use Slack markdown: **bold** for important numbers, *italic* for emphasis, `code` for technical terms
- Add relevant emojis: 📊 (metrics), 📈 (trends), 🚨 (critical), ⚠️ (warnings), 💡 (recommendations), 🔍 (analysis)
- Create clear sections with bullet points using •
- Highlight key numbers and percentages with **bold**
- Maximum {self.SLACK_SUMMARY_SECTION_LIMIT} characters total
- Business-focused language that engineers and managers can quickly understand

STRUCTURE:
🤖 *{system_name} {report_type.title()} Report*

📊 *Key Metrics:*
• [extract and bold the most important numbers]

🚨 *Critical Issues:* (if any high-priority problems found)
• [list urgent problems that need immediate attention]

⚠️ *Component Health:* (if component-level issues found)
• [list component status with error counts]

🔍 *Failure Patterns:* (if recurring issues identified)
• [common error patterns or timeout issues]

💡 *Action Required:* (if specific recommendations available)
• [actionable next steps for engineering teams]

GUIDELINES:
- Focus on what engineering teams need to know immediately
- Use bullet points, not paragraphs
- Bold all important numbers (request counts, percentages, error counts)
- Skip technical implementation details
- Prioritize business impact and actionable insights
- If this is infrastructure monitoring, emphasize system health and performance
"""

                messages = [ProviderMessage(content=prompt, role="user")]
                response = await ai_provider.chat_completion(messages)

                # Clean and format the response
                slack_content = response.content.strip()

                # Ensure it fits within Slack limits
                if len(slack_content) > self.SLACK_SUMMARY_SECTION_LIMIT:
                    # Intelligently truncate while preserving structure
                    slack_content = self._truncate_slack_content(
                        slack_content, self.SLACK_SUMMARY_SECTION_LIMIT
                    )

                logger.debug(
                    f"AI-generated Slack summary: {len(slack_content)} characters"
                )
                return slack_content

            finally:
                await ai_provider.shutdown()

        except Exception as e:
            logger.error(f"Failed to generate AI Slack summary: {e}")
            # Return None to trigger fallback
            return None

    def _extract_system_name(self, ai_analysis: str, project_name: str) -> str:
        """Extract system name from analysis for better header formatting."""
        # Look for common system names in the analysis
        analysis_lower = ai_analysis.lower()

        system_indicators = {
            "datarouter": "DataRouter",
            "kubernetes": "Kubernetes",
            "k8s": "Kubernetes",
            "prometheus": "Prometheus",
            "elasticsearch": "Elasticsearch",
            "redis": "Redis",
            "postgresql": "PostgreSQL",
            "mysql": "MySQL",
            "nginx": "Nginx",
            "apache": "Apache",
            "monitoring": "System Monitoring",
            "infrastructure": "Infrastructure",
        }

        for indicator, name in system_indicators.items():
            if indicator in analysis_lower:
                return name

        # Fallback to project name or generic
        if (
            project_name
            and project_name != "Unknown"
            and project_name != "External Analysis"
        ):
            return project_name

        return "System"

    def _truncate_slack_content(self, content: str, max_length: int) -> str:
        """Intelligently truncate Slack content while preserving structure."""
        if len(content) <= max_length:
            return content

        lines = content.split("\n")
        essential_lines = []
        optional_lines = []

        for line in lines:
            # Keep headers and critical sections
            if any(
                essential in line
                for essential in [
                    "🤖 *",
                    "📊 *Key Metrics",
                    "🚨 *Critical Issues",
                    "💡 *Action Required",
                ]
            ):
                essential_lines.append(line)
            # Mark detailed sections as optional
            elif any(
                optional in line
                for optional in ["⚠️ *Component Health", "🔍 *Failure Patterns"]
            ):
                optional_lines.append(line)
            else:
                essential_lines.append(line)

        # Rebuild with essential content first
        truncated = "\n".join(essential_lines)

        # Add optional content if space allows
        remaining_space = (
            max_length - len(truncated) - 50
        )  # Buffer for truncation notice
        if remaining_space > 100 and optional_lines:
            optional_content = "\n".join(optional_lines)
            if len(optional_content) <= remaining_space:
                truncated += "\n" + optional_content
            else:
                truncated += (
                    "\n"
                    + optional_content[: remaining_space - 30]
                    + "\n\n*[Details truncated]*"
                )

        return truncated

    async def _generate_ai_summary(
        self, custom_analysis: List[Dict[str, Any]], report: Dict[str, Any]
    ) -> str:
        """Generate AI-powered summary of analysis results."""
        try:
            settings = load_settings()
            provider_config = get_provider_config(settings)
            provider_name = settings.ai_provider or "gemini"

            # Create AI provider
            ai_provider = create_provider(provider_name, provider_config)
            await ai_provider.initialize()

            try:
                # Prepare analysis data for AI
                analysis_data = self._prepare_analysis_for_ai(custom_analysis, report)

                summary_prompt = f"""
Analyze the following AI agent analysis results and provide a concise executive summary.

Analysis Results:
{analysis_data}

Please provide a summary in 2-3 sentences that covers:
1. What type of analysis was performed (based on the tools used)
2. Key findings or insights discovered
3. Overall status or any notable issues

Keep it concise and business-focused for a Slack notification. Be generic - don't assume this is
system monitoring.
"""

                messages = [ProviderMessage(content=summary_prompt, role="user")]
                response = await ai_provider.chat_completion(messages)

                # Clean and format the response
                summary = response.content.strip()

                # Limit length for Slack
                if len(summary) > 300:
                    summary = summary[:297] + "..."

                return summary

            finally:
                await ai_provider.shutdown()

        except Exception as e:
            logger.error(f"Failed to generate AI summary: {e}")
            # Fallback to basic summary
            return self._generate_basic_summary(custom_analysis, report)

    def _prepare_analysis_for_ai(
        self, custom_analysis: List[Dict[str, Any]], report: Dict[str, Any]
    ) -> str:
        """Prepare analysis data in a format suitable for AI processing."""
        summary_parts = []

        summary_parts.append(f"Project: {report.get('project', 'Unknown')}")
        summary_parts.append(
            f"Analysis Duration: {report.get('execution_time', 0):.1f} seconds"
        )
        summary_parts.append(f"Analysis Type: {report.get('task_type', 'custom')}")

        if custom_analysis:
            for i, analysis in enumerate(
                custom_analysis[:3], 1
            ):  # Limit to first 3 analyses
                summary_parts.append(f"\nAnalysis {i}:")

                # Add execution details
                execution_time = analysis.get("execution_time", 0)
                summary_parts.append(f"  Execution time: {execution_time:.1f}s")

                # Add tool calls information
                tool_calls = analysis.get("tool_calls", [])
                if tool_calls:
                    summary_parts.append(f"  MCP Tools executed: {len(tool_calls)}")
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("tool_name", "Unknown")
                        arguments = tool_call.get("arguments", {})
                        # Format tool parameters generically
                        param_summary = []
                        for key, value in arguments.items():
                            if key == "numOfDays":
                                param_summary.append(f"last {value} days")
                            elif key == "numOfHours":
                                param_summary.append(f"last {value} hours")
                            elif key in ["accountName", "requestUUID"]:
                                param_summary.append(f"{key}: {value}")
                            else:
                                param_summary.append(f"{key}: {value}")

                        if param_summary:
                            summary_parts.append(
                                f"    - {tool_name}: {', '.join(param_summary)}"
                            )
                        else:
                            summary_parts.append(f"    - {tool_name}: executed")

                # Add AI analysis if available
                ai_analysis = analysis.get("ai_analysis", "")
                if ai_analysis and len(ai_analysis) > 50:
                    # Extract first sentence or first 100 chars
                    truncated = (
                        ai_analysis[:100] + "..."
                        if len(ai_analysis) > 100
                        else ai_analysis
                    )
                    summary_parts.append(f"  AI Analysis: {truncated}")

        return "\n".join(summary_parts)

    def _generate_basic_summary(
        self, custom_analysis: List[Dict[str, Any]], report: Dict[str, Any]
    ) -> str:
        """Generate basic summary as fallback when AI is unavailable."""
        if not custom_analysis:
            return "Analysis completed but no results were generated."

        successful_count = len([a for a in custom_analysis if not a.get("error")])
        total_count = len(custom_analysis)

        # Count total tool calls
        total_tools = 0
        for analysis in custom_analysis:
            tool_calls = analysis.get("tool_calls", [])
            total_tools += len(tool_calls)

        summary = "AI agent analysis completed successfully. "
        summary += f"Executed {total_tools} MCP tools across {successful_count}/{total_count} analysis runs. "

        if successful_count == total_count:
            summary += "All tools executed without errors."
        else:
            failed_count = total_count - successful_count
            summary += f"{failed_count} analysis runs encountered errors."

        return summary

    def _count_vulnerability_severity(
        self, vulnerabilities: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Count vulnerabilities by severity."""
        counts: Dict[str, int] = {}
        for vuln in vulnerabilities:
            severity = vuln.get("severity", "unknown").lower()
            counts[severity] = counts.get(severity, 0) + 1
        return counts

    def _calculate_quality_score(self, metrics: List[Dict[str, Any]]) -> int:
        """Calculate overall quality score from metrics."""
        if not metrics:
            return 0

        scores = []
        for metric in metrics:
            result = metric.get("result", {})
            if "score" in result:
                scores.append(result["score"])

        return int(sum(scores) / len(scores)) if scores else 0

    def _get_quality_emoji(self, score: int) -> str:
        """Get emoji based on quality score."""
        if score >= 90:
            return "🟢"
        elif score >= 75:
            return "🟡"
        elif score >= 60:
            return "🟠"
        else:
            return "🔴"

    def _extract_business_insights(self, ai_analyses: List[str]) -> str:
        """Extract business insights from AI analyses while preserving detailed metrics."""
        if not ai_analyses:
            return "Analysis completed successfully."

        # Get the most comprehensive analysis
        full_analysis = (
            max(ai_analyses, key=len) if len(ai_analyses) > 1 else ai_analyses[0]
        )

        # Extract insights while preserving structure and detail
        lines = full_analysis.split("\n")
        business_lines = []
        in_metrics_section = False

        for line in lines:
            line = line.strip()

            # Detect metrics/statistics sections to preserve them fully
            if any(
                section in line.lower()
                for section in [
                    "system metrics",
                    "detailed statistics",
                    "metrics",
                    "statistics",
                    "infrastructure monitoring",
                    "past day",
                    "past hour",
                ]
            ):
                in_metrics_section = True
                business_lines.append(line)
                continue

            # End metrics section on new major section
            if in_metrics_section and (
                line.startswith("##") or (line.startswith("**") and line.endswith("**"))
            ):
                in_metrics_section = False

            # Skip only very specific tool execution lines, not general mentions
            if any(
                skip_term in line.lower()
                for skip_term in [
                    "following tools were utilized",
                    "tool was executed successfully",
                    "mcp tool call completed",
                    "executed the following mcp tools",
                ]
            ):
                continue

            # Include lines with business value or in metrics sections
            if line and (
                in_metrics_section
                or any(
                    keep_term in line.lower()
                    for keep_term in [
                        "recommendation",
                        "analysis",
                        "status",
                        "request",
                        "component",
                        "issue",
                        "finding",
                        "total",
                        "success",
                        "failure",
                        "warning",
                        "overall",
                        "daily",
                        "monitoring",
                        "system",
                        "metrics",
                        "rate",
                        "processed",
                        "error",
                        "timeout",
                        "infrastructure",
                        "health",
                        "report",
                        "summary",
                        "statistics",
                        "past",
                        "predominant",
                        "pattern",
                        "significant",
                        "required",
                        "investigate",
                        "resolve",
                        "action",
                        "concerns",
                        "indicates",
                        "shows",
                        "reveals",
                        "%",
                    ]
                )
            ):
                business_lines.append(line)

        # Return more comprehensive insights instead of just top 5
        if business_lines:
            # Preserve more content but still limit for Slack
            return "\n".join(business_lines[:15])  # Increased from 5 to 15 lines
        else:
            # Fallback: create a concise summary
            return "System analysis completed. Key metrics and status information have been collected and evaluated."

    def _check_if_truncated(
        self, ai_analysis: str, analysis_result: Dict[str, Any]
    ) -> bool:
        """Check if analysis was truncated due to token limits."""
        # Look for truncation indicators in the text
        for indicator in self.TRUNCATION_INDICATORS:
            if indicator in ai_analysis:
                return True

        # Check execution steps for token limit errors
        execution_steps = analysis_result.get("execution_steps", [])
        for step in execution_steps:
            if step.get("step_type") == "inference":
                error = step.get("error", "")
                if "token" in error.lower() and (
                    "limit" in error.lower() or "exceed" in error.lower()
                ):
                    return True

        return False

    def _create_truncation_notice(self, analysis_result: Dict[str, Any]) -> str:
        """Create a user-friendly notice about truncation."""
        tool_calls = analysis_result.get("tool_calls", [])
        successful_calls = [tc for tc in tool_calls if tc.get("status") == "success"]

        if successful_calls:
            return (
                f"⚠️ *Note:* Analysis was limited by AI model token constraints. "
                f"Successfully processed {len(successful_calls)} data queries before reaching limits."
            )
        else:
            return (
                "⚠️ *Note:* Analysis was limited by AI model token constraints. "
                "Consider reducing scope for more detailed results."
            )

    def _create_tool_call_summary(self, analysis_result: Dict[str, Any]) -> str:
        """Create a summary of tool calls when analysis was truncated."""
        tool_calls = analysis_result.get("tool_calls", [])
        if not tool_calls:
            return ""

        successful_calls = [tc for tc in tool_calls if tc.get("status") == "success"]

        summary_parts = [
            f"📊 *Completed Tasks:* {len(successful_calls)}/{len(tool_calls)} tools executed successfully"
        ]

        # Add brief summary of what was done
        if successful_calls:
            tool_names = [tc.get("tool_name", "Unknown") for tc in successful_calls[:3]]
            summary_parts.append(f"✅ Key tools executed: {', '.join(tool_names)}")

            if len(successful_calls) > 3:
                summary_parts.append(
                    f"   + {len(successful_calls) - 3} additional tools"
                )

        return "\n".join(summary_parts) + "\n\n"

    # Data preservation methods (using accumulated_knowledge)

    def _format_generic_tool_summary(self, tool_results: List[Dict]) -> str:
        """
        Format ANY MCP tool results into readable summary (template-based).

        Works for any combination of MCP servers (DataRouter, GitHub, Sourcebot, etc.)
        Uses actual tool results from knowledge store, not compacted conversation.
        """
        # Group by server (works for any MCP server)
        by_server: Dict[str, List[Dict[str, Any]]] = {}
        for result in tool_results:
            server = result["server"]
            if server not in by_server:
                by_server[server] = []
            by_server[server].append(result)

        summary = "🤖 *MCP Tool Analysis Report*\n\n"

        # Generic metrics
        summary += "📊 *Execution Summary:*\n"
        summary += f"• Total Tools Executed: **{len(tool_results)}**\n"
        summary += f"• MCP Servers Used: **{', '.join(by_server.keys())}**\n"
        summary += f"• Iterations Completed: **{max(r['iteration'] for r in tool_results)}**\n\n"

        # Per-server breakdown (generic)
        for server, results in by_server.items():
            summary += f"🔧 *{server.title()} Server:*\n"

            # Show tool names used (generic)
            tool_names = list(set(r["tool"] for r in results))
            for tool_name in tool_names[:5]:  # Top 5 tools
                tool_count = len([r for r in results if r["tool"] == tool_name])
                summary += f"• `{tool_name}`: {tool_count} call(s)\n"

            if len(tool_names) > 5:
                summary += f"• ... and {len(tool_names) - 5} more tools\n"
            summary += "\n"

        return summary

    async def _generate_summary_from_tool_data(
        self,
        tool_results: List[Dict[str, Any]],
        servers_used: List[str],
        report_context: Dict[str, Any],
    ) -> str:
        """
        Generate AI summary from ANY MCP tool results using actual data.

        This replaces the broken _generate_slack_summary() that used compacted text.
        Instead, it sends actual tool result data to AI for rich, accurate summaries.
        """
        settings = load_settings()
        provider_config = get_provider_config(settings)
        provider_name = settings.ai_provider or "gemini"

        # Create AI provider
        ai_provider = create_provider(provider_name, provider_config)
        await ai_provider.initialize()

        try:
            # Prepare data summary
            data_summary = {
                "servers_used": servers_used,
                "total_tools": len(tool_results),
                "tools_by_server": self._group_results_by_server(tool_results),
                "sample_results": self._get_sample_results(
                    tool_results, max_samples=10
                ),
            }

            # Build prompt with ACTUAL DATA, not compacted conversation
            prompt = f"""Create a concise Slack notification summarizing this MCP tool analysis.

MCP TOOL EXECUTION DATA:
Servers Used: {", ".join(servers_used)}
Total Tool Calls: {len(tool_results)}

TOOL RESULTS BY SERVER:
{json.dumps(data_summary["tools_by_server"], sort_keys=True, indent=2)[:2000]}

SAMPLE RESULTS (actual data from tools):
{json.dumps(data_summary["sample_results"], sort_keys=True, indent=2)[:2000]}

Create a business-focused summary that:
1. Identifies what was analyzed (infer from tool names and actual results)
2. Highlights key findings from the data above (use specific numbers)
3. Uses actual metrics and statistics from the results
4. Keeps it under {self.SLACK_SUMMARY_SECTION_LIMIT} characters for Slack

Format with Slack markdown (**bold**, *italic*, `code`) and relevant emojis.
Do NOT make up data - only use information from the actual results above.
Focus on business impact and actionable insights.
"""

            messages = [ProviderMessage(content=prompt, role="user")]
            response = await ai_provider.chat_completion(messages)

            slack_content = response.content.strip()

            # Ensure it fits within Slack limits
            if len(slack_content) > self.SLACK_SUMMARY_SECTION_LIMIT:
                slack_content = self._truncate_slack_content(
                    slack_content, self.SLACK_SUMMARY_SECTION_LIMIT
                )

            logger.debug(
                f"AI-generated Slack summary from tool data: {len(slack_content)} characters"
            )
            return slack_content

        finally:
            await ai_provider.shutdown()

    def _truncate_json_intelligently(
        self, data: Union[Dict[str, Any], List[Any]], max_length: int
    ) -> str:
        """
        Intelligently truncate JSON data to fit within max_length.

        Attempts to preserve structure by truncating at object/array boundaries.
        """
        # Try full serialization first
        # Phase 1 KV-Cache Optimization: Use sort_keys=True for deterministic JSON
        full_json = json.dumps(data, sort_keys=True, indent=2)
        if len(full_json) <= max_length:
            return full_json

        # Truncate with structure preservation
        result = json.dumps(data, sort_keys=True, indent=2)[:max_length]

        # Try to truncate at last complete JSON boundary
        last_brace = max(result.rfind("}"), result.rfind("]"))
        last_comma = result.rfind(",")

        if last_brace > max_length * 0.5:  # Keep if we preserve >50%
            result = result[: last_brace + 1]
        elif last_comma > max_length * 0.5:
            result = result[:last_comma]

        return result + "\n... (truncated for brevity)"

    def _truncate_text_intelligently(self, text: str, max_length: int) -> str:
        """
        Intelligently truncate text at sentence or paragraph boundaries.
        """
        if len(text) <= max_length:
            return text

        # Try to truncate at sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind(". ")
        last_newline = truncated.rfind("\n")

        # Prefer paragraph break, then sentence break
        if last_newline > max_length * 0.7:
            return text[: last_newline + 1] + "\n... (truncated)"
        elif last_period > max_length * 0.7:
            return text[: last_period + 2] + "... (truncated)"
        else:
            return truncated + "... (truncated)"

    async def _generate_summary_from_combined_sources(
        self,
        tool_results: List[Dict[str, Any]],
        ai_analysis: str,
        servers_used: List[str],
        report_context: Dict[str, Any],
    ) -> str:
        """
        Generate Slack summary by combining raw tool data with AI analysis insights.

        This produces the richest possible summary by leveraging:
        - Specific numbers/metrics from tool results
        - Context/insights/recommendations from AI analysis

        Args:
            tool_results: Raw MCP tool execution results
            ai_analysis: AI-generated analysis with insights and recommendations
            servers_used: List of MCP servers that were used
            report_context: Context information (task_type, project, agent_type)

        Returns:
            Slack-formatted summary combining both sources
        """
        settings = load_settings()
        provider_config = get_provider_config(settings)
        provider_name = settings.ai_provider or "gemini"

        # Create AI provider
        ai_provider = create_provider(provider_name, provider_config)
        await ai_provider.initialize()

        try:
            # Prepare data summary from tool results
            data_summary = {
                "servers_used": servers_used,
                "total_tools": len(tool_results),
                "tools_by_server": self._group_results_by_server(tool_results),
                "sample_results": self._get_sample_results(
                    tool_results, max_samples=10
                ),
            }

            # Extract typed values for the prompt
            tools_by_server = cast(Dict[str, Any], data_summary["tools_by_server"])
            sample_results = cast(List[Dict[str, Any]], data_summary["sample_results"])

            # Build enhanced prompt combining both sources
            prompt = f"""Create a concise Slack notification by combining raw data with AI insights.

DATA SOURCES AVAILABLE:

1. RAW TOOL DATA (for specific numbers and metrics):
Servers Used: {", ".join(servers_used)}
Total Tool Calls: {len(tool_results)}

TOOL RESULTS BY SERVER:
{self._truncate_json_intelligently(tools_by_server, 2000)}

SAMPLE RESULTS (actual data):
{self._truncate_json_intelligently(sample_results, 2000)}

2. AI ANALYSIS (for context, insights, and recommendations):
{self._truncate_text_intelligently(ai_analysis, 8000)}

TASK:
Create a business-focused Slack summary that:
1. Uses SPECIFIC NUMBERS from raw tool data (request counts, percentages, error rates)
2. Incorporates INSIGHTS and CONTEXT from AI analysis (patterns, root causes, trends)
3. Combines RECOMMENDATIONS from AI analysis with supporting data
4. Keeps it under {self.SLACK_SUMMARY_SECTION_LIMIT} characters for Slack

FORMAT:
- Use Slack markdown: **bold** for numbers, *italic* for emphasis, `code` for technical terms
- Add relevant emojis: 📊 📈 🚨 ⚠️ 💡 🔍
- Create clear sections with bullet points using •
- Highlight key metrics with **bold**

STRUCTURE EXAMPLE:
🤖 *System Analysis Report*

📊 **Key Metrics:** (use actual numbers from tool data)
• Total requests: **1,247** (from raw data)
• Failure rate: **12.2%** (from raw data)

🔍 **Analysis Insights:** (from AI analysis)
• Pattern identified: timeout issues in payment component
• Root cause: connection pool exhaustion

💡 **Recommendations:** (from AI analysis + supporting data)
• Increase connection pool size (currently **50** connections)
• Monitor payment service latency (avg **2.3s**)

Focus on actionable insights that combine hard data with intelligent analysis.
Do NOT make up data - only use information from the sources above.
"""

            messages = [ProviderMessage(content=prompt, role="user")]
            response = await ai_provider.chat_completion(messages)

            slack_content = response.content.strip()

            # Ensure it fits within Slack limits
            if len(slack_content) > self.SLACK_SUMMARY_SECTION_LIMIT:
                slack_content = self._truncate_slack_content(
                    slack_content, self.SLACK_SUMMARY_SECTION_LIMIT
                )

            logger.debug(
                f"AI-generated combined Slack summary: {len(slack_content)} characters"
            )
            return slack_content

        finally:
            await ai_provider.shutdown()

    def _group_results_by_server(
        self, tool_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Group tool results by server - works for any MCP server."""
        grouped: Dict[str, Any] = {}
        for result in tool_results:
            server = result["server"]
            if server not in grouped:
                grouped[server] = {
                    "tool_count": 0,
                    "tools_used": set(),
                    "sample_data": [],
                }
            server_data = grouped[server]
            server_data["tool_count"] += 1
            server_data["tools_used"].add(result["tool"])

            # Include sample of actual result data (first 200 chars)
            result_preview = str(result["result"])[:200]
            server_data["sample_data"].append(
                {"tool": result["tool"], "preview": result_preview}
            )

        # Convert sets to lists for JSON serialization
        for server in grouped:
            grouped[server]["tools_used"] = list(grouped[server]["tools_used"])

        return grouped

    def _get_sample_results(
        self, tool_results: List[Dict], max_samples: int = 10
    ) -> List[Dict]:
        """Get sample of tool results for AI analysis - generic."""
        samples = []
        for result in tool_results[:max_samples]:
            samples.append(
                {
                    "server": result["server"],
                    "tool": result["tool"],
                    "arguments": result["arguments"],
                    "result_preview": str(result["result"])[
                        :500
                    ],  # Limit size for prompt
                }
            )
        return samples
