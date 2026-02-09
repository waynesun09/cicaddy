"""Message formatting utilities for notifications."""

import re
from typing import Any, Dict, List


class MarkdownToSlackConverter:
    """Convert markdown to Slack mrkdwn format."""

    @staticmethod
    def convert(markdown: str) -> str:
        """Convert markdown text to Slack mrkdwn format."""
        # Convert headers
        markdown = re.sub(r"^### (.*?)$", r"*\1*", markdown, flags=re.MULTILINE)
        markdown = re.sub(r"^## (.*?)$", r"*\1*", markdown, flags=re.MULTILINE)
        markdown = re.sub(r"^# (.*?)$", r"*\1*", markdown, flags=re.MULTILINE)

        # Convert bold
        markdown = re.sub(r"\*\*(.*?)\*\*", r"*\1*", markdown)

        # Convert italic
        markdown = re.sub(r"\*(.*?)\*", r"_\1_", markdown)

        # Convert inline code
        markdown = re.sub(r"`([^`]+)`", r"`\1`", markdown)

        # Convert code blocks to Slack format
        markdown = re.sub(
            r"```(\w+)?\n(.*?)\n```", r"```\n\2\n```", markdown, flags=re.DOTALL
        )

        # Convert links
        markdown = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", markdown)

        # Convert lists
        markdown = re.sub(r"^- (.*?)$", r"• \1", markdown, flags=re.MULTILINE)
        markdown = re.sub(r"^\* (.*?)$", r"• \1", markdown, flags=re.MULTILINE)
        markdown = re.sub(r"^\d+\. (.*?)$", r"1. \1", markdown, flags=re.MULTILINE)

        # Convert blockquotes
        markdown = re.sub(r"^> (.*?)$", r"> \1", markdown, flags=re.MULTILINE)

        # Convert horizontal rules
        markdown = re.sub(
            r"^---+$", r"────────────────────", markdown, flags=re.MULTILINE
        )

        return markdown


class MessageSectionBuilder:
    """Builder for common message sections."""

    @staticmethod
    def build_header(title: str, emoji: str = "🤖") -> str:
        """Build standard message header."""
        return f"{emoji} *{title}*\n\n"

    @staticmethod
    def build_project_info(
        project_name: str, duration: float, analysis_type: str = ""
    ) -> str:
        """Build project information section."""
        message = f"🔍 *Project:* {project_name}\n"
        message += f"⏱️ *Duration:* {duration:.1f}s\n"
        if analysis_type:
            message += f"🎯 *Analysis Type:* {analysis_type}\n"
        return message

    @staticmethod
    def build_mr_info(mr_data: Dict[str, Any]) -> str:
        """Build merge request information section."""
        message = f"📋 *Merge Request:* {mr_data['title']}\n"
        message += f"👤 *Author:* {mr_data['author']['name']}\n"
        source_branch = mr_data["source_branch"]
        target_branch = mr_data["target_branch"]
        message += f"🌿 *Branch:* `{source_branch}` → `{target_branch}`\n"
        message += f"🔗 *MR Link:* <{mr_data['web_url']}|View MR>\n"
        return message

    @staticmethod
    def build_analysis_results_summary(analysis_results: Dict[str, Any]) -> str:
        """Build analysis results summary section."""
        message = "*Analysis Results:*\n"
        for task, result in analysis_results.items():
            if result.get("status") == "success":
                message += f"✅ {task.replace('_', ' ').title()}\n"
            else:
                message += f"❌ {task.replace('_', ' ').title()} (Error)\n"
        return message

    @staticmethod
    def build_key_findings(analysis_results: Dict[str, Any]) -> str:
        """Build key findings section."""
        message = "\n*Key Findings:*\n"
        for task, result in analysis_results.items():
            if result.get("status") == "success":
                # Extract first few lines of analysis
                content = result.get("content", "")
                lines = content.split("\n")
                summary = "\n".join(lines[:3])  # First 3 lines
                message += f"• *{task.replace('_', ' ').title()}:* {summary}\n"
        return message

    @staticmethod
    def build_security_findings_summary(
        findings: List[Dict[str, Any]], severity_counts: Dict[str, int]
    ) -> str:
        """Build security findings summary."""
        total_findings = len(findings)
        message = f"🚨 *Security Findings:* {total_findings}\n"

        if severity_counts:
            for severity, count in severity_counts.items():
                emoji = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🟢",
                }.get(severity.lower(), "⚪")
                message += f"  {emoji} {severity.title()}: {count}\n"

        return message

    @staticmethod
    def build_quality_metrics(metrics: Dict[str, Any]) -> str:
        """Build quality metrics section."""
        message = "📊 *Quality Metrics:*\n"
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                message += f"• {metric.replace('_', ' ').title()}: {value}\n"
            else:
                message += f"• {metric.replace('_', ' ').title()}: {value}\n"
        return message

    @staticmethod
    def build_dependency_summary(
        vuln_count: int, updates_available: int, security_updates: List[str]
    ) -> str:
        """Build dependency summary section."""
        message = "📦 *Dependencies:*\n"
        message += f"⚠️ *Vulnerabilities:* {vuln_count}\n"
        message += f"📈 *Updates Available:* {updates_available}\n"
        if security_updates:
            message += f"🛡️ *Security Updates:* {len(security_updates)}\n"
        return message

    @staticmethod
    def build_trends_section(trends: Dict[str, float]) -> str:
        """Build trends section."""
        message = "\n📈 *Trends:*\n"
        for metric, change in trends.items():
            trend_emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            message += f"{trend_emoji} {metric}: {change:+.1f}%\n"
        return message

    @staticmethod
    def build_top_packages_section(packages: List[Dict[str, str]]) -> str:
        """Build top vulnerable packages section."""
        message = "\n🎯 *Top Vulnerable Packages:*\n"
        for pkg in packages[:3]:
            message += f"• `{pkg['name']}` ({pkg['severity']})\n"
        return message

    @staticmethod
    def build_action_items(vuln_count: int, security_updates: List[str]) -> str:
        """Build action items section."""
        message = "\n🔧 *Action Required:*\n"
        if security_updates:
            message += f"• Update {len(security_updates)} security patches\n"
        if vuln_count > 5:
            message += f"• Review and fix {vuln_count} vulnerabilities\n"
        return message

    @staticmethod
    def build_ai_analysis_section(analysis_text: str) -> str:
        """Build AI analysis section."""
        if not analysis_text:
            return ""
        return f"\n🤖 *AI Analysis:*\n{analysis_text}\n"


class SeverityFormatter:
    """Formatter for severity-based content."""

    @staticmethod
    def get_severity_emoji(findings_count: int) -> str:
        """Get emoji based on findings count."""
        if findings_count > 10:
            return "🔴"  # High
        elif findings_count > 5:
            return "🟡"  # Medium
        elif findings_count > 0:
            return "🟠"  # Low
        else:
            return "🟢"  # None

    @staticmethod
    def get_severity_color(severity: str) -> str:
        """Get color indicator for severity."""
        return {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
        }.get(severity.lower(), "⚪")


class MessageLengthManager:
    """Manage message length for platform constraints."""

    @staticmethod
    def truncate_for_slack(message: str, max_length: int = 2500) -> str:
        """Truncate message intelligently for Slack."""
        footer_length = 100  # Approximate length of footer content

        if len(message) <= max_length - footer_length:
            return message

        # Message is too long, need to truncate intelligently
        lines = message.split("\n")
        essential_lines = []
        optional_lines = []

        for line in lines:
            # Keep essential information
            if any(
                essential in line
                for essential in [
                    "AI Agent Analysis Report",
                    "Project:",
                    "Duration:",
                    "Analysis Type:",
                    "CI Job:",
                    "Analysis Results:",
                    "AI Analysis Summary:",
                ]
            ):
                essential_lines.append(line)
            # Mark detailed sections as optional
            elif any(
                optional in line
                for optional in [
                    "Tool Execution Details:",
                    "Failed Analysis:",
                    "∘",
                    "•",
                ]
            ):
                optional_lines.append(line)
            else:
                essential_lines.append(line)

        # Rebuild message with essential content first
        truncated_message = "\n".join(essential_lines)

        # Add optional content if space allows
        remaining_space = max_length - footer_length - len(truncated_message)
        if remaining_space > 200 and optional_lines:
            # Add some optional content
            optional_content = "\n".join(optional_lines)
            if len(optional_content) <= remaining_space:
                truncated_message += "\n" + optional_content
            else:
                # Add truncated optional content
                truncated_message += (
                    "\n"
                    + optional_content[: remaining_space - 50]
                    + "\n\n*[Details truncated - view full report in CI job]*"
                )

        return truncated_message
