"""Task AI Agent for general-purpose analysis tasks."""

import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

import httpx

from cicaddy.config.settings import Settings
from cicaddy.notifications.email import EmailNotifier
from cicaddy.notifications.rich_slack import RichSlackNotifier
from cicaddy.utils.env_substitution import substitute_env_variables
from cicaddy.utils.logger import get_logger

from .base import BaseAIAgent

if TYPE_CHECKING:
    from cicaddy.reports.html_formatter import HTMLReportFormatter

logger = get_logger(__name__)


class TaskAgent(BaseAIAgent):
    """AI Agent for general-purpose tasks (scheduled analysis, custom prompts, DSPy tasks)."""

    def __init__(self, settings: Optional[Settings] = None):
        super().__init__(settings)

        # Task-specific configuration
        self.task_type = os.getenv("TASK_TYPE") or os.getenv(
            "CRON_TASK_TYPE", "scheduled_analysis"
        )
        # TASK_SCOPE is only relevant for code/project analysis tasks, not external MCP monitoring
        scope_env = os.getenv("TASK_SCOPE") or os.getenv("CRON_SCOPE")
        if self.task_type == "custom" and not scope_env:
            # For custom tasks without explicit scope, use external_tools to indicate MCP-focused work
            self.scope = "external_tools"
        else:
            # Use provided scope or default for non-custom tasks
            self.scope = scope_env or "full_project"
        self.report_format = os.getenv("TASK_REPORT_FORMAT") or os.getenv(
            "CRON_REPORT_FORMAT", "detailed"
        )

    async def initialize(self):
        """Initialize all components for task mode."""
        logger.info(
            "Initializing AI Agent for task mode",
            task_type=self.task_type,
            scope=self.scope,
        )

        # Use base class initialization but customize GitLab analyzer setup
        await super().initialize()

        # Override platform analyzer setup based on scope
        project_id = getattr(self.settings, "project_id", None)
        if (
            self.scope not in ["full_project", "main_branch", "recent_changes"]
            or not project_id
        ):
            if self.scope == "external_tools":
                logger.info("Platform analyzer not needed for external_tools scope")
            else:
                logger.warning("No project ID available - platform analyzer disabled")
            self.platform_analyzer = None

        # Replace base Slack notifier with enhanced task notifier
        slack_webhook_urls = self.settings.get_slack_webhook_urls()
        if (
            isinstance(slack_webhook_urls, (list, tuple))
            and len(slack_webhook_urls) > 0
        ):
            logger.info(
                f"Initializing Task Slack notifier with {len(slack_webhook_urls)} webhook(s)"
            )
            self.slack_notifier = RichSlackNotifier(
                slack_webhook_urls, ssl_verify=self.settings.ssl_verify
            )

        # Initialize email notifier if enabled
        self.email_notifier = None
        email_config = self.settings.get_email_config()
        if email_config:
            logger.info(
                f"Initializing Email notifier for {len(email_config['recipients'])} recipient(s)"
            )
            self.email_notifier = EmailNotifier(
                recipients=email_config["recipients"],
                sender_email=email_config["sender_email"],
                use_gmail_api=email_config["use_gmail_api"],
                ssl_verify=self.settings.ssl_verify,
            )

        logger.info("Task Agent initialized successfully")

    async def run_scheduled_analysis(self) -> Dict[str, Any]:
        """Run scheduled analysis based on task type."""
        logger.info(f"Starting scheduled analysis: {self.task_type}")

        # Note: Agent should already be initialized by the CLI
        # No need to initialize again here to avoid duplicate MCP connections

        # Use the base class analyze method which implements the full pipeline
        result = await self.analyze()

        # Return in expected format with task-specific metadata
        return {
            "task_type": self.task_type,
            "scope": self.scope,
            "analysis_result": result["analysis_result"],
            "report": result["report"],
            "execution_time": result["execution_time"],
        }

    async def get_analysis_context(self) -> Dict[str, Any]:
        """Gather project context based on scope for task analysis."""
        logger.info(f"Gathering project context for scope: {self.scope}")

        if self.platform_analyzer:
            try:
                project_info = await self.platform_analyzer.get_project_info()
            except Exception as e:
                # Handle various platform-related errors
                error_type = type(e).__name__
                if "AuthenticationError" in error_type or "Unauthorized" in str(e):
                    logger.error(
                        f"Platform authentication failed - check API token: {e}"
                    )
                    project_info = {
                        "name": "Unknown Project (Auth Failed)",
                        "id": getattr(self.settings, "project_id", None) or "external",
                    }
                elif isinstance(e, (httpx.ConnectError, httpx.TimeoutException)):
                    logger.error(f"Network error connecting to platform API: {e}")
                    project_info = {
                        "name": "Unknown Project (Network Error)",
                        "id": getattr(self.settings, "project_id", None) or "external",
                    }
                else:
                    # Catch-all for unexpected errors with full traceback for debugging tasks
                    logger.error(
                        f"Unexpected error getting project info ({error_type}): {e}",
                        exc_info=True,
                    )
                    project_info = {
                        "name": "Unknown Project",
                        "id": getattr(self.settings, "project_id", None) or "external",
                    }
        else:
            # Minimal project info when GitLab analyzer is not available
            project_info = {"name": "External Analysis", "id": "external"}

        context = {
            "project": project_info,
            "scope": self.scope,
            "task_type": self.task_type,
            "timestamp": self.start_time.isoformat(),
            "platform_available": self.platform_analyzer is not None,
        }

        if self.scope == "full_project":
            # Full project analysis
            context.update(await self._get_full_project_context())
        elif self.scope == "main_branch":
            # Main branch analysis
            context.update(await self._get_main_branch_context())
        elif self.scope == "recent_changes":
            # Recent changes analysis
            context.update(await self._get_recent_changes_context())
        elif self.scope == "external_tools":
            # External MCP tool analysis (no code context needed)
            context.update(await self._get_external_tools_context())

        return context

    def build_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Build analysis prompt based on task type and context."""
        if self.task_type == "custom":
            return self._build_custom_prompt(context)
        elif self.task_type == "security_audit":
            return self._build_security_prompt(context)
        elif self.task_type == "quality_report":
            return self._build_quality_prompt(context)
        elif self.task_type == "dependency_check":
            return self._build_dependency_prompt(context)
        else:
            return self._build_general_prompt(context)

    def _build_custom_prompt(self, context: Dict[str, Any]) -> str:
        """Build custom analysis prompt using AI_TASK_FILE (DSPy) or AI_TASK_PROMPT.

        Priority:
        1. AI_TASK_FILE - YAML task definition (DSPy mode)
        2. AI_TASK_PROMPT - Legacy monolithic prompt string
        """
        # Check for DSPy task file first
        task_file = os.getenv("AI_TASK_FILE")
        if task_file:
            dspy_prompt = self.build_dspy_prompt(task_file, context)
            if dspy_prompt:
                return dspy_prompt
            # Fall back to basic prompt if DSPy fails
            logger.warning(
                f"DSPy task file {task_file} failed to load, falling back to AI_TASK_PROMPT"
            )

        # Fall back to legacy AI_TASK_PROMPT
        custom_prompt = os.getenv(
            "AI_TASK_PROMPT", "Analyze the project comprehensively"
        )
        # Substitute environment variables in prompt (e.g., {{ANALYSIS_DAYS}} -> 30)
        custom_prompt = substitute_env_variables(custom_prompt)
        return self._build_task_prompt(custom_prompt, context)

    def _build_security_prompt(self, context: Dict[str, Any]) -> str:
        """Build security audit prompt."""
        security_prompt = """
Perform comprehensive security analysis:
1. Scan for security vulnerabilities and weaknesses
2. Identify potential attack vectors and security risks
3. Analyze authentication and authorization mechanisms
4. Check for sensitive data exposure
5. Provide actionable security recommendations

Focus on critical security issues that require immediate attention.
"""
        return self._build_task_prompt(security_prompt, context)

    def _build_quality_prompt(self, context: Dict[str, Any]) -> str:
        """Build quality report prompt."""
        quality_prompt = """
Perform comprehensive code quality analysis:
1. Assess code maintainability and readability
2. Identify technical debt and code smells
3. Analyze code complexity and structure
4. Check adherence to best practices
5. Provide improvement recommendations

Focus on actionable quality improvements.
"""
        return self._build_task_prompt(quality_prompt, context)

    def _build_dependency_prompt(self, context: Dict[str, Any]) -> str:
        """Build dependency check prompt."""
        dependency_prompt = """
Perform comprehensive dependency analysis:
1. Identify outdated dependencies and available updates
2. Scan for security vulnerabilities in dependencies
3. Analyze license compatibility and compliance
4. Check for deprecated or unmaintained packages
5. Provide dependency management recommendations

Focus on security and maintenance priorities.
"""
        return self._build_task_prompt(dependency_prompt, context)

    def _build_general_prompt(self, context: Dict[str, Any]) -> str:
        """Build general analysis prompt."""
        general_prompt = """
Perform general project analysis:
1. Assess overall project health and status
2. Identify areas for improvement
3. Analyze recent changes and their impact
4. Provide general recommendations

Focus on maintaining project quality and health.
"""
        return self._build_task_prompt(general_prompt, context)

    def get_session_id(self) -> str:
        """
        Get unique session ID for this task analysis session.

        Uses job_id (from CI_JOB_ID) if available for deterministic session IDs
        that enable KV-cache hits. Falls back to task_type for local development.
        """
        # Phase 1 KV-Cache Optimization: Use job_id for deterministic session IDs
        job_id = os.getenv("CI_JOB_ID") or os.getenv("JOB_ID")
        if job_id:
            return f"task_{self.task_type}_{job_id}"
        # Fallback for local development (still non-deterministic)
        return f"task_{self.task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    async def send_notifications(
        self, report: Dict[str, Any], analysis_result: Dict[str, Any]
    ):
        """Send notifications via Slack and/or Email based on configuration."""
        # Send Slack notification if configured
        if self.slack_notifier:
            await self.slack_notifier.send_cron_report(
                report,
                analysis_result,
                trends={},  # TODO: Implement trend tracking
            )

        # Send email notification if configured
        if self.email_notifier:
            await self._send_email_report(report, analysis_result)

    async def _send_email_report(
        self, report: Dict[str, Any], analysis_result: Dict[str, Any]
    ):
        """Send report via email notification.

        Creates a simplified email-friendly HTML format with AI Analysis
        section only (no charts, data sources, or footer - charts are not
        supported in email clients).
        """
        try:
            # Use simplified email format with AI Analysis only
            html_content = self._format_email_html(report, analysis_result)

            # Build subject line aligned with report/job context
            subject = report.get("email_subject") or self._build_email_subject(report)

            # Send email
            result = await self.email_notifier.send_notification(
                message=html_content,
                subject=subject,
                html=True,
            )

            if result.get("status") == "sent":
                logger.info(
                    f"Email report sent successfully via {result.get('method')} "
                    f"to {len(result.get('recipients', []))} recipient(s)"
                )
            else:
                logger.error(f"Email report failed: {result.get('error')}")

        except Exception as e:
            logger.error(f"Failed to send email report: {e}")

    def _format_email_html(
        self, report: Dict[str, Any], analysis_result: Dict[str, Any]
    ) -> str:
        """Render email-ready HTML based on the AI response format."""
        from cicaddy.reports.html_formatter import HTMLReportFormatter

        formatter = HTMLReportFormatter()

        response_format = (
            analysis_result.get("ai_response_format", "markdown") or "markdown"
        )
        response_format = response_format.lower()
        ai_analysis = analysis_result.get("ai_analysis", "")
        title = self._derive_report_title(report, formatter)
        metadata_html = self._build_email_metadata_block(report)
        footer_html = self._build_email_footer(report, formatter)

        if response_format == "html":
            direct_html = formatter._prepare_direct_response_content(
                ai_analysis, "html"
            )
            return (
                direct_html
                if direct_html
                else "<p><em>No analysis content available.</em></p>"
            )

        if response_format == "json":
            json_payload = formatter._prepare_direct_response_content(
                ai_analysis, "json"
            )
            escaped_json = (
                formatter._escape_html(json_payload)
                if json_payload
                else "<em>No analysis content available.</em>"
            )

            analysis_section = (
                f"<div class='analysis-content'><pre>{escaped_json}</pre></div>"
            )
            return self._render_email_template(
                title, metadata_html, analysis_section, footer_html
            )

        # Default: markdown → convert to HTML snippet
        analysis_html = (
            formatter._format_analysis_text(ai_analysis) if ai_analysis else ""
        )
        analysis_section = f"""
            <div class="analysis-content">
                {analysis_html if analysis_html else "<p><em>No analysis content available.</em></p>"}
            </div>
        """
        return self._render_email_template(
            title, metadata_html, analysis_section, footer_html
        )

    def _render_email_template(
        self,
        title: str,
        metadata_html: str,
        body_html: str,
        footer_html: str,
    ) -> str:
        styles = """
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 { color: #1a1a2e; border-bottom: 2px solid #4a4a8a; padding-bottom: 10px; }
            .metadata {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                border-left: 4px solid #4a4a8a;
            }
            .metadata p { margin: 5px 0; }
            .analysis-content { margin-top: 20px; }
            pre {
                background: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            a { color: #007bff; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .footer {
                margin-top: 30px;
                font-size: 0.85rem;
                color: #6c757d;
                border-top: 1px solid #e9ecef;
                padding-top: 15px;
            }
        """
        return f"""
        <html>
        <head>
            <style>
                {styles}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            {metadata_html}
            {body_html}
            {footer_html}
        </body>
        </html>
        """

    def _build_email_subject(self, report: Dict[str, Any]) -> str:
        # Use HTML_REPORT_HEADER if set to a custom value
        default_header = "AI Agent Analysis Report"
        if (
            self.settings.html_report_header
            and self.settings.html_report_header != default_header
        ):
            title = self.settings.html_report_header
        else:
            from cicaddy.reports.html_formatter import HTMLReportFormatter

            formatter = HTMLReportFormatter()
            title = self._derive_report_title(report, formatter)
        return f"{title} - {datetime.now().strftime('%Y-%m-%d')}"

    def _derive_report_title(
        self, report: Dict[str, Any], formatter: "HTMLReportFormatter"
    ) -> str:
        report_id = report.get("report_id", "")
        job_name = formatter._extract_job_name_from_report_id(report_id)
        if job_name and job_name != "Unknown":
            return f"{job_name.replace('_', ' ').title()} AI Analysis"
        project = report.get("project") or report.get("context_summary", {}).get(
            "project_name", ""
        )
        if project and project != "Unknown":
            return f"{project} AI Analysis"
        return f"{self.task_type.replace('_', ' ').title()} AI Analysis"

    def _build_email_metadata_block(self, report: Dict[str, Any]) -> str:
        project = report.get("project") or report.get("context_summary", {}).get(
            "project_name", "Unknown Project"
        )
        generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        return f"""
            <div class="metadata">
                <p><strong>Project:</strong> {project}</p>
                <p><strong>Task Type:</strong> {self.task_type}</p>
                <p><strong>Scope:</strong> {self.scope}</p>
                <p><strong>Generated:</strong> {generated}</p>
            </div>
        """

    def _build_email_footer(
        self, report: Dict[str, Any], formatter: "HTMLReportFormatter"
    ) -> str:
        pipeline_lines = []
        pipeline_url = os.getenv("CI_PIPELINE_URL")
        pipeline_id = os.getenv("CI_PIPELINE_ID")
        job_url = os.getenv("CI_JOB_URL")

        if pipeline_url and pipeline_id:
            pipeline_lines.append(
                f'<p><strong>Pipeline:</strong> <a href="{pipeline_url}">{pipeline_id}</a></p>'
            )
        elif pipeline_id:
            pipeline_lines.append(f"<p><strong>Pipeline:</strong> {pipeline_id}</p>")

        if job_url:
            pipeline_lines.append(
                f'<p><strong>Job:</strong> <a href="{job_url}">View job logs</a></p>'
            )

        html_report_url = self._build_html_report_url(report, formatter)
        if html_report_url:
            pipeline_lines.append(
                f'<p><strong>Full HTML report:</strong> <a href="{html_report_url}">{html_report_url}</a></p>'
            )

        pipeline_html = "".join(pipeline_lines) if pipeline_lines else ""
        return f"""
            <div class="footer">
                {pipeline_html}
            </div>
        """

    def _build_html_report_url(
        self, report: Dict[str, Any], formatter: "HTMLReportFormatter"
    ) -> Optional[str]:
        report_id = report.get("report_id", "")
        if not report_id or report_id == "unknown":
            return None
        ci_project_url = os.getenv("CI_PROJECT_URL")
        ci_job_id = os.getenv("CI_JOB_ID")
        if ci_project_url and ci_job_id:
            return f"{ci_project_url}/-/jobs/{ci_job_id}/artifacts/external_file/{report_id}.html"
        html_path = report.get("html_report_path")
        if html_path and os.path.exists(html_path):
            return os.path.abspath(html_path)
        return None

    async def _get_full_project_context(self) -> Dict[str, Any]:
        """Get full project context for comprehensive analysis."""
        context = {
            "analysis_type": "full_project",
            "description": (
                "Comprehensive project analysis including all files and history"
            ),
            "focus_areas": ["security", "quality", "dependencies", "architecture"],
        }

        if not self.platform_analyzer:
            logger.warning(
                "Platform analyzer not available for full project context - using basic context"
            )
        else:
            # Future enhancement: could fetch repository statistics, branch info, etc.
            pass

        return context

    async def _get_main_branch_context(self) -> Dict[str, Any]:
        """Get main branch context."""
        context = {
            "analysis_type": "main_branch",
            "description": "Analysis of current main branch state",
            "focus_areas": ["current_state", "deployment_readiness"],
        }

        if not self.platform_analyzer:
            logger.warning(
                "Platform analyzer not available for main branch context - using basic context"
            )
        else:
            # Future enhancement: could fetch branch status, latest commits, etc.
            pass

        return context

    async def _get_recent_changes_context(self) -> Dict[str, Any]:
        """Get recent changes context (last week)."""
        context = {
            "analysis_type": "recent_changes",
            "description": "Analysis of changes in the last 7 days",
            "focus_areas": ["recent_commits", "impact_assessment"],
        }

        if not self.platform_analyzer:
            logger.warning(
                "Platform analyzer not available for recent changes context - using basic context"
            )
        else:
            # Future enhancement: could fetch recent commits, merge requests, etc.
            pass

        return context

    async def _get_external_tools_context(self) -> Dict[str, Any]:
        """Get context for external MCP tool analysis (infrastructure monitoring, etc.)."""
        context = {
            "analysis_type": "external_tools",
            "description": "External system monitoring and analysis using MCP tools",
            "focus_areas": [
                "infrastructure_health",
                "system_metrics",
                "service_status",
            ],
        }

        logger.info(
            "Using external tools context - GitLab integration not required for this scope"
        )
        return context

    async def generate_report(
        self, analysis_result: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate formatted report with task-specific details."""
        # Use base class report generation
        report = await super().generate_report(analysis_result, context)

        # Add task-specific metadata
        report.update(
            {
                "task_type": self.task_type,
                "scope": self.scope,
            }
        )

        # Add format-specific content
        if self.report_format == "summary":
            report["summary"] = await self._generate_summary(analysis_result)
        elif self.report_format == "detailed":
            detailed_report = await self._generate_detailed_report(analysis_result)
            report["detailed_analysis"] = detailed_report
        elif self.report_format == "metrics_only":
            report["metrics"] = await self._extract_metrics(analysis_result)

        return report

    def _build_task_prompt(self, custom_prompt: str, context: Dict[str, Any]) -> str:
        """Build prompt for AI model including available MCP tools and context."""
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

        # Phase 1 KV-Cache Optimization: Build STATIC system prompt + separate context message
        # Static system prompt (cacheable, never changes)
        system_prompt = f"""
You are an intelligent infrastructure monitoring AI agent performing scheduled analysis.

Task: {self.task_type}
Scope: {self.scope}
Analysis Type: {context.get("analysis_type", "external_tools")}

User Request: {custom_prompt}

Available MCP Tools:
{tools_list}

Instructions:
1. Analyze the user request to understand what data/analysis is needed
2. Identify which MCP tools would be most useful for this request
3. Use the available MCP tools to gather data. Pay attention to:
   - Parameter types (string, integer, boolean, etc.) as specified in the tool schemas
   - Required vs optional parameters
   - Parameter descriptions to understand their purpose
4. When calling tools, ensure you provide the correct parameter types and values
5. Focus on tools that provide overview/summary data rather than specific IDs

Please provide your analysis and recommendations for monitoring this infrastructure.
"""

        # Dynamic execution context (passed as separate user message, not cached)
        execution_context = f"""
Execution Context:
- Project: {context.get("project", {}).get("name", "Unknown")}
- Timestamp: {context.get("timestamp", "Unknown")}
- Platform Available: {context.get("platform_available", False)}
"""

        # Return combined prompt (system + context)
        # Note: ExecutionEngine will receive this and convert to proper message format
        return system_prompt + "\n" + execution_context

    async def _generate_summary(self, analysis_result: Dict[str, Any]) -> str:
        """Generate executive summary."""
        key_areas = len(analysis_result)
        return f"Analysis completed for {self.task_type}. Found {key_areas} key areas."

    async def _generate_detailed_report(self, analysis_result: Dict[str, Any]) -> str:
        """Generate detailed markdown report."""
        import json

        # Phase 1 KV-Cache Optimization: Use sort_keys=True for deterministic JSON
        return f"# Detailed Analysis Report\n\n{json.dumps(analysis_result, sort_keys=True, indent=2)}"

    async def _extract_metrics(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from analysis."""
        return {
            "total_findings": len(str(analysis_result)),
            "analysis_time": (datetime.now() - self.start_time).total_seconds(),
        }
