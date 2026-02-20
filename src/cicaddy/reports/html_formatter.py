"""HTML report formatter for Cicaddy analysis results."""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)


class HTMLReportFormatter:
    """Formats AI agent analysis reports as HTML artifacts for CI/CD pipelines."""

    # Default values for report customization
    DEFAULT_HEADER = "AI Agent Analysis Report"
    DEFAULT_SUBHEADER = 'Analysis results from <a href="https://github.com/waynesun09/cicaddy" target="_blank">Cicaddy</a>'
    DEFAULT_HEADER_EMOJI = "🤖"
    DEFAULT_PROJECT_URL = "https://github.com/waynesun09/cicaddy"

    # Default CI environment variables to display in reports.
    # Each tuple is (env_var_name, display_label).
    DEFAULT_CI_ENV_VARS: List[tuple] = [
        ("CI_PIPELINE_ID", "Pipeline ID"),
        ("CI_JOB_ID", "Job ID"),
        ("CI_COMMIT_SHA", "Commit SHA"),
        ("CI_COMMIT_REF_NAME", "Branch/Tag"),
        ("CI_PIPELINE_SOURCE", "Pipeline Source"),
        ("CI_JOB_URL", "Job URL"),
    ]

    def __init__(
        self,
        header: Optional[str] = None,
        subheader: Optional[str] = None,
        logo_url: Optional[str] = None,
        logo_height: int = 48,
        header_emoji: Optional[str] = None,
        project_url: Optional[str] = None,
        ci_env_vars: Optional[List[tuple]] = None,
    ):
        """
        Initialize HTML report formatter with customizable headers and logo.

        Args:
            header: Main header text. Defaults to "AI Agent Analysis Report".
            subheader: Sub-header text/HTML. Defaults to Cicaddy link.
            logo_url: URL or path to logo image. If provided, displayed before headers.
            logo_height: Height of logo in pixels (default: 48, validated by Settings).
            header_emoji: Emoji prefix for header. Defaults to "🤖". Ignored when logo is present.
            project_url: URL for footer attribution link. Defaults to GitLab agent project.
            ci_env_vars: List of (env_var, label) tuples for CI info section.
                         Defaults to GitLab CI variables.
        """
        self.header = header or self.DEFAULT_HEADER
        self.subheader = subheader or self.DEFAULT_SUBHEADER
        self.logo_url = logo_url
        self.logo_height = logo_height
        self.header_emoji = (
            header_emoji if header_emoji is not None else self.DEFAULT_HEADER_EMOJI
        )
        self.project_url = project_url or self.DEFAULT_PROJECT_URL
        self.ci_env_vars = (
            ci_env_vars if ci_env_vars is not None else self.DEFAULT_CI_ENV_VARS
        )
        self.template = self._get_html_template()

    @classmethod
    def from_settings(cls, settings) -> "HTMLReportFormatter":
        """
        Create HTMLReportFormatter from Settings object.

        Args:
            settings: Settings object containing HTML report configuration.

        Returns:
            Configured HTMLReportFormatter instance.
        """
        return cls(
            header=settings.html_report_header,
            subheader=settings.html_report_subheader,
            logo_url=settings.html_report_logo_url,
            logo_height=settings.html_report_logo_height,
            header_emoji=settings.html_report_emoji,
        )

    def _extract_job_name_from_report_id(self, report_id: str) -> str:
        """
        Extract job name from report_id by removing agent_type prefix and timestamp postfix.

        Report ID format: {agent_type}_{job_name}_{timestamp} or {agent_type}_{timestamp}
        Example: task_sourcebot_llama_20251105_211002 -> sourcebot_llama

        Args:
            report_id: The report ID string

        Returns:
            The extracted job name, or "Unknown" if parsing fails
        """
        if not report_id or report_id == "unknown":
            return "Unknown"

        try:
            # Split by underscore
            parts = report_id.split("_")

            # Need at least 3 parts: agent_type, timestamp_part1, timestamp_part2
            # Format: agent_type_[job_name_parts]_YYYYMMDD_HHMMSS
            if len(parts) < 3:
                return report_id

            # Last two parts are timestamp (YYYYMMDD and HHMMSS)
            # First part is agent_type
            # Everything in between is job_name
            if len(parts) >= 3:
                # Remove first part (agent_type) and last two parts (timestamp)
                job_parts = parts[1:-2]
                if job_parts:
                    return "_".join(job_parts)

            # Fallback: return report_id if we can't parse it
            return report_id
        except Exception:
            return report_id

    def format_report(
        self, report: Dict[str, Any], json_report_path: Optional[str] = None
    ) -> str:
        """Convert JSON report to HTML format with link to external JSON file."""
        # Extract report metadata
        report_id = report.get("report_id", "unknown")
        project = report.get("project", "Unknown Project")
        execution_time = report.get("execution_time", 0)
        task_type = report.get("task_type", "custom")
        start_time = report.get("start_time", "")

        # Format timestamps
        if start_time:
            try:
                dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                formatted_time = start_time
        else:
            formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        # Extract analysis results
        analysis_result = report.get("analysis_result", {})
        ai_response_format = (
            analysis_result.get("ai_response_format", "markdown") or "markdown"
        )
        if isinstance(ai_response_format, str):
            ai_response_format = ai_response_format.lower()
        else:
            ai_response_format = "markdown"
        if ai_response_format not in {"markdown", "html", "json"}:
            ai_response_format = "markdown"

        direct_response_context = None

        tool_execution_html = self._format_tool_execution(analysis_result)
        summary_html = self._format_summary(
            report,
            analysis_result,
            report_id,
            project,
            task_type.replace("_", " ").title(),
            formatted_time,
        )

        # Get CI environment info (pass report_id to construct HTML report URL)
        ci_info = self._get_ci_info(report_id)

        # Get JSON file information
        if json_report_path:
            json_filename = os.path.basename(json_report_path)

            # Calculate file size if file exists
            json_file_size = "Unknown size"
            if os.path.exists(json_report_path):
                size_bytes = os.path.getsize(json_report_path)
                json_file_size = self._format_human_file_size(size_bytes)

            # Construct URL for CI artifacts or local path
            json_report_url = self._build_artifact_url(json_filename)
        else:
            json_filename = "report.json"
            json_file_size = "Unknown size"
            json_report_url = json_filename

        # Build AI analysis sections using new structure based on response format
        if ai_response_format == "markdown":
            ai_analysis_html = self._format_ai_analysis(analysis_result)
        else:
            direct_response_context = self._persist_direct_ai_response(
                analysis_result.get("ai_analysis", ""),
                ai_response_format,
                json_report_path,
                report_id,
            )
            ai_analysis_html = self._format_direct_response_section(
                analysis_result, ai_response_format, direct_response_context
            )

        # Extract job name from report_id for title
        job_name = self._extract_job_name_from_report_id(report_id)
        if job_name and job_name != "Unknown":
            # Convert snake_case to Title Case (e.g., sourcebot_llama -> Sourcebot Llama)
            readable_job_name = job_name.replace("_", " ").title()
            report_title = f"{readable_job_name} AI Analysis Report"
        else:
            # Fallback to original format if no job name available
            report_title = f"AI Agent Analysis Report - {project}"

        # Build logo HTML if logo URL is provided
        logo_html = ""
        if self.logo_url:
            logo_html = (
                f'<div class="header-logo"><img src="{self.logo_url}" alt="Logo"></div>'
            )

        # Fill template (with JSON file info instead of raw_json)
        # Don't use emoji prefix if logo is provided
        header_emoji = "" if self.logo_url else self.header_emoji
        html_content = self.template.format(
            title=report_title,
            report_id=report_id,
            project=project,
            task_type=task_type.replace("_", " ").title(),
            execution_time=f"{execution_time:.1f}",
            formatted_time=formatted_time,
            summary_html=summary_html,
            ai_analysis_html=ai_analysis_html,
            tool_execution_html=tool_execution_html,
            ci_info_html=ci_info,
            json_filename=json_filename,
            json_file_size=json_file_size,
            json_report_url=json_report_url,
            # Header customization
            logo_html=logo_html,
            logo_height=self.logo_height,
            header_emoji=header_emoji,
            main_header=self.header,
            sub_header=self.subheader,
            # Footer customization
            project_url=self.project_url,
        )

        return html_content

    def _format_summary(
        self,
        report: Dict[str, Any],
        analysis_result: Dict[str, Any],
        report_id: str,
        project: str,
        task_type: str,
        formatted_time: str,
    ) -> str:
        """Format unified badge section with metadata and stats."""
        status = analysis_result.get("status", "unknown")

        # Use accumulated_knowledge for tool count if available (data preservation)
        knowledge = analysis_result.get("accumulated_knowledge", {})
        if knowledge and knowledge.get("tool_results"):
            tool_count = len(knowledge["tool_results"])
        else:
            # Fallback to tool_calls for backward compatibility
            tool_calls = analysis_result.get("tool_calls", [])
            tool_count = len(tool_calls)

        # Use report execution time (total), not turn time
        execution_time = report.get(
            "execution_time", analysis_result.get("execution_time", 0)
        )

        # Get token usage information
        token_info = self._extract_token_info(analysis_result)

        # Build unified badge grid HTML
        summary_html = '<div class="unified-stats">'

        # Plain text metadata
        summary_html += '<div class="metadata-text">'
        summary_html += f"<span><strong>Report ID:</strong> {report_id}</span>"
        summary_html += f"<span><strong>Generated:</strong> {formatted_time}</span>"
        summary_html += "</div>"

        # Single row: All badges with same width
        summary_html += '<div class="stats-row">'

        # Status badge
        status_icon = "✅" if status == "success" else "❌"
        status_class = "success" if status == "success" else "danger"
        summary_html += f"""
            <div class="stat-badge-item">
                <span class="stat-badge {status_class}">{status_icon} {status.title()}</span>
            </div>
        """

        # Tool Calls badge
        summary_html += f"""
            <div class="stat-badge-item">
                <span class="stat-badge info">🔧 {tool_count} Tool Calls</span>
            </div>
        """

        # Duration badge
        summary_html += f"""
            <div class="stat-badge-item">
                <span class="stat-badge primary">⏱️ {execution_time:.1f}s Duration</span>
            </div>
        """

        # Add token usage badges if token data is available
        if token_info["has_token_data"]:
            summary_html += f"""
            <div class="stat-badge-item">
                <span class="stat-badge token">🤖 {token_info["total_tokens_formatted"]} Tokens</span>
            </div>
            """

            summary_html += f"""
            <div class="stat-badge-item">
                <span class="stat-badge token-detail">
                    📊 {token_info["input_tokens_formatted"]} input / {token_info["output_tokens_formatted"]} output
                </span>
            </div>
            """

            # Add inference calls badge (includes recovery AI calls)
            summary_html += f"""
            <div class="stat-badge-item">
                <span class="stat-badge inference">🔄 {token_info["inference_calls"]} Inference Calls</span>
            </div>
            """

            summary_html += f"""
            <div class="stat-badge-item">
                <span class="stat-badge model">{token_info["model_display"]}</span>
            </div>
            """

        summary_html += "</div>"  # End stats-row
        summary_html += "</div>"  # End unified-stats

        # Add error alert if failed
        if status == "failed":
            error = analysis_result.get("error", "Unknown error")
            summary_html += f"""
            <div class="alert alert-warning">
                <h4>❌ Analysis Failed</h4>
                <p>{error}</p>
            </div>
            """

        return summary_html

    def _extract_token_info(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and format token usage information from analysis result.

        This method uses the shared token utility and returns structured data
        for HTML formatting.
        """
        from cicaddy.utils.token_utils import TokenUsageExtractor

        # Default token info structure for HTML
        token_info = {  # nosec
            "has_token_data": False,
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "inference_calls": 0,
            "model_name": "Unknown Model",
            "provider": "unknown",
            "model_display": "Unknown Model",
            "total_tokens_formatted": "0",
            "input_tokens_formatted": "0",
            "output_tokens_formatted": "0",
        }

        try:
            # Determine AI provider from analysis_result.ai_provider, defaulting to "unknown"
            default_provider = analysis_result.get(
                "ai_provider"
            ) or analysis_result.get("provider", "unknown")

            # Extract token data using shared utility
            token_data = TokenUsageExtractor.extract_token_usage(
                analysis_result,
                default_provider=default_provider,
                include_model_info=True,
            )

            # Only update if we have meaningful usage data
            if TokenUsageExtractor.has_meaningful_usage(token_data):
                token_info["has_token_data"] = True
                token_info["total_tokens"] = token_data["total_tokens"]
                token_info["input_tokens"] = token_data["prompt_tokens"]
                token_info["output_tokens"] = token_data["completion_tokens"]
                token_info["inference_calls"] = token_data["inference_calls"]
                token_info["provider"] = token_data["provider"]
                token_info["model_name"] = token_data.get("model_used", "Unknown Model")

                # Format model display name
                if token_info["provider"] != "unknown":
                    provider_name = str(token_info["provider"]).title()
                    if (
                        provider_name.lower()
                        not in str(token_info["model_name"]).lower()
                    ):
                        token_info["model_display"] = (
                            f"{provider_name} {token_info['model_name']}"
                        )
                    else:
                        token_info["model_display"] = token_info["model_name"]
                else:
                    token_info["model_display"] = token_info["model_name"]

                # Format token counts with thousands separators
                token_info["total_tokens_formatted"] = f"{token_info['total_tokens']:,}"
                token_info["input_tokens_formatted"] = f"{token_info['input_tokens']:,}"
                token_info["output_tokens_formatted"] = (
                    f"{token_info['output_tokens']:,}"
                )

        except Exception as e:
            # Log the error but don't fail the report generation
            logger.warning(f"Failed to extract token info for HTML report: {e}")

        return token_info

    def _format_ai_analysis(self, analysis_result: Dict[str, Any]) -> str:
        """Format detailed AI analysis sections with enhanced handling for truncated responses."""
        # Priority 1: Use full AI analysis from main MCP tool iteration
        # This contains the AI's reasoning about the prompt using tool data
        ai_analysis = analysis_result.get("ai_analysis", "")
        is_intelligent_summary = False

        # Priority 2: Only if ai_analysis is missing/empty, fall back to tool summary
        if not ai_analysis or len(ai_analysis.strip()) == 0:
            knowledge = analysis_result.get("accumulated_knowledge", {})
            if knowledge and knowledge.get("tool_results"):
                # Generate tool execution summary as fallback when ai_analysis unavailable
                ai_analysis = self._generate_html_summary_from_knowledge(
                    knowledge, analysis_result
                )
                is_intelligent_summary = True
        else:
            # Check if the ai_analysis is an intelligent summary from token limit handling
            is_intelligent_summary = (
                self._is_intelligent_summary(ai_analysis) if ai_analysis else False
            )

        if not ai_analysis:
            return "<p>No AI analysis data available.</p>"

        execution_time = analysis_result.get("execution_time", 0)
        turn_id = analysis_result.get("turn_id", "analysis")
        status = analysis_result.get("status", "unknown")

        if analysis_result.get("error"):
            return f"""
            <div class="analysis-section error">
                <h3>Analysis - Error</h3>
                <div class="error-message">
                    <strong>Error:</strong> {analysis_result.get("error", "Unknown error")}
                </div>
            </div>
            """

        # Convert markdown-like formatting to HTML
        formatted_analysis = self._format_analysis_text(ai_analysis)

        # Add special styling and header for intelligent summaries
        if is_intelligent_summary:
            analysis_html = f"""
            <div class="analysis-section intelligent-summary">
                <h3>AI Analysis - Key Findings <span class="execution-time">({execution_time:.1f}s)</span></h3>
                <div class="summary-badge">
                    <span class="badge badge-info">📊 Intelligent Summary</span>
                    <small>Analysis focused on key findings from successful tool executions</small>
                </div>
                <div class="analysis-content enhanced-summary">
                    {formatted_analysis}
                </div>
                <div class="analysis-meta">
                    <small><strong>Turn ID:</strong> {turn_id} | <strong>Status:</strong> {status}</small>
                </div>
            </div>
            """
        else:
            analysis_html = f"""
            <div class="analysis-section">
                <h3>AI Analysis <span class="execution-time">({execution_time:.1f}s)</span></h3>
                <div class="analysis-content">
                    {formatted_analysis}
                </div>
                <div class="analysis-meta">
                    <small><strong>Turn ID:</strong> {turn_id} | <strong>Status:</strong> {status}</small>
                </div>
            </div>
            """

        return analysis_html

    def _is_intelligent_summary(self, ai_analysis: str) -> bool:
        """
        Detect if the AI analysis contains an intelligent summary from token limit handling.

        Args:
            ai_analysis: The AI analysis text

        Returns:
            True if this appears to be an intelligent summary
        """
        if not ai_analysis:
            return False

        # Look for markers that indicate intelligent summary
        intelligent_markers = [
            "Analysis completed",
            "operations before reaching token limits",
            "## Key Findings:",
            "## Data Summary:",
            "## Metrics Discovered:",
            "## Analysis Status:",
            "Successfully gathered data for:",
            "Key insights have been extracted",
        ]

        ai_analysis_lower = ai_analysis.lower()
        for marker in intelligent_markers:
            if marker.lower() in ai_analysis_lower:
                return True

        return False

    def _format_tool_execution(self, analysis_result: Dict[str, Any]) -> str:
        """Format tool execution details with inference-level grouping."""
        # Use accumulated_knowledge for tool results if available (data preservation)
        knowledge = analysis_result.get("accumulated_knowledge", {})
        if knowledge and knowledge.get("tool_results"):
            tool_results = knowledge["tool_results"]
            tool_count = len(tool_results)
            # Use total execution time from knowledge if available
            execution_time = knowledge.get(
                "total_execution_time", analysis_result.get("execution_time", 0)
            )
        else:
            # Fallback to tool_calls for backward compatibility
            tool_calls = analysis_result.get("tool_calls", [])
            tool_results = tool_calls
            tool_count = len(tool_calls)
            execution_time = analysis_result.get("execution_time", 0)

        if not tool_results:
            return "<p>No tool execution data available.</p>"

        # Sort by inference_id first, then iteration within each inference
        # Backward compatibility: default inference_id to 1 for old data
        sorted_results = sorted(
            tool_results,
            key=lambda x: (x.get("inference_id", 1), x.get("iteration", 0)),
        )

        # Group by inference for clearer display
        by_inference: Dict[int, List[Dict[str, Any]]] = {}
        for tool in sorted_results:
            inf_id = tool.get("inference_id", 1)  # Backward compatibility
            if inf_id not in by_inference:
                by_inference[inf_id] = []
            by_inference[inf_id].append(tool)

        tool_html = f"""
        <div class="tool-section">
            <h4>Tool Execution - {tool_count} tools across {len(by_inference)} inference(s) ({execution_time:.1f}s)</h4>
        """

        # Iterate through each inference group
        for inf_id, tools in sorted(by_inference.items()):
            # Generate inference label
            if inf_id == 1:
                inference_label = "Initial Inference"
            else:
                inference_label = f"Recovery Inference #{inf_id - 1}"

            tool_html += f"""
            <div class="inference-group">
                <h5>{inference_label} - {len(tools)} tools</h5>
                <table class="tool-execution-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Ref</th>
                            <th>Iteration</th>
                            <th>Server</th>
                            <th>Tool Name</th>
                            <th>Arguments</th>
                            <th>Result</th>
                            <th>Time (s)</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            for idx, tool_result in enumerate(tools, 1):
                # Handle both accumulated_knowledge format and tool_calls format
                if "tool" in tool_result and "arguments" in tool_result:
                    # accumulated_knowledge format
                    tool_name = tool_result.get("tool", "Unknown")
                    tool_server = tool_result.get("server", "")
                    arguments = tool_result.get("arguments", {})
                    iteration = tool_result.get("iteration", "")
                    unique_ref = tool_result.get(
                        "unique_ref", ""
                    )  # NEW: get unique_ref
                    result = tool_result.get("result", {})
                    exec_time = tool_result.get("execution_time", 0)
                elif tool_result.get("step_type") == "tool_execution":
                    # tool_calls format (execution step) - backward compatibility
                    tool_name = tool_result.get("tool_name", "Unknown")
                    tool_server = tool_result.get("tool_server", "")
                    arguments = tool_result.get("arguments", {})
                    iteration = ""
                    unique_ref = ""  # No unique_ref in old format
                    result = tool_result.get("result", {})
                    exec_time = tool_result.get("execution_time", 0)
                else:
                    # Skip invalid entries
                    continue

                # Format arguments as expandable JSON (mirror Result column)
                args_id = f"args_{inf_id}_{idx}"
                args_json = self._format_json_result(arguments)
                args_preview = self._get_args_preview(arguments)

                args_html = f"""
                    <button class="expand-btn" onclick="toggleResult('{args_id}')">
                        <span class="expand-icon">▶</span> {args_preview}
                    </button>
                    <pre id="{args_id}" class="result-json" style="display: none;">{args_json}</pre>
                """

                # Format result as expandable JSON
                result_id = f"result_{inf_id}_{idx}"
                result_json = self._format_json_result(result)

                tool_html += f"""
                        <tr>
                            <td class="tool-seq">{idx}</td>
                            <td class="tool-ref">{unique_ref}</td>
                            <td class="tool-iteration">{iteration}</td>
                            <td class="tool-server">{tool_server}</td>
                            <td class="tool-name">{tool_name}</td>
                            <td class="tool-args">{args_html}</td>
                            <td class="tool-result">
                                <button class="expand-btn" onclick="toggleResult('{result_id}')">
                                    <span class="expand-icon">▶</span> Show Result
                                </button>
                                <pre id="{result_id}" class="result-json" style="display: none;">{result_json}</pre>
                            </td>
                            <td class="tool-time">{exec_time:.2f}</td>
                        </tr>
                """

            tool_html += """
                    </tbody>
                </table>
            </div>
            """

        tool_html += """
        </div>
        """

        return tool_html

    def _format_direct_response_section(
        self,
        analysis_result: Dict[str, Any],
        response_format: str,
        artifact_context: Dict[str, Any],
    ) -> str:
        """Render AI analysis section for non-markdown responses."""
        execution_time = analysis_result.get("execution_time", 0)
        turn_id = analysis_result.get("turn_id", "analysis")
        status = analysis_result.get("status", "unknown")
        format_label = response_format.upper()

        content_html = ""
        if artifact_context.get("status") == "success":
            # Try to read and embed HTML content for inline display with charts
            embedded_content = ""
            if response_format == "html" and artifact_context.get("file_path"):
                embedded_content = self._embed_html_content(
                    artifact_context["file_path"]
                )

            if embedded_content:
                # Show embedded HTML with iframe for chart rendering
                content_html = f"""
                    <p>The AI model returned a direct {format_label} document with embedded charts:</p>
                    <div class="embedded-html-container" style="margin: 20px 0;">
                        {embedded_content}
                    </div>
                    <p style="margin-top: 15px;">
                        <strong>File:</strong> <code>{artifact_context["file_name"]}</code> ({artifact_context["file_size"]})
                        <a href="{artifact_context["file_url"]}" class="download-link" style="margin-left: 15px; padding: 8px 16px; font-size: 0.9rem;" download="{artifact_context["file_name"]}">
                            📥 Download {format_label}
                        </a>
                    </p>
                """
            else:
                # Fallback to download link only
                content_html = f"""
                    <p>The AI model returned a direct {format_label} document. Download the preserved output below.</p>
                    <p><strong>File:</strong> <code>{artifact_context["file_name"]}</code> ({artifact_context["file_size"]})</p>
                    <p>
                        <a href="{artifact_context["file_url"]}" class="download-link" download="{artifact_context["file_name"]}">
                            Open {format_label} Response
                        </a>
                    </p>
                    <p style="font-size: 0.9rem; color: #6c757d;">
                        Saved automatically because the response was not markdown and cannot be safely embedded here.
                    </p>
                """
        elif artifact_context.get("status") == "empty":
            content_html = "<p>No AI analysis data was available to save.</p>"
        else:
            error_message = artifact_context.get(
                "message", "Unable to persist AI response artifact."
            )
            content_html = f"""
                <div class="alert alert-warning">
                    <h4>⚠️ Unable to save direct AI response</h4>
                    <p>{self._escape_html(error_message)}</p>
                </div>
            """

        analysis_html = f"""
            <div class="analysis-section">
                <h3>AI Analysis ({format_label} response) <span class="execution-time">({execution_time:.1f}s)</span></h3>
                <div class="analysis-content">
                    {content_html}
                </div>
                <div class="analysis-meta">
                    <small><strong>Turn ID:</strong> {turn_id} | <strong>Status:</strong> {status}</small>
                </div>
            </div>
        """

        return analysis_html

    def _embed_html_content(self, file_path: str) -> str:
        """
        Read HTML file and create an iframe embed for chart rendering.

        Uses srcdoc attribute to embed the full HTML document in an iframe,
        allowing Chart.js scripts to execute in an isolated context.

        Args:
            file_path: Path to the saved HTML file

        Returns:
            HTML string with iframe embed, or empty string on failure
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"HTML file not found for embedding: {file_path}")
                return ""

            with open(file_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            if not html_content.strip():
                return ""

            # Escape the HTML content for use in srcdoc attribute
            # srcdoc requires HTML entities for quotes and special chars
            escaped_content = (
                html_content.replace("&", "&amp;")
                .replace('"', "&quot;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

            # Create iframe with srcdoc for isolated script execution
            # sandbox="allow-scripts" enables JavaScript while keeping it isolated
            iframe_html = f"""
                <iframe
                    srcdoc="{escaped_content}"
                    style="width: 100%; min-height: 800px; border: 1px solid #e9ecef; border-radius: 6px; background: white;"
                    sandbox="allow-scripts"
                    loading="lazy"
                    title="Embedded AI Analysis Report"
                ></iframe>
                <script>
                    // Auto-resize iframe to fit content
                    (function() {{
                        var iframe = document.querySelector('.embedded-html-container iframe');
                        if (iframe) {{
                            iframe.onload = function() {{
                                try {{
                                    var height = iframe.contentDocument.body.scrollHeight;
                                    if (height > 0) {{
                                        iframe.style.height = Math.min(height + 50, 2000) + 'px';
                                    }}
                                }} catch (e) {{
                                    // Cross-origin restriction, keep default height
                                }}
                            }};
                        }}
                    }})();
                </script>
            """

            return iframe_html

        except Exception as e:
            logger.error(f"Failed to embed HTML content from {file_path}: {e}")
            return ""

    def _format_json_result(self, result: Any) -> str:
        """Format result as pretty-printed JSON with HTML escaping."""

        try:
            # Pretty print JSON with 2-space indentation
            # Phase 1 KV-Cache Optimization: Use sort_keys=True for deterministic JSON
            json_str = json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False)
            # Escape HTML to prevent rendering issues
            return self._escape_html(json_str)
        except (TypeError, ValueError):
            # Fallback for non-serializable objects
            return self._escape_html(str(result))

    def _get_args_preview(self, arguments: dict) -> str:
        """Generate preview text for arguments expand button.

        Args:
            arguments: Dictionary of tool arguments

        Returns:
            Preview text suitable for expand button label
        """
        if not arguments:
            return "No arguments"

        arg_count = len(arguments)
        if arg_count == 1:
            # For single argument, show key and truncated value
            key, value = next(iter(arguments.items()))
            value_str = str(value)
            if len(value_str) <= 40:
                return f"{key}: {value_str}"
            return f"{key}: {value_str[:37]}..."

        # For multiple arguments, just show count
        return f"{arg_count} arguments"

    def _generate_html_summary_from_knowledge(
        self, knowledge: Dict, analysis_result: Dict
    ) -> str:
        """
        Generate HTML summary from accumulated knowledge tool results.

        This method creates a rich summary based on actual MCP tool execution data,
        replacing the compacted conversation text with detailed findings.

        Args:
            knowledge: Accumulated knowledge dictionary with tool_results
            analysis_result: Analysis result for metadata (execution time, etc.)

        Returns:
            Formatted markdown text summarizing tool execution and findings
        """
        tool_results = knowledge.get("tool_results", [])
        servers_used = knowledge.get("servers_used", [])

        if not tool_results:
            return "No tool results available."

        # Build summary sections
        summary_parts = []

        # Header with server information
        server_names = (
            ", ".join(servers_used).title() if servers_used else "MCP Servers"
        )
        summary_parts.append(f"## MCP Tool Analysis - {server_names}")
        summary_parts.append("")
        summary_parts.append(
            f"Executed **{len(tool_results)}** tool calls across **{len(servers_used)}** MCP server(s)."
        )
        summary_parts.append("")

        # Group by server for detailed breakdown
        by_server: Dict[str, List[Dict]] = {}
        for result in tool_results:
            server = result["server"]
            if server not in by_server:
                by_server[server] = []
            by_server[server].append(result)

        # Server-by-server breakdown
        summary_parts.append("### Tool Execution by Server:")
        summary_parts.append("")
        for server, results in by_server.items():
            tool_names = list(set(r["tool"] for r in results))
            summary_parts.append(
                f"**{server.title()}**: {len(results)} calls ({len(tool_names)} unique tools)"
            )
            summary_parts.append("")

            # List unique tools executed
            tool_count_map: Dict[str, int] = {}
            for result in results:
                tool_name = result["tool"]
                tool_count_map[tool_name] = tool_count_map.get(tool_name, 0) + 1

            for tool_name, count in sorted(
                tool_count_map.items(), key=lambda x: x[1], reverse=True
            )[:5]:
                summary_parts.append(f"- `{tool_name}`: {count} call(s)")

            if len(tool_count_map) > 5:
                summary_parts.append(f"- ... and {len(tool_count_map) - 5} more tools")

            summary_parts.append("")

        # Add execution metadata
        total_execution_time = knowledge.get("total_execution_time", 0)
        if total_execution_time > 0:
            summary_parts.append(
                f"**Total Tool Execution Time**: {total_execution_time:.1f}s"
            )
            summary_parts.append("")

        # Note about full data availability
        summary_parts.append("---")
        summary_parts.append("")
        summary_parts.append(
            "*Note*: This summary is generated from complete MCP tool execution data. "
            "See **Tool Execution Details** section below for full results and **Raw JSON Data** "
            "for the complete dataset."
        )

        return "\n".join(summary_parts)

    def _format_analysis_text(self, text: str) -> str:
        """Convert Markdown analysis text to HTML with comprehensive formatting."""
        if not text:
            return ""

        # Preprocess: normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Convert markdown headers (order matters - start with more specific)
        text = re.sub(r"^#### (.*?)$", r"<h4>\1</h4>", text, flags=re.MULTILINE)
        text = re.sub(r"^### (.*?)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
        text = re.sub(r"^## (.*?)$", r"<h2>\1</h2>", text, flags=re.MULTILINE)
        text = re.sub(r"^# (.*?)$", r"<h1>\1</h1>", text, flags=re.MULTILINE)

        # Convert markdown links (do this before other processing)
        text = re.sub(
            r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" target="_blank">\1</a>', text
        )

        # Split into lines for list processing
        lines = text.split("\n")
        formatted_lines = []
        in_ordered_list = False
        in_unordered_list = False
        in_code_block = False
        code_block_lines: List[str] = []

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Handle code blocks
            if stripped.startswith("```") and len(stripped) >= 3:
                if not in_code_block:
                    # Start code block
                    in_code_block = True
                    language = stripped[3:].strip()
                    code_block_lines = []
                    if language:
                        formatted_lines.append(
                            f'<pre class="code-block language-{language}"><code>'
                        )
                    else:
                        formatted_lines.append('<pre class="code-block"><code>')
                else:
                    # End code block
                    in_code_block = False
                    # Add accumulated code lines
                    for code_line in code_block_lines:
                        formatted_lines.append(self._escape_html(code_line))
                    formatted_lines.append("</code></pre>")
                    code_block_lines = []
                i += 1
                continue

            if in_code_block:
                code_block_lines.append(line)
                i += 1
                continue

            # Handle numbered lists (1. 2. etc.)
            if re.match(r"^\s*\d+\.\s+", stripped):
                if in_unordered_list:
                    formatted_lines.append("</ul>")
                    in_unordered_list = False
                if not in_ordered_list:
                    formatted_lines.append('<ol class="analysis-list">')
                    in_ordered_list = True

                # Extract list item content (everything after the number and dot)
                match = re.match(r"^\s*\d+\.\s+(.*)", stripped)
                if match:
                    item_content = match.group(1)
                    formatted_lines.append(
                        f"<li>{self._apply_smart_highlighting(item_content)}</li>"
                    )

            # Handle bullet lists (- or * or +, including nested with spaces)
            elif re.match(r"^\s*[-*+]\s+", stripped):
                if in_ordered_list:
                    formatted_lines.append("</ol>")
                    in_ordered_list = False
                if not in_unordered_list:
                    formatted_lines.append('<ul class="analysis-list">')
                    in_unordered_list = True

                # Extract list item content (everything after the bullet)
                match = re.match(r"^\s*[-*+]\s+(.*)", stripped)
                if match:
                    item_content = match.group(1)
                    # Check if this is a nested item (starts with spaces and then more content)
                    if line.startswith("  ") and not line.startswith("   "):
                        # This is a nested item, format it differently
                        style = "margin-left: 20px; font-size: 0.95em;"
                        content = self._apply_smart_highlighting(item_content)
                        formatted_lines.append(f"<li style='{style}'>{content}</li>")
                    else:
                        formatted_lines.append(
                            f"<li>{self._apply_smart_highlighting(item_content)}</li>"
                        )

            # Handle empty lines
            elif not stripped:
                # Close any open lists
                if in_ordered_list:
                    formatted_lines.append("</ol>")
                    in_ordered_list = False
                if in_unordered_list:
                    formatted_lines.append("</ul>")
                    in_unordered_list = False
                # Add spacing only if we have content before and after
                if formatted_lines and i < len(lines) - 1:
                    formatted_lines.append('<div class="paragraph-break"></div>')

            # Handle headers (already converted above)
            elif re.match(r"^<h[1-6]", stripped):
                # Close any open lists
                if in_ordered_list:
                    formatted_lines.append("</ol>")
                    in_ordered_list = False
                if in_unordered_list:
                    formatted_lines.append("</ul>")
                    in_unordered_list = False
                formatted_lines.append(stripped)

            # Handle markdown tables
            elif "|" in stripped and self._is_table_row(stripped):
                # Close any open lists for tables
                if in_ordered_list:
                    formatted_lines.append("</ol>")
                    in_ordered_list = False
                if in_unordered_list:
                    formatted_lines.append("</ul>")
                    in_unordered_list = False

                # Parse and render the table
                table_html, lines_consumed, headers, rows = self._parse_markdown_table(
                    lines, i
                )
                formatted_lines.append(table_html)
                i += lines_consumed - 1  # -1 because the loop will increment i

            # Handle regular paragraphs
            elif stripped:
                # Close any open lists for regular paragraphs
                if in_ordered_list:
                    formatted_lines.append("</ol>")
                    in_ordered_list = False
                if in_unordered_list:
                    formatted_lines.append("</ul>")
                    in_unordered_list = False

                # Format as paragraph with smart highlighting
                formatted_content = self._apply_smart_highlighting(stripped)
                formatted_lines.append(f"<p>{formatted_content}</p>")

            i += 1

        # Close any remaining open lists
        if in_ordered_list:
            formatted_lines.append("</ol>")
        if in_unordered_list:
            formatted_lines.append("</ul>")
        if in_code_block and code_block_lines:
            # Close unclosed code block
            for code_line in code_block_lines:
                formatted_lines.append(self._escape_html(code_line))
            formatted_lines.append("</code></pre>")

        return "\n".join(formatted_lines)

    def _format_inline_markdown(self, text: str) -> str:
        """Format inline markdown elements within text."""
        if not text:
            return ""

        # Bold and italic (already done in main function, but ensure it's applied)
        text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"\*(.*?)\*", r"<em>\1</em>", text)

        # Inline code
        text = re.sub(r"`(.*?)`", r"<code>\1</code>", text)

        # Links (already done in main function, but ensure it's applied)
        text = re.sub(
            r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" target="_blank">\1</a>', text
        )

        return text

    def _classify_percentage(self, percentage_str: str) -> str:
        """Classify percentage value and return appropriate CSS class."""
        try:
            # Extract numeric value from percentage string
            import re

            match = re.search(r"(\d+(?:\.\d+)?)", percentage_str)
            if not match:
                return ""

            value = float(match.group(1))

            # Apply threshold-based classification
            if value == 0:
                return "percentage-perfect"  # 0% failure is perfect
            elif value < 20:
                return "percentage-good"  # Low failure rate
            elif value < 80:
                return "percentage-warning"  # Moderate failure rate
            else:
                return "percentage-danger"  # High failure rate

        except (ValueError, AttributeError):
            return ""

    def _classify_status_word(self, word: str) -> str:
        """Classify status word and return appropriate CSS class."""
        word_upper = word.upper()

        # Success indicators
        success_words = {
            "SUCCESS",
            "COMPLETED",
            "PASS",
            "PASSED",
            "OK",
            "GOOD",
            "ACTIVE",
            "ENABLED",
        }
        if word_upper in success_words:
            return "status-success"

        # Warning indicators
        warning_words = {"WARNING", "PARTIAL", "PENDING", "CAUTION", "SLOW", "DEGRADED"}
        if word_upper in warning_words:
            return "status-warning"

        # Danger indicators
        danger_words = {
            "ERROR",
            "FAILED",
            "FAILURE",
            "CRITICAL",
            "DANGER",
            "BAD",
            "BROKEN",
            "DOWN",
            "DISABLED",
        }
        if word_upper in danger_words:
            return "status-danger"

        return ""

    def _apply_smart_highlighting(self, text: str) -> str:
        """Enhanced inline markdown processing with intelligent status and percentage coloring."""
        if not text:
            return ""

        # First apply standard inline markdown
        text = self._format_inline_markdown(text)

        # Detect and highlight status words FIRST (to avoid conflicts with percentage CSS classes)
        def highlight_status(match):
            word = match.group(0)
            css_class = self._classify_status_word(word)
            if css_class:
                return f'<span class="{css_class}">{word}</span>'
            return word

        # Pattern for status words (case-insensitive, word boundaries)
        status_pattern = (
            r"\b(?:SUCCESS|COMPLETED|PASS|PASSED|OK|GOOD|ACTIVE|ENABLED|WARNING|PARTIAL|PENDING|"
            r"CAUTION|SLOW|DEGRADED|ERROR|FAILED|FAILURE|CRITICAL|DANGER|BAD|BROKEN|DOWN|DISABLED)\b"
        )
        text = re.sub(status_pattern, highlight_status, text, flags=re.IGNORECASE)

        # Detect and highlight percentages AFTER status words (to avoid nested spans)
        def highlight_percentage(match):
            percentage = match.group(0)
            css_class = self._classify_percentage(percentage)
            if css_class:
                return f'<span class="{css_class}">{percentage}</span>'
            return percentage

        # Pattern for percentages: number followed by % (with proper word boundaries)
        text = re.sub(r"(?<!\w)\d+(?:\.\d+)?%(?!\w)", highlight_percentage, text)

        return text

    def _is_table_row(self, line: str) -> bool:
        """Check if a line appears to be a markdown table row."""

        # Must contain pipes and have at least one character between pipes
        if "|" not in line:
            return False

        # Split by pipe and check if we have reasonable content
        parts = line.split("|")

        # Need at least 3 parts (empty, content, empty) for a valid table row
        if len(parts) < 3:
            return False

        # Check if it looks like a separator row (contains only dashes, colons, spaces, and pipes)
        separator_pattern = r"^[\s\|\:\-]+$"
        if re.match(separator_pattern, line.strip()):
            return True

        # Check if it has actual content between pipes (not all empty)
        content_parts = [
            part.strip() for part in parts[1:-1]
        ]  # Exclude first and last empty parts
        has_content = any(part for part in content_parts)

        return has_content

    def _parse_markdown_table(self, lines: List[str], start_index: int) -> tuple:
        """Parse a markdown table starting at the given index and return HTML and lines consumed."""

        table_lines = []
        current_index = start_index

        # Collect all consecutive table lines
        while current_index < len(lines):
            line = lines[current_index].strip()
            if line and self._is_table_row(line):
                table_lines.append(line)
                current_index += 1
            else:
                break

        if not table_lines:
            return "", 0

        # Parse the table structure
        headers = []
        alignments = []
        rows = []

        # First line should be headers
        if table_lines:
            header_line = table_lines[0]
            headers = [cell.strip() for cell in header_line.split("|")[1:-1]]

        # Second line might be alignment specifiers
        alignment_line = None
        data_start = 1
        if len(table_lines) > 1:
            potential_alignment = table_lines[1]
            # Check if this is an alignment row (contains only dashes, colons, spaces, pipes)
            if re.match(r"^[\s\|\:\-]+$", potential_alignment):
                alignment_line = potential_alignment
                data_start = 2
                # Parse alignments
                alignment_parts = [
                    part.strip() for part in alignment_line.split("|")[1:-1]
                ]
                for part in alignment_parts:
                    if part.startswith(":") and part.endswith(":"):
                        alignments.append("center")
                    elif part.endswith(":"):
                        alignments.append("right")
                    else:
                        alignments.append("left")

        # Ensure we have alignments for all columns
        while len(alignments) < len(headers):
            alignments.append("left")

        # Parse data rows
        for line in table_lines[data_start:]:
            row_data = [cell.strip() for cell in line.split("|")[1:-1]]
            # Pad or trim row to match header count
            while len(row_data) < len(headers):
                row_data.append("")
            rows.append(row_data[: len(headers)])

        # Generate HTML table
        html_parts = ['<table class="markdown-table">']

        # Headers
        if headers:
            html_parts.append("<thead><tr>")
            for i, header in enumerate(headers):
                align = alignments[i] if i < len(alignments) else "left"
                header_content = self._apply_smart_highlighting(header)
                html_parts.append(
                    f'<th style="text-align: {align};">{header_content}</th>'
                )
            html_parts.append("</tr></thead>")

        # Data rows
        if rows:
            html_parts.append("<tbody>")
            for row in rows:
                html_parts.append("<tr>")
                for i, cell in enumerate(row):
                    align = alignments[i] if i < len(alignments) else "left"
                    cell_content = self._apply_smart_highlighting(cell)
                    html_parts.append(
                        f'<td style="text-align: {align};">{cell_content}</td>'
                    )
                html_parts.append("</tr>")
            html_parts.append("</tbody>")

        html_parts.append("</table>")

        return "\n".join(html_parts), current_index - start_index, headers, rows

    def _persist_direct_ai_response(
        self,
        ai_content: Any,
        response_format: str,
        json_report_path: Optional[str],
        report_id: str,
    ) -> Dict[str, Any]:
        """Persist raw AI response when it's not markdown so it can be linked from the report."""
        if not ai_content:
            return {"status": "empty", "message": "AI analysis body was empty."}

        try:
            if json_report_path:
                base_path = os.path.splitext(json_report_path)[0]
            else:
                # Fallback to report_id in current working directory
                fallback_base = report_id or "ai_analysis_report"
                base_path = os.path.join(".", fallback_base)

            extension = "html" if response_format == "html" else "json"
            target_path = f"{base_path}_ai_direct_resp.{extension}"
            abs_target_dir = os.path.dirname(os.path.abspath(target_path))
            if abs_target_dir:
                os.makedirs(abs_target_dir, exist_ok=True)

            serialized_content = self._prepare_direct_response_content(
                ai_content, response_format
            )

            with open(target_path, "w", encoding="utf-8") as f:
                f.write(serialized_content)

            size_bytes = os.path.getsize(target_path)
            file_name = os.path.basename(target_path)
            return {
                "status": "success",
                "file_path": target_path,
                "file_name": file_name,
                "file_size": self._format_human_file_size(size_bytes),
                "file_url": self._build_artifact_url(file_name),
            }
        except Exception as e:
            logger.error(
                f"Failed to persist direct AI response ({response_format}): {e}",
                exc_info=True,
            )
            return {
                "status": "error",
                "message": f"Failed to save {response_format} response: {e}",
            }

    def _prepare_direct_response_content(
        self, ai_content: Any, response_format: str
    ) -> str:
        """Normalize AI response before persisting as standalone artifact."""
        text = ""

        if response_format == "json" and not isinstance(ai_content, str):
            try:
                text = json.dumps(ai_content, indent=2, ensure_ascii=False)
            except (TypeError, ValueError):
                text = str(ai_content)
        else:
            try:
                text = str(ai_content)
            except Exception:
                text = ""

        text = text.strip()
        if response_format == "html":
            return self._extract_html_payload(text)
        if response_format == "json":
            return self._extract_json_payload(text)
        return text

    def _extract_html_payload(self, text: str) -> str:
        """Remove narrative text/code fences preceding actual HTML markup."""
        if not text:
            return text

        fenced = self._extract_code_fence_payload(text, ["html", "htm"])
        if fenced:
            return self._trim_html_terminator(fenced.strip())

        lowered = text.lower()
        for marker in ("<!doctype", "<html"):
            idx = lowered.find(marker)
            if idx != -1:
                snippet = text[idx:].strip()
                return self._trim_html_terminator(snippet)

        return self._trim_html_terminator(text)

    def _extract_json_payload(self, text: str) -> str:
        """Remove preamble before JSON payload or code fences."""
        if not text:
            return text

        fenced = self._extract_code_fence_payload(text, ["json"])
        if fenced:
            return self._trim_json_terminator(fenced.strip())

        for i, ch in enumerate(text):
            if ch in "{[":
                return self._trim_json_terminator(text[i:].strip())
        return self._trim_json_terminator(text)

    def _extract_code_fence_payload(
        self, text: str, languages: List[str]
    ) -> Optional[str]:
        """Extract payload from markdown code fences."""
        if not text or "```" not in text:
            return None

        if languages:
            fence_pattern = r"```(?:{langs})\s*(.*?)```".format(
                langs="|".join([re.escape(lang) for lang in languages])
            )
            match = re.search(fence_pattern, text, flags=re.DOTALL | re.IGNORECASE)
            if match:
                candidate = match.group(1)
                if self._looks_like_language(candidate, languages):
                    return candidate

        generic_match = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
        if generic_match:
            return generic_match.group(1)
        return None

    def _looks_like_language(self, text: str, languages: List[str]) -> bool:
        """Heuristic check to ensure fenced payload matches the expected language."""
        sample = text.strip().lower()
        if "html" in languages:
            return sample.startswith("<") or sample.startswith("!")
        if "json" in languages:
            return sample.startswith("{") or sample.startswith("[")
        return True

    def _trim_html_terminator(self, text: str) -> str:
        """Ensure HTML payload contains only content through </html> if present."""
        closing_tag = re.search(r"</html\s*>", text, flags=re.IGNORECASE)
        if closing_tag:
            end_idx = closing_tag.end()
            return text[:end_idx].strip()
        return text

    def _trim_json_terminator(self, text: str) -> str:
        """Trim trailing narrative after final closing brace/bracket."""
        stack = []
        last_valid_index = None
        for idx, ch in enumerate(text):
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()
                last_valid_index = idx

            # Early exit if we've closed everything
            if not stack and last_valid_index is not None and ch in "}]":
                remainder = text[last_valid_index + 1 :].strip()
                if remainder:
                    continue

        if last_valid_index is not None:
            return text[: last_valid_index + 1].strip()

        return text

    def _build_artifact_url(self, filename: str) -> str:
        """Construct artifact URL for CI environments or fallback to filename."""
        ci_project_url = os.getenv("CI_PROJECT_URL")
        ci_job_id = os.getenv("CI_JOB_ID")
        if ci_project_url and ci_job_id:
            return f"{ci_project_url}/-/jobs/{ci_job_id}/artifacts/external_file/{filename}"
        return filename

    def _format_human_file_size(self, size_bytes: int) -> str:
        """Return human-readable file size."""
        if size_bytes >= 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        elif size_bytes > 0:
            return f"{size_bytes / 1024:.2f} KB"
        return "0 bytes"

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )

    def _get_ci_info(self, report_id: str) -> str:
        """Get CI/CD environment information from configurable env vars."""
        ci_info = []

        for var, label in self.ci_env_vars:
            value = os.getenv(var)
            if value:
                if var == "CI_JOB_URL":
                    ci_info.append(
                        f'<li><strong>{label}:</strong> <a href="{value}" target="_blank">View Job</a></li>'
                    )
                elif var == "CI_COMMIT_SHA":
                    ci_info.append(
                        f"<li><strong>{label}:</strong> <code>{value[:8]}...</code></li>"
                    )
                else:
                    ci_info.append(f"<li><strong>{label}:</strong> {value}</li>")

        # Add HTML Report URL if running in CI with job artifacts
        ci_project_url = os.getenv("CI_PROJECT_URL")
        ci_job_id = os.getenv("CI_JOB_ID")
        if ci_project_url and ci_job_id and report_id and report_id != "unknown":
            # Construct the direct link to this HTML report in job artifacts
            # Format: {CI_PROJECT_URL}/-/jobs/{CI_JOB_ID}/artifacts/external_file/{report_id}.html
            html_filename = f"{report_id}.html"
            html_report_url = f"{ci_project_url}/-/jobs/{ci_job_id}/artifacts/external_file/{html_filename}"
            ci_info.append(
                f'<li><strong>HTML Report URL:</strong> <a href="{html_report_url}" target="_blank">{html_filename}</a></li>'
            )

        if ci_info:
            return f'<ul class="ci-info">{"".join(ci_info)}</ul>'
        else:
            return "<p><em>No CI environment information available.</em></p>"

    def _get_html_template(self) -> str:
        """Get the HTML template for reports."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            border-bottom: 3px solid #007bff;
            margin-bottom: 30px;
            padding-bottom: 20px;
        }}
        .header-content {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}
        .header-logo {{
            flex-shrink: 0;
        }}
        .header-logo img {{
            display: block;
            height: {logo_height}px;
            width: auto;
            max-width: 200px;
            object-fit: contain;
        }}
        .header-text {{
            flex-grow: 1;
        }}
        .header h1 {{
            color: #007bff;
            margin: 0;
            font-size: 2.2rem;
        }}
        /* Unified stats section with badges */
        .unified-stats {{
            margin: 25px 0;
            padding: 15px 20px;
            background: white;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }}
        .metadata-text {{
            display: flex;
            gap: 30px;
            margin-bottom: 12px;
            font-size: 0.85rem;
            color: #495057;
        }}
        .metadata-text span {{
            display: inline-block;
        }}
        .metadata-text strong {{
            color: #212529;
            margin-right: 5px;
        }}
        .stats-row {{
            display: flex;
            flex-wrap: nowrap;
            gap: 8px;
            overflow-x: auto;
        }}
        .stat-badge-item {{
            flex: 0 0 auto;
        }}
        .stat-badge {{
            display: inline-block;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 500;
            white-space: nowrap;
            text-align: center;
        }}
        .stat-badge.success {{
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        .stat-badge.danger {{
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        .stat-badge.info {{
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }}
        .stat-badge.primary {{
            background: #cfe2ff;
            color: #084298;
            border: 1px solid #b6d4fe;
        }}
        .stat-badge.token {{
            background: #d1f4e0;
            color: #0f5132;
            border: 1px solid #badbcc;
        }}
        .stat-badge.token-detail {{
            background: #e7f3ff;
            color: #004085;
            border: 1px solid #b8daff;
            font-size: 0.75rem;
        }}
        .stat-badge.inference {{
            background: #e0e7ff;
            color: #3730a3;
            border: 1px solid #c7d2fe;
            font-weight: 500;
        }}
        .stat-badge.model {{
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
            font-weight: 600;
        }}
        .primary-section {{
            border-left: 4px solid #007bff;
            background: #f8f9fa;
        }}
        .section {{
            margin: 40px 0;
            padding: 25px;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            background: #fdfdfd;
        }}
        .section h2 {{
            color: #495057;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .analysis-section {{
            margin: 20px 0;
            padding: 20px;
            border-left: 4px solid #007bff;
            background: #f8f9fa;
            border-radius: 0 6px 6px 0;
        }}
        .analysis-section.error {{
            border-left-color: #dc3545;
            background: #f8d7da;
        }}
        .analysis-section.intelligent-summary {{
            border-left-color: #28a745;
            background: linear-gradient(135deg, #d4edda 0%, #f8f9fa 100%);
            border: 1px solid #c3e6cb;
        }}
        .summary-badge {{
            margin: 10px 0 20px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-radius: 12px;
        }}
        .badge-info {{
            background: linear-gradient(135deg, #17a2b8 0%, #20c997 100%);
            color: white;
            box-shadow: 0 2px 4px rgba(23, 162, 184, 0.3);
        }}
        .summary-badge small {{
            color: #495057;
            font-style: italic;
        }}
        .enhanced-summary {{
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border: 1px solid #c3e6cb;
            box-shadow: 0 2px 8px rgba(40, 167, 69, 0.1);
        }}
        .analysis-section h3 {{
            margin: 0 0 15px 0;
            color: #495057;
        }}
        .execution-time {{
            font-size: 0.9rem;
            color: #6c757d;
            font-weight: normal;
        }}
        .analysis-content {{
            background: white;
            padding: 20px;
            border-radius: 6px;
            border: 1px solid #e9ecef;
            line-height: 1.7;
        }}
        .analysis-content p {{
            margin: 10px 0;
        }}
        .analysis-content h1 {{
            font-size: 1.8rem;
            color: #343a40;
            margin: 20px 0 15px 0;
            border-bottom: 2px solid #007bff;
            padding-bottom: 5px;
        }}
        .analysis-content h2 {{
            font-size: 1.5rem;
            color: #495057;
            margin: 18px 0 12px 0;
            border-bottom: 1px solid #dee2e6;
            padding-bottom: 3px;
        }}
        .analysis-content h3 {{
            font-size: 1.3rem;
            color: #6c757d;
            margin: 15px 0 10px 0;
        }}
        .analysis-content h4 {{
            font-size: 1.1rem;
            color: #868e96;
            margin: 12px 0 8px 0;
            font-weight: 600;
        }}
        .analysis-content ol {{
            padding-left: 20px;
            margin: 15px 0;
        }}
        .analysis-content ul {{
            padding-left: 20px;
            margin: 15px 0;
        }}
        .analysis-content .analysis-list {{
            background: #f8f9fa;
            padding: 15px 20px;
            border-radius: 4px;
            border-left: 3px solid #007bff;
        }}
        .analysis-content .analysis-list li {{
            margin: 8px 0;
            line-height: 1.6;
        }}
        .analysis-content li {{
            margin: 8px 0;
        }}
        .analysis-content strong {{
            color: #495057;
            font-weight: 600;
        }}
        .analysis-content em {{
            color: #6c757d;
            font-style: italic;
        }}
        .analysis-content code {{
            background: #f1f3f4;
            color: #d73a49;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
        }}
        .analysis-content .code-block {{
            background: #f6f8fa;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            padding: 16px;
            margin: 15px 0;
            overflow-x: auto;
        }}
        .analysis-content .code-block code {{
            background: none;
            color: #24292e;
            padding: 0;
            font-size: 0.85rem;
            line-height: 1.45;
        }}
        .analysis-content .paragraph-break {{
            margin: 15px 0;
        }}
        .analysis-content a {{
            color: #007bff;
            text-decoration: none;
            border-bottom: 1px dotted #007bff;
        }}
        .analysis-content a:hover {{
            color: #0056b3;
            border-bottom: 1px solid #0056b3;
        }}
        .analysis-content .markdown-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-radius: 6px;
            overflow: hidden;
        }}
        .analysis-content .markdown-table th {{
            background: #f8f9fa;
            color: #495057;
            font-weight: 600;
            padding: 12px 15px;
            border-bottom: 2px solid #dee2e6;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .analysis-content .markdown-table td {{
            padding: 10px 15px;
            border-bottom: 1px solid #e9ecef;
            font-size: 0.9rem;
            line-height: 1.5;
        }}
        .analysis-content .markdown-table tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        .analysis-content .markdown-table tr:hover {{
            background: #e3f2fd;
        }}
        .analysis-content .markdown-table tbody tr:last-child td {{
            border-bottom: none;
        }}
        /* Status and threshold-based coloring */
        .status-success {{
            color: #28a745;
            font-weight: 600;
        }}
        .status-warning {{
            color: #ffc107;
            font-weight: 600;
        }}
        .status-danger {{
            color: #dc3545;
            font-weight: 600;
        }}
        .percentage-perfect {{
            background: #d4edda;
            color: #155724;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
        }}
        .percentage-good {{
            background: #d1ecf1;
            color: #0c5460;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 500;
        }}
        .percentage-warning {{
            background: #fff3cd;
            color: #856404;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
        }}
        .percentage-danger {{
            background: #f8d7da;
            color: #721c24;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
        }}
        .status-badge {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .status-badge.success {{
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        .status-badge.warning {{
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }}
        .status-badge.danger {{
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        .analysis-meta {{
            margin-top: 15px;
            padding-top: 10px;
            border-top: 1px solid #e9ecef;
            font-size: 0.9rem;
            color: #6c757d;
        }}
        .tool-section {{
            margin: 20px 0;
            padding: 20px;
            background: #e3f2fd;
            border-radius: 6px;
            border: 1px solid #bbdefb;
        }}
        .inference-group {{
            margin: 15px 0;
            padding: 15px;
            background: #ffffff;
            border-radius: 6px;
            border-left: 4px solid #1976d2;
        }}
        .inference-group h5 {{
            margin: 0 0 10px 0;
            color: #1976d2;
            font-size: 1rem;
            font-weight: 600;
        }}
        .tool-execution-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 6px;
            overflow: hidden;
        }}
        .tool-execution-table thead {{
            background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%);
            color: white;
        }}
        .tool-execution-table th {{
            padding: 12px 10px;
            text-align: left;
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid #0d47a1;
        }}
        .tool-execution-table td {{
            padding: 10px;
            border-bottom: 1px solid #e3f2fd;
            font-size: 0.9rem;
            vertical-align: top;
        }}
        .tool-execution-table tbody tr:hover {{
            background: #f5f5f5;
        }}
        .tool-execution-table tbody tr:last-child td {{
            border-bottom: none;
        }}
        .tool-seq {{
            width: 40px;
            text-align: center;
            font-weight: 600;
            color: #1976d2;
        }}
        .tool-ref {{
            width: 90px;
            text-align: center;
            font-weight: 600;
            color: #28a745;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.85rem;
        }}
        .tool-iteration {{
            width: 80px;
            text-align: center;
            font-weight: 500;
        }}
        .tool-server {{
            width: 120px;
            color: #495057;
            font-weight: 500;
        }}
        .tool-name {{
            min-width: 150px;
            color: #1976d2;
            font-weight: 600;
        }}
        .tool-args {{
            min-width: 180px;
            font-size: 0.85rem;
        }}
        .tool-args code {{
            background: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-size: 0.8rem;
            color: #d73a49;
        }}
        .tool-result {{
            min-width: 150px;
        }}
        .tool-time {{
            width: 80px;
            text-align: right;
            font-family: 'Monaco', 'Courier New', monospace;
            color: #495057;
        }}
        .expand-btn {{
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 500;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}
        .expand-btn:hover {{
            background: linear-gradient(135deg, #218838 0%, #1fa383 100%);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .expand-btn:active {{
            transform: translateY(1px);
        }}
        .expand-icon {{
            display: inline-block;
            transition: transform 0.2s ease;
            font-size: 0.7rem;
        }}
        .expand-icon.expanded {{
            transform: rotate(90deg);
        }}
        .result-json {{
            margin-top: 10px;
            background: #f6f8fa;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            padding: 12px;
            overflow-x: auto;
            max-height: 400px;
            overflow-y: auto;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.8rem;
            line-height: 1.5;
            color: #24292e;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .alert {{
            padding: 15px;
            margin: 20px 0;
            border-radius: 6px;
            border: 1px solid transparent;
        }}
        .alert-warning {{
            background-color: #fff3cd;
            border-color: #ffeaa7;
            color: #856404;
        }}
        .error-message {{
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #f5c6cb;
        }}
        .ci-info {{
            list-style: none;
            padding: 0;
        }}
        .ci-info li {{
            padding: 8px 0;
            border-bottom: 1px solid #e9ecef;
        }}
        .ci-info li:last-child {{
            border-bottom: none;
        }}
        code {{
            background: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.9rem;
            word-wrap: break-word;
            overflow-wrap: break-word;
            word-break: break-all;
            white-space: normal;
        }}
        .json-section {{
            margin-top: 30px;
        }}
        .json-content {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 20px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.8rem;
            overflow-x: auto;
            max-height: 500px;
            overflow-y: auto;
        }}
        pre {{
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        a {{
            color: #007bff;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .download-link {{
            display: inline-block;
            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
            color: white;
            padding: 12px 24px;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.2s ease;
            box-shadow: 0 2px 4px rgba(0,123,255,0.3);
        }}
        .download-link:hover {{
            background: linear-gradient(135deg, #0056b3 0%, #004085 100%);
            box-shadow: 0 4px 8px rgba(0,123,255,0.4);
            transform: translateY(-1px);
            color: white;
            text-decoration: none;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e9ecef;
            text-align: center;
            color: #6c757d;
            font-size: 0.9rem;
        }}
    </style>
    <!-- Chart.js library for embedded charts -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-content">
                {logo_html}
                <div class="header-text">
                    <h1>{header_emoji} {main_header}</h1>
                    <p>{sub_header}</p>
                </div>
            </div>
        </div>

        {summary_html}

        <div class="section primary-section">
            <h2>🧠 AI Analysis Results</h2>
            <p>Detailed analysis insights and findings from the AI agent:</p>
            {ai_analysis_html}
        </div>

        <div class="section">
            <h2>🔧 Tool Execution Details</h2>
            <p>MCP tools executed during the analysis:</p>
            {tool_execution_html}
        </div>

        <div class="section">
            <h2>🏗️ CI Environment</h2>
            {ci_info_html}
        </div>

        <div class="section json-section">
            <h2>📄 Raw JSON Data</h2>
            <p>Complete report data is available in JSON format:</p>
            <div style="padding: 20px; background: #f8f9fa; border-radius: 6px; border: 1px solid #e9ecef;">
                <p><strong>JSON Report File:</strong> <code>{json_filename}</code></p>
                <p style="margin-top: 10px;">
                    <a href="{json_report_url}" class="download-link" download="{json_filename}">
                        📥 Download JSON Report ({json_file_size})
                    </a>
                </p>
                <p style="margin-top: 10px; font-size: 0.9rem; color: #6c757d;">
                    <em>The JSON file contains the complete analysis data including all tool results and accumulated knowledge.</em>
                </p>
            </div>
        </div>

        <div class="footer">
            <p>Generated by <strong><a href="{project_url}" target="_blank">Cicaddy</a></strong> | Report ID: <code>{report_id}</code></p>
        </div>
    </div>

    <script>
        function toggleResult(resultId) {{
            const resultElement = document.getElementById(resultId);
            const button = resultElement.previousElementSibling;
            const icon = button.querySelector('.expand-icon');

            if (resultElement.style.display === 'none') {{
                resultElement.style.display = 'block';
                button.innerHTML = '<span class="expand-icon expanded">▶</span> Hide Result';
                icon.classList.add('expanded');
            }} else {{
                resultElement.style.display = 'none';
                button.innerHTML = '<span class="expand-icon">▶</span> Show Result';
                icon.classList.remove('expanded');
            }}
        }}
    </script>
</body>
</html>"""  # noqa: E501
