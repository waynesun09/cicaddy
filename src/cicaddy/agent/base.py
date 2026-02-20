"""Base AI Agent with shared functionality for all agent types."""

import json
import os
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from cicaddy.ai_providers.base import BaseProvider, ProviderMessage
from cicaddy.ai_providers.factory import (
    DEFAULT_AI_PROVIDER,
    create_provider,
    get_default_model,
    get_provider_config,
)
from cicaddy.config.settings import Settings, load_settings
from cicaddy.execution.engine import ExecutionEngine
from cicaddy.mcp_client.client import OfficialMCPClientManager
from cicaddy.notifications.rich_slack import RichSlackNotifier
from cicaddy.tools import ToolRegistry, create_local_file_registry
from cicaddy.utils.logger import get_captured_logs, get_logger

logger = get_logger(__name__)


class BaseAIAgent(ABC):
    """Base AI Agent with shared initialization and core analysis pipeline."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or load_settings()
        self.ai_provider: Optional[BaseProvider] = None
        self.mcp_manager: Optional[OfficialMCPClientManager] = None
        self.local_tool_registry: Optional[ToolRegistry] = None
        self.platform_analyzer = (
            None  # Set by _setup_platform_integration() in subclasses
        )
        self.slack_notifier: Optional[RichSlackNotifier] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.start_time = datetime.now()
        # Maximum inference iterations for multi-step execution (like Llama Stack)
        # Provide sane default if settings mock omits this attribute in tests
        self.max_infer_iters = getattr(self.settings, "max_infer_iters", 10)

    async def initialize(self):
        """Initialize all shared components."""
        logger.info(f"Initializing {self.__class__.__name__}")

        # Initialize AI provider
        await self._setup_ai_provider()

        # Initialize MCP client manager
        await self._setup_mcp_manager()

        # Initialize local tool registry
        await self._setup_local_tools()

        # Initialize platform-specific integration (GitLab, GitHub, etc.)
        await self._setup_platform_integration()

        # Initialize Slack notifier
        await self._setup_slack_notifier()

        # Initialize execution engine
        await self._setup_execution_engine()

        logger.info(f"{self.__class__.__name__} initialized successfully")

    async def analyze(self) -> Dict[str, Any]:
        """
        Main analysis pipeline using template method pattern.

        This method orchestrates the entire analysis process:
        1. Gather context (MR data, project info, etc.)
        2. Build analysis prompt
        3. Execute analysis using AI + MCP tools
        4. Generate reports (JSON + HTML)
        5. Send notifications
        """
        logger.info(f"Starting analysis with {self.__class__.__name__}")

        context = None
        analysis_result = None
        report = None

        try:
            # Step 1: Gather analysis context (subclass-specific)
            context = await self.get_analysis_context()

            # Step 2: Get available MCP tools
            logger.info("Getting available MCP tools...")
            mcp_tools = await self._get_available_tools()
            logger.info(f"Retrieved {len(mcp_tools)} MCP tools")
            context["mcp_tools"] = mcp_tools

            # Step 3: Build analysis prompt (subclass-specific)
            prompt = self.build_analysis_prompt(context)

            # Step 4: Execute analysis using execution engine
            logger.info("Executing AI analysis...")
            analysis_result = await self.execute_analysis(prompt, context)
            logger.info("Analysis completed")

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            # Create failure context if none exists
            if context is None:
                context = await self._create_failure_context()

            # Create failure analysis result
            analysis_result = {
                "error": str(e),
                "error_type": type(e).__name__,
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "ai_response_format": self.settings.ai_response_format,
            }

        try:
            # Step 5: Generate and save report
            report = await self.generate_report(analysis_result, context)
            await self._save_report(report)
            logger.info(f"Report saved: {report['report_id']}")

        except Exception as e:
            logger.error(f"Failed to save report: {e}", exc_info=True)
            # Create minimal fallback report
            report = await self._create_fallback_report(analysis_result, e)

        try:
            # Step 6: Send notifications
            await self.send_notifications(report, analysis_result)
            logger.info("Notifications sent successfully")
        except Exception as e:
            logger.error(f"Failed to send notification: {e}", exc_info=True)

        # Step 7: Save log file at the very end to capture all logs
        # This ensures HTML report generation, notifications, and all other phases are logged
        try:
            if report and report.get("report_id"):
                logger.info("Saving execution log file...")
                await self._save_log_file(report["report_id"])
                logger.info("Execution log file saved successfully")
        except Exception as e:
            logger.error(f"Failed to save log file: {e}", exc_info=True)

        logger.info(
            f"Analysis pipeline completed in {(datetime.now() - self.start_time).total_seconds():.2f}s"
        )

        return {
            "analysis_result": analysis_result,
            "report": report,
            "execution_time": (datetime.now() - self.start_time).total_seconds(),
        }

    # Shared setup methods

    async def _setup_ai_provider(self):
        """Setup AI provider with direct API connections."""
        provider_config = get_provider_config(self.settings)
        provider_name = self.settings.ai_provider or "gemini"

        # Create and initialize provider
        self.ai_provider = create_provider(provider_name, provider_config)
        await self.ai_provider.initialize()

        logger.info(f"AI provider {provider_name} configured")

    async def _setup_mcp_manager(self):
        """Initialize MCP client manager."""
        logger.info(
            f"MCP_SERVERS_CONFIG length: {len(self.settings.mcp_servers_config)}"
        )
        logger.info(
            f"MCP_SERVERS_CONFIG type: {type(self.settings.mcp_servers_config)}"
        )

        mcp_servers = self.settings.get_mcp_servers()
        logger.info(f"Loaded MCP servers configuration: {len(mcp_servers)} servers")

        if mcp_servers:
            for i, server in enumerate(mcp_servers):
                logger.info(f"MCP Server {i + 1}: {server.name} ({server.protocol})")
        else:
            logger.warning("No MCP servers configured in MCP_SERVERS_CONFIG")

        self.mcp_manager = OfficialMCPClientManager(
            mcp_servers, ssl_verify=self.settings.ssl_verify
        )
        await self.mcp_manager.initialize()

    async def _setup_local_tools(self):
        """Setup local tool registry if enabled."""
        enable_local_tools = getattr(self.settings, "enable_local_tools", False)
        if not enable_local_tools:
            logger.debug("Local tools disabled (ENABLE_LOCAL_TOOLS=false)")
            return

        # Determine working directory for local file tools
        working_dir = getattr(self.settings, "local_tools_working_dir", None)
        if not working_dir:
            # Fall back to git_working_directory if set
            working_dir = getattr(self.settings, "git_working_directory", None)

        try:
            self.local_tool_registry = create_local_file_registry(working_dir)
            tool_names = self.local_tool_registry.list_tool_names()
            logger.info(
                f"Local tools enabled: {tool_names} "
                f"(working_dir={working_dir or 'cwd'})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize local tools: {e}", exc_info=True)
            self.local_tool_registry = None

    async def _setup_platform_integration(self):
        """Setup platform-specific integration (e.g., GitLab, GitHub).

        Override in subclasses or platform-specific packages to set up
        platform analyzers. Base implementation is a no-op.
        """
        logger.warning("No project ID available - platform analyzer disabled")

    async def _setup_slack_notifier(self):
        """Setup Slack notifier if configured."""
        slack_webhook_urls = self.settings.get_slack_webhook_urls()
        if (
            isinstance(slack_webhook_urls, (list, tuple))
            and len(slack_webhook_urls) > 0
        ):
            logger.info(
                f"Initializing Slack notifier with {len(slack_webhook_urls)} webhook(s)"
            )
            # Use rich notifier for all cases (supports all report types with enhanced formatting)
            self.slack_notifier = RichSlackNotifier(
                slack_webhook_urls, ssl_verify=self.settings.ssl_verify
            )

    async def _setup_execution_engine(self):
        """Setup execution engine for multi-step analysis."""
        # Import here to avoid circular imports
        from cicaddy.execution.token_aware_executor import ExecutionLimits
        from cicaddy.utils.token_utils import TokenLimitManager

        # Get model-specific token limits
        provider = self.settings.ai_provider or DEFAULT_AI_PROVIDER
        model = self.settings.ai_model or get_default_model(provider)
        token_limits = TokenLimitManager.get_limits(provider, model)

        # Calculate dynamic sub-limits based on model capabilities and recommended ratios
        # These ratios are optimized for effective context utilization:
        # - 6.25% of input for per-iteration (allows ~16 iterations within budget)
        # - 50% of output for tool results (optimized for comprehensive code/data capture)
        #   Early results get full budget; progressive degradation manages high-volume scenarios
        max_tokens_per_iteration = int(token_limits["input"] * 0.0625)  # 6.25% of input
        max_tokens_per_tool_result = int(token_limits["output"] * 0.50)  # 50% of output

        # Ensure minimum sane values for models with smaller capacities
        max_tokens_per_iteration = max(max_tokens_per_iteration, 4096)  # Minimum 4K
        max_tokens_per_tool_result = max(max_tokens_per_tool_result, 1024)  # Minimum 1K

        # Create execution limits with fully dynamic token configuration
        execution_limits = ExecutionLimits(
            max_infer_iters=self.max_infer_iters,
            max_tokens_total=token_limits["input"],  # Use model's max input tokens
            max_tokens_per_iteration=max_tokens_per_iteration,  # Dynamic: ~6.25% of input
            max_tokens_per_tool_result=max_tokens_per_tool_result,  # Dynamic: 50% of output
            max_execution_time=self.settings.max_execution_time,  # Configurable via MAX_EXECUTION_TIME env var
        )

        logger.info(
            f"Execution engine configured with dynamic token limits: "
            f"provider={provider}, model={model}, "
            f"max_tokens_total={execution_limits.max_tokens_total:,}, "
            f"max_tokens_per_iteration={execution_limits.max_tokens_per_iteration:,}, "
            f"max_tokens_per_tool_result={execution_limits.max_tokens_per_tool_result:,}, "
            f"max_infer_iters={self.max_infer_iters}"
        )

        self.execution_engine = ExecutionEngine(
            ai_provider=self.ai_provider,
            mcp_manager=self.mcp_manager,
            local_tool_registry=self.local_tool_registry,
            session_id=self.get_session_id(),
            execution_limits=execution_limits,
            context_safety_factor=self.settings.context_safety_factor,  # Configurable via CONTEXT_SAFETY_FACTOR env var
        )

    # Shared analysis methods

    def _aggregate_token_usage(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate token usage from all inference steps in the analysis result.

        Args:
            analysis_result: The analysis result containing execution steps

        Returns:
            Dict containing aggregated token usage statistics and metadata
        """
        from cicaddy.utils.token_utils import TokenUsageExtractor

        return TokenUsageExtractor.extract_token_usage(
            analysis_result,
            default_provider=self.settings.ai_provider or "unknown",
            include_model_info=True,
        )

    async def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from MCP servers and local tool registry."""
        tools = []

        # Get tools from local tool registry
        if self.local_tool_registry:
            local_tools = self.local_tool_registry.get_tools()
            logger.info(f"Local tool registry returned {len(local_tools)} tools")
            tools.extend(local_tools)

        # Get tools from MCP servers
        if self.mcp_manager:
            server_names = self.mcp_manager.get_server_names()
            logger.info(
                f"Checking tools from {len(server_names)} connected MCP servers: {server_names}"
            )

            for server_name in server_names:
                try:
                    logger.info(f"Listing tools from server: {server_name}")
                    server_tools = await self.mcp_manager.list_tools(server_name)
                    logger.info(
                        f"Server {server_name} returned {len(server_tools)} tools"
                    )
                    for tool in server_tools:
                        tool["server"] = server_name
                    tools.extend(server_tools)
                except Exception as e:
                    logger.error(
                        f"Failed to list tools from server {server_name}: {e}",
                        exc_info=True,
                    )
        else:
            logger.debug("No MCP manager available")

        if not tools:
            logger.warning("No tools available (no local tools or MCP servers)")
            return []

        # Phase 1 KV-Cache Optimization: Sort tools alphabetically for deterministic ordering
        # This ensures consistent tool presentation order across runs, enabling KV-cache hits
        tools.sort(key=lambda t: t.get("name", ""))

        logger.info(f"Available tools: {len(tools)} (sorted alphabetically)")
        return tools

    async def execute_analysis(
        self, prompt: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute analysis using multi-step execution engine."""
        try:
            # Use execution engine for multi-step planning and execution
            messages = [ProviderMessage(content=prompt, role="user")]

            turn = await self.execution_engine.execute_turn(
                messages=messages,
                available_tools=context.get("mcp_tools", []),
                max_infer_iters=self.max_infer_iters,
            )

            # Extract results from the turn execution
            return {
                "turn_id": turn.turn_id,
                "ai_analysis": turn.output_message,
                "ai_response_format": self.settings.ai_response_format,
                "execution_steps": [step.to_dict() for step in turn.steps],
                "tool_calls": [
                    step.to_dict()
                    for step in turn.steps
                    if step.step_type.value == "tool_execution"
                ],
                # Data preservation: include accumulated knowledge for reports and notifications
                "accumulated_knowledge": (
                    turn.accumulated_knowledge.to_dict()
                    if turn.accumulated_knowledge
                    else None
                ),
                "model_used": self.settings.ai_model
                or get_default_model(self.settings.ai_provider),
                "ai_provider": self.settings.ai_provider or DEFAULT_AI_PROVIDER,
                "execution_time": (
                    (turn.completed_at - turn.started_at).total_seconds()
                    if turn.completed_at
                    else 0
                ),
                "execution_summary": turn.execution_summary,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Analysis execution failed: {e}")
            return {
                "error": f"Analysis execution failed: {e}",
                "status": "failed",
                "execution_engine_available": self.execution_engine is not None,
                "model_used": self.settings.ai_model
                or get_default_model(self.settings.ai_provider),
                "ai_provider": self.settings.ai_provider or DEFAULT_AI_PROVIDER,
                "ai_response_format": self.settings.ai_response_format,
            }

    # Shared report generation

    async def generate_report(
        self, analysis_result: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate formatted report."""
        # Get CI job name if available for report filename
        job_name = os.getenv("CI_JOB_NAME")

        report = {
            "report_id": self._generate_report_id(job_name=job_name),
            "generated_at": datetime.now().isoformat(),
            "agent_type": self.__class__.__name__,
            "project": context.get("project", {}).get("name", "Unknown"),
            "execution_time": (datetime.now() - self.start_time).total_seconds(),
            "analysis_result": analysis_result,
            "context_summary": self._create_context_summary(context),
        }

        return report

    def _normalize_agent_type(self, agent_type: str) -> str:
        """Normalize agent type for consistent file naming and URLs."""
        normalized = agent_type.lower()
        # Handle specific agent types first, then generic patterns
        if "branchreview" in normalized:
            return "branchreview"
        elif "mergerequest" in normalized or "mr" in normalized:
            return "mr"
        elif "cron" in normalized:
            return "cron"
        else:
            # Generic cleanup for other agent types
            return (
                normalized.replace("aiagent", "")
                .replace("agent", "")
                .replace("review", "")
                or "analysis"
            )

    def _generate_report_id(self, job_name: Optional[str] = None) -> str:
        """Generate unique report ID, optionally including a job name.

        Args:
            job_name: The name of the CI job, if available.

        Returns:
            String report ID in format: {agent_type}_{job_name}_{timestamp} or {agent_type}_{timestamp}
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent_type = self._normalize_agent_type(self.__class__.__name__)
        if job_name:
            return f"{agent_type}_{job_name}_{timestamp}"
        return f"{agent_type}_{timestamp}"

    def _create_context_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of analysis context for reporting."""
        return {
            "tools_available": len(context.get("mcp_tools", [])),
            "platform_available": context.get("platform_available", False),
            "project_name": context.get("project", {}).get("name", "Unknown"),
        }

    async def _save_report(self, report: Dict[str, Any]):
        """Save report to file in the pipeline working directory."""
        # Use report_id directly since it already contains normalized agent type
        filename = f"{report['report_id']}.json"

        # Determine the correct path for saving reports
        if os.getenv("CI_PROJECT_DIR"):
            # In CI: Agent runs from cicaddy/, but artifacts collected from parent directory
            report_path = os.path.join("..", filename)
            logger.info(f"CI mode: saving report to parent directory: {report_path}")
        else:
            # Local development: save to parent directory
            report_path = os.path.join("..", filename)
            logger.info(f"Local mode: saving report to parent directory: {report_path}")

        try:
            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)

            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(
                f"JSON report saved successfully to {os.path.abspath(report_path)}"
            )

            # Generate HTML report
            logger.info("Starting HTML report generation...")
            await self._save_html_report(report, report_path)
            logger.info("HTML report generation completed")

            # Note: Log file is saved at the end of analyze() to capture all logs

        except Exception as e:
            logger.error(f"Failed to save report to {report_path}: {e}")
            # Fallback: save to current directory
            await self._save_report_fallback(report, filename)

    async def _save_html_report(self, report: Dict[str, Any], json_report_path: str):
        """Generate and save HTML version of the report."""
        try:
            from cicaddy.reports.html_formatter import HTMLReportFormatter

            # Generate HTML filename from JSON path
            html_filename = json_report_path.replace(".json", ".html")

            # Create HTML formatter from settings for customizable headers/logo
            formatter = HTMLReportFormatter.from_settings(self.settings)
            html_content = formatter.format_report(
                report, json_report_path=json_report_path
            )

            # Save HTML file
            with open(html_filename, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(
                f"HTML report saved successfully to {os.path.abspath(html_filename)}"
            )

            # Log file sizes for comparison
            html_size = os.path.getsize(html_filename) / (1024 * 1024)
            json_size = os.path.getsize(json_report_path) / (1024 * 1024)
            logger.info(
                f"HTML report size: {html_size:.2f} MB, JSON report size: {json_size:.2f} MB"
            )

            # Store HTML path and content in report for later reference
            # html_content is used by email notification to send the full report
            report["html_report_path"] = os.path.abspath(html_filename)
            report["html_content"] = html_content

        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}", exc_info=True)
            # Don't raise - HTML report is optional, JSON report is primary

    async def _save_report_fallback(self, report: Dict[str, Any], filename: str):
        """Save report to fallback locations when primary save fails."""
        fallback_path = filename
        try:
            with open(fallback_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(
                f"Report saved to fallback location: {os.path.abspath(fallback_path)}"
            )

            # Try to save HTML in fallback location too
            # Note: Log file is saved at the end of analyze() to capture all logs
            await self._save_html_report(report, fallback_path)

        except Exception as e2:
            logger.error(f"Failed to save report to fallback location: {e2}")
            # Last resort: try saving to /tmp
            try:
                tmp_dir = tempfile.gettempdir()
                tmp_path = os.path.join(tmp_dir, filename)
                with open(tmp_path, "w") as f:
                    json.dump(report, f, indent=2)
                logger.warning(f"Report saved to temporary location: {tmp_path}")
            except Exception as e3:
                logger.error(f"Failed to save report anywhere: {e3}")
                raise

    async def _save_log_file(self, report_id: str, fallback: bool = False):
        """Save execution log for CI artifacts."""
        # Use report_id directly since it already contains normalized agent type
        log_filename = f"{report_id}.log"

        if fallback:
            log_path = log_filename
        else:
            log_path = os.path.join("..", log_filename)

        try:
            # Create execution log header
            execution_time = (datetime.now() - self.start_time).total_seconds()
            log_header = f"""Cicaddy Execution Log
===========================================
Report ID: {report_id}
Agent Type: {self.__class__.__name__}
Start Time: {self.start_time.isoformat()}
End Time: {datetime.now().isoformat()}
Execution Time: {execution_time:.2f} seconds
"""

            # Check if running locally (not in CI) and include captured logs
            is_ci = os.getenv("CI_PROJECT_DIR") is not None
            captured_logs = get_captured_logs()

            if is_ci or captured_logs is None:
                # In CI mode or no logs captured: just include header with reference
                log_content = (
                    log_header
                    + """
Log created for CI artifact collection.
For detailed logs, check the CI job output.
"""
                )
            else:
                # Local mode: include full captured logs
                log_content = (
                    log_header
                    + """
===========================================
Detailed Execution Logs
===========================================

"""
                    + captured_logs
                )

            with open(log_path, "w") as f:
                f.write(log_content)
            logger.info(f"Log file saved to {os.path.abspath(log_path)}")

        except Exception as e:
            logger.error(f"Failed to save log file: {e}")
            # Don't raise - log file is optional

    # Shared notification methods

    async def send_notifications(
        self, report: Dict[str, Any], analysis_result: Dict[str, Any]
    ):
        """Send notifications via Slack."""
        if self.slack_notifier:
            await self.slack_notifier.send_formatted_report(
                report,
                analysis_result,
                trends={},  # TODO: Implement trend tracking
            )

    # Failure handling methods

    async def _create_failure_context(self) -> Dict[str, Any]:
        """Create minimal context when analysis fails."""
        return {
            "project": {"name": "Analysis Failed", "id": "unknown"},
            "timestamp": self.start_time.isoformat(),
            "platform_available": False,
            "mcp_tools": [],
        }

    async def _create_fallback_report(
        self, analysis_result: Dict[str, Any], error: Exception
    ) -> Dict[str, Any]:
        """Create minimal fallback report when report generation fails."""
        try:
            fallback_report = {
                "report_id": f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "generated_at": datetime.now().isoformat(),
                "agent_type": self.__class__.__name__,
                "project": "Report Generation Failed",
                "execution_time": (datetime.now() - self.start_time).total_seconds(),
                "error": str(error),
                "status": "report_failed",
                "analysis_result": analysis_result,
            }
            await self._save_report(fallback_report)
            logger.info(f"Fallback report saved: {fallback_report['report_id']}")
            return fallback_report
        except Exception as e2:
            logger.error(f"Failed to save fallback report: {e2}", exc_info=True)
            return {
                "report_id": "emergency_fallback",
                "status": "critical_failure",
                "error": str(error),
            }

    # Cleanup

    async def cleanup(self):
        """Simple, non-interfering cleanup."""
        # Clean up MCP manager
        if self.mcp_manager:
            try:
                await self.mcp_manager.cleanup()
            except Exception as e:
                logger.debug(f"Expected MCP cleanup error: {e}")

        # Clean up AI provider
        if self.ai_provider:
            try:
                await self.ai_provider.shutdown()
            except Exception as e:
                logger.debug(f"Expected AI provider error: {e}")

        logger.info(f"{self.__class__.__name__} cleanup completed")

    # Helper methods for prompt processing

    def build_dspy_prompt(
        self, task_file: str, context: Dict[str, Any]
    ) -> Optional[str]:
        """Build prompt from DSPy YAML task definition.

        Args:
            task_file: Path to YAML task definition file
            context: Execution context with MCP tools, project info, etc.

        Returns:
            Structured prompt built from task definition, or None on failure
        """
        from cicaddy.dspy import PromptBuilder, TaskLoader, TaskLoadError

        try:
            loader = TaskLoader()
            task, resolved_inputs = loader.load(task_file)

            # Build prompt using the task definition and resolved inputs
            builder = PromptBuilder(task, resolved_inputs=resolved_inputs)
            prompt = builder.build(
                context=context,
                mcp_tools=context.get("mcp_tools", []),
            )

            # Propagate task output_format to settings so the report
            # formatter knows to extract and save standalone HTML/JSON
            if task.output_format and task.output_format != "markdown":
                self.settings.ai_response_format = task.output_format
                logger.info(
                    f"Set ai_response_format='{task.output_format}' from DSPy task"
                )

            logger.info(
                f"Built DSPy prompt from {task_file}",
                extra={
                    "task_name": task.name,
                    "task_type": task.type,
                    "prompt_length": len(prompt),
                },
            )

            return prompt

        except TaskLoadError as e:
            logger.error(f"Failed to load DSPy task file: {e}")
            return None

    # Abstract methods for subclass specialization

    @abstractmethod
    async def get_analysis_context(self) -> Dict[str, Any]:
        """
        Gather analysis context (subclass-specific).

        Returns:
            Dict containing context information needed for analysis.
            Must include 'project' key with project information.
        """
        pass

    @abstractmethod
    def build_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """
        Build analysis prompt based on context (subclass-specific).

        Args:
            context: Analysis context from get_analysis_context()

        Returns:
            String prompt for AI analysis
        """
        pass

    @abstractmethod
    def get_session_id(self) -> str:
        """
        Get unique session ID for this analysis session.

        Returns:
            String session identifier
        """
        pass
