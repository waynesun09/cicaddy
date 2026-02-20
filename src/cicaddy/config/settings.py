"""Configuration management for Cicaddy."""

import json
import logging
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Use standard logging for settings module to avoid circular imports
# This logger will be reconfigured by setup_logging() in CLI commands
logger = logging.getLogger(__name__)


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    model_config = {"populate_by_name": True}  # Enable parsing by field alias names

    name: str
    protocol: str = Field(..., pattern="^(sse|http|stdio|websocket)$")

    # Remote server fields (sse/http/websocket)
    endpoint: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)

    # Local server fields (stdio)
    command: Optional[str] = None  # e.g., "uv", "node", "python", "./binary"
    args: Optional[List[str]] = None  # e.g., ["run", "server.py"] or ["--port", "8080"]
    working_directory: Optional[str] = None  # Absolute path to server directory
    env_vars: Optional[Dict[str, str]] = Field(
        None, alias="env"
    )  # Additional environment variables (JSON field: "env")

    # Common fields
    tools: Optional[List[str]] = None
    timeout: int = 30
    retry_count: int = 3

    # Retry configuration for connection resilience
    retry_delay: float = 1.0  # Initial delay between retries in seconds
    retry_max_delay: float = 60.0  # Maximum delay between retries in seconds
    max_retry_time_seconds: float = (
        300.0  # Maximum total time for all retry attempts (0=disabled)
    )
    retry_backoff_factor: float = 2.0  # Exponential backoff multiplier
    retry_jitter: bool = True  # Add random jitter to retry delays

    # Connection health monitoring
    heartbeat_interval: Optional[float] = (
        None  # Heartbeat/ping interval in seconds (None=disabled)
    )
    connection_timeout: float = 10.0  # Initial connection timeout in seconds
    read_timeout: Optional[float] = None  # Read timeout in seconds (None=use timeout)
    idle_timeout: Optional[float] = (
        None  # Idle timeout for SSE/long-running operations (None=use read_timeout)
    )

    # Error handling
    retry_on_connection_error: bool = True  # Retry on connection failures
    retry_on_timeout: bool = True  # Retry on timeout errors
    retry_on_server_error: bool = True  # Retry on 5xx HTTP errors (enabled by default)

    def model_post_init(self, __context) -> None:
        """Validate configuration after initialization."""
        if self.protocol in ["sse", "http", "websocket"] and not self.endpoint:
            raise ValueError(f"endpoint is required for {self.protocol} protocol")
        elif self.protocol == "stdio" and not self.command:
            raise ValueError("command is required for stdio protocol")


# Sensitive environment variable names - centralized for DRY principle
# Used by Settings.__repr__ for masking and CLI arg_mapping for exclusion
SENSITIVE_ENV_VAR_NAMES: frozenset = frozenset(
    {
        "GITLAB_TOKEN",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_KEY",
        "OLLAMA_API_KEY",
        "SLACK_WEBHOOK_URL",
        "SLACK_WEBHOOK_URLS",
        "MCP_SERVERS_CONFIG",  # May contain API keys in headers
    }
)

# Corresponding field names (lowercase with underscores)
_SENSITIVE_FIELD_NAMES: frozenset = frozenset(
    {name.lower() for name in SENSITIVE_ENV_VAR_NAMES}
)


class CoreSettings(BaseSettings):
    """Platform-agnostic core settings for the AI agent.

    Contains all settings that are not specific to any particular CI/CD platform.
    Platform-specific settings (e.g., GitLab) should extend this class.
    """

    model_config = SettingsConfigDict(extra="ignore")

    # Sensitive fields that should be masked in logs/tracebacks
    _SENSITIVE_FIELDS: frozenset = _SENSITIVE_FIELD_NAMES

    def __repr__(self) -> str:
        """Return a representation with sensitive fields masked."""
        field_strs = []
        for field_name in self.model_fields:
            value = getattr(self, field_name, None)
            if field_name in self._SENSITIVE_FIELDS:
                if value:
                    # Mask sensitive values, showing only length
                    masked = f"<{len(str(value))} chars>"
                else:
                    masked = "None"
                field_strs.append(f"{field_name}={masked!r}")
            else:
                field_strs.append(f"{field_name}={value!r}")
        class_name = self.__class__.__name__
        return f"{class_name}({', '.join(field_strs)})"

    def __str__(self) -> str:
        """Return a string representation with sensitive fields masked."""
        return self.__repr__()

    # AI provider configuration
    ai_provider: str = Field(
        default="gemini", validation_alias="AI_PROVIDER"
    )  # gemini, openai, claude, azure, ollama
    ai_model: str = Field(default="gemini-3-flash-preview", validation_alias="AI_MODEL")
    ai_response_format: str = Field(
        default="markdown",
        validation_alias="AI_RESPONSE_FORMAT",
        description="Preferred AI response format for reports (markdown, html, json)",
    )

    # Token limit management
    dynamic_token_limits: bool = Field(
        default=True, validation_alias="DYNAMIC_TOKEN_LIMITS"
    )
    token_limit_cache_ttl: int = Field(
        default=7200, validation_alias="TOKEN_LIMIT_CACHE_TTL"
    )  # 2 hours
    ai_temperature: str = Field(default="0.0", validation_alias="AI_TEMPERATURE")

    # AI API keys
    gemini_api_key: Optional[str] = Field(None, validation_alias="GEMINI_API_KEY")
    openai_api_key: Optional[str] = Field(None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, validation_alias="ANTHROPIC_API_KEY")
    azure_openai_key: Optional[str] = Field(None, validation_alias="AZURE_OPENAI_KEY")
    azure_endpoint: Optional[str] = Field(None, validation_alias="AZURE_ENDPOINT")

    # Ollama configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL"
    )
    ollama_api_key: Optional[str] = Field(None, validation_alias="OLLAMA_API_KEY")

    # MCP server configuration (YAML list)
    mcp_servers_config: str = Field("[]", validation_alias="MCP_SERVERS_CONFIG")

    # Slack configuration
    slack_webhook_url: Optional[str] = Field(
        None, validation_alias="SLACK_WEBHOOK_URL"
    )  # Backward compatibility
    slack_webhook_urls: str = Field(
        "[]", validation_alias="SLACK_WEBHOOK_URLS"
    )  # JSON list of webhook URLs

    # Email notification configuration
    email_enabled: bool = Field(
        False, validation_alias="EMAIL_ENABLED"
    )  # Enable email notifications
    email_recipients: Optional[str] = Field(
        None, validation_alias="EMAIL_RECIPIENTS"
    )  # Comma-separated or JSON list of recipients
    sender_email: Optional[str] = Field(
        None, validation_alias="SENDER_EMAIL"
    )  # Sender email address
    use_gmail_api: bool = Field(
        False, validation_alias="USE_GMAIL_API"
    )  # Use Gmail API (True) or SMTP (False)

    # Agent configuration
    agent_tasks: str = Field("code_review", validation_alias="AGENT_TASKS")
    analysis_focus: Optional[str] = Field(
        None, validation_alias="ANALYSIS_FOCUS"
    )  # security, performance, maintainability
    review_prompt: Optional[str] = Field(None, validation_alias="AI_TASK_PROMPT")
    task_file: Optional[str] = Field(
        None,
        validation_alias="AI_TASK_FILE",
        description="Path to YAML task definition file (DSPy mode). Takes precedence over AI_TASK_PROMPT.",
    )

    # Git configuration
    git_diff_context_lines: int = Field(10, validation_alias="GIT_DIFF_CONTEXT_LINES")
    git_working_directory: Optional[str] = Field(
        None, validation_alias="GIT_WORKING_DIRECTORY"
    )

    # Merge request / pull request fields (platform-agnostic)
    merge_request_iid: Optional[str] = Field(
        None, validation_alias="CI_MERGE_REQUEST_IID"
    )
    merge_request_source_branch: Optional[str] = Field(
        None, validation_alias="CI_MERGE_REQUEST_SOURCE_BRANCH_NAME"
    )
    merge_request_target_branch: Optional[str] = Field(
        None, validation_alias="CI_MERGE_REQUEST_TARGET_BRANCH_NAME"
    )

    # Logging configuration
    log_level: str = Field(
        "INFO",
        validation_alias="LOG_LEVEL",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
    )
    json_logs: bool = Field(False, validation_alias="JSON_LOGS")

    # Report configuration
    enable_report_chart: bool = Field(
        False,
        validation_alias="ENABLE_REPORT_CHART",
        description="Enable chart image generation in HTML reports using Playwright",
    )

    # SSL/TLS configuration
    ssl_verify: bool = Field(True, validation_alias="SSL_VERIFY")

    # Cron-specific configuration
    cron_task_type: str = Field("scheduled_analysis", validation_alias="CRON_TASK_TYPE")
    cron_scope: str = Field("recent_changes", validation_alias="CRON_SCOPE")
    cron_schedule_name: Optional[str] = Field(
        None, validation_alias="CRON_SCHEDULE_NAME"
    )

    # Local tools configuration
    enable_local_tools: bool = Field(
        False,
        validation_alias="ENABLE_LOCAL_TOOLS",
        description="Enable built-in local file tools (glob_files, read_file) for AI agent.",
    )
    local_tools_working_dir: Optional[str] = Field(
        None,
        validation_alias="LOCAL_TOOLS_WORKING_DIR",
        description="Working directory for local file tools. Defaults to GIT_WORKING_DIRECTORY or current directory.",
    )

    # Execution configuration
    max_infer_iters: int = Field(
        10,
        validation_alias="MAX_INFER_ITERS",
        description="Maximum AI planning iterations for multi-step analysis (minimum: 1)",
    )

    # Execution time limits (seconds)
    max_execution_time: int = Field(
        600,
        validation_alias="MAX_EXECUTION_TIME",
        ge=60,  # Minimum 1 minute
        le=7200,  # Maximum 2 hours
        description="Maximum total execution time in seconds (60-7200)",
    )

    # Token budget management
    context_safety_factor: float = Field(
        0.85,  # Changed from 0.7 to allow more token usage with recovery support
        validation_alias="CONTEXT_SAFETY_FACTOR",
        ge=0.5,  # Minimum 50% safety margin
        le=0.97,  # Changed from 0.95 to allow higher utilization with recovery
        description="Token budget safety factor (0.5-0.97). Higher values allow more tokens. Recovery mechanism handles overflow.",
    )

    # Recovery mechanism configuration
    max_tokens_recovery_limit: int = Field(
        3,
        validation_alias="MAX_TOKENS_RECOVERY_LIMIT",
        ge=1,
        le=10,
        description="Maximum number of fresh context recovery attempts when token limit is exceeded (1-10).",
    )
    max_iterations_recovery_limit: int = Field(
        2,
        validation_alias="MAX_ITERATIONS_RECOVERY_LIMIT",
        ge=1,
        le=5,
        description="Maximum number of fresh context recovery attempts when iteration limit is exceeded (1-5).",
    )
    min_tokens_for_iteration_recovery: int = Field(
        10000,
        validation_alias="MIN_TOKENS_FOR_ITERATION_RECOVERY",
        ge=1000,
        le=100000,
        description="Minimum remaining token budget required to attempt iteration recovery (1000-100000).",
    )
    recovery_content_truncation_length: int = Field(
        10000,
        validation_alias="RECOVERY_CONTENT_TRUNCATION_LENGTH",
        ge=500,
        le=100000,
        description="Maximum characters to keep per tool result in recovery prompts (500-100000).",
    )

    # HTML Report customization
    html_report_header: str = Field(
        default="AI Agent Analysis Report",
        validation_alias="HTML_REPORT_HEADER",
        description="Main header text for HTML reports.",
    )
    html_report_subheader: Optional[str] = Field(
        default=None,
        validation_alias="HTML_REPORT_SUBHEADER",
        description="Sub-header text for HTML reports. If not set (or set to None), the HTMLReportFormatter's default sub-header will be used.",
    )
    html_report_logo_url: Optional[str] = Field(
        default=None,
        validation_alias="HTML_REPORT_LOGO_URL",
        description="URL or path to logo image displayed before report headers.",
    )
    html_report_logo_height: int = Field(
        default=48,
        validation_alias="HTML_REPORT_LOGO_HEIGHT",
        ge=16,
        le=200,
        description="Height of the logo image in pixels (16-200).",
    )
    html_report_emoji: str = Field(
        default="🤖",
        validation_alias="HTML_REPORT_EMOJI",
        description="Emoji prefix for HTML report header. Set to empty string to disable. Ignored when logo is present.",
    )

    @field_validator(
        "use_gmail_api",
        "email_enabled",
        "json_logs",
        "enable_report_chart",
        "ssl_verify",
        "dynamic_token_limits",
        "enable_local_tools",
        mode="before",
    )
    @classmethod
    def parse_bool_from_env(cls, v: Any) -> bool:
        """Handle empty strings and various boolean representations from env vars."""
        if v is None or v == "":
            return False  # Default to False for empty/unset values
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower().strip() in ("true", "1", "yes")
        return bool(v)

    def _log_safe_mcp_info(self, mcp_configs: List[MCPServerConfig]):
        """Log safe MCP server information (exclude sensitive data like headers)."""
        for config in mcp_configs:
            safe_info = {
                "name": config.name,
                "protocol": config.protocol,
                "endpoint": config.endpoint,
                "tools": (
                    config.tools[:3]
                    if config.tools and len(config.tools) > 3
                    else config.tools
                ),  # Limit tools for readability
                "tool_count": len(config.tools) if config.tools else 0,
                "timeout": config.timeout,
                "connection_timeout": config.connection_timeout,
                "read_timeout": config.read_timeout,
                "idle_timeout": config.idle_timeout,
            }

            # Add environment variable information for stdio servers (for debugging)
            if config.protocol == "stdio" and config.env_vars:
                env_info = {}
                for key, value in config.env_vars.items():
                    if (
                        key.lower() in ["api_key", "token", "password"]
                        or "key" in key.lower()
                    ):
                        # Mask sensitive values
                        env_info[key] = f"<{len(value)} chars>" if value else "<empty>"
                    else:
                        # Show non-sensitive values
                        env_info[key] = value
                safe_info["env_vars"] = env_info
                logger.info(f"MCP Server: {safe_info}")
            else:
                logger.info(f"MCP Server: {safe_info}")

    def get_mcp_servers(self) -> List[MCPServerConfig]:
        """Parse MCP servers configuration from YAML."""
        logger.debug(
            f"Parsing MCP_SERVERS_CONFIG, type: {type(self.mcp_servers_config)}, length: {len(self.mcp_servers_config)}"
        )

        # Check for empty or whitespace-only config
        if not self.mcp_servers_config or not self.mcp_servers_config.strip():
            logger.warning("MCP_SERVERS_CONFIG is empty or whitespace-only")
            return []

        try:
            # Try YAML first (preferred format)
            servers_data = yaml.safe_load(self.mcp_servers_config)
            logger.debug(f"YAML parsing result type: {type(servers_data)}")

            if servers_data is None:
                logger.warning("YAML parsing returned None - config may be empty")
                return []
            if not isinstance(servers_data, list):
                logger.error(f"Expected list from YAML, got {type(servers_data)}")
                return []

            logger.info(
                f"Successfully parsed {len(servers_data)} MCP server configurations"
            )
            mcp_configs = [MCPServerConfig(**server) for server in servers_data]
            self._log_safe_mcp_info(mcp_configs)
            return mcp_configs
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            # Fallback to JSON for backward compatibility
            try:
                servers_data = json.loads(self.mcp_servers_config)
                logger.info(
                    f"JSON fallback successful with {len(servers_data)} servers"
                )
                mcp_configs = [MCPServerConfig(**server) for server in servers_data]
                self._log_safe_mcp_info(mcp_configs)
                return mcp_configs
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"JSON parsing also failed: {e}")
                return []
        except Exception as e:
            logger.error(f"Error creating MCP server configs: {e}")
            # Return empty list if server config creation fails
            return []

    def get_slack_webhook_urls(self) -> List[str]:
        """Parse Slack webhook URLs with backward compatibility support."""
        webhook_urls = []

        # First, try the new JSON list format
        if self.slack_webhook_urls and self.slack_webhook_urls.strip() != "[]":
            try:
                # Try YAML first (more flexible)
                urls_data = yaml.safe_load(self.slack_webhook_urls)
                if isinstance(urls_data, list):
                    webhook_urls.extend(
                        [url for url in urls_data if url and isinstance(url, str)]
                    )
                    logger.info(
                        f"Successfully parsed {len(webhook_urls)} Slack webhook URLs from JSON/YAML list"
                    )
                elif isinstance(urls_data, str):
                    # Single string in YAML format
                    webhook_urls.append(urls_data)
                    logger.info("Parsed single Slack webhook URL from YAML string")
            except yaml.YAMLError:
                # Fallback to JSON parsing
                try:
                    urls_data = json.loads(self.slack_webhook_urls)
                    if isinstance(urls_data, list):
                        webhook_urls.extend(
                            [url for url in urls_data if url and isinstance(url, str)]
                        )
                        logger.info(
                            f"Successfully parsed {len(webhook_urls)} Slack webhook URLs from JSON list"
                        )
                    elif isinstance(urls_data, str):
                        webhook_urls.append(urls_data)
                        logger.info("Parsed single Slack webhook URL from JSON string")
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"Failed to parse SLACK_WEBHOOK_URLS: {e}")

        # Backward compatibility: check single URL field
        if not webhook_urls and self.slack_webhook_url:
            webhook_urls.append(self.slack_webhook_url)
            logger.info("Using backward compatibility: single SLACK_WEBHOOK_URL")

        # Validate URLs
        valid_urls = []
        for url in webhook_urls:
            if self._is_valid_slack_webhook_url(url):
                valid_urls.append(url)
            else:
                logger.warning(f"Invalid Slack webhook URL format: {url[:50]}...")

        logger.debug(f"Final Slack webhook URLs count: {len(valid_urls)}")
        return valid_urls

    def get_email_config(self) -> Optional[Dict[str, Any]]:
        """Get email configuration if enabled.

        Returns:
            Dict with email config if enabled, None otherwise.
            Keys: recipients (list), sender_email (str), use_gmail_api (bool)
        """
        if not self.email_enabled:
            logger.debug("Email notifications disabled (EMAIL_ENABLED=false)")
            return None

        if not self.email_recipients:
            logger.warning("EMAIL_ENABLED=true but EMAIL_RECIPIENTS not set")
            return None

        # Parse recipients (comma-separated or JSON list)
        recipients = []
        if self.email_recipients:
            try:
                # Try JSON list first
                parsed = json.loads(self.email_recipients)
                if isinstance(parsed, list):
                    recipients = [r.strip() for r in parsed if r and isinstance(r, str)]
                elif isinstance(parsed, str):
                    recipients = [parsed.strip()]
            except (json.JSONDecodeError, TypeError):
                # Fallback to comma-separated string
                recipients = [
                    r.strip() for r in self.email_recipients.split(",") if r.strip()
                ]

        if not recipients:
            logger.warning("No valid email recipients found")
            return None

        config = {
            "recipients": recipients,
            "sender_email": self.sender_email,
            "use_gmail_api": self.use_gmail_api,
        }

        logger.info(
            f"Email notifications enabled: {len(recipients)} recipient(s), "
            f"method={'Gmail API' if self.use_gmail_api else 'SMTP'}"
        )
        return config

    def _is_valid_slack_webhook_url(self, url: str) -> bool:
        """
        Validate Slack webhook URL format with strict security checks.

        Only allows genuine Slack webhook URLs to prevent malicious redirects.
        """
        if not url or not isinstance(url, str):
            return False

        return url.startswith("https://hooks.slack.com/services/")

    def get_enabled_tasks(self) -> List[str]:
        """Get list of enabled agent tasks."""
        return [task.strip() for task in self.agent_tasks.split(",")]

    def get_ai_temperature(self) -> float:
        """Get AI temperature as float with safe conversion and validation."""
        try:
            temp = float(self.ai_temperature)
            # Clamp temperature to valid range [0.0, 2.0]
            return max(0.0, min(2.0, temp))
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Invalid AI_TEMPERATURE '{self.ai_temperature}': {e}. Using default 0.0"
            )
            return 0.0

    @field_validator("ai_response_format", mode="before")
    @classmethod
    def validate_ai_response_format(cls, value: Any) -> str:
        """Normalize and validate AI response format setting."""
        if value is None:
            return "markdown"

        allowed = {"markdown", "html", "json"}
        normalized = str(value).strip().lower()
        if normalized not in allowed:
            logger.warning(
                f"Invalid AI_RESPONSE_FORMAT '{value}'. "
                "Falling back to 'markdown'. Allowed values: markdown, html, json."
            )
            return "markdown"
        return normalized


# Alias for backward compatibility and for platform wrappers that import Settings
Settings = CoreSettings


def load_core_settings() -> CoreSettings:
    """Load platform-agnostic core settings from environment variables.

    Returns:
        CoreSettings instance with all platform-agnostic fields populated.
    """
    import os

    # Handle MCP_SERVERS_CONFIG - default to empty array if missing
    current_mcp_config = os.getenv("MCP_SERVERS_CONFIG")
    if not current_mcp_config:
        os.environ["MCP_SERVERS_CONFIG"] = "[]"

    return CoreSettings()


def load_settings() -> Settings:
    """Load settings from environment variables.

    This is the main entry point for loading configuration. Platform-specific
    packages should override this function to add
    platform-specific defaults and env var handling.
    """
    return load_core_settings()
