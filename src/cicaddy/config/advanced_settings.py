"""Advanced configuration management with environment-specific settings and validation."""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)


class Environment(str, Enum):
    """Deployment environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class RetrySettings:
    """Retry configuration settings."""

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retry_on_connection_error: bool = True
    retry_on_timeout: bool = True
    retry_on_server_error: bool = False


@dataclass
class CircuitBreakerSettings:
    """Circuit breaker configuration settings."""

    enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    timeout: float = 10.0
    monitor_calls: int = 10
    success_rate_threshold: float = 50.0


@dataclass
class ConnectionPoolSettings:
    """Connection pool configuration settings."""

    enabled: bool = True
    pool_size: int = 5
    max_retries: int = 3
    load_balancing_strategy: str = (
        "weighted_random"  # round_robin, least_connections, weighted_random
    )
    health_check_interval: float = 30.0
    cleanup_interval: float = 300.0
    max_idle_time: float = 600.0


@dataclass
class CacheSettings:
    """Cache configuration settings."""

    enabled: bool = True
    backend: str = "memory"  # memory, redis (future)
    default_ttl: float = 300.0
    max_size: int = 1000
    max_memory: int = 100 * 1024 * 1024  # 100MB
    enable_compression: bool = False
    namespace: str = "mcp"


@dataclass
class TelemetrySettings:
    """Telemetry configuration settings for OpenTelemetry."""

    enabled: bool = True
    service_name: str = "gitlab-ai-agent"
    service_version: str = "1.0.0"
    environment: str = "development"

    # OTLP endpoint configuration (for GitLab observability)
    otlp_endpoint: Optional[str] = (
        None  # e.g., "https://gitlab.example.com/api/v4/projects/123/observability/traces"
    )
    otlp_headers: Dict[str, str] = field(
        default_factory=dict
    )  # e.g., {"Authorization": "Bearer <token>"}

    # Sampling configuration
    trace_sample_rate: float = 1.0  # Sample all traces for actions
    metrics_export_interval: float = 30.0  # Export metrics every 30 seconds

    # Resource attributes
    resource_attributes: Dict[str, str] = field(
        default_factory=lambda: {
            "service.name": "gitlab-ai-agent",
            "deployment.environment": "development",
        }
    )

    # Alert thresholds (kept for compatibility)
    alert_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "error_rate_threshold": 10.0,
            "response_time_threshold": 5.0,
            "cache_hit_rate_threshold": 50.0,
            "connection_failure_rate": 20.0,
        }
    )


@dataclass
class SecuritySettings:
    """Security configuration settings."""

    ssl_verify: bool = True
    timeout: float = 30.0
    max_redirects: int = 3
    allowed_hosts: Optional[List[str]] = None
    blocked_hosts: Optional[List[str]] = None
    enable_command_validation: bool = True
    allowed_commands: Optional[List[str]] = None
    blocked_commands: Optional[List[str]] = None


@dataclass
class LoggingSettings:
    """Logging configuration settings."""

    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_structured: bool = True
    enable_file_logging: bool = False
    log_file: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


class AdvancedMCPConfig(BaseModel):
    """Advanced configuration model with environment-specific settings."""

    # Environment configuration
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False

    # Feature flags
    enable_retry: bool = True
    enable_circuit_breaker: bool = True
    enable_connection_pool: bool = True
    enable_cache: bool = True
    enable_telemetry: bool = True

    # Component configurations
    retry: RetrySettings = Field(default_factory=RetrySettings)
    circuit_breaker: CircuitBreakerSettings = Field(
        default_factory=CircuitBreakerSettings
    )
    connection_pool: ConnectionPoolSettings = Field(
        default_factory=ConnectionPoolSettings
    )
    cache: CacheSettings = Field(default_factory=CacheSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # Performance tuning
    max_concurrent_requests: int = 100
    request_timeout: float = 30.0
    connection_timeout: float = 10.0
    read_timeout: float = 30.0
    write_timeout: float = 30.0

    # Resource limits
    max_memory_usage: int = 512 * 1024 * 1024  # 512MB
    max_cpu_usage: float = 80.0  # 80%
    max_open_files: int = 1000

    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment setting."""
        if isinstance(v, str):
            try:
                return Environment(v.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid environment: {v}. Must be one of {list(Environment)}"
                )
        return v

    @field_validator("max_concurrent_requests")
    @classmethod
    def validate_max_concurrent_requests(cls, v):
        """Validate max concurrent requests."""
        if v <= 0:
            raise ValueError("max_concurrent_requests must be positive")
        if v > 10000:
            raise ValueError("max_concurrent_requests should not exceed 10000")
        return v

    @field_validator(
        "request_timeout", "connection_timeout", "read_timeout", "write_timeout"
    )
    @classmethod
    def validate_timeouts(cls, v):
        """Validate timeout values."""
        if v <= 0:
            raise ValueError("Timeout values must be positive")
        if v > 300:  # 5 minutes
            raise ValueError("Timeout values should not exceed 300 seconds")
        return v

    def apply_environment_overrides(self):
        """Apply environment-specific configuration overrides."""
        if self.environment == Environment.PRODUCTION:
            # Production optimizations
            self.debug = False
            self.logging.level = LogLevel.WARNING
            self.connection_pool.pool_size = 10
            self.cache.max_size = 5000
            self.telemetry.metrics_export_interval = 30.0
            self.security.ssl_verify = True

        elif self.environment == Environment.DEVELOPMENT:
            # Development settings
            self.debug = True
            self.logging.level = LogLevel.DEBUG
            self.connection_pool.pool_size = 2
            self.cache.max_size = 100
            self.telemetry.metrics_export_interval = 60.0
            self.security.ssl_verify = False

        elif self.environment == Environment.TESTING:
            # Testing settings
            self.debug = True
            self.logging.level = LogLevel.INFO
            self.connection_pool.enabled = False
            self.cache.enabled = False
            self.telemetry.enabled = False
            self.circuit_breaker.enabled = False

        elif self.environment == Environment.STAGING:
            # Staging settings (similar to production but with more logging)
            self.debug = False
            self.logging.level = LogLevel.INFO
            self.connection_pool.pool_size = 5
            self.cache.max_size = 1000
            self.telemetry.metrics_export_interval = 45.0

    def get_feature_flags(self) -> Dict[str, bool]:
        """Get all feature flags."""
        return {
            "retry": self.enable_retry,
            "circuit_breaker": self.enable_circuit_breaker,
            "connection_pool": self.enable_connection_pool,
            "cache": self.enable_cache,
            "telemetry": self.enable_telemetry,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.model_dump()

    def to_yaml(self) -> str:
        """Convert to YAML representation."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)


class ConfigurationManager:
    """Manager for advanced configuration with multiple sources."""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path.cwd() / "config"
        self._config_cache: Dict[str, AdvancedMCPConfig] = {}
        self._file_watchers: Dict[Path, float] = {}

    def load_config(
        self,
        config_name: str = "default",
        environment: Optional[Environment] = None,
        config_file: Optional[Path] = None,
    ) -> AdvancedMCPConfig:
        """
        Load configuration from multiple sources with precedence.

        Precedence order (highest to lowest):
        1. Environment variables
        2. Command line arguments (if provided)
        3. Environment-specific config file
        4. Default config file
        5. Built-in defaults

        Args:
            config_name: Name of the configuration
            environment: Target environment
            config_file: Specific config file to load

        Returns:
            Loaded configuration
        """
        cache_key = f"{config_name}:{environment}:{config_file}"

        # Check cache first
        if cache_key in self._config_cache:
            config_path = config_file or self._get_config_file(config_name, environment)
            if config_path and self._is_file_modified(config_path):
                # File was modified, reload
                del self._config_cache[cache_key]
            else:
                return self._config_cache[cache_key]

        logger.info(
            f"Loading configuration: {config_name} for environment: {environment}"
        )

        # Start with defaults
        config = AdvancedMCPConfig()

        # Load from file if specified
        config_path = config_file or self._get_config_file(config_name, environment)
        if config_path and config_path.exists():
            file_config = self._load_from_file(config_path)
            if file_config:
                config = self._merge_configs(config, file_config)

        # Override with environment variables
        env_config = self._load_from_environment()
        if env_config:
            config = self._merge_configs(config, env_config)

        # Set environment and apply overrides
        if environment:
            config.environment = environment
        config.apply_environment_overrides()

        # Cache the config
        self._config_cache[cache_key] = config
        if config_path:
            self._file_watchers[config_path] = config_path.stat().st_mtime

        logger.info(
            f"Configuration loaded successfully for {config_name}:{environment}"
        )
        return config

    def save_config(self, config: AdvancedMCPConfig, config_file: Path):
        """Save configuration to file."""
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, "w") as f:
                yaml.dump(
                    config.to_dict(), f, default_flow_style=False, sort_keys=False
                )
            logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_file}: {e}")
            raise

    def _get_config_file(
        self, config_name: str, environment: Optional[Environment]
    ) -> Optional[Path]:
        """Get configuration file path."""
        # Try environment-specific file first
        if environment:
            env_file = self.config_dir / f"{config_name}.{environment.value}.yaml"
            if env_file.exists():
                return env_file

        # Try default file
        default_file = self.config_dir / f"{config_name}.yaml"
        if default_file.exists():
            return default_file

        return None

    def _load_from_file(self, config_file: Path) -> Optional[Dict[str, Any]]:
        """Load configuration from YAML file."""
        try:
            with open(config_file, "r") as f:
                data = yaml.safe_load(f)
            logger.debug(f"Loaded configuration from {config_file}")
            return data
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")
            return None

    def _load_from_environment(self) -> Optional[Dict[str, Any]]:
        """Load configuration from environment variables."""
        env_config: Dict[str, Any] = {}

        # Map environment variables to config paths
        env_mappings = {
            "MCP_ENVIRONMENT": ["environment"],
            "MCP_DEBUG": ["debug"],
            "MCP_LOG_LEVEL": ["logging", "level"],
            "MCP_POOL_SIZE": ["connection_pool", "pool_size"],
            "MCP_CACHE_TTL": ["cache", "default_ttl"],
            "MCP_REQUEST_TIMEOUT": ["request_timeout"],
            "MCP_MAX_RETRIES": ["retry", "max_retries"],
            "MCP_CIRCUIT_BREAKER_ENABLED": ["enable_circuit_breaker"],
            "MCP_CACHE_ENABLED": ["enable_cache"],
            "MCP_METRICS_ENABLED": ["enable_metrics"],
            "MCP_SSL_VERIFY": ["security", "ssl_verify"],
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                converted_value = self._convert_env_value(value)
                self._set_nested_value(env_config, config_path, converted_value)

        return env_config if env_config else None

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        elif value.lower() in ("false", "no", "0", "off"):
            return False

        # Number conversion
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _set_nested_value(self, config: Dict[str, Any], path: List[str], value: Any):
        """Set nested value in configuration dictionary."""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _merge_configs(
        self, base: AdvancedMCPConfig, override: Dict[str, Any]
    ) -> AdvancedMCPConfig:
        """Merge configuration override into base configuration."""
        try:
            # Convert base to dict, merge, and create new config
            base_dict = base.to_dict()
            merged_dict = self._deep_merge(base_dict, override)
            return AdvancedMCPConfig(**merged_dict)
        except Exception as e:
            logger.warning(f"Failed to merge configuration: {e}, using base config")
            return base

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _is_file_modified(self, file_path: Path) -> bool:
        """Check if file was modified since last load."""
        if file_path not in self._file_watchers:
            return True

        try:
            current_mtime = file_path.stat().st_mtime
            return current_mtime > self._file_watchers[file_path]
        except OSError:
            return True

    def clear_cache(self):
        """Clear configuration cache."""
        self._config_cache.clear()
        self._file_watchers.clear()
        logger.info("Configuration cache cleared")

    def get_cached_configs(self) -> List[str]:
        """Get list of cached configuration keys."""
        return list(self._config_cache.keys())

    def validate_config(self, config: AdvancedMCPConfig) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Check resource limits
        if config.max_memory_usage < 64 * 1024 * 1024:  # 64MB minimum
            issues.append("max_memory_usage is too low (minimum 64MB)")

        if config.max_cpu_usage > 100.0:
            issues.append("max_cpu_usage cannot exceed 100%")

        # Check pool settings
        if config.connection_pool.enabled and config.connection_pool.pool_size <= 0:
            issues.append(
                "connection_pool.pool_size must be positive when pool is enabled"
            )

        # Check cache settings
        if config.cache.enabled and config.cache.max_size <= 0:
            issues.append("cache.max_size must be positive when cache is enabled")

        # Check timeout consistency
        if config.connection_timeout > config.request_timeout:
            issues.append("connection_timeout should not exceed request_timeout")

        return issues


# Global configuration manager
config_manager = ConfigurationManager()

# Default configuration instance
default_config = config_manager.load_config()
