"""OpenTelemetry instrumentation for MCP client operations in GitLab Actions."""

import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)

# OpenTelemetry imports with fallback
try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    logger.warning("OpenTelemetry not available. Telemetry features will be disabled.")
    OPENTELEMETRY_AVAILABLE = False

    # Create mock classes for when OpenTelemetry is not available
    class MockTracer:
        def start_as_current_span(self, name, **kwargs):
            return MockSpan()

        def start_span(self, name, **kwargs):
            return MockSpan()

    class MockSpan:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def set_attribute(self, key, value):
            pass

        def set_status(self, status):
            pass

        def record_exception(self, exception):
            pass

    class MockMeter:
        def create_counter(self, name, **kwargs):
            return MockInstrument()

        def create_histogram(self, name, **kwargs):
            return MockInstrument()

        def create_gauge(self, name, **kwargs):
            return MockInstrument()

    class MockInstrument:
        def add(self, amount, attributes=None):
            pass

        def record(self, amount, attributes=None):
            pass

        def set(self, amount, attributes=None):
            pass


class GitLabActionTelemetry:
    """
    OpenTelemetry implementation optimized for GitLab Actions.

    Designed for on-demand execution rather than long-running services.
    Exports telemetry data to GitLab observability or OTLP endpoints.
    """

    def __init__(self, config=None):
        """Initialize telemetry for GitLab Action execution."""
        self.config = config
        self.enabled = OPENTELEMETRY_AVAILABLE and (config and config.enabled)

        if not self.enabled:
            logger.info("Telemetry disabled or OpenTelemetry not available")
            self._setup_mock_instruments()
            return

        # Set up OpenTelemetry for GitLab Action
        self._setup_resource()
        self._setup_tracing()
        self._setup_metrics()

        logger.info(
            f"OpenTelemetry initialized for GitLab Action: {self.config.service_name}"
        )

    def _setup_resource(self):
        """Set up resource with CI/CD context."""
        resource_attributes = {
            "service.name": self.config.service_name,
            "service.version": self.config.service_version,
            "deployment.environment": self.config.environment,
        }

        # Add CI/CD environment variables if available
        gitlab_vars = {
            "gitlab.project.id": os.getenv("CI_PROJECT_ID"),
            "gitlab.project.name": os.getenv("CI_PROJECT_NAME"),
            "gitlab.pipeline.id": os.getenv("CI_PIPELINE_ID"),
            "gitlab.job.id": os.getenv("CI_JOB_ID"),
            "gitlab.job.name": os.getenv("CI_JOB_NAME"),
            "gitlab.commit.sha": os.getenv("CI_COMMIT_SHA"),
            "gitlab.commit.ref": os.getenv("CI_COMMIT_REF_NAME"),
            "gitlab.runner.id": os.getenv("CI_RUNNER_ID"),
        }

        # Add non-null GitLab variables
        resource_attributes.update({k: v for k, v in gitlab_vars.items() if v})

        # Add custom resource attributes from config
        resource_attributes.update(self.config.resource_attributes)

        self.resource = Resource.create(resource_attributes)

    def _setup_tracing(self):
        """Set up OpenTelemetry tracing."""
        if not OPENTELEMETRY_AVAILABLE:
            return

        # Create tracer provider
        trace.set_tracer_provider(
            TracerProvider(
                resource=self.resource,
                sampler=trace.sampling.TraceIdRatioBased(self.config.trace_sample_rate),
            )
        )

        # Set up OTLP exporter if endpoint is configured
        if self.config.otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=f"{self.config.otlp_endpoint}/v1/traces",
                headers=self.config.otlp_headers,
            )
            span_processor = BatchSpanProcessor(otlp_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)

        self.tracer = trace.get_tracer(__name__)

    def _setup_metrics(self):
        """Set up OpenTelemetry metrics."""
        if not OPENTELEMETRY_AVAILABLE:
            return

        # Set up metric reader
        metric_readers = []

        if self.config.otlp_endpoint:
            otlp_metric_exporter = OTLPMetricExporter(
                endpoint=f"{self.config.otlp_endpoint}/v1/metrics",
                headers=self.config.otlp_headers,
            )
            metric_reader = PeriodicExportingMetricReader(
                exporter=otlp_metric_exporter,
                export_interval_millis=int(self.config.metrics_export_interval * 1000),
            )
            metric_readers.append(metric_reader)

        # Create meter provider
        metrics.set_meter_provider(
            MeterProvider(
                resource=self.resource,
                metric_readers=metric_readers,
            )
        )

        self.meter = metrics.get_meter(__name__)

        # Create instruments
        self._create_instruments()

    def _create_instruments(self):
        """Create OpenTelemetry instruments for MCP operations."""
        if not OPENTELEMETRY_AVAILABLE:
            return

        # Request metrics
        self.request_counter = self.meter.create_counter(
            name="mcp_requests_total",
            description="Total number of MCP requests",
            unit="1",
        )

        self.request_duration = self.meter.create_histogram(
            name="mcp_request_duration_seconds",
            description="MCP request duration in seconds",
            unit="s",
        )

        self.error_counter = self.meter.create_counter(
            name="mcp_errors_total", description="Total number of MCP errors", unit="1"
        )

        # Connection metrics
        self.connection_counter = self.meter.create_counter(
            name="mcp_connections_total",
            description="Total number of MCP connections created",
            unit="1",
        )

        # Cache metrics
        self.cache_hits = self.meter.create_counter(
            name="mcp_cache_hits_total",
            description="Total number of cache hits",
            unit="1",
        )

        self.cache_misses = self.meter.create_counter(
            name="mcp_cache_misses_total",
            description="Total number of cache misses",
            unit="1",
        )

        # Circuit breaker metrics
        self.circuit_breaker_trips = self.meter.create_counter(
            name="mcp_circuit_breaker_trips_total",
            description="Total number of circuit breaker trips",
            unit="1",
        )

    def _setup_mock_instruments(self):
        """Set up mock instruments when OpenTelemetry is not available."""
        self.tracer = MockTracer()
        self.meter = MockMeter()

        # Create mock instruments
        self.request_counter = MockInstrument()
        self.request_duration = MockInstrument()
        self.error_counter = MockInstrument()
        self.connection_counter = MockInstrument()
        self.cache_hits = MockInstrument()
        self.cache_misses = MockInstrument()
        self.circuit_breaker_trips = MockInstrument()

    @contextmanager
    def trace_operation(
        self, operation_name: str, attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for tracing MCP operations.

        Args:
            operation_name: Name of the operation being traced
            attributes: Additional attributes to add to the span
        """
        if not self.enabled:
            yield None
            return

        with self.tracer.start_as_current_span(operation_name) as span:
            try:
                # Add common attributes
                span.set_attribute("mcp.operation", operation_name)

                # Add custom attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                yield span

                span.set_status(Status(StatusCode.OK))

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def record_request(
        self,
        server: str,
        method: str,
        success: bool = True,
        duration: Optional[float] = None,
    ):
        """Record a request metric."""
        attributes = {
            "server": server,
            "method": method,
            "success": str(success).lower(),
        }

        self.request_counter.add(1, attributes)

        if duration is not None:
            self.request_duration.record(duration, attributes)

        if not success:
            self.error_counter.add(1, attributes)

    def record_connection(self, server: str, success: bool = True):
        """Record a connection metric."""
        attributes = {
            "server": server,
            "success": str(success).lower(),
        }
        self.connection_counter.add(1, attributes)

    def record_cache_hit(self, server: str, method: str):
        """Record a cache hit."""
        attributes = {"server": server, "method": method}
        self.cache_hits.add(1, attributes)

    def record_cache_miss(self, server: str, method: str):
        """Record a cache miss."""
        attributes = {"server": server, "method": method}
        self.cache_misses.add(1, attributes)

    def record_circuit_breaker_trip(self, server: str):
        """Record a circuit breaker trip."""
        attributes = {"server": server}
        self.circuit_breaker_trips.add(1, attributes)

    def record_error(self, server: str, error_type: str, operation: str = "unknown"):
        """Record an error metric."""
        attributes = {
            "server": server,
            "error_type": error_type,
            "operation": operation,
        }
        self.error_counter.add(1, attributes)

    def shutdown(self):
        """Shutdown telemetry and flush remaining data."""
        if not self.enabled or not OPENTELEMETRY_AVAILABLE:
            return

        try:
            # Force export of remaining metrics and traces
            if hasattr(trace.get_tracer_provider(), "force_flush"):
                trace.get_tracer_provider().force_flush(timeout_millis=5000)

            if hasattr(metrics.get_meter_provider(), "force_flush"):
                metrics.get_meter_provider().force_flush(timeout_millis=5000)

            logger.info("Telemetry data exported successfully")
        except Exception as e:
            logger.warning(f"Error during telemetry shutdown: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get telemetry statistics."""
        return {
            "enabled": self.enabled,
            "opentelemetry_available": OPENTELEMETRY_AVAILABLE,
            "service_name": self.config.service_name if self.config else None,
            "environment": self.config.environment if self.config else None,
            "otlp_endpoint": self.config.otlp_endpoint if self.config else None,
        }


class TelemetryInstrumentation:
    """
    High-level instrumentation wrapper for MCP operations.
    Provides decorators and context managers for easy telemetry integration.
    """

    def __init__(self, telemetry: GitLabActionTelemetry):
        self.telemetry = telemetry

    def instrument_mcp_call(self, func):
        """Decorator to instrument MCP calls with telemetry."""

        async def wrapper(*args, **kwargs):
            # Extract server and method from function call
            server = kwargs.get("server_name") or (
                args[1] if len(args) > 1 else "unknown"
            )
            method = func.__name__

            start_time = time.time()

            with self.telemetry.trace_operation(
                f"mcp.{method}",
                {
                    "mcp.server": server,
                    "mcp.method": method,
                },
            ) as span:
                try:
                    result = await func(*args, **kwargs)

                    # Record successful request
                    duration = time.time() - start_time
                    self.telemetry.record_request(
                        server, method, success=True, duration=duration
                    )

                    if span:
                        span.set_attribute("mcp.success", True)
                        span.set_attribute("mcp.duration", duration)

                    return result

                except Exception as e:
                    # Record failed request
                    duration = time.time() - start_time
                    self.telemetry.record_request(
                        server, method, success=False, duration=duration
                    )
                    self.telemetry.record_error(server, type(e).__name__, method)

                    if span:
                        span.set_attribute("mcp.success", False)
                        span.set_attribute("mcp.error_type", type(e).__name__)

                    raise

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    @contextmanager
    def trace_cache_operation(self, operation: str, server: str, method: str):
        """Context manager for tracing cache operations."""
        with self.telemetry.trace_operation(
            f"cache.{operation}",
            {
                "cache.operation": operation,
                "mcp.server": server,
                "mcp.method": method,
            },
        ) as span:
            try:
                yield span
            except Exception as e:
                self.telemetry.record_error(
                    server, type(e).__name__, f"cache.{operation}"
                )
                raise

    @contextmanager
    def trace_circuit_breaker_operation(self, server: str, state: str):
        """Context manager for tracing circuit breaker operations."""
        with self.telemetry.trace_operation(
            "circuit_breaker.state_change",
            {
                "circuit_breaker.server": server,
                "circuit_breaker.state": state,
            },
        ) as span:
            try:
                yield span
            except Exception as e:
                self.telemetry.record_error(server, type(e).__name__, "circuit_breaker")
                raise


# Global telemetry instance (will be initialized by configuration)
gitlab_telemetry: Optional[GitLabActionTelemetry] = None
telemetry_instrumentation: Optional[TelemetryInstrumentation] = None


def initialize_telemetry(config):
    """Initialize global telemetry instance."""
    global gitlab_telemetry, telemetry_instrumentation

    gitlab_telemetry = GitLabActionTelemetry(config)
    telemetry_instrumentation = TelemetryInstrumentation(gitlab_telemetry)

    logger.info("Global telemetry initialized")


def get_telemetry() -> Optional[GitLabActionTelemetry]:
    """Get the global telemetry instance."""
    return gitlab_telemetry


def get_instrumentation() -> Optional[TelemetryInstrumentation]:
    """Get the global telemetry instrumentation."""
    return telemetry_instrumentation


def shutdown_telemetry():
    """Shutdown global telemetry."""
    global gitlab_telemetry
    if gitlab_telemetry:
        gitlab_telemetry.shutdown()
        gitlab_telemetry = None
    logger.info("Global telemetry shutdown")
