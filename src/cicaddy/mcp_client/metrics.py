"""Comprehensive metrics and monitoring for MCP client operations using OpenTelemetry."""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from cicaddy.utils.logger import get_logger

from .telemetry import get_telemetry

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """A single metric data point."""

    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class HistogramBucket:
    """Histogram bucket for latency measurements."""

    upper_bound: float
    count: int = 0


class MetricCollector:
    """Base class for metric collection."""

    def __init__(self, name: str, metric_type: MetricType, description: str = ""):
        self.name = name
        self.metric_type = metric_type
        self.description = description
        self.created_at = time.time()

    def collect(self) -> List[MetricPoint]:
        """Collect current metric values."""
        raise NotImplementedError


class Counter(MetricCollector):
    """Counter metric for tracking cumulative values."""

    def __init__(self, name: str, description: str = ""):
        super().__init__(name, MetricType.COUNTER, description)
        self._value = 0.0
        self._labels_values: Dict[str, float] = defaultdict(float)

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment counter."""
        if labels:
            label_key = self._labels_to_key(labels)
            self._labels_values[label_key] += value
        else:
            self._value += value

    def collect(self) -> List[MetricPoint]:
        """Collect counter values."""
        points = []

        # Main counter
        if self._value > 0:
            points.append(MetricPoint(timestamp=time.time(), value=self._value))

        # Labeled counters
        for label_key, value in self._labels_values.items():
            if value > 0:
                labels = self._key_to_labels(label_key)
                points.append(
                    MetricPoint(timestamp=time.time(), value=value, labels=labels)
                )

        return points

    def _labels_to_key(self, labels: Dict[str, str]) -> str:
        """Convert labels dict to string key."""
        return "|".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def _key_to_labels(self, key: str) -> Dict[str, str]:
        """Convert string key back to labels dict."""
        if not key:
            return {}
        return dict(item.split("=", 1) for item in key.split("|"))


class Gauge(MetricCollector):
    """Gauge metric for tracking current values."""

    def __init__(self, name: str, description: str = ""):
        super().__init__(name, MetricType.GAUGE, description)
        self._value = 0.0
        self._labels_values: Dict[str, float] = defaultdict(float)

    def set(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Set gauge value."""
        if labels:
            label_key = self._labels_to_key(labels)
            self._labels_values[label_key] = value
        else:
            self._value = value

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment gauge value."""
        if labels:
            label_key = self._labels_to_key(labels)
            self._labels_values[label_key] += value
        else:
            self._value += value

    def dec(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Decrement gauge value."""
        self.inc(-value, labels)

    def collect(self) -> List[MetricPoint]:
        """Collect gauge values."""
        points = []

        # Main gauge
        points.append(MetricPoint(timestamp=time.time(), value=self._value))

        # Labeled gauges
        for label_key, value in self._labels_values.items():
            labels = self._key_to_labels(label_key)
            points.append(
                MetricPoint(timestamp=time.time(), value=value, labels=labels)
            )

        return points

    def _labels_to_key(self, labels: Dict[str, str]) -> str:
        """Convert labels dict to string key."""
        return "|".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def _key_to_labels(self, key: str) -> Dict[str, str]:
        """Convert string key back to labels dict."""
        if not key:
            return {}
        return dict(item.split("=", 1) for item in key.split("|"))


class Histogram(MetricCollector):
    """Histogram metric for tracking distributions."""

    def __init__(
        self, name: str, description: str = "", buckets: Optional[List[float]] = None
    ):
        super().__init__(name, MetricType.HISTOGRAM, description)
        self.buckets = buckets or [
            0.005,
            0.01,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.5,
            5.0,
            10.0,
            float("inf"),
        ]
        self._histogram_buckets = [HistogramBucket(upper_bound=b) for b in self.buckets]
        self._sum = 0.0
        self._count = 0

    def observe(self, value: float):
        """Observe a value."""
        self._sum += value
        self._count += 1

        # Update buckets
        for bucket in self._histogram_buckets:
            if value <= bucket.upper_bound:
                bucket.count += 1

    def collect(self) -> List[MetricPoint]:
        """Collect histogram values."""
        points = []
        timestamp = time.time()

        # Bucket counts
        for bucket in self._histogram_buckets:
            labels = {"le": str(bucket.upper_bound)}
            points.append(
                MetricPoint(timestamp=timestamp, value=bucket.count, labels=labels)
            )

        # Sum and count
        points.append(
            MetricPoint(timestamp=timestamp, value=self._sum, labels={"type": "sum"})
        )
        points.append(
            MetricPoint(
                timestamp=timestamp, value=self._count, labels={"type": "count"}
            )
        )

        return points

    @property
    def average(self) -> float:
        """Get average value."""
        return self._sum / self._count if self._count > 0 else 0.0


class Timer:
    """Context manager for timing operations."""

    def __init__(self, histogram: Histogram, labels: Optional[Dict[str, str]] = None):
        self.histogram = histogram
        self.labels = labels
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.histogram.observe(duration)


class MCPMetrics:
    """
    Main metrics collector for MCP operations using OpenTelemetry.

    This class provides a compatibility layer over OpenTelemetry instruments
    while maintaining the same interface as the previous Prometheus-based implementation.
    """

    def __init__(self):
        self.telemetry = get_telemetry()

        # Local tracking for summary statistics
        self._request_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_duration": 0.0,
        }
        self._connection_stats = {
            "active_connections": 0,
            "total_connections": 0,
            "failed_connections": 0,
        }
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "size": 0,
        }
        # Server health tracking with gauge metrics
        self._health_stats = defaultdict(
            lambda: {"healthy": False, "last_check": 0.0, "consecutive_failures": 0}
        )
        self._created_at = time.time()

    def record_request(
        self,
        server: str,
        method: str,
        success: bool = True,
        duration: Optional[float] = None,
    ):
        """Record a request."""
        # Update local stats
        self._request_stats["total_requests"] += 1
        if success:
            self._request_stats["successful_requests"] += 1
        else:
            self._request_stats["failed_requests"] += 1

        if duration is not None:
            self._request_stats["total_duration"] += duration

        # Record via OpenTelemetry
        if self.telemetry:
            self.telemetry.record_request(server, method, success, duration)

    def record_connection(self, server: str, success: bool = True):
        """Record a connection attempt."""
        self._connection_stats["total_connections"] += 1
        if success:
            self._connection_stats["active_connections"] += 1
        else:
            self._connection_stats["failed_connections"] += 1

        if self.telemetry:
            self.telemetry.record_connection(server, success)

    def record_disconnection(self, server: str):
        """Record a disconnection."""
        self._connection_stats["active_connections"] = max(
            0, self._connection_stats["active_connections"] - 1
        )

    def update_server_health(self, server: str, healthy: bool):
        """Update server health status with gauge metrics tracking."""
        current_time = time.time()
        health_data = self._health_stats[server]

        # Update health tracking
        was_healthy = health_data["healthy"]
        health_data["healthy"] = healthy
        health_data["last_check"] = current_time

        if healthy:
            health_data["consecutive_failures"] = 0
        else:
            health_data["consecutive_failures"] += 1

        # Record health change events via OpenTelemetry
        if self.telemetry:
            # Record gauge metric for current health status (1.0 = healthy, 0.0 = unhealthy)
            self.telemetry.record_gauge(
                "mcp_server_health", 1.0 if healthy else 0.0, {"server": server}
            )

            # Record consecutive failures as a gauge
            self.telemetry.record_gauge(
                "mcp_server_consecutive_failures",
                float(health_data["consecutive_failures"]),
                {"server": server},
            )

            # Record health check events and transitions
            if not healthy:
                self.telemetry.record_error(
                    server, "health_check_failed", "health_check"
                )

            # Record health state transitions
            if was_healthy != healthy:
                status = "healthy" if healthy else "unhealthy"
                self.telemetry.record_counter(
                    "mcp_server_health_transitions",
                    1.0,
                    {
                        "server": server,
                        "from_status": "healthy" if was_healthy else "unhealthy",
                        "to_status": status,
                    },
                )

    def record_circuit_breaker_state(self, server: str, state: str):
        """Record circuit breaker state."""
        # For GitLab Actions, we record state changes as events
        pass

    def record_circuit_breaker_trip(self, server: str):
        """Record circuit breaker trip."""
        if self.telemetry:
            self.telemetry.record_circuit_breaker_trip(server)

    def record_cache_hit(self, server: str, method: str):
        """Record cache hit."""
        self._cache_stats["hits"] += 1
        if self.telemetry:
            self.telemetry.record_cache_hit(server, method)

    def record_cache_miss(self, server: str, method: str):
        """Record cache miss."""
        self._cache_stats["misses"] += 1
        if self.telemetry:
            self.telemetry.record_cache_miss(server, method)

    def update_cache_size(self, size: int):
        """Update cache size."""
        self._cache_stats["size"] = size

    def update_pool_stats(
        self, server: str, active: int, idle: int, queue_size: int = 0
    ):
        """Update connection pool statistics."""
        # For GitLab Actions, pool stats are more informational
        pass

    def record_error(self, server: str, error_type: str):
        """Record an error."""
        if self.telemetry:
            self.telemetry.record_error(server, error_type)

    def record_timeout(self, server: str, operation: str):
        """Record a timeout."""
        if self.telemetry:
            self.telemetry.record_error(server, "timeout", operation)

    def get_server_health(self, server: str) -> Dict[str, Any]:
        """Get health statistics for a specific server."""
        health_data = self._health_stats.get(server, {})
        return {
            "healthy": health_data.get("healthy", False),
            "last_check": health_data.get("last_check", 0.0),
            "consecutive_failures": health_data.get("consecutive_failures", 0),
            "time_since_last_check": time.time() - health_data.get("last_check", 0.0),
        }

    def get_all_server_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health statistics for all servers."""
        return {
            server: self.get_server_health(server)
            for server in self._health_stats.keys()
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics for GitLab Action summary."""
        return {
            "request_stats": self._request_stats.copy(),
            "connection_stats": self._connection_stats.copy(),
            "cache_stats": self._cache_stats.copy(),
            "health_stats": self.get_all_server_health(),
            "telemetry_stats": self.telemetry.get_stats() if self.telemetry else {},
        }

    def export_opentelemetry_summary(self) -> str:
        """Export metrics summary for GitLab Action logs."""
        stats = self.get_summary_stats()

        lines = [
            "📊 MCP Action Telemetry Summary",
            "=" * 40,
            f"📋 Total Requests: {stats.get('total_requests', 0)}",
            f"✅ Successful: {stats.get('successful_requests', 0)}",
            f"❌ Failed: {stats.get('failed_requests', 0)}",
            f"📈 Success Rate: {stats.get('success_rate', 0):.1f}%",
        ]

        if stats.get("avg_response_time", 0) > 0:
            lines.append(f"⏱️  Avg Response Time: {stats['avg_response_time']:.3f}s")

        if stats.get("total_cache_requests", 0) > 0:
            lines.extend(
                [
                    f"🗄️  Cache Hits: {stats.get('cache_hits', 0)}",
                    f"🔍 Cache Misses: {stats.get('cache_misses', 0)}",
                    f"📊 Cache Hit Rate: {stats.get('cache_hit_rate', 0):.1f}%",
                ]
            )

        lines.extend(
            [
                f"🔗 Active Connections: {stats.get('active_connections', 0)}",
                f"⏳ Action Duration: {stats.get('uptime', 0):.1f}s",
            ]
        )

        return "\n".join(lines)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for GitLab Action execution."""
        try:
            # Calculate summary values from local stats
            total_requests = self._request_stats["total_requests"]
            successful_requests = self._request_stats["successful_requests"]
            failed_requests = self._request_stats["failed_requests"]

            success_rate = 0.0
            if total_requests > 0:
                success_rate = (successful_requests / total_requests) * 100

            # Calculate average response time
            avg_response_time = 0.0
            if total_requests > 0 and self._request_stats["total_duration"] > 0:
                avg_response_time = (
                    self._request_stats["total_duration"] / total_requests
                )

            # Cache statistics
            cache_hits = self._cache_stats["hits"]
            cache_misses = self._cache_stats["misses"]
            total_cache_requests = cache_hits + cache_misses
            cache_hit_rate = 0.0
            if total_cache_requests > 0:
                cache_hit_rate = (cache_hits / total_cache_requests) * 100

            return {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": success_rate,
                "active_connections": self._connection_stats["active_connections"],
                "total_connections": self._connection_stats["total_connections"],
                "avg_response_time": avg_response_time,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "cache_hit_rate": cache_hit_rate,
                "total_cache_requests": total_cache_requests,
                "uptime": time.time() - self._created_at,
            }
        except Exception as e:
            logger.error(f"Error generating summary stats: {e}")
            return {}


class PerformanceMonitor:
    """Performance monitoring with alerting capabilities."""

    def __init__(
        self, metrics: MCPMetrics, alert_thresholds: Optional[Dict[str, float]] = None
    ):
        self.metrics = metrics
        self.alert_thresholds = alert_thresholds or {
            "error_rate_threshold": 10.0,  # 10% error rate
            "response_time_threshold": 5.0,  # 5 seconds
            "cache_hit_rate_threshold": 50.0,  # 50% cache hit rate
            "connection_failure_rate": 20.0,  # 20% connection failure rate
        }
        self.alerts: List[Dict[str, Any]] = []
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False

    async def start_monitoring(self, interval: float = 60.0):
        """Start performance monitoring."""
        if self._running:
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
        logger.info(f"Started performance monitoring with {interval}s interval")

    async def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self._running:
            return

        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped performance monitoring")

    async def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(interval)
                await self._check_performance()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    async def _check_performance(self):
        """Check performance metrics and generate alerts."""
        summary = self.metrics.get_summary_stats()

        # Check error rate
        error_rate = 100.0 - summary.get("success_rate", 100.0)
        if error_rate > self.alert_thresholds["error_rate_threshold"]:
            await self._create_alert(
                "high_error_rate",
                f"Error rate {error_rate:.1f}% exceeds threshold",
                {
                    "error_rate": error_rate,
                    "threshold": self.alert_thresholds["error_rate_threshold"],
                },
            )

        # Check response time
        avg_response_time = summary.get("avg_response_time", 0.0)
        if avg_response_time > self.alert_thresholds["response_time_threshold"]:
            await self._create_alert(
                "slow_response",
                f"Average response time {avg_response_time:.2f}s exceeds threshold",
                {
                    "avg_response_time": avg_response_time,
                    "threshold": self.alert_thresholds["response_time_threshold"],
                },
            )

        # Check cache hit rate
        cache_hit_rate = summary.get("cache_hit_rate", 100.0)
        if cache_hit_rate < self.alert_thresholds["cache_hit_rate_threshold"]:
            await self._create_alert(
                "low_cache_hit_rate",
                f"Cache hit rate {cache_hit_rate:.1f}% below threshold",
                {
                    "cache_hit_rate": cache_hit_rate,
                    "threshold": self.alert_thresholds["cache_hit_rate_threshold"],
                },
            )

    async def _create_alert(
        self, alert_type: str, message: str, context: Dict[str, Any]
    ):
        """Create a performance alert."""
        alert = {
            "type": alert_type,
            "message": message,
            "context": context,
            "timestamp": time.time(),
            "severity": "warning",
        }

        self.alerts.append(alert)
        logger.warning(f"Performance alert: {message}")

        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

    def get_recent_alerts(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent performance alerts."""
        return self.alerts[-count:] if self.alerts else []

    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts.clear()


# Global metrics instance
mcp_metrics = MCPMetrics()
performance_monitor = PerformanceMonitor(mcp_metrics)
