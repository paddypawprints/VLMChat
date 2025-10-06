"""
Lightweight metrics collection system compatible with OpenTelemetry conventions.

This module provides a simple metrics collection interface that follows OpenTelemetry
naming and structure patterns, outputting to JSON files. It can be easily migrated
to full OpenTelemetry in the future.
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Any, Optional
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import psutil
import platform


@dataclass
class MetricPoint:
    """A single metric data point following OpenTelemetry conventions."""
    name: str
    value: Union[int, float]
    timestamp: float
    attributes: Dict[str, Any]
    unit: Optional[str] = None


@dataclass
class MetricSession:
    """Container for all metrics from a single model session."""
    resource_attributes: Dict[str, str]  # Platform, runtime, etc.
    start_time: float
    end_time: Optional[float] = None
    counters: List[MetricPoint] = None
    gauges: List[MetricPoint] = None
    histograms: List[MetricPoint] = None

    def __post_init__(self):
        if self.counters is None:
            self.counters = []
        if self.gauges is None:
            self.gauges = []
        if self.histograms is None:
            self.histograms = []


class MetricsCollector:
    """
    Lightweight metrics collector following OpenTelemetry patterns.

    Provides Counter, Gauge, and Histogram metric types that can be easily
    migrated to OpenTelemetry in the future.
    """

    def __init__(self, service_name: str, model_path: str, runtime: str):
        """
        Initialize metrics collector.

        Args:
            service_name: Name of the service (e.g., "smol-vlm")
            model_path: Path to the model being used
            runtime: Runtime type (onnx, transformers, tensorrt, etc.)
        """
        self.service_name = service_name
        self.session_id = str(uuid.uuid4())[:8]

        # Resource attributes following OpenTelemetry semantic conventions
        self.resource_attributes = {
            "service.name": service_name,
            "service.version": "1.0.0",
            "telemetry.sdk.name": "custom-metrics",
            "telemetry.sdk.version": "1.0.0",
            "host.name": platform.node(),
            "host.arch": platform.machine(),
            "os.type": platform.system().lower(),
            "model.path": str(model_path),
            "model.runtime": runtime,
            "session.id": self.session_id
        }

        self.session = MetricSession(
            resource_attributes=self.resource_attributes,
            start_time=time.time()
        )

        # Initialize system monitoring
        self.process = psutil.Process()

    def create_counter(self, name: str, description: str = "", unit: str = "") -> 'Counter':
        """Create a Counter metric (monotonically increasing value)."""
        return Counter(name, description, unit, self)

    def create_gauge(self, name: str, description: str = "", unit: str = "") -> 'Gauge':
        """Create a Gauge metric (point-in-time value)."""
        return Gauge(name, description, unit, self)

    def create_histogram(self, name: str, description: str = "", unit: str = "") -> 'Histogram':
        """Create a Histogram metric (distribution of values)."""
        return Histogram(name, description, unit, self)

    def _record_metric(self, metric_type: str, point: MetricPoint):
        """Record a metric point to the appropriate collection."""
        if metric_type == "counter":
            self.session.counters.append(point)
        elif metric_type == "gauge":
            self.session.gauges.append(point)
        elif metric_type == "histogram":
            self.session.histograms.append(point)

    def export_to_json(self, output_path: Path) -> str:
        """
        Export collected metrics to JSON file.

        Args:
            output_path: Directory to save the JSON file

        Returns:
            Path to the created JSON file
        """
        self.session.end_time = time.time()

        # Add final system metrics
        self._record_system_metrics()

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_{self.service_name}_{self.session_id}_{timestamp}.json"
        filepath = output_path / filename

        # Convert to dict and save
        data = asdict(self.session)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return str(filepath)

    def _record_system_metrics(self):
        """Record final system resource metrics."""
        current_time = time.time()

        # Memory metrics
        memory_info = self.process.memory_info()
        self.create_gauge("system.memory.usage", unit="bytes").record(
            memory_info.rss, {"type": "rss"}
        )

        # CPU metrics
        cpu_percent = self.process.cpu_percent()
        self.create_gauge("system.cpu.usage", unit="percent").record(
            cpu_percent, {"type": "process"}
        )

        # Session duration
        duration_ms = (current_time - self.session.start_time) * 1000
        self.create_histogram("session.duration", unit="ms").record(
            duration_ms, {"session_id": self.session_id}
        )


class Counter:
    """Counter metric - monotonically increasing value."""

    def __init__(self, name: str, description: str, unit: str, collector: MetricsCollector):
        self.name = name
        self.description = description
        self.unit = unit
        self.collector = collector
        self._value = 0

    def add(self, value: Union[int, float], attributes: Dict[str, Any] = None):
        """Add to the counter value."""
        if attributes is None:
            attributes = {}

        self._value += value
        point = MetricPoint(
            name=self.name,
            value=self._value,
            timestamp=time.time(),
            attributes=attributes,
            unit=self.unit
        )
        self.collector._record_metric("counter", point)


class Gauge:
    """Gauge metric - point-in-time value that can go up or down."""

    def __init__(self, name: str, description: str, unit: str, collector: MetricsCollector):
        self.name = name
        self.description = description
        self.unit = unit
        self.collector = collector

    def record(self, value: Union[int, float], attributes: Dict[str, Any] = None):
        """Record a gauge value."""
        if attributes is None:
            attributes = {}

        point = MetricPoint(
            name=self.name,
            value=value,
            timestamp=time.time(),
            attributes=attributes,
            unit=self.unit
        )
        self.collector._record_metric("gauge", point)


class Histogram:
    """Histogram metric - distribution of values."""

    def __init__(self, name: str, description: str, unit: str, collector: MetricsCollector):
        self.name = name
        self.description = description
        self.unit = unit
        self.collector = collector

    def record(self, value: Union[int, float], attributes: Dict[str, Any] = None):
        """Record a histogram value."""
        if attributes is None:
            attributes = {}

        point = MetricPoint(
            name=self.name,
            value=value,
            timestamp=time.time(),
            attributes=attributes,
            unit=self.unit
        )
        self.collector._record_metric("histogram", point)

    @contextmanager
    def time(self, attributes: Dict[str, Any] = None):
        """Context manager to time an operation."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.record(duration_ms, attributes)


# Convenience function for creating collectors
def create_metrics_collector(service_name: str, model_path: str, runtime: str) -> MetricsCollector:
    """Create a metrics collector with standard configuration."""
    return MetricsCollector(service_name, model_path, runtime)