"""
System metrics sampler.

This module implements a singleton-like SystemMetricsSampler which, when
started with a Collector, will spawn a background thread that samples CPU
and memory usage every `sample_duration` seconds (minimum 1s) and records
those as datapoints in the provided collector. GPU metrics are queried from
the platform adapter layer when available.

If the provided collector is None or is a null collector, sampling is a no-op
and the sampler will not start.
"""
from __future__ import annotations

import threading
import time
import logging
from typing import Optional, Dict

from .metrics_collector import Collector, ValueType, UnknownTimeSeriesError, null_collector
from .platform_adapter import get_adapter

logger = logging.getLogger(__name__)


class SystemMetricsSampler:
    """Background sampler for CPU, memory, and optional GPU metrics.

    Usage:
        sampler = SystemMetricsSampler.get_instance(collector)
        sampler.start()
        ...
        sampler.stop()
    """

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, collector: Optional[Collector] = None, sample_duration: float = 5.0):
        with cls._lock:
            if cls._instance is None:
                cls._instance = SystemMetricsSampler(collector, sample_duration)
            return cls._instance

    def __init__(self, collector: Optional[Collector] = None, sample_duration: float = 5.0):
        # Respect instruction: if collector is None or null_collector, do nothing
        if collector is None:
            logger.info("SystemMetricsSampler: no collector provided; sampler will be a no-op")
            self._collector = None
            self._noop = True
            return

        # Guard against null collector sentinel
        try:
            if collector is null_collector():
                logger.info("SystemMetricsSampler: null_collector provided; sampler will be a no-op")
                self._collector = None
                self._noop = True
                return
        except Exception:
            # null_collector may not be comparable in some odd cases; ignore
            pass

        self._collector = collector
        self._noop = False

        # enforce minimum sample duration of 1 second
        self.sample_duration = max(float(sample_duration or 5.0), 1.0)

        # platform adapter for GPU metrics
        self.adapter = get_adapter()

        # internal threading primitives
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # create timeseries up-front (per requirement)
        try:
            self._collector.register_timeseries("system.cpu", registered_attribute_keys=["core"], max_count=0, ttl_seconds=None)
            self._collector.register_timeseries("system.memory", registered_attribute_keys=["type"], max_count=0, ttl_seconds=None)
            # GPU timeseries will be registered but may be no-op on platforms without GPU
            self._collector.register_timeseries("system.gpu.util", registered_attribute_keys=["adapter", "device"], max_count=0, ttl_seconds=None)
            self._collector.register_timeseries("system.gpu.mem_used", registered_attribute_keys=["adapter", "device"], max_count=0, ttl_seconds=None)
            self._collector.register_timeseries("system.gpu.mem_total", registered_attribute_keys=["adapter", "device"], max_count=0, ttl_seconds=None)
        except Exception as e:
            logger.exception("Failed to register system timeseries: %s", e)

    def start(self) -> None:
        if self._noop:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="SystemMetricsSampler", daemon=True)
        self._thread.start()

    def stop(self, join: bool = False) -> None:
        if self._noop:
            return
        self._stop_event.set()
        if join and self._thread:
            self._thread.join(timeout=2.0)

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _run_loop(self) -> None:
        # import psutil lazily
        try:
            import psutil
        except Exception:
            psutil = None
            logger.exception("psutil not available; CPU/memory sampling will be skipped")

        while not self._stop_event.is_set():
            start = time.time()
            try:
                self.sample_once(psutil)
            except Exception:
                logger.exception("Error during system metrics sampling")

            # sleep the remainder of interval
            elapsed = time.time() - start
            wait = max(0.0, self.sample_duration - elapsed)
            # allow early exit
            self._stop_event.wait(wait)

    def sample_once(self, psutil_module=None) -> None:
        if self._noop or self._collector is None:
            return

        # CPU: all cores (per user request). Use psutil if available; otherwise skip.
        try:
            if psutil_module is None:
                import psutil as _ps
                psutil_module = _ps
        except Exception:
            psutil_module = None

        try:
            if psutil_module:
                percore = psutil_module.cpu_percent(percpu=True)
                for idx, pct in enumerate(percore):
                    try:
                        self._collector.data_point("system.cpu", {"core": str(idx)}, pct)
                    except Exception:
                        # ignore per-core write errors
                        pass
            else:
                logger.debug("psutil not available; skipping CPU sampling")
        except Exception:
            logger.exception("Failed to sample CPU usage")

        # Memory: percent virtual
        try:
            if psutil_module:
                vm = psutil_module.virtual_memory()
                try:
                    self._collector.data_point("system.memory", {"type": "virtual"}, float(vm.percent))
                except Exception:
                    pass
            else:
                logger.debug("psutil not available; skipping memory sampling")
        except Exception:
            logger.exception("Failed to sample memory usage")

        # GPU: ask adapter for metrics and record if present
        try:
            gm = self.adapter.get_gpu_metrics() if self.adapter is not None else None
            if gm:
                adapter_name = getattr(self.adapter, "name", "unknown")
                device = "0"
                if "util" in gm:
                    try:
                        self._collector.data_point("system.gpu.util", {"adapter": adapter_name, "device": device}, float(gm["util"]))
                    except Exception:
                        pass
                if "mem_used" in gm:
                    try:
                        self._collector.data_point("system.gpu.mem_used", {"adapter": adapter_name, "device": device}, float(gm["mem_used"]))
                    except Exception:
                        pass
                if "mem_total" in gm:
                    try:
                        self._collector.data_point("system.gpu.mem_total", {"adapter": adapter_name, "device": device}, float(gm["mem_total"]))
                    except Exception:
                        pass
        except Exception:
            logger.exception("Failed to sample GPU metrics")
