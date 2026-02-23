"""Simple metrics collection with observer pattern for evolving requirements."""

import collections
import threading
import time
import weakref
from abc import ABC, abstractmethod
from typing import Any, Deque, Dict, List, Optional


class DataPoint:
    """A single metric observation with value and timestamp."""
    
    def __init__(self, value: float, timestamp: float, attributes: Optional[Dict[str, str]] = None):
        self.value = value
        self.timestamp = timestamp
        self.attributes = dict(attributes or {})


class Instrument(ABC):
    """Abstract instrument that observes datapoints and computes aggregates.
    
    Subclass to create custom aggregations (avg, percentiles, rates, etc).
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def on_datapoint(self, dp: DataPoint) -> None:
        """Called when a matching datapoint is recorded."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return current statistics as a dict."""
        pass


class AvgDurationInstrument(Instrument):
    """Computes average, min, max, count, last value."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self._sum = 0.0
        self._count = 0
        self._min: Optional[float] = None
        self._max: Optional[float] = None
        self._last: Optional[float] = None
        self._lock = threading.Lock()
    
    def on_datapoint(self, dp: DataPoint) -> None:
        with self._lock:
            self._sum += dp.value
            self._count += 1
            self._last = dp.value
            if self._min is None or dp.value < self._min:
                self._min = dp.value
            if self._max is None or dp.value > self._max:
                self._max = dp.value
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "avg": self._sum / self._count if self._count > 0 else 0.0,
                "min": self._min,
                "max": self._max,
                "count": self._count,
                "last": self._last,
            }


class RecentSamplesInstrument(Instrument):
    """Keeps last N samples in a circular buffer for graphing."""
    
    def __init__(self, name: str, max_samples: int = 100):
        super().__init__(name)
        self._samples: Deque[DataPoint] = collections.deque(maxlen=max_samples)
        self._lock = threading.Lock()
    
    def on_datapoint(self, dp: DataPoint) -> None:
        with self._lock:
            self._samples.append(dp)
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "samples": [(dp.timestamp, dp.value) for dp in self._samples],
                "count": len(self._samples),
            }


class CounterInstrument(Instrument):
    """Simple counter that increments on each datapoint."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self._count = 0
        self._lock = threading.Lock()
    
    def on_datapoint(self, dp: DataPoint) -> None:
        with self._lock:
            self._count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {"count": self._count}


class RateInstrument(Instrument):
    """Computes rate per second since creation."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self._count = 0
        self._created = time.time()
        self._lock = threading.Lock()
    
    def on_datapoint(self, dp: DataPoint) -> None:
        with self._lock:
            self._count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            elapsed = time.time() - self._created
            rate = self._count / elapsed if elapsed > 0 else 0.0
            return {
                "count": self._count,
                "elapsed": elapsed,
                "rate": rate,
            }


class MemoryInstrument(Instrument):
    """Tracks object lifetimes using weakref for leak detection.
    
    Records datapoints when objects are tracked (value=size_bytes).
    Uses weakref to detect when objects are garbage collected.
    
    Usage:
        collector = Collector()
        mem_inst = MemoryInstrument("memory.contexts")
        collector.add_instrument(mem_inst, "memory.track")
        
        # Track object
        message = {}
        collector.record("memory.track", size_bytes, attributes={
            "type": "Context",
            "obj_id": str(id(ctx))
        })
        mem_inst.track_object(ctx, id(ctx))  # Must call after record
    """
    
    def __init__(self, name: str, leak_threshold_seconds: float = 60.0):
        super().__init__(name)
        self.leak_threshold = leak_threshold_seconds
        self._tracked: Dict[int, Dict[str, Any]] = {}  # obj_id -> {type, created_at, size, attributes}
        self._weakrefs: Dict[int, weakref.ref] = {}  # obj_id -> weakref (keep refs alive)
        self._lock = threading.Lock()
        
        # Statistics
        self.total_tracked = 0
        self.total_cleaned = 0
    
    def on_datapoint(self, dp: DataPoint) -> None:
        """Record object tracking metadata."""
        with self._lock:
            obj_id = int(dp.attributes.get("obj_id", "0"))
            if obj_id != 0:
                self._tracked[obj_id] = {
                    "type": dp.attributes.get("type", "Unknown"),
                    "created_at": dp.timestamp,
                    "size_bytes": dp.value,
                    "attributes": dict(dp.attributes)
                }
                self.total_tracked += 1
    
    def track_object(self, obj: Any, obj_id: int) -> None:
        """Create weakref to track when object is GC'd.
        
        Must be called AFTER recording datapoint with same obj_id.
        """
        try:
            def on_cleanup(ref):
                self._on_object_deleted(obj_id)
            
            # Store the weakref so it doesn't get GC'd immediately
            with self._lock:
                self._weakrefs[obj_id] = weakref.ref(obj, on_cleanup)
        except TypeError:
            # Object doesn't support weakref - remove from tracking
            with self._lock:
                if obj_id in self._tracked:
                    del self._tracked[obj_id]
    
    def _on_object_deleted(self, obj_id: int) -> None:
        """Callback when tracked object is GC'd."""
        with self._lock:
            if obj_id in self._tracked:
                del self._tracked[obj_id]
                self.total_cleaned += 1
            if obj_id in self._weakrefs:
                del self._weakrefs[obj_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current memory statistics including potential leaks."""
        with self._lock:
            now = time.time()
            
            # Group by type
            by_type: Dict[str, List[Dict]] = {}
            for obj_id, info in self._tracked.items():
                obj_type = info["type"]
                if obj_type not in by_type:
                    by_type[obj_type] = []
                by_type[obj_type].append({
                    "obj_id": obj_id,
                    "age_sec": now - info["created_at"],
                    "size_bytes": info["size_bytes"]
                })
            
            # Detect potential leaks
            leaks = []
            for obj_type, objects in by_type.items():
                old_objects = [o for o in objects if o["age_sec"] > self.leak_threshold]
                if old_objects:
                    total_size = sum(o["size_bytes"] for o in old_objects)
                    max_age = max(o["age_sec"] for o in old_objects)
                    leaks.append({
                        "type": obj_type,
                        "count": len(old_objects),
                        "total_size_bytes": total_size,
                        "max_age_sec": max_age
                    })
            
            return {
                "total_alive": len(self._tracked),
                "total_tracked": self.total_tracked,
                "total_cleaned": self.total_cleaned,
                "by_type": {t: len(objs) for t, objs in by_type.items()},
                "potential_leaks": leaks,
                "leak_threshold_sec": self.leak_threshold
            }


class DurationTimer:
    """Context manager for recording durations."""
    
    def __init__(self, collector: 'Collector', name: str, attributes: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.name = name
        self.attributes = attributes
        self._start: Optional[float] = None
    
    def __enter__(self) -> 'DurationTimer':
        self._start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        start = self._start if self._start is not None else time.time()
        duration_ms = (time.time() - start) * 1000.0
        self.collector.record(self.name, duration_ms, self.attributes)


class Collector:
    """Manages timeseries and notifies instruments of new datapoints."""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self._instruments: Dict[str, List[Instrument]] = {}  # timeseries_name -> [instruments]
        self._lock = threading.RLock()
    
    def add_instrument(self, instrument: Instrument, timeseries_name: str) -> None:
        """Attach an instrument to observe a timeseries."""
        with self._lock:
            if timeseries_name not in self._instruments:
                self._instruments[timeseries_name] = []
            self._instruments[timeseries_name].append(instrument)
    
    def get_stats(self, instrument_name: str) -> Optional[Dict[str, Any]]:
        """Get stats for a specific instrument by name."""
        with self._lock:
            for instruments in self._instruments.values():
                for inst in instruments:
                    if inst.name == instrument_name:
                        return inst.get_stats()
        return None
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats from all instruments."""
        with self._lock:
            stats = {}
            for instruments in self._instruments.values():
                for inst in instruments:
                    stats[inst.name] = inst.get_stats()
            return stats
    
    def record(self, name: str, value: float, attributes: Optional[Dict[str, str]] = None, timestamp: Optional[float] = None) -> None:
        """Record a datapoint."""
        timestamp = timestamp if timestamp is not None else time.time()
        dp = DataPoint(value, timestamp, attributes)
        
        with self._lock:
            # Notify all instruments observing this timeseries
            instruments = self._instruments.get(name, [])
            for inst in instruments:
                inst.on_datapoint(dp)  # Let exceptions propagate
    
    def duration_timer(self, name: str, attributes: Optional[Dict[str, str]] = None) -> DurationTimer:
        """Create a duration timer context manager."""
        return DurationTimer(self, name, attributes)


class NullCollector(Collector):
    """No-op collector for when metrics are disabled."""
    
    def __init__(self):
        self.name = "null"
    
    def add_instrument(self, instrument: Instrument, timeseries_name: str) -> None:
        pass
    
    def get_stats(self, instrument_name: str) -> Optional[Dict[str, Any]]:
        return None
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        return {}
    
    def record(self, name: str, value: float, attributes: Optional[Dict[str, str]] = None, timestamp: Optional[float] = None) -> None:
        pass
    
    def duration_timer(self, name: str, attributes: Optional[Dict[str, str]] = None) -> DurationTimer:
        return DurationTimer(self, name, attributes)


# Singleton null collector
_null_collector = NullCollector()


def null_collector() -> NullCollector:
    """Return singleton null collector."""
    return _null_collector
