"""
Metrics system implementation matching the user's specification.

Key points implemented:
- Timeseries: named collection of DataPoints indexed by time.
- DataPoint supports int, float, duration (stored as float seconds).
- Timeseries registers allowed attribute keys (string keys, string values).
- Collector owns timeseries and enforces bounds: max_count and ttl_seconds.
- Updates go through Collector.add_datapoint(name, value, type, attributes, timestamp?)
- Sessions observe a Collector and receive add/remove notifications.
- Instruments are bound to a timeseries by name + attributes and receive updates.
- Thread-safe with per-timeseries locks and collector-level lock.

This module is intentionally self-contained and minimal; further helpers
and exporters can be added if desired.
"""
from __future__ import annotations

import collections
import threading
import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Union, Set
from enum import Enum

import logging

logger = logging.getLogger(__name__)


# ----- Types and small utilities
class ValueType(Enum):
    INT = "int"
    FLOAT = "float"
    DURATION = "duration"


@dataclass(frozen=True)
class DataPoint:
    value_type: ValueType
    value: Union[int, float]
    timestamp: float
    attributes: Dict[str, str]


class InvalidAttributesError(ValueError):
    pass


class UnknownTimeSeriesError(KeyError):
    pass


# ----- TimeSeries
class TimeSeries:
    """Named time-ordered collection of DataPoint objects.

    - registered_attribute_keys: set of string keys allowed on datapoints
    - max_count: optional, maximum number of points to retain
    - ttl_seconds: optional, time-window to retain points (old points evicted)
    """

    def __init__(
        self,
        name: str,
        registered_attribute_keys: Optional[Set[str]] = None,
        max_count: Optional[int] = None,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        self.name = name
        self.registered_attribute_keys: Set[str] = set(registered_attribute_keys or [])
        self.max_count = max_count
        self.ttl_seconds = ttl_seconds

        # store datapoints in a deque ordered oldest->newest
        self._points: Deque[DataPoint] = collections.deque()
        self._lock = threading.Lock()

    def _evict_if_needed(self) -> List[DataPoint]:
        """Evict points based on max_count and ttl_seconds.

        Returns list of evicted datapoints (oldest-first) so caller can notify.
        """
        evicted: List[DataPoint] = []
        now = time.time()

        # Evict by TTL first
        if self.ttl_seconds is not None:
            cutoff = now - self.ttl_seconds
            while self._points and self._points[0].timestamp < cutoff:
                evicted.append(self._points.popleft())

        # Evict by count
        if self.max_count is not None:
            while len(self._points) > self.max_count:
                evicted.append(self._points.popleft())

        return evicted

    def append(self, dp: DataPoint) -> List[DataPoint]:
        """Append a datapoint; return list of evicted datapoints (if any).

        Note: caller must validate that dp.attributes keys are subset of
        registered_attribute_keys.
        """
        with self._lock:
            # maintain order by timestamp: append normally if timestamp >= last
            if not self._points or dp.timestamp >= self._points[-1].timestamp:
                self._points.append(dp)
            else:
                # insert in time order (rare path). Keep stable insertion.
                # Find right insertion index from right side (usually near end).
                idx = len(self._points)
                while idx > 0 and self._points[idx - 1].timestamp > dp.timestamp:
                    idx -= 1
                self._points.insert(idx, dp)

            evicted = self._evict_if_needed()
            return evicted

    def remove_exact(self, dp: DataPoint) -> bool:
        """Remove a datapoint equal to dp if present. Return True if removed."""
        with self._lock:
            try:
                self._points.remove(dp)
                return True
            except ValueError:
                return False

    def snapshot(self) -> List[DataPoint]:
        with self._lock:
            return list(self._points)


# ----- Instrument base class
class Instrument(ABC):
    """Abstract instrument attached to a session and bound to a timeseries by name
    and a set of binding attributes (subset of those registered on the timeseries).

    Concrete instruments should implement on_datapoint_added and
    on_datapoint_removed. The default match() performs exact equality on the
    binding attributes; subclasses may override match() for different logic.
    """

    def __init__(self, name: str, binding_attributes: Optional[Dict[str, str]] = None):
        self.name = name
        self.binding_attributes = dict(binding_attributes or {})

    def matches(self, dp: DataPoint) -> bool:
        # quick name check (dp already from timeseries with same name)
        for k, v in self.binding_attributes.items():
            if dp.attributes.get(k) != v:
                return False
        return True

    @abstractmethod
    def on_datapoint_added(self, dp: DataPoint) -> None:
        raise NotImplementedError()

    @abstractmethod
    def on_datapoint_removed(self, dp: DataPoint) -> None:
        raise NotImplementedError()

    def export(self) -> Dict:
        """Return a serializable dict representing this instrument's state.

        Subclasses should extend this with instrument-specific aggregates.
        """
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "binding_attributes": dict(self.binding_attributes),
        }


# ----- Example concrete instruments
class CounterInstrument(Instrument):
    """Counts total of numeric values (adds value on datapoint add, subtracts on remove).

    Keeps internal matched point deque so removals can be handled.
    """

    def __init__(self, name: str, binding_attributes: Optional[Dict[str, str]] = None):
        super().__init__(name, binding_attributes)
        self._matched: Deque[DataPoint] = collections.deque()
        self.total: float = 0.0
        self._lock = threading.Lock()

    def on_datapoint_added(self, dp: DataPoint) -> None:
        if not self.matches(dp):
            return
        # only numeric types supported
        if dp.value_type not in (ValueType.INT, ValueType.FLOAT):
            return
        with self._lock:
            self._matched.append(dp)
            self.total += float(dp.value)

    def on_datapoint_removed(self, dp: DataPoint) -> None:
        if not self.matches(dp):
            return
        with self._lock:
            # remove first matching point (datapoints are unique by timestamp+value+attrs)
            try:
                self._matched.remove(dp)
                self.total -= float(dp.value)
            except ValueError:
                # may not be present if it was removed earlier
                pass

    def export(self) -> Dict:
        base = super().export()
        base.update({"total": self.total, "matched_count": len(self._matched)})
        return base


class HistogramInstrument(Instrument):
    """Simple running histogram-like aggregates: count, sum, min, max.

    For percentiles a reservoir would be added; for now we maintain exact
    aggregates and a deque of matched points so removals can be handled.
    """

    def __init__(self, name: str, binding_attributes: Optional[Dict[str, str]] = None):
        super().__init__(name, binding_attributes)
        self._matched: Deque[DataPoint] = collections.deque()
        self.count: int = 0
        self.sum: float = 0.0
        self.min: Optional[float] = None
        self.max: Optional[float] = None
        self._lock = threading.Lock()

    def on_datapoint_added(self, dp: DataPoint) -> None:
        if not self.matches(dp):
            return
        # numeric or duration
        if dp.value_type not in (ValueType.INT, ValueType.FLOAT, ValueType.DURATION):
            return
        v = float(dp.value)
        with self._lock:
            self._matched.append(dp)
            self.count += 1
            self.sum += v
            if self.min is None or v < self.min:
                self.min = v
            if self.max is None or v > self.max:
                self.max = v

    def on_datapoint_removed(self, dp: DataPoint) -> None:
        if not self.matches(dp):
            return
        with self._lock:
            try:
                self._matched.remove(dp)
                v = float(dp.value)
                self.count -= 1
                self.sum -= v
                # recompute min/max lazily if needed
                if self.count == 0:
                    self.min = None
                    self.max = None
                else:
                    if self.min == v or self.max == v:
                        vals = [float(d.value) for d in self._matched]
                        self.min = min(vals)
                        self.max = max(vals)
            except ValueError:
                pass

    def export(self) -> Dict:
        base = super().export()
        base.update({
            "count": self.count,
            "sum": self.sum,
            "min": self.min,
            "max": self.max,
        })
        return base


class AverageDurationInstrument(Instrument):
    """Computes sum(values) / total_duration of matched datapoints.

    total_duration = last_timestamp - first_timestamp across matched samples.
    Instrument stores matched samples so removals are possible.
    """

    def __init__(self, name: str, binding_attributes: Optional[Dict[str, str]] = None):
        super().__init__(name, binding_attributes)
        self._matched: Deque[DataPoint] = collections.deque()
        self._sum: float = 0.0
        self._first_ts: Optional[float] = None
        self._last_ts: Optional[float] = None
        self._lock = threading.Lock()

    def on_datapoint_added(self, dp: DataPoint) -> None:
        if not self.matches(dp):
            return
        # duration or numeric type acceptable for sum
        if dp.value_type not in (ValueType.INT, ValueType.FLOAT, ValueType.DURATION):
            return
        v = float(dp.value)
        with self._lock:
            self._matched.append(dp)
            self._sum += v
            if self._first_ts is None or dp.timestamp < self._first_ts:
                self._first_ts = dp.timestamp
            if self._last_ts is None or dp.timestamp > self._last_ts:
                self._last_ts = dp.timestamp

    def on_datapoint_removed(self, dp: DataPoint) -> None:
        if not self.matches(dp):
            return
        with self._lock:
            try:
                self._matched.remove(dp)
                v = float(dp.value)
                self._sum -= v
                if not self._matched:
                    self._first_ts = None
                    self._last_ts = None
                else:
                    ts = [d.timestamp for d in self._matched]
                    self._first_ts = min(ts)
                    self._last_ts = max(ts)
            except ValueError:
                pass

    def average_duration(self) -> float:
        with self._lock:
            if self._first_ts is None or self._last_ts is None or self._last_ts <= self._first_ts:
                return 0.0
            total_duration = self._last_ts - self._first_ts
            return self._sum / total_duration if total_duration > 0 else 0.0

    def export(self) -> Dict:
        base = super().export()
        base.update({
            "sum": self._sum,
            "first_timestamp": self._first_ts,
            "last_timestamp": self._last_ts,
            "average_duration": self.average_duration(),
            "matched_count": len(self._matched),
        })
        return base


# ----- New instruments requested by user
class CountInstrument(Instrument):
    """Counts the number of matched datapoints (increments on add, decrements on remove)."""

    def __init__(self, name: str, binding_attributes: Optional[Dict[str, str]] = None):
        super().__init__(name, binding_attributes)
        self.count: int = 0
        self._matched: Deque[DataPoint] = collections.deque()
        self._lock = threading.Lock()

    def on_datapoint_added(self, dp: DataPoint) -> None:
        if not self.matches(dp):
            return
        with self._lock:
            self._matched.append(dp)
            self.count += 1

    def on_datapoint_removed(self, dp: DataPoint) -> None:
        if not self.matches(dp):
            return
        with self._lock:
            try:
                self._matched.remove(dp)
                self.count -= 1
            except ValueError:
                pass

    def export(self) -> Dict:
        base = super().export()
        base.update({"count": self.count})
        return base


class AverageInstrument(Instrument):
    """Maintains sum/count and exposes average = sum / count."""

    def __init__(self, name: str, binding_attributes: Optional[Dict[str, str]] = None):
        super().__init__(name, binding_attributes)
        self.sum: float = 0.0
        self.count: int = 0
        self._matched: Deque[DataPoint] = collections.deque()
        self._lock = threading.Lock()

    def on_datapoint_added(self, dp: DataPoint) -> None:
        if not self.matches(dp):
            return
        if dp.value_type not in (ValueType.INT, ValueType.FLOAT, ValueType.DURATION):
            return
        v = float(dp.value)
        with self._lock:
            self._matched.append(dp)
            self.sum += v
            self.count += 1

    def on_datapoint_removed(self, dp: DataPoint) -> None:
        if not self.matches(dp):
            return
        with self._lock:
            try:
                self._matched.remove(dp)
                v = float(dp.value)
                self.sum -= v
                self.count -= 1
            except ValueError:
                pass

    def average(self) -> float:
        with self._lock:
            if self.count == 0:
                return 0.0
            return self.sum / self.count

    def export(self) -> Dict:
        base = super().export()
        base.update({"sum": self.sum, "count": self.count, "average": self.average()})
        return base


class HistogramByAttributeInstrument(Instrument):
    """Buckets values by a specified attribute key and maintains sum and count per bucket.

    bucket_key: attribute name to use for bucketing (string). If a datapoint lacks
    the attribute, the bucket name '__unknown__' is used.
    """

    def __init__(self, name: str, bucket_key: str, binding_attributes: Optional[Dict[str, str]] = None):
        super().__init__(name, binding_attributes)
        self.bucket_key = bucket_key
        # buckets: bucket_value -> {'sum': float, 'count': int}
        self.buckets: Dict[str, Dict[str, Union[float, int]]] = {}
        self._lock = threading.Lock()

    def _bucket_name(self, dp: DataPoint) -> str:
        return dp.attributes.get(self.bucket_key, "__unknown__")

    def on_datapoint_added(self, dp: DataPoint) -> None:
        if not self.matches(dp):
            return
        if dp.value_type not in (ValueType.INT, ValueType.FLOAT, ValueType.DURATION):
            return
        v = float(dp.value)
        bname = self._bucket_name(dp)
        with self._lock:
            entry = self.buckets.get(bname)
            if entry is None:
                entry = {"sum": 0.0, "count": 0}
                self.buckets[bname] = entry
            entry["sum"] = float(entry["sum"]) + v
            entry["count"] = int(entry["count"]) + 1

    def on_datapoint_removed(self, dp: DataPoint) -> None:
        if not self.matches(dp):
            return
        if dp.value_type not in (ValueType.INT, ValueType.FLOAT, ValueType.DURATION):
            return
        v = float(dp.value)
        bname = self._bucket_name(dp)
        with self._lock:
            entry = self.buckets.get(bname)
            if not entry:
                return
            entry["sum"] = float(entry["sum"]) - v
            entry["count"] = int(entry["count"]) - 1
            if entry["count"] <= 0:
                # remove empty bucket
                del self.buckets[bname]

    def export(self) -> Dict:
        base = super().export()
        with self._lock:
            base.update({"buckets": {k: {"sum": float(v["sum"]), "count": int(v["count"])} for k, v in self.buckets.items()}})
        return base


# ----- Convenience context manager for recording durations
class DurationTimer:
    """Context manager that records elapsed time as a DURATION datapoint.

    Usage:
        with DurationTimer(collector, "session.duration", attributes={...}):
            do_work()

    On exit this will call collector.add_datapoint(..., value_type=ValueType.DURATION)
    with the measured duration in milliseconds. Exceptions during recording are logged.
    """

    def __init__(self, collector: 'Collector', timeseries_name: str, attributes: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.timeseries_name = timeseries_name
        self.attributes = dict(attributes or {})
        self._start: Optional[float] = None

    def __enter__(self) -> 'DurationTimer':
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        end = time.time()
        start = self._start or end
        duration_s = end - start
        duration_ms = duration_s * 1000.0
        try:
            # record duration in milliseconds
            self.collector.add_datapoint(self.timeseries_name, ValueType.DURATION, duration_ms, attributes=self.attributes, timestamp=end)
        except Exception:
            logger.exception("Failed to record duration datapoint for %s", self.timeseries_name)


# ----- Session and Collector
class Session:
    """Observer of a Collector. Holds instruments and receives notifications.

    A session registers itself with a collector and receives add/remove events.
    """

    def __init__(self, collector: 'Collector') -> None:
        self.collector = collector
        # start_time is set when the session is started; keep constructor
        # backwards-compatible by starting immediately
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self._instruments: List[Tuple[str, Instrument]] = []
        self._lock = threading.Lock()
        collector.register_session(self)

    def start(self) -> None:
        """Start or restart the session.

        This sets the session start_time and ensures the session is registered
        with the collector.
        """
        with self._lock:
            self.start_time = time.time()
            self.end_time = None
            try:
                self.collector.register_session(self)
            except Exception:
                # ignore registration errors if already registered
                pass

    def stop(self) -> None:
        self.end_time = time.time()
        self.collector.unregister_session(self)

    def add_instrument(self, inst: Instrument, timeseries_name: str) -> None:
        # instrument is bound conceptually to a timeseries name; binding attrs are
        # part of the instrument itself (Instrument.binding_attributes)
        with self._lock:
            self._instruments.append((timeseries_name, inst))

    def to_dict(self) -> Dict:
        """Serialize the session and its instruments into a JSON-serializable dict.

        Includes start_time, end_time, and a list of instruments with their exports
        and the timeseries name they are bound to.
        """
        with self._lock:
            instruments = []
            for (ts_name, inst) in list(self._instruments):
                try:
                    instruments.append({"timeseries": ts_name, "instrument": inst.export()})
                except Exception:
                    instruments.append({"timeseries": ts_name, "instrument": {"error": "export failed"}})

        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "instruments": instruments,
        }

    def export_to_json(self, output_dir: Union[str, Path]) -> str:
        """Write the session export to a JSON file inside output_dir and return path."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        filename = f"session_{int(self.start_time)}_{ts}.json"
        filepath = out / filename
        data = self.to_dict()
        # ensure serializable (datetimes are floats already)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        return str(filepath)

    def _on_datapoint_added(self, ts_name: str, dp: DataPoint) -> None:
        with self._lock:
            for (tsn, inst) in list(self._instruments):
                if tsn != ts_name:
                    continue
                try:
                    inst.on_datapoint_added(dp)
                except Exception:
                    # swallow instrument errors to avoid breaking collector flow
                    pass

    def _on_datapoint_removed(self, ts_name: str, dp: DataPoint) -> None:
        with self._lock:
            for (tsn, inst) in list(self._instruments):
                if tsn != ts_name:
                    continue
                try:
                    inst.on_datapoint_removed(dp)
                except Exception:
                    pass


class Collector:
    """Manages timeseries and sessions; accepts datapoint updates."""

    def __init__(self) -> None:
        self._ts_map: Dict[str, TimeSeries] = {}
        self._sessions: List[Session] = []
        self._lock = threading.RLock()

    def register_timeseries(self, name: str, registered_attribute_keys: Optional[List[str]] = None, max_count: Optional[int] = None, ttl_seconds: Optional[float] = None) -> None:
        with self._lock:
            if name in self._ts_map:
                raise KeyError(f"TimeSeries already registered: {name}")
            ts = TimeSeries(name, set(registered_attribute_keys or []), max_count, ttl_seconds)
            self._ts_map[name] = ts

    def unregister_timeseries(self, name: str) -> None:
        with self._lock:
            if name in self._ts_map:
                del self._ts_map[name]

    def register_session(self, session: Session) -> None:
        with self._lock:
            self._sessions.append(session)

    def duration_timer(self, timeseries_name: str, attributes: Optional[Dict[str, str]] = None) -> 'DurationTimer':
        """Convenience factory returning a DurationTimer bound to this collector."""
        return DurationTimer(self, timeseries_name, attributes=attributes)

    def data_point(self, name: str, attributes: Optional[Dict[str, str]], value: Union[int, float], timestamp: Optional[float] = None) -> None:
        """Convenience helper to record a numeric datapoint (infers int vs float).

        Example: collector.data_point("requests", {"route": "/"}, 1)
        """
        # infer type (bool is a subclass of int, avoid treating bool as int)
        if isinstance(value, bool):
            # treat booleans as ints (0/1)
            vtype = ValueType.INT
        elif isinstance(value, int):
            vtype = ValueType.INT
        else:
            vtype = ValueType.FLOAT
        self.add_datapoint(name, vtype, value, attributes=attributes, timestamp=timestamp)

    def unregister_session(self, session: Session) -> None:
        with self._lock:
            try:
                self._sessions.remove(session)
            except ValueError:
                pass

    def add_datapoint(self, name: str, value_type: ValueType, value: Union[int, float], attributes: Optional[Dict[str, str]] = None, timestamp: Optional[float] = None) -> None:
        attributes = dict(attributes or {})
        timestamp = timestamp if timestamp is not None else time.time()

        with self._lock:
            ts = self._ts_map.get(name)
            if ts is None:
                raise UnknownTimeSeriesError(name)

            # Validate attributes are subset
            if not set(attributes.keys()).issubset(ts.registered_attribute_keys):
                raise InvalidAttributesError(f"Attributes {list(attributes.keys())} not registered for timeseries {name}")

            dp = DataPoint(value_type=value_type, value=value, timestamp=timestamp, attributes=attributes)

            # Append and get evicted datapoints
            evicted = ts.append(dp)

            # Notify sessions about the addition
            for s in list(self._sessions):
                try:
                    s._on_datapoint_added(name, dp)
                except Exception:
                    pass

            # Notify sessions about evicted datapoints (removals)
            for ev in evicted:
                for s in list(self._sessions):
                    try:
                        s._on_datapoint_removed(name, ev)
                    except Exception:
                        pass

    def snapshot_timeseries(self, name: str) -> List[DataPoint]:
        with self._lock:
            ts = self._ts_map.get(name)
            if ts is None:
                raise UnknownTimeSeriesError(name)
            return ts.snapshot()
