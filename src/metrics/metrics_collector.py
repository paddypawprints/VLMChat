"""
Metrics system implementation matching the user's specification, now including
full support for factory-based deserialization of Instruments and Sessions.

Key updates:
- Instrument base class includes a class registry and factory methods.
- All concrete Instrument classes implement the _deserialize class method.
- Collector now has a 'name' attribute.
- Session includes a static deserialize factory method.
"""
from __future__ import annotations

import collections
import threading
import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Union, Set, Type, Any
from enum import Enum

import logging
from contextlib import nullcontext

logger = logging.getLogger(__name__)

# ----- Types and small utilities
class ValueType(Enum):
    INT = "int"
    FLOAT = "float"
    DURATION = "duration"

class DataPoint:
    value_type: ValueType
    value: Union[int, float]
    timestamp: float
    attributes: Dict[str, str]

    def __init__(self, value_type: ValueType, value: Union[int, float], timestamp: float, attributes: Optional[Dict[str, str]] = None):
        self.value_type = value_type
        self.value = value
        self.timestamp = timestamp
        self.attributes = dict(attributes or {})

class InvalidAttributesError(ValueError):
    pass

class UnknownTimeSeriesError(KeyError):
    pass

# ----- TimeSeries (unchanged)
class TimeSeries:
    """Named time-ordered collection of DataPoint objects."""

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

        self._points: Deque[DataPoint] = collections.deque()
        self._lock = threading.Lock()

    def _evict_if_needed(self) -> List[DataPoint]:
        """Evict points based on max_count and ttl_seconds."""
        evicted: List[DataPoint] = []
        now = time.time()

        if self.ttl_seconds is not None:
            cutoff = now - self.ttl_seconds
            while self._points and self._points[0].timestamp < cutoff:
                evicted.append(self._points.popleft())

        if self.max_count is not None:
            while len(self._points) > self.max_count:
                evicted.append(self._points.popleft())

        return evicted

    def append(self, dp: DataPoint) -> List[DataPoint]:
        """Append a datapoint; return list of evicted datapoints (if any)."""
        with self._lock:
            if not self._points or dp.timestamp >= self._points[-1].timestamp:
                self._points.append(dp)
            else:
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


# ----- Instrument base class (Updated with Factory)
class Instrument(ABC):
    """Abstract instrument attached to a session and bound to a timeseries by name.
    Includes methods for state export and factory-based deserialization.
    """

    # --- Deserialization Registry and Factory ---
    _registry: Dict[str, Type['Instrument']] = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register every concrete subclass."""
        super().__init_subclass__(**kwargs)
        # Only register concrete classes that are not the base ABC itself
        if cls.__name__ != 'Instrument' and not ABC in cls.__bases__:
            Instrument._registry[cls.__name__] = cls

    @classmethod
    @abstractmethod
    def create(cls, data: Dict[str, Any]) -> 'Instrument':
        """Internal method: Create an instance of the specific subclass from JSON data.
        Subclasses MUST implement this.
        """
        raise NotImplementedError()

    @staticmethod
    def load_instrument(json_str: str) -> 'Instrument':
        data = json.loads(json_str)
        return Instrument.create(data)
    
    @staticmethod
    def create(data: Dict[str, Any]) -> 'Instrument':
        """
        CLASS FACTORY: Create an Instrument instance (of the correct subclass) from a dictionary.
        The input dict is expected to be the output of Instrument.export().
        """
        instrument_type, instrument_data = list(data.items())[0]

        subclass = Instrument._registry.get(instrument_type)
        if not subclass:
            raise ValueError(f"Unknown instrument type: {instrument_type}")
        
        # Call the specific subclass's _deserialize method
        return subclass.create(instrument_data)
    # --------------------------------------------

    def __init__(self, name: str, binding_keys: Optional[List[str]] = None):
        """Create an instrument.

        Args:
            name: instrument name
            binding_keys: list of attribute keys the instrument requires.
        """
        self.name = name
        self.binding_keys = list(binding_keys or [])

    def matches(self, dp: DataPoint) -> bool:
        """Match datapoints by presence of required attribute keys only."""
        if not self.binding_keys:
            return True
        for k in self.binding_keys:
            if k not in dp.attributes:
                return False
        return True

    @abstractmethod
    def on_datapoint_added(self, dp: DataPoint) -> None:
        raise NotImplementedError()

    @abstractmethod
    def on_datapoint_removed(self, dp: DataPoint) -> None:
        raise NotImplementedError()

    def export(self) -> Dict:
        """Return a serializable dict representing this instrument's state."""
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "binding_keys": list(self.binding_keys),
        }




# ----- Convenience context manager for recording durations (unchanged)
class DurationTimer:
    """Context manager that records elapsed time as a DURATION datapoint."""

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
            self.collector.add_datapoint(self.timeseries_name, ValueType.DURATION, duration_ms, attributes=self.attributes, timestamp=end)
        except Exception:
            logger.exception("Failed to record duration datapoint for %s", self.timeseries_name)


# ----- Session and Collector (Updated for name and deserialization)
class Session:
    """Observer of a Collector. Holds instruments and receives notifications."""

    def __init__(self, collector: 'Collector', start_time: Optional[float] = None, end_time: Optional[float] = None):
        self.collector = collector
        self.start_time = start_time if start_time is not None else time.time()
        self.end_time: Optional[float] = end_time
        # Stores (timeseries_name, Instrument) tuples
        self._instruments: List[Tuple[str, Instrument]] = []
        self._lock = threading.Lock()
        collector.register_session(self)

    @classmethod
    def load_instruments_from_json(cls, collector: 'Collector', json_str: str) -> Session:
        """
        Factory method to reconstruct a Session instance from a dictionary export.
        """
        data = json.loads(json_str)
        session = cls(collector=collector)
        session.load_instruments(data)
        return session


    def load_instruments(self, data: Dict[str, Any]) -> None:
        for metrics in data:
            item = metrics.get("metrics")
            ts_name = item.get("timeseries")
            instrument_data = item.get("instruments")
            # Use the Instrument factory method
            for instrument_json in instrument_data:
                instrument = Instrument.create(instrument_json)
                self.add_instrument(instrument, ts_name)

    def start(self) -> None:
        """Start or restart the session."""
        with self._lock:
            self.start_time = time.time()
            self.end_time = None
            try:
                self.collector.register_session(self)
            except Exception:
                pass

    def stop(self) -> None:
        self.end_time = time.time()
        self.collector.unregister_session(self)

    def add_instrument(self, inst: Instrument, timeseries_name: str) -> None:
        """Add a bound instrument to the session."""
        with self._lock:
            self._instruments.append((timeseries_name, inst))

    def to_dict(self) -> Dict:
        """Serialize the session and its instruments into a JSON-serializable dict."""
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

    def __init__(self, name: str = "default_collector") -> None:
        self.name = name # Added name attribute
        self._ts_map: Dict[str, TimeSeries] = {}
        self._sessions: List[Session] = []
        self._lock = threading.RLock()

    def register_timeseries(self, name: str, registered_attribute_keys: Optional[List[str]] = None, max_count: Optional[int] = None, ttl_seconds: Optional[float] = None) -> None:
        with self._lock:
            if name in self._ts_map:
                raise KeyError(f"TimeSeries already registered: {name}")
            ts = TimeSeries(name, set(registered_attribute_keys or []), max_count, ttl_seconds)
            self._ts_map[name] = ts

    # ... (other Collector methods remain the same)

    def unregister_timeseries(self, name: str) -> None:
        with self._lock:
            if name in self._ts_map:
                del self._ts_map[name]

    def register_session(self, session: Session) -> None:
        with self._lock:
            # Check if session is already registered to avoid duplicates
            if session not in self._sessions:
                self._sessions.append(session)

    def duration_timer(self, timeseries_name: str, attributes: Optional[Dict[str, str]] = None) -> 'DurationTimer':
        """Convenience factory returning a DurationTimer bound to this collector."""
        return DurationTimer(self, timeseries_name, attributes=attributes)

    def data_point(self, name: str, attributes: Optional[Dict[str, str]], value: Union[int, float], timestamp: Optional[float] = None) -> None:
        """Convenience helper to record a numeric datapoint (infers int vs float)."""
        if isinstance(value, bool):
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
                logger.info(f"Collector.add_datapoint: unknown timeseries {name}, ignoring datapoint")
                return # silently ignore unknown timeseries

            if not set(attributes.keys()).issubset(ts.registered_attribute_keys):
                raise InvalidAttributesError(f"Attributes {list(attributes.keys())} not registered for timeseries {name}")

            dp = DataPoint(value_type=value_type, value=value, timestamp=timestamp, attributes=attributes)
            evicted = ts.append(dp)

            for s in list(self._sessions):
                try:
                    s._on_datapoint_added(name, dp)
                except Exception:
                    pass

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


# ----- Null (no-op) Collector (Updated with name)
class NullCollector:
    """A no-op collector that implements the same public API as Collector."""

    def __init__(self, name: str = "null_collector") -> None:
        self.name = name

    def register_timeseries(self, name: str, registered_attribute_keys: Optional[List[str]] = None, max_count: Optional[int] = None, ttl_seconds: Optional[float] = None) -> None:
        return None

    def unregister_timeseries(self, name: str) -> None:
        return None

    def register_session(self, session: Session) -> None:
        return None

    def unregister_session(self, session: Session) -> None:
        return None

    def duration_timer(self, timeseries_name: str, attributes: Optional[Dict[str, str]] = None):
        return nullcontext()

    def data_point(self, name: str, attributes: Optional[Dict[str, str]], value: Union[int, float], timestamp: Optional[float] = None) -> None:
        return None

    def add_datapoint(self, name: str, value_type: ValueType, value: Union[int, float], attributes: Optional[Dict[str, str]] = None, timestamp: Optional[float] = None) -> None:
        return None

    def snapshot_timeseries(self, name: str) -> List[DataPoint]:
        return []


# module-level singleton factory
_NULL_COLLECTOR = NullCollector()


def null_collector() -> NullCollector:
    """Return a singleton NullCollector instance."""
    return _NULL_COLLECTOR