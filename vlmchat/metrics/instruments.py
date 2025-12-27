import collections
import threading
import time
from typing import Deque, Dict, List, Optional, Tuple, Union
from .metrics_collector import Instrument, DataPoint, ValueType

# ----- Example concrete instruments (Updated with _deserialize)



class CounterInstrument(Instrument):
    """Counts total of numeric values (adds value on datapoint add, subtracts on remove)."""

    def __init__(self, name: str, binding_keys: Optional[List[str]] = None, total: float = 0.0):
        super().__init__(name, binding_keys=binding_keys)
        self.total: float = total
        self._lock = threading.Lock()

    def on_datapoint_added(self, dp: DataPoint) -> None:
        if not self.matches(dp):
            return
        with self._lock:
            self.total += 1.0

    def on_datapoint_removed(self, dp: DataPoint) -> None:
        if not self.matches(dp):
            return
        with self._lock:
            self.total -= 1.0

    def export(self) -> Dict:
        base = super().export()
        base.update({"total": self.total})
        return base
        
    @classmethod
    def create(cls, data: Dict) -> 'CounterInstrument':
        return cls(
            name=data["name"],
            binding_keys=data["binding_keys"],
            total=data.get("total", 0.0)
        )


class AverageDurationInstrument(Instrument):
    """Computes sum(duration) / count of matched datapoints."""

    def __init__(self, name: str, binding_keys: Optional[List[str]] = None, initial_sum: float = 0.0, initial_count: float = 0.0):
        super().__init__(name, binding_keys=binding_keys)
        self._matched: Deque[DataPoint] = collections.deque() # Not restored on deserialization
        self._sum: float = initial_sum
        self._count: float = initial_count
        self._lock = threading.Lock()

    # ... on_datapoint_added, on_datapoint_removed, average_duration (unchanged)

    def on_datapoint_added(self, dp: DataPoint) -> None:
        if not self.matches(dp) or dp.value_type != ValueType.DURATION:
            return
        v = float(dp.value)
        with self._lock:
            self._matched.append(dp)
            self._sum += v
            self._count += 1.0

    def on_datapoint_removed(self, dp: DataPoint) -> None:
        if not self.matches(dp):
            return
        # Since we don't restore _matched, we only restore sum/count if we remove from collector
        # This implementation requires _matched to be accurate for removal, so deserialization loses removeability.
        # We will keep the original logic, accepting the limitation on removal after deserialization.
        with self._lock:
            try:
                self._matched.remove(dp)
                v = float(dp.value)
                self._sum -= v
                self._count -= 1.0
            except ValueError:
                pass
    
    def average_duration(self) -> float:
        with self._lock:
            if self._count is None or self._count == 0:
                return 0.0
            return self._sum / self._count

    def export(self) -> Dict:
        base = super().export()
        base.update({
            "sum": self._sum,
            "count": self._count,
            "average_duration": self.average_duration(),
        })
        return base
        
    @classmethod
    def create(cls, data: Dict) -> 'AverageDurationInstrument':
        return cls(
            name=data["name"],
            binding_keys=data["binding_keys"],
            initial_sum=data.get("sum", 0.0),
            initial_count=data.get("count", 0.0)
        )


class HistogramByAttributeInstrument(Instrument):
    """Buckets values by a specified attribute key and maintains sum and count per bucket."""

    def __init__(self, name: str, binding_keys: Optional[List[str]] = None, bucket_key: Optional[str] = None, initial_buckets: Optional[Dict] = None):
        if binding_keys and len(binding_keys) > 1:
            raise ValueError("HistogramByAttributeInstrument accepts at most one binding key")
        super().__init__(name, binding_keys=binding_keys)
        
        # Determine the bucket key source
        if bucket_key is not None:
            self.bucket_key = bucket_key
        else:
            self.bucket_key = binding_keys[0] if binding_keys and len(binding_keys) == 1 else None
        
        # buckets: bucket_value -> {'sum': float, 'count': int}
        self.buckets: Dict[str, Dict[str, Union[float, int]]] = initial_buckets if initial_buckets is not None else {}
        self._lock = threading.Lock()

    # ... _bucket_name, on_datapoint_added, on_datapoint_removed (unchanged)

    def _bucket_name(self, dp: DataPoint) -> str:
        if self.bucket_key is None:
            if not dp.attributes:
                return "__unknown__"
            return list(dp.attributes.keys())[0]
        else:
            return dp.attributes.get(self.bucket_key, "__unknown__")

    def on_datapoint_added(self, dp: DataPoint) -> None:
        if not self.matches(dp) or dp.value_type not in (ValueType.INT, ValueType.FLOAT, ValueType.DURATION):
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
        if not self.matches(dp) or dp.value_type not in (ValueType.INT, ValueType.FLOAT, ValueType.DURATION):
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
                del self.buckets[bname]

    def export(self) -> Dict:
        base = super().export()
        # Export bucket_key explicitly for correct reconstruction logic
        if self.bucket_key:
             base["bucket_key"] = self.bucket_key
        with self._lock:
            base.update({"buckets": {k: {"sum": float(v["sum"]), "count": int(v["count"])} for k, v in self.buckets.items()}})
        return base
        
    @classmethod
    def create(cls, data: Dict) -> 'HistogramByAttributeInstrument':
        return cls(
            name=data["name"],
            binding_keys=data["binding_keys"],
            bucket_key=data.get("bucket_key"),
            initial_buckets=data.get("buckets")
        )


class CountInstrument(Instrument):
    """CountInstrument that counts occurrences per attribute value for the first binding key."""

    def __init__(self, name: str, binding_keys: Optional[List[str]] = None, initial_counts: Optional[Dict[str, int]] = None):
        super().__init__(name, binding_keys=binding_keys)
        self._counts: Dict[str, int] = initial_counts if initial_counts is not None else {}
        self._lock = threading.Lock()

    # ... on_datapoint_added, on_datapoint_removed, count (unchanged)

    def on_datapoint_added(self, dp: DataPoint) -> None:
        if not self.matches(dp):
            return
        key = self.binding_keys[0] if self.binding_keys else None
        bucket = '__all__' if key is None else dp.attributes.get(key, '__unknown__')
        with self._lock:
            self._counts[bucket] = self._counts.get(bucket, 0) + 1

    def on_datapoint_removed(self, dp: DataPoint) -> None:
        if not self.matches(dp):
            return
        key = self.binding_keys[0] if self.binding_keys else None
        bucket = '__all__' if key is None else dp.attributes.get(key, '__unknown__')
        with self._lock:
            if bucket in self._counts:
                self._counts[bucket] -= 1
                if self._counts[bucket] <= 0:
                    del self._counts[bucket]

    @property
    def count(self) -> int:
        with self._lock:
            if not self._counts:
                return 0
            return max(self._counts.values())

    def export(self) -> Dict:
        base = super().export()
        # Export internal counts for full state restoration
        with self._lock:
            base.update({"counts": self._counts})
        return base
        
    @classmethod
    def create(cls, data: Dict) -> 'CountInstrument':
        return cls(
            name=data["name"],
            binding_keys=data["binding_keys"],
            initial_counts=data.get("counts")
        )


class AverageInstrument(Instrument):
    """Average instrument that computes per-attribute-value averages for the first binding key."""

    def __init__(self, name: str, binding_keys: Optional[List[str]] = None, initial_buckets: Optional[Dict[str, Tuple[float, int]]] = None):
        super().__init__(name, binding_keys=binding_keys)
        # map bucket -> (sum, count)
        self._buckets: Dict[str, Tuple[float, int]] = initial_buckets if initial_buckets is not None else {}
        self._lock = threading.Lock()

    # ... _bucket_for_dp, on_datapoint_added, on_datapoint_removed, average (unchanged)

    def _bucket_for_dp(self, dp: DataPoint) -> str:
        key = self.binding_keys[0] if self.binding_keys else None
        if key is None:
            return '__all__'
        return dp.attributes.get(key, '__unknown__')

    def on_datapoint_added(self, dp: DataPoint) -> None:
        if not self.matches(dp) or dp.value_type not in (ValueType.INT, ValueType.FLOAT):
            return
        v = float(dp.value)
        b = self._bucket_for_dp(dp)
        with self._lock:
            s, c = self._buckets.get(b, (0.0, 0))
            self._buckets[b] = (s + v, c + 1)

    def on_datapoint_removed(self, dp: DataPoint) -> None:
        if not self.matches(dp):
            return
        b = self._bucket_for_dp(dp)
        v = float(dp.value)
        with self._lock:
            if b not in self._buckets:
                return
            s, c = self._buckets[b]
            s -= v
            c -= 1
            if c <= 0:
                del self._buckets[b]
            else:
                self._buckets[b] = (s, c)
    
    def average(self) -> float:
        """Return the average for the most frequent bucket (max count)."""
        with self._lock:
            if not self._buckets:
                return 0.0
            best = None
            best_count = -1
            for b, (s, c) in self._buckets.items():
                if c > best_count:
                    best = b
                    best_count = c
            s, c = self._buckets[best]
            return float(s) / float(c)

    def export(self) -> Dict:
        with self._lock:
            base = super().export()
            # Export internal buckets for full state restoration
            export_buckets = {b: [s, c] for b, (s, c) in self._buckets.items()}
            base.update({"buckets": export_buckets})

            # export only aggregate totals across buckets for convenience
            total_sum = sum(s for s, c in self._buckets.values())
            total_count = sum(c for s, c in self._buckets.values())
            base.update({"sum": total_sum, "count": total_count, "average": (float(total_sum)/total_count) if total_count>0 else 0.0})
            return base
        
    @classmethod
    def create(cls, data: Dict) -> 'AverageInstrument':
        # Convert exported list [sum, count] back to tuple (sum, count)
        restored_buckets = {b: (s, c) for b, (s, c) in data.get("buckets", {}).items()}
        return cls(
            name=data["name"],
            binding_keys=data["binding_keys"],
            initial_buckets=restored_buckets
        )


class HistogramInstrument(Instrument):
    """Simple histogram that aggregates sum and count across matching datapoints."""

    def __init__(self, name: str, binding_keys: Optional[List[str]] = None, initial_sum: float = 0.0, initial_count: int = 0):
        super().__init__(name, binding_keys=binding_keys)
        self.sum: float = initial_sum
        self.count: int = initial_count
        self._lock = threading.Lock()

    # ... on_datapoint_added, on_datapoint_removed (unchanged)
    
    def on_datapoint_added(self, dp: DataPoint) -> None:
        if not self.matches(dp) or dp.value_type not in (ValueType.INT, ValueType.FLOAT, ValueType.DURATION):
            return
        v = float(dp.value)
        with self._lock:
            self.sum += v
            self.count += 1

    def on_datapoint_removed(self, dp: DataPoint) -> None:
        if not self.matches(dp):
            return
        v = float(dp.value)
        with self._lock:
            self.sum -= v
            self.count -= 1

    def export(self) -> Dict:
        base = super().export()
        base.update({"sum": self.sum, "count": self.count})
        return base
        
    @classmethod
    def create(cls, data: Dict) -> 'HistogramInstrument':
        return cls(
            name=data["name"],
            binding_keys=data["binding_keys"],
            initial_sum=data.get("sum", 0.0),
            initial_count=data.get("count", 0)
        )


class MinMaxAvgLastInstrument(Instrument):
    """Tracks min, max, sum, count, last value and average rate since creation."""

    def __init__(self, name: str, binding_keys: Optional[List[str]] = None, 
                 initial_sum: float = 0.0, initial_count: int = 0, 
                 initial_min: Optional[float] = None, initial_max: Optional[float] = None, 
                 initial_last_value: Optional[float] = None, created_ts: Optional[float] = None):
        super().__init__(name, binding_keys=binding_keys)
        # Note: _matched is not restored on deserialization
        self._matched: Deque[DataPoint] = collections.deque() 
        self.sum: float = initial_sum
        self.count: int = initial_count
        self.min: Optional[float] = initial_min
        self.max: Optional[float] = initial_max
        self.last_value: Optional[float] = initial_last_value
        self.created_ts: float = created_ts if created_ts is not None else time.time()
        self._lock = threading.Lock()

    # ... on_datapoint_added, on_datapoint_removed, average_rate (unchanged)

    def on_datapoint_added(self, dp: DataPoint) -> None:
        if not self.matches(dp) or dp.value_type not in (ValueType.INT, ValueType.FLOAT, ValueType.DURATION):
            return
        v = float(dp.value)
        with self._lock:
            self._matched.append(dp)
            self.sum += v
            self.count += 1
            self.last_value = v
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
                self.sum -= v
                self.count -= 1
                # update last_value if necessary
                if self.last_value == v:
                    if self._matched:
                        self.last_value = float(self._matched[-1].value)
                    else:
                        self.last_value = None
                # recompute min/max lazily if needed
                if self.count == 0:
                    self.min = None
                    self.max = None
                else:
                    # Note: This is an expensive operation on removal; generally avoided in production metrics.
                    if self.min == v or self.max == v:
                        vals = [float(d.value) for d in self._matched]
                        self.min = min(vals)
                        self.max = max(vals)
            except ValueError:
                pass
    
    def average_rate(self) -> float:
        with self._lock:
            elapsed = time.time() - self.created_ts
            if elapsed <= 0.0:
                return 0.0
            return float(self.sum) / elapsed

    def export(self) -> Dict:
        base = super().export()
        base.update({
            "min": self.min,
            "max": self.max,
            "sum": self.sum,
            "count": self.count,
            "last_value": self.last_value,
            "average_rate": self.average_rate(),
            "created_ts": self.created_ts,
        })
        return base
        
    @classmethod
    def create(cls, data: Dict) -> 'MinMaxAvgLastInstrument':
        return cls(
            name=data["name"],
            binding_keys=data["binding_keys"],
            initial_sum=data.get("sum", 0.0),
            initial_count=data.get("count", 0),
            initial_min=data.get("min"),
            initial_max=data.get("max"),
            initial_last_value=data.get("last_value"),
            created_ts=data.get("created_ts")
        )
