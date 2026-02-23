## Production Memory Monitoring with weakref

### Summary

Memory monitoring is integrated into the metrics collector using `MemoryInstrument`. It uses `weakref`-based tracking to detect memory leaks **without** affecting refcounts or requiring manual cleanup.

### Quick Start

```python
from camera_framework import (
    Collector,
    MemoryInstrument,
    AvgDurationInstrument,
)

# Create collector
collector = Collector("pipeline")

# Add memory instrument
mem_inst = MemoryInstrument("memory.objects", leak_threshold_seconds=60.0)
collector.add_instrument(mem_inst, "memory.track")

# Track objects in your tasks
class CameraTask(BaseTask):
    def process(self):
        frame = capture_frame()
        
        # Record to collector (triggers MemoryInstrument.on_datapoint)
        obj_id = id(frame)
        self.collector.record("memory.track", frame.nbytes, attributes={
            "type": "numpy.ndarray",
            "obj_id": str(obj_id)
        })
        
        # Create weakref (must be after record())
        mem_inst.track_object(frame, obj_id)

# Get stats
stats = collector.get_all_stats()
if stats['memory.objects']['potential_leaks']:
    print("LEAK DETECTED!")
```

### Why weakref?

**Problem with refcount checks:**
- ❌ Python caching adds unpredictable +1 refs
- ❌ Fragile across Python versions
- ❌ Doesn't prove object deletion

**Advantage of weakref:**
- ✅ **Proves deletion**: If `weak() is None`, object is gone
- ✅ **Detects leaks**: If object lives too long, we have a leak
- ✅ **Zero overhead**: Doesn't increment refcount
- ✅ **Production-safe**: Can run in production without affecting behavior

### Test Usage

```python
import weakref
from camera_framework import Context, Buffer, drop_oldest_policy

def test_buffer_no_leak():
    """Verify buffer releases objects when dropped."""
    buffer = Buffer(size=2, policy=drop_oldest_policy)
    
    # Track objects with weakrefs
    weak_refs = []
    for i in range(5):
        ctx = Context()
        weak_refs.append(weakref.ref(ctx))
        buffer.put(ctx)
        del ctx  # Remove our reference
    
    # Buffer size=2, so items 0-2 were dropped
    assert weak_refs[0]() is None  # ✅ Cleaned up
    assert weak_refs[1]() is None  # ✅ Cleaned up
    assert weak_refs[2]() is None  # ✅ Cleaned up
    assert weak_refs[3]() is not None  # Still alive
    assert weak_refs[4]() is not None  # Still alive
```

### Production Usage

```python
from camera_framework import (
    Collector,
    MemoryInstrument,
    AvgDurationInstrument,
)

# Setup collector with memory tracking
collector = Collector("pipeline")
mem_inst = MemoryInstrument("memory.images", leak_threshold_seconds=60.0)
collector.add_instrument(mem_inst, "memory.track")

class CameraTask(BaseTask):
    def __init__(self, collector: Collector):
        super().__init__(name="camera")
        self.collector = collector
        self.mem_inst = collector.get_instrument("memory.images")
    
    def process(self):
        # Create frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Track in metrics system
        obj_id = id(frame)
        self.collector.record("memory.track", frame.nbytes, attributes={
            "type": "numpy.ndarray",
            "obj_id": str(obj_id),
            "source": "camera"
        })
        self.mem_inst.track_object(frame, obj_id)
        
        # Normal processing...
        ctx = Context()
        ctx.append("frame", frame)
        for buf in self.outputs:
            buf.put(ctx)

# Check for leaks periodically
stats = collector.get_all_stats()
if "memory.images" in stats:
    mem_stats = stats["memory.images"]
    if mem_stats['potential_leaks']:
        logger.warning(f"Leaks detected: {mem_stats['potential_leaks']}")
```

### How It Works

1. **Create MemoryInstrument**: Add to collector observing "memory.track" timeseries
2. **Record datapoint**: `collector.record("memory.track", size_bytes, attributes={"obj_id": ...})`
3. **Track object**: `mem_inst.track_object(obj, obj_id)` creates weakref
4. **Cleanup callback**: When object is GC'd, callback fires automatically
5. **Leak detection**: Objects alive > threshold are flagged in `get_stats()`

### Integration with Metrics

**Benefits of using Collector:**
- ✅ **Unified API**: Same `collector.record()` interface for all metrics
- ✅ **Centralized stats**: `collector.get_all_stats()` includes memory + performance
- ✅ **Easy export**: All metrics available for Prometheus/JSON export
- ✅ **Flexible instruments**: Add custom aggregations per metric

**Example stats output:**
```python
stats = collector.get_all_stats()
# {
#   "camera.capture_avg": {"avg": 12.5, "min": 10.2, "max": 15.3, "count": 100},
#   "memory.images": {
#     "total_alive": 3,
#     "total_tracked": 1543,
#     "total_cleaned": 1540,
#     "by_type": {"numpy.ndarray": 2, "Context": 1},
#     "potential_leaks": [
#       {"type": "Context", "count": 1, "total_size_bytes": 1024, "max_age_sec": 127.3}
#     ]
#   }
# }
```

### Example Output

```
--- Iteration 100 (1.0s) ---
Camera capture: 2.34ms avg, 30 samples
Frames: 30

Memory:
  Alive: 5 objects
  Tracked: 30
  Cleaned: 25
  By type: {'numpy.ndarray': 3, 'Context': 2}

--- Iteration 200 (2.0s) ---
  ⚠️  LEAKS DETECTED:
    Context: 2 objects, 2.0 KB, max age 1.5s
```

This shows 2 Context objects have been alive >1s - investigate why they're not being consumed!

### Integration Points

**Optional tracking locations:**
1. ✅ **Large images** - Track in camera/processing tasks
2. ✅ **Contexts** - Track when created by sources
3. ⚠️ **Small objects** - Don't track (overhead not worth it)
4. ⚠️ **Pooled objects** - Don't track (intentionally long-lived)

**Sampling for performance:**
```python
if self.frame_count % 10 == 0:  # Track every 10th frame
    track_image(frame, size_bytes=frame.nbytes)
```

### No Production Code Changes Required

The beauty: **production code needs zero `del` or cleanup calls**. Python's scope-based cleanup handles everything:

```python
def process(self):
    ctx = self.inputs[0].get()  # Get reference
    # ... use ctx ...
    # Function ends → ctx goes out of scope (automatic cleanup)
```

Tests use explicit `del` to **simulate** end-of-scope and verify cleanup works.

### See Also

- [`camera_framework/metrics.py`](../camera_framework/metrics.py) - MemoryInstrument implementation
- [`examples/metrics_with_memory.py`](../examples/metrics_with_memory.py) - Full working example
- [`tests/test_buffer_architecture.py`](../tests/test_buffer_architecture.py) - Test patterns

### Migration from Standalone MemoryMonitor

If you were using the standalone `memory_monitor` module, migrate to integrated approach:

**Old (standalone):**
```python
from camera_framework import memory_monitor, track_image
track_image(frame, size_bytes=frame.nbytes)
memory_monitor.report()
```

**New (integrated with collector):**
```python
# Setup once
collector = Collector()
mem_inst = MemoryInstrument("memory.images")
collector.add_instrument(mem_inst, "memory.track")

# In your task
obj_id = id(frame)
self.collector.record("memory.track", frame.nbytes, 
                      attributes={"type": "Image", "obj_id": str(obj_id)})
mem_inst.track_object(frame, obj_id)

# Get stats
stats = collector.get_all_stats()["memory.images"]
```
