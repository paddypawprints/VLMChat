# Zero-Copy Camera Architecture Design

**Status**: Design Complete, Implementation Planned  
**Created**: 2025-12-08  
**Target**: Post label-based provenance & package refactoring

## Overview

Design for zero-copy video frame handling from GStreamer camera sources to VLM pipeline tasks, with automatic memory promotion on buffer pool exhaustion. Eliminates memcpy overhead while maintaining clean abstractions.

## Problem Statement

### Current Challenge
- VLM inference takes 2-10 seconds per frame
- Camera captures at 30 fps (33ms intervals)
- During one VLM inference: 60-300 frames captured
- Need bounded memory without dropping frames during processing
- Want zero-copy access for performance

### Key Insight
**GStreamer GstBuffer uses reference counting** - we can hold refs to prevent recycling without implementing custom allocators.

## Architecture

### Three-Phase Implementation Plan

#### Phase 1: Copy-Based (Initial Implementation)
**Status**: Start here for simplicity

```python
GStreamer buffer → memcpy → Pool buffer (numpy array)
```

**Characteristics**:
- Simple ownership model
- One memcpy per frame (~6.2 MB for 1080p)
- Cost: ~1% CPU core at 30fps
- Works everywhere, easy to debug

**When to use**: Initial implementation, validation, non-performance-critical

#### Phase 2: GstBuffer Reference Holding (Recommended)
**Status**: Sweet spot for most use cases

```python
GStreamer buffer → ref++ → Hold reference → Zero-copy numpy view
```

**Characteristics**:
- Zero bytes copied
- No custom allocator needed
- Standard GStreamer reference counting
- Works with any pipeline configuration
- Medium complexity

**When to use**: Production CPU pipelines, after Phase 1 validated

#### Phase 3: Custom GstAllocator (Advanced)
**Status**: Only if Phase 2 insufficient

```python
Pool allocates → GStreamer writes directly → True zero-copy
```

**Characteristics**:
- True zero-copy (GStreamer writes to our memory)
- Required for GPU-only paths (NVMM → CUDA)
- High complexity (implement GstAllocator)
- Platform-specific

**When to use**: GPU-accelerated pipelines, hardware encoder paths

## Core Design Components

### 1. Memory Backend Abstraction

```python
class MemoryBackend(ABC):
    """Abstract interface for different memory implementations"""
    
    @abstractmethod
    def allocate_buffer(self, width: int, height: int, channels: int) -> BufferHandle:
        """Allocate a buffer in backend-specific memory"""
        pass
    
    @abstractmethod
    def copy_from_gstreamer(self, gst_buffer, gst_caps, target: BufferHandle):
        """Copy from GStreamer buffer to our buffer (Phase 1)"""
        pass
    
    @abstractmethod
    def get_numpy_view(self, handle: BufferHandle) -> np.ndarray:
        """Get numpy-compatible view (may require copy for GPU)"""
        pass
    
    @abstractmethod
    def get_native_data(self, handle: BufferHandle) -> Any:
        """Get backend-native data (numpy, cupy, tensor, etc.)"""
        pass
    
    @abstractmethod
    def copy_buffer(self, source: BufferHandle) -> Any:
        """Create owned copy of buffer data"""
        pass
```

### 2. Buffer Pool with Auto-Promotion

```python
class BufferPool:
    """Reference-counted buffer pool with automatic promotion on eviction"""
    
    def __init__(self, num_buffers: int, width: int, height: int, 
                 backend: MemoryBackend):
        self.backend = backend
        self.buffers = [
            PooledBuffer(
                data=backend.allocate_buffer(width, height, 3),
                backend=backend,
                index=i
            )
            for i in range(num_buffers)
        ]
    
    def acquire(self) -> Optional[PooledBuffer]:
        """Get free buffer, or promote oldest if pool exhausted"""
        # Find free buffer
        for buf in self.buffers:
            if buf.refcount == 0:
                buf.refcount = 1
                return buf
        
        # Pool full - auto-promote oldest borrowed buffer
        oldest = min(
            (b for b in self.buffers if b.refcount > 0),
            key=lambda b: b.capture_time
        )
        
        self._promote_buffer(oldest)  # Copy to owned memory
        
        # Reclaim buffer
        oldest.refcount = 0
        oldest.valid = False
        oldest.refcount = 1
        oldest.valid = True
        return oldest
    
    def _promote_buffer(self, buffer: PooledBuffer):
        """Auto-promote all ImageContainers using this buffer"""
        for container in list(buffer.containers):
            container._promote_to_owned()
```

### 3. Transparent ImageContainer

```python
class ImageContainer(CachedItem):
    """
    Wraps either a pooled buffer (zero-copy) or owned data (copied)
    Automatically promotes on pool eviction
    """
    
    def __init__(self, pooled_buffer=None, owned_data=None):
        self._pooled = pooled_buffer  # Volatile, zero-copy
        self._owned = owned_data       # Stable, copied
        
        if self._pooled:
            self._pooled.register_container(self)
    
    def get_numpy(self) -> np.ndarray:
        """Get numpy array - transparent to caller"""
        if self._owned is not None:
            return self._owned  # Promoted/owned copy
        
        if self._pooled and self._pooled.valid:
            return self._pooled.backend.get_numpy_view(self._pooled.data)
        
        raise BufferRecycledError("Buffer recycled")
    
    def _promote_to_owned(self):
        """Called automatically by pool during eviction"""
        if self._owned is not None:
            return  # Already promoted
        
        if self._pooled and self._pooled.valid:
            self._owned = self._pooled.backend.copy_buffer(self._pooled.data)
            self._pooled.unregister_container(self)
            self._pooled = None
```

### 4. Cache as Virtual Ring Buffer

```python
class ItemCache:
    """Cache with per-label LRU eviction (virtual ring buffer)"""
    
    def __init__(self, max_images_per_label=30):
        self.max_images_per_label = max_images_per_label
        self.data: Dict[ContextDataType, Dict[str, List[CachedItem]]] = {}
    
    def add(self, data_type: ContextDataType, label: str, item: CachedItem):
        """Add with automatic FIFO eviction per label"""
        if data_type not in self.data:
            self.data[data_type] = {}
        if label not in self.data[data_type]:
            self.data[data_type][label] = []
        
        items = self.data[data_type][label]
        items.append(item)
        
        # Virtual ring buffer: keep last N images per label
        if data_type == ContextDataType.IMAGE:
            while len(items) > self.max_images_per_label:
                old_item = items.pop(0)
                old_item.release()  # Trigger GC
```

## Phase 2 Implementation Details (GstBuffer Ref Holding)

### GstBuffer Reference Counting

GStreamer's GstBuffer has built-in reference counting:

```python
buffer = sample.get_buffer()  # refcount = 1 (owned by sample)
buffer.ref()                   # refcount = 2 (we hold a ref)
# Sample released                → refcount = 1
# GStreamer CANNOT recycle until we call unref()
buffer.unref()                 # refcount = 0 → returned to GStreamer pool
```

### Pressure-Based Promotion Strategy

**Critical Insight**: Monitor pool fullness and proactively promote buffers BEFORE exhaustion to prevent frame drops.

```python
Pool Pressure Levels:
  0-50%:   HEALTHY    → No promotion needed
  50-75%:  MODERATE   → Monitor closely
  75-90%:  HIGH       → Proactive promotion (10% oldest buffers)
  90-100%: CRITICAL   → Aggressive promotion (25% oldest buffers)
```

**Promotion Triggers**:
1. **Reactive**: Pool exhausted (last resort)
2. **Proactive**: High water mark reached (preferred)
3. **Background**: Continuous monitoring thread (optimal)

### Zero-Copy Pool Implementation

```python
class GstBufferPool:
    """Pool with pressure-based promotion to prevent frame drops"""
    
    def __init__(self, max_buffers: int = 60,
                 high_water_mark: float = 0.75,
                 critical_mark: float = 0.90,
                 background_promotion: bool = True):
        self.active_buffers: List[PooledGstBuffer] = []
        self.max_buffers = max_buffers
        self.high_water_mark = high_water_mark      # 75% → start promoting
        self.critical_mark = critical_mark          # 90% → aggressive promotion
        self.lock = threading.Lock()
        self.metrics = defaultdict(int)
        
        # Optional background promotion thread
        self.background_promotion = background_promotion
        if background_promotion:
            self.promotion_thread = threading.Thread(
                target=self._promotion_worker,
                daemon=True
            )
            self.promotion_thread.start()
    
    def get_pressure(self) -> float:
        """Return pool utilization (0.0 to 1.0)"""
        with self.lock:
            return len(self.active_buffers) / self.max_buffers
    
    def add_buffer(self, gst_buffer, caps) -> Optional[PooledGstBuffer]:
        """Add buffer with pressure-aware promotion"""
        with self.lock:
            pressure = len(self.active_buffers) / self.max_buffers
            
            # Proactive promotion based on pressure
            if pressure >= self.high_water_mark:
                self._promote_by_pressure(pressure)
            
            # If still at capacity after promotion, evict oldest
            if len(self.active_buffers) >= self.max_buffers:
                oldest = min(self.active_buffers, key=lambda b: b.capture_time)
                self._promote_buffer(oldest)
                self.active_buffers.remove(oldest)
                self.metrics['frames_evicted'] += 1
            
            # Add new buffer
            gst_buffer.ref()  # Hold reference - GStreamer can't recycle
            pooled = PooledGstBuffer(gst_buffer, caps)
            self.active_buffers.append(pooled)
            return pooled
    
    def _promote_by_pressure(self, pressure: float):
        """Proactively promote buffers based on pool pressure"""
        if pressure < self.high_water_mark:
            return  # No pressure, no action
        
        # Determine how many to promote
        if pressure >= self.critical_mark:
            # Critical: promote oldest 25%
            num_to_promote = max(1, len(self.active_buffers) // 4)
            self.metrics['critical_promotions'] += 1
        else:
            # High: promote oldest 10%
            num_to_promote = max(1, len(self.active_buffers) // 10)
            self.metrics['pressure_promotions'] += 1
        
        # Find candidates for promotion
        # Use policy to select best candidates
        policy = PromotionPolicy()
        candidates = policy.select_candidates(
            self.active_buffers, 
            num_to_promote
        )
        
        for buffer in candidates:
            self._promote_buffer(buffer)
            self.active_buffers.remove(buffer)
            self.metrics['frames_promoted'] += 1
    
    def _promotion_worker(self):
        """Background thread that promotes buffers preemptively"""
        while True:
            try:
                time.sleep(0.1)  # Check every 100ms
                
                pressure = self.get_pressure()
                
                if pressure >= self.high_water_mark:
                    with self.lock:
                        candidates = sorted(
                            [b for b in self.active_buffers if b.containers],
                            key=lambda b: b.capture_time
                        )
                    
                    # Promote oldest in background
                    if candidates:
                        oldest = candidates[0]
                        self._promote_buffer_async(oldest)
                        
            except Exception as e:
                logging.error(f"Promotion worker error: {e}")
    
    def _promote_buffer_async(self, buffer: PooledGstBuffer):
        """Promote buffer in background without blocking camera thread"""
        for container in list(buffer.containers):
            if not container._owned:
                # Do the copy in background thread
                view = buffer.get_numpy_view()
                owned_copy = view.copy()
                
                # Update container atomically
                with container._lock:
                    if not container._owned:  # Double-check
                        container._owned = owned_copy
                        container._pooled_gst = None
                        buffer.unregister_container(container)

class PooledGstBuffer:
    """Wrapper around held GstBuffer reference"""
    
    def __init__(self, gst_buffer, caps):
        self.gst_buffer = gst_buffer
        self.caps = caps
        self.capture_time = time.monotonic()
        self.last_access_time = time.monotonic()
        self.containers: List[ImageContainer] = []
        self._map_info = None
        self._numpy_view = None
    
    def get_numpy_view(self) -> np.ndarray:
        """Zero-copy numpy view via GstBuffer.map()"""
        self.last_access_time = time.monotonic()  # Track access for policy
        
        if self._numpy_view is not None:
            return self._numpy_view
        
        success, map_info = self.gst_buffer.map(Gst.MapFlags.READ)
        if not success:
            raise RuntimeError("Failed to map GstBuffer")
        
        struct = self.caps.get_structure(0)
        width = struct.get_value("width")
        height = struct.get_value("height")
        
        # Zero-copy numpy view of GstBuffer memory
        self._numpy_view = np.ndarray(
            shape=(height, width, 3),
            dtype=np.uint8,
            buffer=map_info.data
        )
        self._map_info = map_info
        return self._numpy_view
    
    def register_container(self, container):
        self.containers.append(container)
    
    def unregister_container(self, container):
        if container in self.containers:
            self.containers.remove(container)
    
    def __del__(self):
        """Release GstBuffer when pool buffer destroyed"""
        if self._map_info:
            self.gst_buffer.unmap(self._map_info)
        if self.gst_buffer:
            self.gst_buffer.unref()  # Now GStreamer can recycle

class PromotionPolicy:
    """Smart selection of buffers to promote based on usage patterns"""
    
    def select_candidates(self, buffers: List[PooledGstBuffer], 
                          num_needed: int) -> List[PooledGstBuffer]:
        """Select buffers to promote using scoring algorithm"""
        scored = []
        for buf in buffers:
            if not buf.containers:
                continue  # No one using it, will be freed naturally
            
            score = self._promotion_score(buf)
            scored.append((score, buf))
        
        # Sort by score (higher = more urgent to promote)
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [buf for score, buf in scored[:num_needed]]
    
    def _promotion_score(self, buffer: PooledGstBuffer) -> float:
        """Calculate urgency score for promotion"""
        now = time.monotonic()
        age = now - buffer.capture_time
        time_since_access = now - buffer.last_access_time
        
        score = 0.0
        
        # Older buffers score higher (more likely to be held long-term)
        score += age * 10
        
        # Buffers with many containers score higher (widely referenced)
        score += len(buffer.containers) * 5
        
        # Buffers that have been mapped score lower
        # (numpy view cached, may be accessed again soon)
        if buffer._numpy_view is not None:
            score -= 20
        
        # Buffers accessed recently score lower
        # (active use, may need zero-copy performance)
        if time_since_access < 0.5:  # Last 500ms
            score -= 30
        
        return score
```

### Camera Integration with Pressure Monitoring

```python
class GStreamerCamera:
    """Camera source with zero-copy frame handling and pressure monitoring"""
    
    def __init__(self, sensor_id: int = 0):
        self.gst_pool = GstBufferPool(
            max_buffers=60,
            high_water_mark=0.75,
            critical_mark=0.90,
            background_promotion=True
        )
        self.metrics = defaultdict(int)
        
        # Standard GStreamer pipeline - no custom allocator needed
        pipeline_str = f"""
            nvarguscamerasrc sensor-id={sensor_id} !
            video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 !
            nvvidconv !
            video/x-raw, format=BGR !
            appsink name=sink emit-signals=true max-buffers=1 drop=true
        """
        
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name("sink")
        self.appsink.connect("new-sample", self._on_new_sample)
    
    def _on_new_sample(self, appsink):
        """GStreamer callback - runs in GStreamer thread"""
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK
        
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        
        # Check pool pressure BEFORE adding
        pressure = self.gst_pool.get_pressure()
        
        if pressure >= 0.95:
            # Critical pressure - log warning
            self.metrics['high_pressure_frames'] += 1
            logging.warning(f"Pool pressure critical: {pressure:.2%}")
        
        # Add buffer (pool handles promotion internally)
        pooled_gst = self.gst_pool.add_buffer(buffer, caps)
        
        if pooled_gst is None:
            # Pool exhausted even after promotion - frame dropped
            self.metrics['frames_dropped_pool_full'] += 1
            logging.error("Frame dropped: pool exhausted after promotion")
            return Gst.FlowReturn.OK
        
        # Log pressure for monitoring (debug level)
        if pressure >= 0.75:
            logging.debug(f"Pool pressure: {pressure:.2%}, "
                         f"{len(self.gst_pool.active_buffers)} buffers active")
        
        # Create ImageContainer wrapping the GstBuffer
        img = ImageContainer(pooled_gst_buffer=pooled_gst)
        
        # Emit to pipeline
        self._emit_to_pipeline(img)
        
        return Gst.FlowReturn.OK
    
    def get_pool_status(self) -> dict:
        """Get real-time pool health metrics"""
        pressure = self.gst_pool.get_pressure()
        
        return {
            'pressure': pressure,
            'status': self._pressure_status(pressure),
            'active_buffers': len(self.gst_pool.active_buffers),
            'max_buffers': self.gst_pool.max_buffers,
            'frames_promoted': self.gst_pool.metrics['frames_promoted'],
            'frames_evicted': self.gst_pool.metrics['frames_evicted'],
            'pressure_promotions': self.gst_pool.metrics['pressure_promotions'],
            'critical_promotions': self.gst_pool.metrics['critical_promotions'],
            'frames_dropped': self.metrics['frames_dropped_pool_full'],
            'high_pressure_events': self.metrics['high_pressure_frames'],
        }
    
    def _pressure_status(self, pressure: float) -> str:
        if pressure < 0.5:
            return "HEALTHY"
        elif pressure < 0.75:
            return "MODERATE"
        elif pressure < 0.90:
            return "HIGH"
        else:
            return "CRITICAL"
```

## Memory Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Camera Hardware                                             │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ GStreamer Pipeline (nvarguscamerasrc → nvvidconv → appsink)│
│   - NVMM (GPU) → CPU memory                                 │
│   - GstBuffer allocated from GStreamer's pool               │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
         _on_new_sample() callback
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 1 (Copy):                                             │
│   memcpy → Pool Buffer (numpy array)                        │
│                                                              │
│ Phase 2 (Zero-Copy):                                        │
│   gst_buffer.ref() → Hold reference                         │
│   buffer.map() → Zero-copy numpy view                       │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ ImageContainer (pooled_buffer or pooled_gst_buffer)         │
│   - Tasks call get_numpy() → zero-copy access               │
│   - Auto-promotion on pool eviction                         │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ Pipeline Tasks (YOLO, VLM, etc.)                            │
│   - Read frames via get_numpy()                             │
│   - Unaware of memory tier (pooled vs owned)                │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
        Pool buffer needed?
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ Auto-Promotion                                              │
│   - Pool calls _promote_to_owned()                          │
│   - Copy to owned memory                                    │
│   - Release pool buffer / GstBuffer ref                     │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ Cache (Virtual Ring Buffer)                                 │
│   - Keeps last N images per label                           │
│   - FIFO eviction on overflow                               │
└─────────────────────────────────────────────────────────────┘
```

## Threading Model

```
┌─────────────────────────┐
│  GStreamer Thread       │  (hardware callback thread)
│                         │
│  _on_new_sample()       │  ← GStreamer calls on every frame
│     ↓                   │
│  pool.add_buffer()      │  (ref++ or memcpy)
│     ↓                   │
│  frame_queue.put()      │  → Handoff to pipeline thread
└─────────────────────────┘
                ↓
┌─────────────────────────┐
│  Pipeline Thread        │  (DSL execution)
│                         │
│  camera_task.execute()  │
│     ↓                   │
│  frame_queue.get()      │  ← Pull frame
│     ↓                   │
│  ctx.add_data(...)      │
│     ↓                   │
│  Tasks process frame    │  (zero-copy access)
└─────────────────────────┘
```

### Configuration

```json
{
  "camera": {
    "sensor_id": 0,
    "buffer_pool": {
      "size": 60,
      "width": 1920,
      "height": 1080,
      "backend": "cpu_zerocopy",
      "high_water_mark": 0.75,
      "critical_mark": 0.90,
      "background_promotion": true,
      "promotion_policy": "age_and_usage"
    }
  },
  "cache": {
    "max_images_per_label": 30,
    "enable_auto_promotion": true
  }
}
```

## Performance Characteristics

### Phase 1 (Copy)
- **Memory copies**: 1 per frame (GStreamer → Pool)
- **Copy cost**: ~6.2 MB @ 1080p, ~1% CPU at 30fps
- **Memory overhead**: Pool + Cache
- **Latency**: ~0.3ms per frame (memcpy time)

### Phase 2 (Zero-Copy)
- **Memory copies**: 0 (until auto-promotion)
- **Copy cost**: Only on eviction (rare)
- **Memory overhead**: Pool holds GstBuffer refs
- **Latency**: ~0.01ms (ref++ operation)

### Pool Sizing
```
Buffer size: 1920 × 1080 × 3 = 6.2 MB
Pool of 60 buffers: ~372 MB

At 30 fps with 5 sec VLM inference:
  Frames during inference: 150
  Need pool ≥ 150 to avoid eviction
  
Practical: 60 buffers (2 seconds @ 30fps)
  Auto-promotion triggers after 2 sec
  Acceptable for most VLM use cases
```

## Metrics & Observability

### Real-Time Metrics

```python
Tracked metrics:
- frames_captured: Total from camera
- frames_zero_copy: Accessed directly from pool
- frames_promoted: Auto-copied during eviction
- frames_dropped_pool_full: Lost due to pool exhaustion
- pressure_promotions: Proactive promotions at high water mark
- critical_promotions: Aggressive promotions at critical mark
- buffer_utilization: Peak/avg borrowed buffers (pressure)
- promotion_rate: Promotions per second
- gst_buffer_held: Number of GstBuffer refs held (Phase 2)
- high_pressure_events: Frames captured during critical pressure
- pool_pressure_history: Time series of pressure values

Alerts:
- pressure > 90% for >5 sec: Pool undersized, increase size
- critical_promotions > 5/min: VLM too slow or pool too small
- frames_dropped_pool_full > 0: Promotion strategy failed
- promotion_rate > 10/sec: Background promotion not keeping up
- pressure oscillating: Tune high_water_mark threshold
```

### Monitoring Dashboard Example

```python
Pool Status: HIGH (78% full)
  Active buffers: 47/60
  Frames captured: 1,247
  Frames zero-copy: 1,100 (88%)
  Frames promoted: 145 (142 pressure, 3 critical)
  Frames evicted: 2
  Frames dropped: 0
  
Pressure history (last 10 sec):
  [0.65, 0.68, 0.72, 0.78, 0.75, 0.73, 0.76, 0.80, 0.78, 0.77]
  
Recommendation: Pool healthy, proactive promotion working
Next action: Monitor for 1 min, no changes needed
```

### Tuning Guidelines

**If pressure stays < 50%**: Pool oversized, can reduce
**If pressure 50-75%**: Optimal range, no action
**If pressure 75-90%**: Monitor promotion effectiveness
**If pressure > 90%**: Either increase pool size OR reduce VLM parallelism
**If frames dropped > 0**: Immediate action - increase pool or add background promotion

## Benefits Summary

### Zero-Copy Advantages
1. **No memcpy overhead** - ~1% CPU saved at 30fps
2. **Lower latency** - ref++ vs memcpy
3. **Simple implementation** - No custom allocator (Phase 2)
4. **Works with any pipeline** - Standard GStreamer

### Auto-Promotion Advantages
1. **Transparent to tasks** - No `.to_owned()` calls needed
2. **Rare frame loss** - Only under extreme load
3. **Bounded memory** - Pool size limits total memory
4. **Graceful degradation** - Promotes oldest frames first

### Virtual Ring Buffer Advantages
1. **Automatic retention** - Keep last N frames per label
2. **Memory bounded** - FIFO eviction prevents growth
3. **Multi-source support** - Per-label limits
4. **Works with provenance** - Label-aware caching

## Implementation Checklist

### Phase 1: Copy-Based (Initial)
- [ ] Implement `CPUMemoryBackend` with `copy_from_gstreamer()`
- [ ] Implement `BufferPool` with numpy arrays
- [ ] Implement `ImageContainer` with pooled/owned modes
- [ ] Implement auto-promotion on pool exhaustion
- [ ] Add GStreamer camera with copy path
- [ ] Add threading (GStreamer → Queue → Pipeline)
- [ ] Test with YOLO detector
- [ ] Test with VLM (long inference time)
- [ ] Validate auto-promotion triggers
- [ ] Add metrics collection

### Phase 2: Zero-Copy (Optimization)
- [ ] Implement `GstBufferPool` with reference holding
- [ ] Implement `PooledGstBuffer` wrapper with access tracking
- [ ] Add pressure monitoring (`get_pressure()` method)
- [ ] Implement `_promote_by_pressure()` for proactive promotion
- [ ] Implement `PromotionPolicy` with scoring algorithm
- [ ] Add background promotion thread (optional)
- [ ] Implement `_promotion_worker()` for async promotion
- [ ] Update `ImageContainer` to support `pooled_gst_buffer`
- [ ] Implement `get_numpy_view()` with `buffer.map()`
- [ ] Test GstBuffer ref counting behavior
- [ ] Validate zero-copy numpy views
- [ ] Test pressure thresholds (high water, critical)
- [ ] Validate proactive promotion prevents drops
- [ ] Test background promotion thread
- [ ] Benchmark: compare Phase 1 vs Phase 2 performance
- [ ] Test auto-promotion with GstBuffer refs
- [ ] Add Phase 2 metrics (pressure, promotions, etc.)
- [ ] Add pressure monitoring dashboard
- [ ] Tune high_water_mark and critical_mark thresholds
- [ ] Make backend selectable via config

### Phase 3: Custom Allocator (Future)
- [ ] Research GstAllocator implementation
- [ ] Implement custom allocator for CPU pool
- [ ] Test with GStreamer pipeline integration
- [ ] Implement NVMM allocator for Jetson GPU
- [ ] Implement CUDA backend with cupy
- [ ] Test GPU → GPU zero-copy path
- [ ] Benchmark GPU pipeline performance

## Integration Points

### With Label-Based Provenance
```python
# Camera produces labeled frames
ctx.add_data(IMAGE, "cabview", frame)   # Front camera
ctx.add_data(IMAGE, "roadview", frame)  # Rear camera

# Cache maintains per-label ring buffers
cache.max_images_per_label = 30  # Last 30 frames each
```

### With Format Conversion
```python
# Format conversion triggers promotion
frame = ctx.get_data(IMAGE, "camera")[0]  # Pooled (zero-copy)
jpeg = frame.get_format(ImageFormat.JPEG)  # Promotes + converts
```

### With Detection Merge
```python
# Detections reference frames - auto-promotion handles long holds
frame = ctx.get_data(IMAGE, "camera")[0]
detections = yolo.detect(frame.get_numpy())  # Zero-copy

# Frame held for detection merge (may get promoted if VLM slow)
ctx.add_data(DETECTION, "objects", detections)
```

## Testing Strategy

### Unit Tests
- Pool acquire/release cycles
- Auto-promotion triggers correctly
- Pressure calculation accuracy
- Proactive promotion at thresholds
- Promotion policy scoring algorithm
- Background promotion thread behavior
- GstBuffer ref counting (Phase 2)
- ImageContainer mode transitions
- Cache LRU eviction per label

### Integration Tests
- Camera → Pool → Task pipeline
- Long VLM inference (simulate 5+ sec)
- Pool exhaustion scenarios
- Pressure-based promotion effectiveness
- Background promotion prevents drops
- Multi-camera with labels
- Format conversion during zero-copy
- Sustained high load (pressure > 75% for 60 sec)

### Performance Tests
- Benchmark memcpy overhead (Phase 1)
- Benchmark ref counting overhead (Phase 2)
- Benchmark promotion overhead (proactive vs reactive)
- Background thread CPU usage
- Memory usage under load
- Frame drop rates at various pool sizes
- Promotion frequency vs pool size
- Pressure response time (time from 75% → promotion)
- Zero-copy effectiveness (% frames never promoted)

### Platform Tests
- Jetson with NVMM memory
- Generic Linux with V4L2
- Raspberry Pi camera
- USB webcam

## Future Enhancements

1. **GPU Memory Pools** - cupy arrays for CUDA tasks
2. **dmabuf Support** - Zero-copy to hardware encoder
3. **Multiple Backends** - CPU/GPU hybrid pipelines
4. **Dynamic Pool Sizing** - Adjust based on load
5. **Compression** - JPEG compress old frames instead of dropping
6. **Tiered Storage** - RAM → Disk for long-term archival
7. **Network Streaming** - Zero-copy to RTSP encoder

## References

- GStreamer Buffer Management: https://gstreamer.freedesktop.org/documentation/plugin-development/advanced/allocation.html
- GStreamer Memory API: https://gstreamer.freedesktop.org/documentation/gstreamer/gstmemory.html
- NumPy Buffer Interface: https://numpy.org/doc/stable/reference/arrays.interface.html
- VLMChat Label Provenance: `ADAPTIVE_SCHEDULING_DESIGN.md`
- Package Structure: `src/vlmchat/pipeline/`

## Notes

- **Start with Phase 1** - Validate architecture with simple copy approach
- **Phase 2 is the sweet spot** - Zero-copy without allocator complexity
- **Phase 3 only if needed** - Custom allocator adds significant complexity
- **GstBuffer ref counting insight** - Key to zero-copy without custom allocator
- **Auto-promotion is critical** - Prevents `BufferRecycledError` for tasks
- **Virtual ring buffer in cache** - Natural fit with per-label eviction
