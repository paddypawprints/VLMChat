# Memory Management Design: Reference Counting & Multiprocessing

## Overview

This document outlines the design for implementing reference counting and garbage collection in the pipeline system, with a clear path to multiprocessing support.

## Current Architecture

### Context Model
- **Shallow copy lists** on split: Each branch gets own list, items are shared references
- **ID-based deduplication** on merge: Uses `item.id` or `id(item)` to prevent duplicates
- **No mutable/immutable flags**: All data treated uniformly
- **Direct access**: Tasks access `context.data[type]` lists directly

### Memory Lifecycle
1. Context created with initial data
2. Split creates shallow copies → items shared across branches
3. Tasks modify lists (add/remove items)
4. Merge combines branches, deduplicates by ID
5. Python GC cleans up contexts when no references remain
6. **Problem**: Large objects (Images, embeddings) stay in memory until ALL branches complete

## Phase 1: Reference Counting with Transparent Proxy

### Design: RefCountedList

```python
class RefCountedList(list):
    """
    List subclass that tracks references to items automatically.
    Completely transparent to tasks - they use normal list operations.
    """
    def __init__(self, items, runner=None):
        super().__init__(items)
        self._runner = runner
        # Acquire refs for initial items
        if self._runner:
            for item in items:
                self._runner._track_object(item)
    
    def append(self, item):
        """Task calls: detections.append(new_det)"""
        if self._runner:
            self._runner._track_object(item)  # +1 ref
        super().append(item)
    
    def extend(self, items):
        """Task calls: detections.extend([det1, det2])"""
        if self._runner:
            for item in items:
                self._runner._track_object(item)
        super().extend(items)
    
    def __setitem__(self, key, value):
        """Task calls: detections[0] = new_det"""
        if self._runner:
            old_item = self[key]
            self._runner._release_object(old_item)  # -1 ref on old
            self._runner._track_object(value)  # +1 ref on new
        super().__setitem__(key, value)
    
    def remove(self, item):
        """Task calls: detections.remove(det)"""
        if self._runner:
            self._runner._release_object(item)  # -1 ref
        super().remove(item)
    
    def pop(self, index=-1):
        """Task calls: det = detections.pop()"""
        item = super().pop(index)
        if self._runner:
            self._runner._release_object(item)  # -1 ref
        return item
    
    def clear(self):
        """Task calls: detections.clear()"""
        if self._runner:
            for item in self:
                self._runner._release_object(item)
        super().clear()
    
    def __delitem__(self, key):
        """Task calls: del detections[0]"""
        if self._runner:
            self._runner._release_object(self[key])
        super().__delitem__(key)
    
    # Shallow copy for split() - creates new list, shares items
    def __copy__(self):
        new_list = RefCountedList([], self._runner)
        for item in self:
            new_list.append(item)  # Automatically tracks refs
        return new_list
```

### PipelineRunner Integration

```python
class PipelineRunner:
    def __init__(self, ...):
        # Reference tracking
        self.object_refs: Dict[int, int] = {}  # object_id -> ref_count
        self.object_registry: Dict[int, Any] = {}  # object_id -> actual object
        self.zero_ref_since: Dict[int, int] = {}  # obj_id -> cycle when hit zero
        self._ref_lock = threading.Lock()
        self._gc_cycle = 0
        
    def _track_object(self, obj: Any):
        """Start tracking a large object."""
        obj_id = id(obj)
        with self._ref_lock:
            if obj_id not in self.object_refs:
                self.object_refs[obj_id] = 0
                self.object_registry[obj_id] = obj
            self.object_refs[obj_id] += 1
            # Remove from zero-ref candidates if it was there
            self.zero_ref_since.pop(obj_id, None)
    
    def _release_object(self, obj: Any):
        """Release reference to tracked object."""
        obj_id = id(obj)
        with self._ref_lock:
            if obj_id in self.object_refs:
                self.object_refs[obj_id] -= 1
                if self.object_refs[obj_id] == 0:
                    # Mark for GC
                    self.zero_ref_since[obj_id] = self._gc_cycle
```

### Context Integration

```python
class Context:
    def __init__(self):
        self.data: Dict[ContextDataType, RefCountedList] = {}
        self.collector = None
        self.config = None
        self.pipeline_runner = None  # Set by runner for ref counting
    
    def split(self, num_branches: int) -> List['Context']:
        """Split with automatic reference tracking."""
        contexts = []
        for _ in range(num_branches):
            ctx = Context()
            ctx.collector = self.collector
            ctx.config = self.config
            ctx.pipeline_runner = self.pipeline_runner
            
            # Shallow copy lists - RefCountedList handles ref tracking
            for data_type, ref_list in self.data.items():
                ctx.data[data_type] = ref_list.__copy__()  # +1 ref per item
            
            contexts.append(ctx)
        return contexts
    
    def __setitem__(self, data_type: ContextDataType, value):
        """Intercept list replacement to maintain tracking."""
        # Release old list
        if data_type in self.data:
            self.data[data_type].clear()  # Releases all refs
        
        # Wrap new list
        if isinstance(value, list) and not isinstance(value, RefCountedList):
            value = RefCountedList(value, self.pipeline_runner)
        
        self.data[data_type] = value
```

## Phase 1: Safe Garbage Collection

### GC Strategy

```python
class PipelineRunner:
    GRACE_PERIOD_CYCLES = 2  # Wait 2 GC cycles before deleting
    
    def garbage_collect(self):
        """
        Collect objects with zero tracked references.
        
        Safety measures:
        1. Grace period - wait multiple cycles at ref_count=0
        2. Python validation - check sys.getrefcount() matches
        3. Selective cleanup - only manually clean large objects
        """
        self._gc_cycle += 1
        to_delete = []
        
        for obj_id, tracked_count in list(self.object_refs.items()):
            if tracked_count == 0:
                # Check grace period
                cycles_at_zero = self._gc_cycle - self.zero_ref_since.get(obj_id, self._gc_cycle)
                if cycles_at_zero < self.GRACE_PERIOD_CYCLES:
                    logger.debug(f"GC: Object {obj_id} at zero refs for {cycles_at_zero} cycles, "
                               f"waiting for grace period")
                    continue
                
                # Validate with Python
                obj = self.object_registry[obj_id]
                actual_refs = sys.getrefcount(obj) - 2  # -1 registry, -1 getrefcount temp
                
                if actual_refs != 0:
                    logger.warning(f"GC: Object {obj_id} has leaked refs! "
                                 f"Tracked=0, Python={actual_refs}")
                    if self.debug:
                        self._debug_leaked_refs(obj)
                    continue
                
                # Safe to delete
                to_delete.append(obj_id)
        
        # Cleanup
        for obj_id in to_delete:
            obj = self.object_registry[obj_id]
            self._cleanup_object(obj)
            del self.object_refs[obj_id]
            del self.object_registry[obj_id]
            del self.zero_ref_since[obj_id]
            
            if self.collector:
                self.collector.data_point("gc.objects_freed", {}, 1)
        
        if to_delete:
            logger.info(f"GC: Freed {len(to_delete)} objects")
    
    def _cleanup_object(self, obj: Any):
        """Manual cleanup for large objects."""
        if isinstance(obj, Image.Image):
            obj.close()
            logger.debug(f"GC: Closed PIL Image")
        elif isinstance(obj, np.ndarray):
            # Let Python GC handle numpy arrays
            pass
        # Let Python GC handle Detections and other small objects
    
    def _debug_leaked_refs(self, obj: Any):
        """Debug helper to find leaked references."""
        import gc
        referrers = gc.get_referrers(obj)
        logger.debug(f"Object is referenced by {len(referrers)} objects:")
        for ref in referrers:
            if isinstance(ref, dict):
                for k, v in ref.items():
                    if v is obj or (isinstance(v, list) and obj in v):
                        logger.debug(f"  - Found in dict key: {k}")
            elif isinstance(ref, list):
                logger.debug(f"  - Found in list of length {len(ref)}")
            elif isinstance(ref, Cursor):
                logger.debug(f"  - Found in Cursor #{ref.id}")
            else:
                logger.debug(f"  - Found in {type(ref).__name__}")
```

### GC Invocation Points

```python
class PipelineRunner:
    def run(self, context: Context) -> Context:
        # ... existing setup ...
        
        # Main execution loop
        while task_queue or ready_queue:
            # ... execute batch ...
            
            # SAFE POINT: Between batches, no active task execution
            if self._gc_cycle % 5 == 0:  # Run GC every 5 cycles
                self.garbage_collect()
        
        # Final cleanup
        self.garbage_collect()
        
        return final_context
```

## Phase 2: Context Handles (Multiprocessing Foundation)

### ContextHandle Design

```python
@dataclass
class ContextHandle:
    """
    Serializable reference to context data in shared memory.
    Enables zero-copy transfer between processes.
    """
    handle_id: str
    data_refs: Dict[ContextDataType, List[str]]  # Map type → SHM names
    metadata: Dict[str, Any]  # Shapes, dtypes, class info
    
    def serialize(self) -> bytes:
        """Pickle for sending to worker process."""
        return pickle.dumps(self)
    
    @staticmethod
    def deserialize(data: bytes) -> 'ContextHandle':
        return pickle.loads(data)

# Example:
handle = ContextHandle(
    handle_id="ctx_001",
    data_refs={
        ContextDataType.IMAGE: ["shm_img_001"],
        ContextDataType.DETECTIONS: ["shm_det_001", "shm_det_002"]
    },
    metadata={
        "shm_img_001": {"shape": (480, 640, 3), "dtype": "uint8", "mode": "RGB"},
        "shm_det_001": {"class": "Detection", "id": 0}
    }
)
```

### SharedMemoryRegistry

```python
class SharedMemoryRegistry:
    """
    Central registry of all shared memory segments.
    Handles serialization, allocation, and reference counting.
    """
    
    def __init__(self):
        self.segments: Dict[str, SharedMemory] = {}
        self.metadata: Dict[str, Dict] = {}
        self.lock = mp.Lock()
        
        # Cross-process reference counting
        self.manager = mp.Manager()
        self.ref_counts = self.manager.dict()
    
    def allocate(self, obj: Any) -> str:
        """
        Put object into shared memory, return handle name.
        
        Handles serialization based on type:
        - PIL Images → numpy array with metadata
        - Numpy arrays → raw bytes with shape/dtype
        - Detections → pickle with class info
        - Other → pickle
        """
        shm_name = f"shm_{uuid.uuid4().hex[:8]}"
        
        # Serialize based on type
        if isinstance(obj, Image.Image):
            arr = np.array(obj)
            data = arr.tobytes()
            metadata = {
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "mode": obj.mode,
                "type": "Image"
            }
        elif isinstance(obj, np.ndarray):
            data = obj.tobytes()
            metadata = {
                "shape": obj.shape,
                "dtype": str(obj.dtype),
                "type": "ndarray"
            }
        elif isinstance(obj, Detection):
            data = pickle.dumps(obj)
            metadata = {
                "class": "Detection",
                "id": obj.id,
                "type": "pickle"
            }
        else:
            data = pickle.dumps(obj)
            metadata = {
                "class": type(obj).__name__,
                "type": "pickle"
            }
        
        # Create shared memory
        shm = SharedMemory(create=True, size=len(data), name=shm_name)
        shm.buf[:len(data)] = data
        
        with self.lock:
            self.segments[shm_name] = shm
            self.ref_counts[shm_name] = 1
            self.metadata[shm_name] = metadata
        
        logger.debug(f"Allocated {len(data)} bytes to {shm_name}")
        return shm_name
    
    def acquire(self, shm_name: str):
        """Increment reference count."""
        with self.lock:
            if shm_name in self.ref_counts:
                self.ref_counts[shm_name] += 1
    
    def release(self, shm_name: str):
        """Decrement reference count, cleanup if zero."""
        with self.lock:
            if shm_name not in self.ref_counts:
                return
            
            self.ref_counts[shm_name] -= 1
            
            if self.ref_counts[shm_name] == 0:
                # Cleanup
                shm = self.segments[shm_name]
                shm.close()
                shm.unlink()
                
                del self.segments[shm_name]
                del self.ref_counts[shm_name]
                del self.metadata[shm_name]
                
                logger.debug(f"Freed shared memory {shm_name}")
    
    def materialize(self, shm_name: str) -> Any:
        """Reconstruct object from shared memory."""
        if shm_name not in self.segments:
            # Open existing shared memory
            shm = SharedMemory(name=shm_name)
        else:
            shm = self.segments[shm_name]
        
        metadata = self.metadata[shm_name]
        
        if metadata["type"] == "Image":
            # Reconstruct PIL Image
            arr = np.frombuffer(
                shm.buf,
                dtype=np.dtype(metadata["dtype"])
            ).reshape(metadata["shape"])
            return Image.fromarray(arr, mode=metadata["mode"])
        
        elif metadata["type"] == "ndarray":
            # Reconstruct numpy array
            arr = np.frombuffer(
                shm.buf,
                dtype=np.dtype(metadata["dtype"])
            ).reshape(metadata["shape"])
            return arr.copy()  # Copy so shared mem can be released
        
        elif metadata["type"] == "pickle":
            # Unpickle object
            return pickle.loads(bytes(shm.buf))
        
        else:
            raise ValueError(f"Unknown type: {metadata['type']}")
```

## Phase 3: Process Pool Execution

### Worker Process Function

```python
def _run_task_in_process(
    task_descriptor: Dict[str, Any],
    context_handle: ContextHandle,
    registry_address: Tuple[str, int]
) -> ContextHandle:
    """
    Worker function that runs in separate process.
    
    Args:
        task_descriptor: How to recreate the task (class, params)
        context_handle: Reference to input context in shared memory
        registry_address: How to connect to shared memory registry
    
    Returns:
        New context handle with result data
    """
    # 1. Connect to shared memory registry
    registry_client = SharedMemoryRegistryClient(registry_address)
    
    # 2. Materialize context from shared memory
    context = Context()
    for data_type, shm_names in context_handle.data_refs.items():
        items = []
        for shm_name in shm_names:
            obj = registry_client.materialize(shm_name)
            items.append(obj)
        context.data[data_type] = items
    
    # 3. Recreate task from descriptor
    task_class = _task_registry[task_descriptor["type"]]
    task = task_class(task_id=task_descriptor["task_id"])
    task.configure(**task_descriptor.get("params", {}))
    
    # 4. Run task
    result_context = task.run(context)
    
    # 5. Serialize result context to shared memory
    result_handle = ContextHandle(
        handle_id=f"ctx_{uuid.uuid4().hex[:8]}",
        data_refs={},
        metadata={}
    )
    
    for data_type, items in result_context.data.items():
        shm_names = []
        for item in items:
            shm_name = registry_client.allocate(item)
            shm_names.append(shm_name)
        result_handle.data_refs[data_type] = shm_names
    
    # 6. Release input context references
    for shm_names in context_handle.data_refs.values():
        for shm_name in shm_names:
            registry_client.release(shm_name)
    
    return result_handle
```

### Runtime Integration

```python
class PipelineRunner:
    def __init__(self, ..., use_multiprocessing=False):
        self.use_multiprocessing = use_multiprocessing
        
        if use_multiprocessing:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
            self.shm_registry = SharedMemoryRegistry()
            # Start registry server in separate process
            self.registry_server = mp.Process(
                target=run_registry_server,
                args=(self.shm_registry,)
            )
            self.registry_server.start()
            self.registry_address = ("localhost", 9999)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            self.shm_registry = None
        
        # ... rest of init ...
    
    def _run_task_with_metrics(self, cursor: Cursor, submission_time: float = None) -> Context:
        """Run task - handles both thread and process execution."""
        
        if self.use_multiprocessing:
            # MULTIPROCESSING PATH
            return self._run_task_multiprocess(cursor, submission_time)
        else:
            # THREADING PATH (existing code)
            task = cursor.current_task
            context = cursor.context
            # ... existing thread execution ...
            return task.run(context)
    
    def _run_task_multiprocess(self, cursor: Cursor, submission_time: float = None) -> Context:
        """Execute task in separate process."""
        
        # 1. Serialize context to shared memory
        handle = self._contextualize(cursor.context)
        
        # 2. Create task descriptor
        task_desc = {
            "type": type(cursor.current_task).__name__,
            "task_id": cursor.current_task.task_id,
            "params": self._get_task_params(cursor.current_task)
        }
        
        # 3. Submit to process pool
        future = self.executor.submit(
            _run_task_in_process,
            task_desc,
            handle,
            self.registry_address
        )
        
        # 4. Get result handle
        result_handle = future.result()
        
        # 5. Materialize result context
        result_context = self._materialize_context(result_handle)
        
        return result_context
    
    def _contextualize(self, context: Context) -> ContextHandle:
        """Convert Context to ContextHandle by moving data to shared memory."""
        handle = ContextHandle(
            handle_id=f"ctx_{uuid.uuid4().hex[:8]}",
            data_refs={},
            metadata={}
        )
        
        for data_type, items in context.data.items():
            shm_names = []
            for item in items:
                shm_name = self.shm_registry.allocate(item)
                shm_names.append(shm_name)
            handle.data_refs[data_type] = shm_names
        
        return handle
    
    def _materialize_context(self, handle: ContextHandle) -> Context:
        """Reconstruct Context from ContextHandle."""
        context = Context()
        context.pipeline_runner = self
        context.collector = self.collector
        context.config = self.config
        
        for data_type, shm_names in handle.data_refs.items():
            items = []
            for shm_name in shm_names:
                obj = self.shm_registry.materialize(shm_name)
                items.append(obj)
            
            # Wrap in RefCountedList if using ref counting
            if isinstance(items, list):
                items = RefCountedList(items, runner=self)
            
            context.data[data_type] = items
        
        return context
    
    def _get_task_params(self, task: BaseTask) -> Dict[str, Any]:
        """Extract task parameters for serialization."""
        # Tasks need to implement _get_config() method
        if hasattr(task, '_get_config'):
            return task._get_config()
        return {}
```

## Implementation Phases

### Phase 1: Reference Counting (Thread-based)
**Goal**: Reduce memory usage in current thread-based system

**Changes**:
1. Implement `RefCountedList` in `task_base.py`
2. Update `Context.split()` to use `RefCountedList.__copy__()`
3. Add `_track_object()` and `_release_object()` to `PipelineRunner`
4. Implement `garbage_collect()` with grace period + Python validation
5. Call GC between task batches in main execution loop

**Benefits**:
- Immediate memory improvements
- No task code changes
- Foundation for multiprocessing

**Effort**: Medium (2-3 days)

### Phase 2: Context Handles (Preparation)
**Goal**: Abstraction layer for multiprocessing without breaking threads

**Changes**:
1. Implement `ContextHandle` dataclass in `task_base.py`
2. Implement `SharedMemoryRegistry` in new `shared_memory.py`
3. Add `_contextualize()` and `_materialize_context()` methods
4. Keep thread-based execution, add optional context serialization for testing

**Benefits**:
- Validates serialization/deserialization works
- No performance impact (only used when enabled)
- Incremental testing without breaking existing system

**Effort**: Medium (3-4 days)

### Phase 3: Process Pool (Scaling)
**Goal**: Enable true multiprocessing for CPU-bound tasks

**Changes**:
1. Implement `_run_task_in_process()` worker function
2. Add `use_multiprocessing` flag to `PipelineRunner.__init__()`
3. Add task parameter serialization (`_get_config()` methods)
4. Add registry server (network or unix socket)
5. Route execution based on flag

**Benefits**:
- Scales to multiple CPU cores
- Reduces GIL contention
- Optional - can fall back to threads

**Effort**: Large (5-7 days)

## Key Design Principles

1. **Transparency**: Tasks don't know about ref counting or multiprocessing
2. **Incrementalism**: Each phase adds value independently
3. **Safety**: Grace periods + Python validation prevent dangling refs
4. **Flexibility**: Can mix thread/process execution as needed
5. **Observability**: Metrics and debug logging throughout

## Testing Strategy

### Phase 1 Testing
- Unit tests for `RefCountedList` operations
- Test GC with various pipeline patterns (fork, merge, sequential)
- Validate against Python's `sys.getrefcount()`
- Memory profiling before/after

### Phase 2 Testing
- Test serialization round-trip for all object types
- Test shared memory allocation/deallocation
- Test reference counting across "fake" processes (threads)

### Phase 3 Testing
- Test simple task execution in process
- Test fork/merge with multiple processes
- Test task failures and cleanup
- Performance benchmarks vs threads

## Risks and Mitigations

### Risk: Leaked References
**Mitigation**: Grace period + Python validation + debug logging

### Risk: Task holds local reference
**Mitigation**: Document best practices, add lint rules, runtime warnings

### Risk: Circular references
**Mitigation**: Use weak references for parent pointers, track bidirectional refs

### Risk: Serialization overhead
**Mitigation**: Profile, optimize hot paths, use memory-mapped files for large data

### Risk: Complexity
**Mitigation**: Phased implementation, extensive testing, clear documentation

## Metrics to Track

- `gc.objects_tracked`: Total objects under ref counting
- `gc.objects_freed`: Objects freed by GC per cycle
- `gc.leaked_refs_detected`: Objects with Python refcount mismatch
- `gc.grace_period_delayed`: Objects waiting for grace period
- `shm.allocated_bytes`: Total shared memory allocated
- `shm.segments_active`: Number of active shared memory segments
- `task.serialization_duration`: Time to serialize context
- `task.deserialization_duration`: Time to materialize context

## Future Optimizations

1. **Lazy serialization**: Only serialize data needed by task
2. **Memory-mapped files**: Use mmap instead of SharedMemory for very large data
3. **Object pooling**: Reuse shared memory segments
4. **Compression**: Compress serialized data for large objects
5. **Network distribution**: Extend to multiple machines with same abstraction
