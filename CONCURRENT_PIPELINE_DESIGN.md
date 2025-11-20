# Concurrent Pipeline Execution - Design Analysis

## Problem Statement

**Goal:** Allow starting a new pipeline run before the current run completes.

**Use Cases:**
1. **Interactive Chat:** User submits new query while VLM is still processing previous one
2. **Real-time Detection:** Continuous camera capture with overlapping detector runs
3. **High-throughput Processing:** Queue multiple inputs without blocking
4. **Cancellation:** Ability to cancel in-flight pipeline and start fresh one

## Current State Analysis

### Current Limitations

The `PipelineRunner` has several issues preventing concurrent runs:

**1. Shared Mutable State (Instance Variables)**
```python
class PipelineRunner:
    def __init__(self, connector: Connector, max_workers: int = 4, ...):
        self.graph: List[BaseTask] = []                      # ❌ Shared across runs
        self.immutable_cache: Dict[ContextDataType, ...] = {} # ❌ Shared across runs
        self.task_contexts: Dict[str, Context] = {}          # ❌ Shared across runs
        self.executor = ThreadPoolExecutor(max_workers=4)    # ✅ Can be shared
        self.collector = collector                           # ✅ Can be shared
```

**Problem:** Instance variables are mutated during `run()`:
- `self.graph` - Rebuilt on each run via `build_graph()`
- `self.task_contexts` - Cleared and populated during execution
- `self.immutable_cache` - Potentially shared (currently unused)

**2. Task State Mutation**
```python
# In Connector.build_graph()
task.upstream_tasks = [...]      # ❌ Modifies task objects
task.split_contexts = [...]      # ❌ Stores per-run state on task
```

**Problem:** Task objects themselves carry run-specific state.

**3. No Isolation Between Runs**
- Two concurrent `runner.run(context1)` and `runner.run(context2)` calls would:
  - Overwrite `self.task_contexts` causing data corruption
  - Race on `completed` set (local to `run()`, but tasks are shared)
  - Mix contexts between the two pipelines

**4. ThreadPoolExecutor Sharing**
- Current design: One executor per runner instance
- Concurrent runs would compete for the same worker threads
- No run prioritization or isolation

## Design Options

### Option 1: Stateless Runner (Recommended)

**Approach:** Move all mutable state from instance to `run()` method scope.

**Changes Required:**

```python
class PipelineRunner:
    def __init__(self, connector: Connector, max_workers: int = 4, 
                 collector: Optional[Collector] = None):
        # Only immutable/shareable state
        self.connector = connector
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.collector = collector
        # Remove: self.graph, self.task_contexts, self.immutable_cache
    
    def run(self, context: Context) -> Context:
        """Execute pipeline - now thread-safe for concurrent calls."""
        # All mutable state is local to this invocation
        run_state = PipelineRunState()
        run_state.graph = self._build_graph()
        run_state.task_contexts = {}
        run_state.completed = set()
        
        # Execute with isolated state
        return self._execute_pipeline(context, run_state)
```

**Pros:**
- ✅ Simple and clean
- ✅ Thread-safe by design
- ✅ No locking needed
- ✅ Multiple runs can share executor naturally

**Cons:**
- ❌ Graph is rebuilt on every run (small overhead)
- ❌ Still need to handle task object mutation

### Option 2: Run-Scoped State Object

**Approach:** Create a `PipelineRun` object for each execution.

```python
class PipelineRun:
    """Represents a single execution of a pipeline."""
    
    def __init__(self, runner: 'PipelineRunner', context: Context, run_id: str):
        self.runner = runner
        self.context = context
        self.run_id = run_id
        
        # Per-run state
        self.graph: List[BaseTask] = []
        self.task_contexts: Dict[str, Context] = {}
        self.completed: Set[str] = set()
        self.start_time = time.time()
        self.status = "running"  # running, completed, failed, cancelled
        
    def execute(self) -> Context:
        """Execute this pipeline run."""
        self.graph = self._build_graph()
        # ... rest of current run() logic ...
        self.status = "completed"
        return result

class PipelineRunner:
    def __init__(self, ...):
        self.connector = connector
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.collector = collector
        self.active_runs: Dict[str, PipelineRun] = {}  # Track active runs
        self.runs_lock = threading.Lock()
    
    def run(self, context: Context, run_id: Optional[str] = None) -> Context:
        """Start a pipeline run."""
        if run_id is None:
            run_id = str(uuid.uuid4())
        
        pipeline_run = PipelineRun(self, context, run_id)
        
        with self.runs_lock:
            self.active_runs[run_id] = pipeline_run
        
        try:
            result = pipeline_run.execute()
            return result
        finally:
            with self.runs_lock:
                del self.active_runs[run_id]
    
    def cancel_run(self, run_id: str):
        """Cancel a specific run."""
        with self.runs_lock:
            if run_id in self.active_runs:
                self.active_runs[run_id].status = "cancelled"
    
    def get_active_runs(self) -> List[str]:
        """Get list of currently executing runs."""
        with self.runs_lock:
            return list(self.active_runs.keys())
```

**Pros:**
- ✅ Clear separation of concerns
- ✅ Easy to track/manage multiple concurrent runs
- ✅ Supports cancellation and monitoring
- ✅ Can implement run prioritization

**Cons:**
- ❌ More complex architecture
- ❌ Requires careful locking for run registry
- ❌ Still need to handle task object mutation

### Option 3: Copy-on-Run

**Approach:** Deep copy the entire pipeline structure for each run.

```python
class PipelineRunner:
    def run(self, context: Context) -> Context:
        # Create isolated copy of pipeline for this run
        import copy
        connector_copy = copy.deepcopy(self.connector)
        
        # Execute with copied pipeline (no shared state)
        return self._execute_with_connector(connector_copy, context)
```

**Pros:**
- ✅ Complete isolation between runs
- ✅ No mutation concerns
- ✅ Simple to implement

**Cons:**
- ❌ Expensive (deep copy of entire pipeline structure)
- ❌ Doesn't work for stateful tasks (camera, model instances)
- ❌ Can't share heavy resources (loaded models)

## Task-Level Concerns

Beyond runner state, tasks themselves need consideration:

### Problem: Stateful Resources

```python
class CameraTask(BaseTask):
    def __init__(self, camera: BaseCamera, ...):
        self.camera = camera  # ❌ Shared hardware resource
    
    def run(self, context: Context) -> Context:
        # Multiple concurrent runs would conflict on camera hardware
        filepath, image = self.camera.capture_single_image()
```

**Solution:** Resource pooling or locking

```python
class CameraTask(BaseTask):
    def run(self, context: Context) -> Context:
        with self.camera.acquire_lock():  # Serialize camera access
            filepath, image = self.camera.capture_single_image()
        return context
```

### Problem: Task-Level State

```python
# In current design
task.upstream_tasks = [...]      # Set during build_graph()
task.split_contexts = [...]      # Set during split operation
```

**Solution:** Separate graph structure from runtime state

```python
class TaskGraphNode:
    """Immutable graph structure."""
    def __init__(self, task: BaseTask, upstream_ids: List[str]):
        self.task = task
        self.upstream_ids = upstream_ids
        self.task_id = task.task_id

class PipelineRunState:
    """Mutable runtime state for a single run."""
    def __init__(self):
        self.graph: List[TaskGraphNode] = []
        self.task_contexts: Dict[str, Context] = {}
        self.split_contexts: Dict[str, List[Context]] = {}
        self.completed: Set[str] = set()
```

## Recommended Implementation Plan

### Phase 1: Stateless Runner (Immediate)

**1.1 Move State to Method Scope**
```python
class PipelineRunner:
    def __init__(self, connector, max_workers, collector):
        self.connector = connector
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.collector = collector
        # Remove instance state variables
    
    def run(self, context: Context) -> Context:
        # Create per-run state
        graph = self._build_graph()
        task_contexts = {}
        completed = set()
        
        # Pass state explicitly through execution
        return self._execute(context, graph, task_contexts, completed)
```

**1.2 Update Helper Methods**
```python
def _execute(self, context, graph, task_contexts, completed):
    while len(completed) < len(graph):
        ready = self._get_ready_tasks(graph, completed)
        # ... execute batch ...
        
def _get_ready_tasks(self, graph, completed):
    # No longer accesses self.graph
    return [t for t in graph if self._deps_satisfied(t, completed)]
```

**Impact:**
- ✅ Enables concurrent runs immediately
- ✅ Minimal code changes
- ✅ No breaking API changes
- ⚠️ Still rebuilds graph on each run (acceptable overhead)

### Phase 2: Separate Graph from Runtime State (Follow-up)

**Critical Note:** Pipelining allows the system to run at the throughput of the **slowest component** rather than being limited by the **sum of all components**. Phase 1 enables concurrency but rebuilds the graph on every `run()` call, adding overhead that artificially slows the system below its true maximum throughput.

**System Design Principle:** The goal is to achieve throughput limited only by task execution (the slowest stage), not artificially degraded by non-productive overhead (graph rebuilding).

**Pipelining Example:**
- YOLO: 5ms
- Clusterer: 5ms
- CLIP (4 inferences): 20ms
- **Sequential execution:** 30ms per frame = 33 fps
- **With pipelining:** Limited by slowest stage (CLIP at 20ms) = **50 fps** (with 30ms latency)

**Why Phase 2 Matters:**
- **Phase 1 Only:** If CLIP takes 20ms and graph building takes 1ms, throughput is artificially limited to ~47 fps instead of 50 fps
- **With Phase 2:** Graph built once, throughput reaches true maximum of 50 fps (limited only by CLIP)
- **Combined with Adaptive Scheduling:** Can skip frames proactively while maintaining multiple concurrent batches at true maximum hardware utilization

This optimization becomes essential when implementing adaptive frame skipping (see `ADAPTIVE_SCHEDULING_DESIGN.md`). Without it, the overhead of rebuilding the graph on each frame skip decision reduces overall system throughput below what the hardware can achieve.

**2.1 Create Immutable Graph Structure**
```python
@dataclass(frozen=True)
class TaskGraphNode:
    task: BaseTask
    task_id: str
    upstream_ids: List[str]
    
class PipelineGraph:
    """Immutable pipeline structure built once."""
    def __init__(self, nodes: List[TaskGraphNode]):
        self.nodes = nodes
        self.node_map = {n.task_id: n for n in nodes}
        self._validate()
```

**2.2 Build Graph Once in Constructor**
```python
class PipelineRunner:
    def __init__(self, connector, ...):
        self.connector = connector
        self.graph = self._build_graph()  # ✅ Built once
        self.executor = ...
```

**Impact:**
- ✅ Graph built once, validated once
- ✅ Faster run() startup (critical for frame skipping decisions)
- ✅ System runs at speed of entire pipeline, not slowest component
- ⚠️ Requires careful handling of task mutations

### Phase 3: Run Tracking & Cancellation (Optional)

**3.1 Add Run Management**
```python
class PipelineRunner:
    def __init__(self, ...):
        # ... existing ...
        self.active_runs: Dict[str, 'RunContext'] = {}
        self.runs_lock = threading.Lock()
    
    def run(self, context: Context, run_id: Optional[str] = None) -> Context:
        run_id = run_id or str(uuid.uuid4())
        run_ctx = RunContext(run_id, context)
        
        with self.runs_lock:
            self.active_runs[run_id] = run_ctx
        
        try:
            return self._execute_run(run_ctx)
        finally:
            with self.runs_lock:
                del self.active_runs[run_id]
    
    def cancel_run(self, run_id: str):
        """Request cancellation of specific run."""
        with self.runs_lock:
            if run_id in self.active_runs:
                self.active_runs[run_id].cancel_requested = True
```

**3.2 Cooperative Cancellation in Tasks**
```python
class BaseTask:
    def run(self, context: Context) -> Context:
        while processing:
            if context.is_cancelled():  # Check cancellation flag
                raise CancellationException()
            # ... do work ...
```

**Impact:**
- ✅ Can monitor active runs
- ✅ Can cancel specific runs
- ✅ Graceful shutdown support

### Phase 4: Resource Management (Critical for Hardware)

**4.1 Add Resource Locks to Tasks**
```python
class CameraTask(BaseTask):
    _camera_lock = threading.Lock()  # Class-level lock
    
    def run(self, context: Context) -> Context:
        with self._camera_lock:
            # Only one camera capture at a time
            filepath, image = self.camera.capture_single_image()
        context.data[ContextDataType.IMAGE] = image
        return context
```

**4.2 Resource Pool Pattern**
```python
class DetectorTaskPool:
    """Pool of detector instances for concurrent runs."""
    def __init__(self, model_path: str, pool_size: int = 2):
        self.detectors = [YoloV8Detector(model_path) for _ in range(pool_size)]
        self.pool = queue.Queue()
        for det in self.detectors:
            self.pool.put(det)
    
    def acquire(self) -> YoloV8Detector:
        return self.pool.get()  # Blocks if all busy
    
    def release(self, detector: YoloV8Detector):
        self.pool.put(detector)

class DetectorTask(BaseTask):
    def __init__(self, detector_pool: DetectorTaskPool, ...):
        self.detector_pool = detector_pool
    
    def run(self, context: Context) -> Context:
        detector = self.detector_pool.acquire()
        try:
            detections = detector.detect(image)
            context.data[ContextDataType.DETECTIONS] = detections
            return context
        finally:
            self.detector_pool.release(detector)
```

**Impact:**
- ✅ Prevents hardware conflicts
- ✅ Enables resource pooling for expensive resources (GPU models)
- ✅ Scales naturally with available resources

## Migration Strategy

### Step 1: Make Current Code Concurrent-Safe (Low Risk)
1. Move `self.graph`, `self.task_contexts` to method scope
2. Pass state explicitly through helper methods
3. Add threading tests
4. **Deliverable:** Can call `runner.run()` concurrently

### Step 2: Optimize Graph Building (Medium Risk)
1. Build graph once in constructor
2. Separate immutable structure from runtime state
3. Remove task object mutations
4. **Deliverable:** Faster runs, cleaner separation

### Step 3: Add Run Management (Low Risk, Optional)
1. Track active runs with run IDs
2. Expose monitoring API
3. Implement cancellation
4. **Deliverable:** Observable, controllable pipelines

### Step 4: Resource Management (High Priority for Hardware)
1. Add locks to camera/hardware tasks
2. Implement resource pools for models
3. Add semaphores for bounded concurrency
4. **Deliverable:** Safe concurrent hardware access

## Testing Strategy

### Concurrent Execution Tests
```python
def test_concurrent_runs():
    """Test multiple simultaneous pipeline runs."""
    runner = PipelineRunner(pipeline)
    
    contexts = [Context() for _ in range(5)]
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(runner.run, ctx) for ctx in contexts]
        results = [f.result() for f in futures]
    
    # Verify all runs completed successfully
    assert len(results) == 5
    assert all(ContextDataType.IMAGE in r.data for r in results)
```

### Resource Contention Tests
```python
def test_camera_serialization():
    """Test that camera access is serialized."""
    camera_task = CameraTask(camera, "cam")
    
    capture_times = []
    
    def timed_run():
        start = time.time()
        camera_task.run(Context())
        end = time.time()
        capture_times.append((start, end))
    
    threads = [threading.Thread(target=timed_run) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Verify no overlap (serialized)
    sorted_times = sorted(capture_times)
    for i in range(len(sorted_times) - 1):
        assert sorted_times[i][1] <= sorted_times[i+1][0]  # No overlap
```

## Performance Considerations

**Overhead of Concurrent Runs:**
- Graph building: ~0.1-1ms (acceptable if done per-run)
- Context copying: Depends on data size
- Thread synchronization: Minimal with proper design

**Throughput Gains:**
- Long-running tasks (VLM inference): Significant improvement
- I/O-bound tasks (camera, network): Can overlap with compute
- Short pipelines: Overhead may exceed benefits

**Resource Constraints:**
- GPU memory: Limit concurrent detector runs
- Camera hardware: Must serialize access
- CPU: ThreadPoolExecutor naturally limits parallelism

## Conclusion

**Recommended Approach:**

1. **Phase 1 (Immediate):** Stateless runner - move state to method scope
   - Low risk, high value
   - Enables concurrent runs immediately
   - Minimal code changes

2. **Phase 2 (Soon):** Add resource locks to hardware tasks
   - Critical for camera/hardware safety
   - Prevents conflicts and crashes

3. **Phase 3 (Optional):** Build graph once in constructor
   - Performance optimization
   - Cleaner architecture

4. **Phase 4 (Future):** Run tracking and cancellation
   - Nice-to-have for monitoring
   - Useful for interactive applications

**Estimated Effort:**
- Phase 1: 2-4 hours (refactor + basic tests)
- Phase 2: 1-2 hours (add locks to tasks)
- Phase 3: 4-8 hours (restructure graph building)
- Phase 4: 4-8 hours (run management + cancellation)

**Total:** ~2-3 days for full implementation with comprehensive testing.
