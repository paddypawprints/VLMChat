# Adaptive Frame Skipping and Throughput Maximization

## Overview

Use measured latencies to predict completion times and **delay starting batches** (skip frames) when necessary, while keeping CPU/GPU fully utilized at maximum system throughput.

**Core Principle:** It's better to skip frames and delay starting batches than to abandon batches mid-execution. Once a batch starts, it always runs to completion.

## Core Concepts

### Two Task Types

**Type 1: Batch-Dependent Tasks (Strict Timing)**
- If predicted to miss deadline: **delay starting batch → frame skip**
- All inputs in the batch must be processed together or none at all
- Once started, batch always completes (no abandoning mid-execution)
- Examples: YOLO→Clusterer→CLIP pipeline (atomic batch, no point in partial results)

**Type 2: Independent Long-Running Tasks (Latency-Tolerant)**
- Can skip/ignore missed inputs between runs
- Process whatever is latest when task becomes available
- Examples: Continuous camera capture, VLM running at its own pace

### Latency-Based Scheduling

**Prediction:**
```
completion_time = current_time + task_ewma_duration
deadline = batch_start_time + batch_timeout
can_complete = completion_time <= deadline
```

**Decision Flow:**
1. **Before batch starts:** Predict if all STRICT tasks can complete within deadline
   - If YES: Commit to batch, execute to completion
   - If NO: Skip frame(s), delay starting batch, wait for system to catch up
2. **During batch execution:** Always continue to completion (no abandoning)
3. **After batch completes:** Update EWMA predictions for future batches

**Key Insight:** **Skip frames to avoid starting doomed batches, never abandon work once started.** This prevents cascading failures where high system load causes every batch to be abandoned mid-execution, resulting in zero throughput. Better to delay starting and maintain high completion rate than to start many batches and complete none.

## Architecture Changes

### 1. Task Behavior Annotation

```python
from enum import Enum

class TaskTimingMode(Enum):
    """How a task behaves under timing pressure."""
    STRICT = "strict"           # Must complete or batch fails
    LATENCY_TOLERANT = "latency_tolerant"  # Can skip inputs, use latest
    BEST_EFFORT = "best_effort" # Try to complete, batch continues if fails

class BaseTask(ABC):
    def __init__(self, task_id: str, time_budget_ms: Optional[int] = None,
                 timing_mode: TaskTimingMode = TaskTimingMode.STRICT):
        self.task_id = task_id
        self.time_budget_ms = time_budget_ms
        self.timing_mode = timing_mode
        self.collector: Optional[Collector] = None
        
        # Latency tracking for prediction
        self.execution_history: List[float] = []
        self.ewma_duration_ms: Optional[float] = None
        self.ewma_alpha = 0.2  # Exponential weight
        
        self.input_contract: Dict[ContextDataType, type] = {}
        self.output_contract: Dict[ContextDataType, type] = {}
    
    def record_execution(self, duration_ms: float):
        """Update execution time statistics."""
        self.execution_history.append(duration_ms)
        
        if self.ewma_duration_ms is None:
            self.ewma_duration_ms = duration_ms
        else:
            self.ewma_duration_ms = (
                self.ewma_alpha * duration_ms + 
                (1 - self.ewma_alpha) * self.ewma_duration_ms
            )
    
    def predict_completion_time(self, current_time: float) -> float:
        """Predict when this task will complete if started now."""
        if self.ewma_duration_ms is None:
            # No history, use budget or conservative estimate
            duration = self.time_budget_ms or 1000.0
        else:
            duration = self.ewma_duration_ms
        
        return current_time + (duration / 1000.0)  # Convert to seconds
```

### 2. Batch Context with Timing

```python
@dataclass
class BatchContext:
    """Execution context for a batch of work."""
    batch_id: str
    start_time: float
    deadline: float  # start_time + batch_timeout
    context: Context
    
    # Tracking
    skipped_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Set[str] = field(default_factory=set)
    completed_tasks: Set[str] = field(default_factory=set)
    
    @property
    def time_remaining(self) -> float:
        """Seconds remaining until deadline."""
        return max(0, self.deadline - time.time())
    
    @property
    def is_valid(self) -> bool:
        """Check if batch is still valid (no strict tasks skipped/failed)."""
        return len(self.failed_tasks) == 0 and len(self.skipped_tasks) == 0
```

### 3. Adaptive Scheduler in Runner

```python
class PipelineRunner:
    def __init__(self, connector: Connector, max_workers: int = 4,
                 collector: Optional[Collector] = None,
                 batch_timeout_ms: int = 5000):
        self.connector = connector
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.collector = collector
        self.batch_timeout_ms = batch_timeout_ms
        
        logger.info(f"PipelineRunner initialized with {batch_timeout_ms}ms batch timeout")
    
    def can_meet_deadline(self, context: Context, deadline_ms: int) -> bool:
        """
        Predict if pipeline can complete within deadline.
        
        Called BEFORE starting a batch to decide whether to skip frames.
        Once a batch is started, it always runs to completion.
        """
        total_predicted_ms = 0.0
        
        # Sum predicted durations for all STRICT tasks in critical path
        for task in self._get_critical_path_tasks():
            if task.timing_mode == TaskTimingMode.STRICT:
                if task.ewma_duration_ms is None:
                    # No history, use budget or conservative estimate
                    predicted = task.time_budget_ms or 1000.0
                else:
                    predicted = task.ewma_duration_ms
                
                total_predicted_ms += predicted
        
        can_meet = total_predicted_ms <= deadline_ms
        
        logger.debug(f"Pipeline prediction: {total_predicted_ms:.1f}ms predicted, "
                    f"{deadline_ms}ms deadline, can_meet={can_meet}")
        
        return can_meet
    
    def run(self, context: Context, batch_timeout_ms: Optional[int] = None) -> Context:
        """Execute pipeline with adaptive scheduling."""
        batch_timeout = batch_timeout_ms or self.batch_timeout_ms
        
        # PREDICTION PHASE: Should we start this batch?
        if not self.can_meet_deadline(context, batch_timeout):
            logger.warning(f"Predicted to miss deadline, skipping frame")
            raise PipelineBatchSkipped("Predicted latency exceeds SLO, skipping frame")
        
        # EXECUTION PHASE: Commit to completing this batch
        batch = BatchContext(
            batch_id=str(uuid.uuid4()),
            start_time=time.time(),
            deadline=time.time() + (batch_timeout / 1000.0),
            context=context
        )
        
        logger.info(f"Starting batch {batch.batch_id} with {batch_timeout}ms timeout "
                   "(committed to completion)")
        
        # Build graph (in future, this will be cached)
        graph = self._build_graph()
        task_contexts = {}
        completed = set()
        
        # Execute all tasks to completion (no abandoning mid-batch)
        while len(completed) < len(graph):
            ready_tasks = self._get_ready_tasks(graph, completed)
            
            if not ready_tasks:
                break
            
            # Execute all ready tasks (no selective skipping during execution)
            futures = []
            for task in ready_tasks:
                # Get task context
                if task.upstream_tasks:
                    upstream_task = task.upstream_tasks[0]
                    if isinstance(upstream_task, Connector) and upstream_task.split_contexts:
                        upstream_index = upstream_task.output_tasks.index(task)
                        task_context = upstream_task.split_contexts[upstream_index]
                    else:
                        task_context = task_contexts[upstream_task.task_id]
                else:
                    task_context = context
                
                future = self.executor.submit(
                    self._run_task_with_timing, task, task_context, batch
                )
                futures.append((future, task))
            
            # Wait for ALL tasks to complete
            for future, task in futures:
                try:
                    result_context, duration_ms = future.result()  # No timeout - must complete
                    task_contexts[task.task_id] = result_context
                    completed.add(task.task_id)
                    batch.completed_tasks.add(task.task_id)
                    
                    # Update task statistics for future predictions
                    task.record_execution(duration_ms)
                    
                    # Warn if task exceeded budget but continue anyway
                    if task.time_budget_ms and duration_ms > task.time_budget_ms:
                        logger.warning(f"Batch {batch.batch_id}: Task {task.task_id} "
                                     f"exceeded budget ({duration_ms:.1f}ms > "
                                     f"{task.time_budget_ms}ms) but completed")
                    else:
                        logger.info(f"Batch {batch.batch_id}: Completed {task.task_id} "
                                  f"in {duration_ms:.1f}ms")
                    
                except Exception as e:
                    logger.error(f"Batch {batch.batch_id}: Task {task.task_id} failed: {e}")
                    batch.failed_tasks.add(task.task_id)
                    
                    if task.timing_mode == TaskTimingMode.STRICT:
                        # STRICT task failed - batch is invalid
                        raise
                    else:
                        # LATENCY_TOLERANT task failed - continue with other work
                        completed.add(task.task_id)
        
        # Measure actual batch duration
        batch_duration_ms = (time.time() - batch.start_time) * 1000
        
        # Log batch results
        logger.info(f"Batch {batch.batch_id} completed in {batch_duration_ms:.1f}ms "
                   f"(deadline: {batch_timeout}ms, "
                   f"{'MISSED SLO' if batch_duration_ms > batch_timeout else 'met SLO'}): "
                   f"{len(batch.completed_tasks)} completed, "
                   f"{len(batch.failed_tasks)} failed")
        
        # Track SLO violations for monitoring
        if self.collector:
            self.collector.data_point("batch.duration", {}, batch_duration_ms)
            if batch_duration_ms > batch_timeout:
                self.collector.data_point("batch.slo_violation", {}, 1)
        
        # Return result from final task
        sink_tasks = [t for t in graph if not any(t in other.upstream_tasks for other in graph)]
        if sink_tasks and sink_tasks[0].task_id in task_contexts:
            return task_contexts[sink_tasks[0].task_id]
        return context
    
    def _schedule_with_prediction(self, ready_tasks: List[BaseTask],
                                  batch: BatchContext,
                                  completed: Set[str]) -> List[BaseTask]:
        """Decide which tasks to schedule based on predicted completion time."""
        current_time = time.time()
        scheduled = []
        
        for task in ready_tasks:
            predicted_completion = task.predict_completion_time(current_time)
            
            if predicted_completion <= batch.deadline:
                scheduled.append(task)
                logger.debug(f"Batch {batch.batch_id}: Scheduling {task.task_id} "
                           f"(predicted: {predicted_completion - current_time:.2f}s, "
                           f"remaining: {batch.time_remaining:.2f}s)")
            else:
                # Task won't make deadline
                if task.timing_mode == TaskTimingMode.STRICT:
                    logger.warning(f"Batch {batch.batch_id}: STRICT task {task.task_id} "
                                 f"cannot meet deadline (predicted: "
                                 f"{predicted_completion - current_time:.2f}s, "
                                 f"remaining: {batch.time_remaining:.2f}s)")
                    # Don't schedule, will fail batch
                    batch.skipped_tasks.add(task.task_id)
                    
                elif task.timing_mode == TaskTimingMode.LATENCY_TOLERANT:
                    # Skip this input, will process next one
                    logger.info(f"Batch {batch.batch_id}: Skipping {task.task_id} "
                              f"(LATENCY_TOLERANT, predicted late)")
                    batch.skipped_tasks.add(task.task_id)
                    # Mark as completed so downstream can continue with stale data
                    completed.add(task.task_id)
                    
                elif task.timing_mode == TaskTimingMode.BEST_EFFORT:
                    # Try anyway
                    scheduled.append(task)
                    logger.debug(f"Batch {batch.batch_id}: Scheduling {task.task_id} "
                               "(BEST_EFFORT, may be late)")
        
        return scheduled
    
    def _run_task_with_timing(self, task: BaseTask, context: Context,
                             batch: BatchContext) -> Tuple[Context, float]:
        """Execute task and return result with duration."""
        start_time = time.time()
        
        try:
            result = task.run(context)
            duration_ms = (time.time() - start_time) * 1000
            
            if self.collector:
                self.collector.data_point("task.execution.duration",
                                         {"task_id": task.task_id}, duration_ms)
            
            return result, duration_ms
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            if self.collector:
                self.collector.data_point("task.failures",
                                         {"task_id": task.task_id,
                                          "error_type": type(e).__name__}, 1)
            raise

class PipelineBatchTimeout(Exception):
    """Raised when a strict task cannot meet its deadline."""
    pass
```

### 4. Latency-Tolerant Camera Example

```python
class CameraTask(BaseTask):
    """Camera that skips frames when falling behind."""
    
    def __init__(self, camera: Optional[BaseCamera] = None, task_id: str = "camera"):
        # Camera is LATENCY_TOLERANT - can skip frames
        super().__init__(task_id, 
                        time_budget_ms=100,  # Target 100ms per capture
                        timing_mode=TaskTimingMode.LATENCY_TOLERANT)
        self.camera = camera
        self.last_capture_time = 0
        
        self.input_contract = {}
        self.output_contract = {ContextDataType.IMAGE: Image.Image}
    
    def run(self, context: Context) -> Context:
        """Capture latest frame, skip if called too quickly."""
        if self.camera is None:
            raise RuntimeError(f"Task {self.task_id}: Camera not configured")
        
        # Check if enough time has passed
        current_time = time.time()
        time_since_last = current_time - self.last_capture_time
        
        if time_since_last < 0.05:  # Don't capture faster than 20 FPS
            logger.debug(f"{self.task_id}: Skipping capture (too soon: {time_since_last:.3f}s)")
            # Reuse previous image from context if available
            if ContextDataType.IMAGE not in context.data:
                # Must capture anyway
                pass
            else:
                return context
        
        filepath, pil_image = self.camera.capture_single_image()
        context.data[ContextDataType.IMAGE] = pil_image
        self.last_capture_time = current_time
        
        return context
```

### 5. Latency-Tolerant VLM Task Example

```python
class SmolVLMTask(BaseTask):
    """VLM inference - grabs latest image when ready to process.
    
    VLM is too slow for live video (30fps), so it runs at its own pace
    and processes whatever image is current when it becomes available.
    """
    
    def __init__(self, model: SmolVLMModel, prompt: Prompt, task_id: str = "smolvlm"):
        # VLM is LATENCY_TOLERANT - grabs latest frame when ready
        super().__init__(task_id,
                        time_budget_ms=3000,  # ~3s per inference
                        timing_mode=TaskTimingMode.LATENCY_TOLERANT)
        self.model = model
        self.prompt = prompt
        self.last_processed_image_id = None
        
        self.input_contract = {
            ContextDataType.IMAGE: Image.Image,
            ContextDataType.PROMPT: str
        }
        self.output_contract = {ContextDataType.RESPONSE: str}
    
    def run(self, context: Context) -> Context:
        """Run inference on whatever image is currently available.
        
        If called while still processing previous frame, this call may be
        skipped by the scheduler. When it does run, it processes the latest
        available image even if multiple frames have passed.
        """
        image = context.data.get(ContextDataType.IMAGE)
        prompt_text = context.data.get(ContextDataType.PROMPT)
        
        if image is None or prompt_text is None:
            raise ValueError(f"Task {self.task_id}: Missing required inputs")
        
        # Track which image we're processing (to avoid duplicates if needed)
        image_id = id(image)
        if image_id == self.last_processed_image_id:
            logger.debug(f"{self.task_id}: Already processed this image, skipping")
            return context
        
        logger.info(f"{self.task_id}: Processing frame (may be several frames old)")
        
        self.prompt._user_input = prompt_text
        self.prompt.current_image = image
        
        messages = self.model.get_messages(self.prompt)
        response = self.model.generate_response(messages, [image], stream_output=False)
        
        context.data[ContextDataType.RESPONSE] = response
        self.last_processed_image_id = image_id
        
        return context
```

### 6. Strict Detection Pipeline Example

```python
class DetectorTask(BaseTask):
    """Object detector - part of YOLO→Clusterer→CLIP pipeline.
    
    This pipeline must complete atomically or skip the entire batch.
    No point running clustering without detections, or CLIP without clusters.
    """
    
    def __init__(self, detector: Optional[ObjectDetector] = None, task_id: str = "detector"):
        # Detector is STRICT - must complete or entire detection pipeline fails
        super().__init__(task_id,
                        time_budget_ms=300,  # ~300ms for YOLO
                        timing_mode=TaskTimingMode.STRICT)
        self.detector = detector
        
        self.input_contract = {ContextDataType.IMAGE: Image.Image}
        self.output_contract = {ContextDataType.DETECTIONS: list}
    
    def run(self, context: Context) -> Context:
        """Run detection - must complete for pipeline to continue."""
        if self.detector is None:
            raise RuntimeError(f"Task {self.task_id}: Detector not configured")
        
        image = context.data.get(ContextDataType.IMAGE)
        if image is None:
            raise ValueError(f"Task {self.task_id}: IMAGE not found in context")
        
        existing_detections = context.data.get(ContextDataType.DETECTIONS, [])
        detections = self.detector.detect(image, existing_detections)
        
        context.data[ContextDataType.DETECTIONS] = detections
        return context


class ClustererTask(BaseTask):
    """Detection clusterer - part of YOLO→Clusterer→CLIP pipeline.
    
    STRICT because clustering without detections is meaningless.
    """
    
    def __init__(self, clusterer: ObjectClusterer, task_id: str = "clusterer"):
        super().__init__(task_id,
                        time_budget_ms=50,  # ~50ms for clustering
                        timing_mode=TaskTimingMode.STRICT)
        self.clusterer = clusterer
        
        self.input_contract = {
            ContextDataType.IMAGE: Image.Image,
            ContextDataType.DETECTIONS: list
        }
        self.output_contract = {ContextDataType.CROPS: list}
    
    def run(self, context: Context) -> Context:
        """Cluster detections - must complete for pipeline to continue."""
        image = context.data.get(ContextDataType.IMAGE)
        detections = context.data.get(ContextDataType.DETECTIONS, [])
        
        if not detections:
            logger.debug(f"{self.task_id}: No detections to cluster")
            context.data[ContextDataType.CROPS] = []
            return context
        
        # Cluster and extract crops
        clusters = self.clusterer.cluster_detections(detections)
        crops = self.clusterer.extract_crops(image, clusters)
        
        context.data[ContextDataType.CROPS] = crops
        return context


class CLIPTask(BaseTask):
    """CLIP embeddings - part of YOLO→Clusterer→CLIP pipeline.
    
    STRICT because embeddings without crops is meaningless.
    """
    
    def __init__(self, clip_model, task_id: str = "clip"):
        super().__init__(task_id,
                        time_budget_ms=200,  # ~200ms for CLIP on batch
                        timing_mode=TaskTimingMode.STRICT)
        self.clip_model = clip_model
        
        self.input_contract = {ContextDataType.CROPS: list}
        self.output_contract = {ContextDataType.EMBEDDINGS: list}
    
    def run(self, context: Context) -> Context:
        """Generate embeddings - must complete for pipeline to continue."""
        crops = context.data.get(ContextDataType.CROPS, [])
        
        if not crops:
            logger.debug(f"{self.task_id}: No crops to embed")
            context.data[ContextDataType.EMBEDDINGS] = []
            return context
        
        # Generate embeddings for all crops
        embeddings = self.clip_model.encode_images(crops)
        
        context.data[ContextDataType.EMBEDDINGS] = embeddings
        return context
```

### Example 1: Real-Time Detection Pipeline with VLM

```python
from src.pipeline.pipeline_factory import create_default_factory
from src.pipeline.task_base import TaskTimingMode

factory = create_default_factory()

# Camera captures continuously (can skip frames)
camera = factory.create_task("camera", "cam0", {
    "type": "imx219",
    "device": "0"
})
camera.timing_mode = TaskTimingMode.LATENCY_TOLERANT
camera.time_budget_ms = 33  # Target 30fps

# Detection pipeline: YOLO→Clusterer→CLIP (all STRICT - atomic batch)
detector = factory.create_task("detector", "yolo", {
    "type": "yolo_cpu",
    "model": "yolov8n.pt"
})
detector.timing_mode = TaskTimingMode.STRICT
detector.time_budget_ms = 300

clusterer = ClustererTask(object_clusterer, "clusterer")
clusterer.timing_mode = TaskTimingMode.STRICT
clusterer.time_budget_ms = 50

clip = CLIPTask(clip_model, "clip")
clip.timing_mode = TaskTimingMode.STRICT
clip.time_budget_ms = 200

# VLM runs independently, grabs latest image when ready
smolvlm = SmolVLMTask(model, prompt, "vlm")
smolvlm.timing_mode = TaskTimingMode.LATENCY_TOLERANT
smolvlm.time_budget_ms = 3000

# Build pipeline:
#
#   Camera (tolerant)
#      ↓
#   [Split]
#      ↓ ↘
#      ↓   VLM (tolerant, slow)
#      ↓
#   YOLO (strict) ────→ Clusterer (strict) ────→ CLIP (strict)
#                                                      ↓
#                                                   [Merge]
#                                                      ↓
#                                                   Viewer
#
pipeline = factory.create_connector("connector", "vision_pipeline")
split_conn = factory.create_connector("connector", "split")
merge_conn = factory.create_connector("connector", "merge")

# Add all tasks
pipeline.add_task(camera)
pipeline.add_task(split_conn)
pipeline.add_task(detector)
pipeline.add_task(clusterer)
pipeline.add_task(clip)
pipeline.add_task(smolvlm)
pipeline.add_task(merge_conn)

# Wire edges
pipeline.add_edge(camera, split_conn)

# Split to detection pipeline and VLM
split_conn.output_tasks = [detector, smolvlm]
pipeline.add_edge(split_conn, detector)
pipeline.add_edge(split_conn, smolvlm)

# Detection chain (strict sequence)
pipeline.add_edge(detector, clusterer)
pipeline.add_edge(clusterer, clip)

# Merge detection results and VLM response
pipeline.add_edge(clip, merge_conn)
pipeline.add_edge(smolvlm, merge_conn)

# Run with 600ms batch timeout
# - If detection pipeline (YOLO+Clusterer+CLIP) can't complete in 600ms: skip frame
# - VLM runs asynchronously, processes when it finishes (may be several frames behind)
runner = PipelineRunner(pipeline, batch_timeout_ms=600)

# Continuous loop
while True:
    try:
        result = runner.run(Context())
        
        # Check what completed
        embeddings = result.data.get(ContextDataType.EMBEDDINGS)
        if embeddings:
            print(f"Detection pipeline: {len(embeddings)} objects embedded")
        
        response = result.data.get(ContextDataType.RESPONSE)
        if response:
            print(f"VLM: {response}")
        
    except PipelineBatchTimeout as e:
        # Detection pipeline couldn't complete - frame skipped
        print(f"Frame skipped (detection pipeline timeout): {e}")
        continue
```

**Behavior:**
- **Camera** runs at 30fps, provides continuous frames
- **YOLO→Clusterer→CLIP** pipeline runs as fast as possible (~550ms total)
  - If predicted to exceed 600ms deadline: **skip entire frame**
  - All three must complete or none do (atomic batch)
  - Throughput: ~1.8 fps for detection pipeline
- **VLM** runs at its own pace (~3 seconds per inference)
  - Grabs whatever image is current when it becomes available
  - May be 5-10 frames behind live video
  - Throughput: ~0.33 fps for VLM
  
**Result:** System runs at maximum throughput with no blocking:
- Detection pipeline processes every ~2nd frame
- VLM processes every ~10th frame  
- Both stay fully utilized without wasting work

### Example 2: Interactive Chat with Strict Timing

```python
# All tasks are STRICT - must complete together
start = factory.create_task("start", "start")
start.timing_mode = TaskTimingMode.STRICT

camera = factory.create_task("camera", "cam", {"type": "none"})
camera.timing_mode = TaskTimingMode.STRICT
camera.time_budget_ms = 100

console = factory.create_task("console_input", "input", {"prompt": "You: "})
console.timing_mode = TaskTimingMode.STRICT
console.time_budget_ms = 30000  # Allow time for user to type

# VLM requires both inputs
smolvlm = SmolVLMTask(model, prompt, "vlm")  # Already STRICT by default

# Build split/merge pipeline
pipeline = factory.create_connector("connector", "chat_pipeline")
split_conn = factory.create_connector("connector", "split")
merge_conn = factory.create_connector("connector", "merge")

# ... build graph ...

# Run with 35 second timeout (includes typing time)
runner = PipelineRunner(pipeline, batch_timeout_ms=35000)

while True:
    try:
        result = runner.run(Context())
        response = result.data.get(ContextDataType.RESPONSE)
        print(f"Assistant: {response}")
    except PipelineBatchTimeout as e:
        print(f"Request timed out: {e}")
        print("Please try again with a shorter question.")
```

### Example 3: Mixed Timing Modes

```python
# Camera continuously captures (tolerant)
camera = CameraTask(cam_device, "camera")
camera.timing_mode = TaskTimingMode.LATENCY_TOLERANT

# Fast detector runs on every frame (tolerant)
detector_fast = DetectorTask(yolo_n, "detector_fast")
detector_fast.timing_mode = TaskTimingMode.LATENCY_TOLERANT

# Slow detector runs when it can (best effort)
detector_accurate = DetectorTask(yolo_x, "detector_accurate")
detector_accurate.timing_mode = TaskTimingMode.BEST_EFFORT

# Merge with FirstComplete
merge = FirstCompleteConnector("first_result")

# Viewer shows results (best effort - nice to have)
viewer = ViewerTask(display, "viewer")
viewer.timing_mode = TaskTimingMode.BEST_EFFORT

# ... build graph with split to both detectors ...

# 200ms batch timeout - fast detector usually wins
runner = PipelineRunner(pipeline, batch_timeout_ms=200)
```

## Metrics and Observability

### Tracked Metrics

```python
# Task timing predictions
"task.predicted_duration_ms" - EWMA prediction
"task.actual_duration_ms" - Actual execution time
"task.prediction_error_ms" - |predicted - actual|

# Batch outcomes
"batch.completed" - Successfully completed batches
"batch.timeout" - Batches that exceeded deadline
"batch.tasks.skipped" - Tasks skipped due to prediction
"batch.tasks.failed" - Tasks that failed execution

# System throughput
"system.batches_per_second" - Batch completion rate
"system.cpu_utilization" - CPU usage
"system.gpu_utilization" - GPU usage (if available)
"system.frame_skip_rate" - Fraction of frames skipped
```

### Monitoring Dashboard

```python
def print_pipeline_stats(runner: PipelineRunner, session: Session):
    """Print pipeline performance statistics."""
    
    # Get task statistics
    for task in runner.connector.internal_tasks:
        if task.ewma_duration_ms:
            print(f"{task.task_id}:")
            print(f"  EWMA Duration: {task.ewma_duration_ms:.1f}ms")
            print(f"  Timing Mode: {task.timing_mode.value}")
            
            if len(task.execution_history) > 0:
                recent = task.execution_history[-10:]
                print(f"  Recent: min={min(recent):.1f}ms, "
                      f"max={max(recent):.1f}ms, "
                      f"avg={sum(recent)/len(recent):.1f}ms")
    
    # Get batch statistics from metrics
    batch_metrics = session.get_instrument("batch.completed")
    if batch_metrics:
        exported = batch_metrics.export()
        print(f"\nBatch Statistics:")
        print(f"  Completed: {exported.get('count', 0)}")
        
    skip_metrics = session.get_instrument("batch.tasks.skipped")
    if skip_metrics:
        exported = skip_metrics.export()
        print(f"  Tasks Skipped: {exported.get('count', 0)}")
```

## Benefits

**1. Maximum Throughput with High Completion Rate**
- CPU/GPU kept fully busy with productive work
- Skip frames proactively rather than abandon batches reactively
- Adaptive to system load without wasting resources
- High batch completion rate even under load

**2. Predictable Behavior**
- STRICT tasks guarantee consistency (all-or-nothing)
- LATENCY_TOLERANT tasks ensure continuous progress
- Clear contract for each task type
- No mid-execution surprises

**3. Graceful Degradation**
- System adapts to overload by delaying batch starts
- Skips frames intelligently based on prediction
- Maintains critical functionality (completes what it starts)
- Prevents cascading failures from abandoned work

**4. Observable Performance**
- EWMA tracks task performance over time
- Metrics show skip rates and completion rates
- Can tune timeouts based on actual data
- Clear visibility into prediction accuracy

## Future Enhancements

**1. Dynamic Timeout Adjustment**
```python
# Adjust batch timeout based on observed latencies
runner.batch_timeout_ms = int(sum(task.ewma_duration_ms for task in critical_path) * 1.5)
```

**2. Priority-Based Scheduling**
```python
class BaseTask:
    priority: int = 0  # Higher = more important
    
# Schedule high-priority tasks first when time is tight
```

**3. GPU Memory-Aware Scheduling**
```python
# Don't run multiple GPU tasks if they won't fit in memory
if gpu_memory_available() < task.gpu_memory_required:
    skip_task()
```

**4. Adaptive Quality Levels**
```python
# Use faster models when falling behind
if batch.time_remaining < 100:
    use_model("yolov8n.pt")  # Fast
else:
    use_model("yolov8x.pt")  # Accurate
```

## Implementation Checklist

- [ ] Add `TaskTimingMode` enum to `task_base.py`
- [ ] Add timing mode and EWMA tracking to `BaseTask`
- [ ] Create `BatchContext` dataclass
- [ ] Implement `_schedule_with_prediction()` in runner
- [ ] Update `run()` to use batch contexts
- [ ] Add timeout handling for strict vs tolerant tasks
- [ ] Update task implementations with appropriate timing modes
- [ ] Add batch metrics collection
- [ ] Add tests for frame skipping scenarios
- [ ] Add monitoring/dashboard utilities
- [ ] Document timing mode selection guidelines
- [ ] Benchmark throughput improvements

**Estimated Effort:** 1-2 days for core implementation + testing
