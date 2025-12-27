# VLMChat Pipeline Architecture

## Overview

The VLMChat pipeline system provides a flexible, DAG-based architecture for composing vision and language processing workflows. It features a **three-phase execution model** with clean separation between syntax (DSL), semantics (task instantiation), and execution (cursor-based runtime).

**Architecture Philosophy:**
- **Parser**: Builds Abstract Syntax Tree (AST) from DSL
- **Builder**: Decorates AST with task instances and wires execution graph  
- **Runner**: Executes via cursor-queue model navigating doubly-linked task graph

This separation enables DSL validation, visualization, and optimization while maintaining efficient runtime execution.

## Implementation Status

### ✅ Fully Implemented Features

**DSL & Parsing:**
- ✅ Lexer with full token support (identifiers, operators, literals, strings)
- ✅ Recursive descent parser building AST (TaskNode, SequenceNode, ParallelNode, LoopNode)
- ✅ Sequential pipelines: `task1 -> task2 -> task3`
- ✅ Parallel branches: `[branch1, branch2, branch3]`
- ✅ Loops with conditions: `{input -> :break_on(code=1): -> process}`
- ✅ Control operators: `:operator:` syntax for loop conditions
- ✅ Timing annotations: `task<100ms>` and `task<<50ms>>` (advisory/enforced)
- ✅ Parser returns `(sources, pipeline)` tuple for stream source support

**Execution Engine:**
- ✅ Cursor-queue execution model with ThreadPoolExecutor
- ✅ Doubly-linked task graph navigation (upstream/downstream tasks)
- ✅ Fork execution with independent context copies per branch
- ✅ Merge coordination with multi-cursor synchronization
- ✅ Loop execution via LoopConnector with break/continue support
- ✅ Exit code propagation through pipeline (0=success, 1=empty, 2=exception, 3+=custom)
- ✅ Trace event recording for debugging and metrics
- ✅ Metrics collection (execution time, task counts)

**Stream Sources:**
- ✅ Stream source polling with configurable intervals
- ✅ `wait(source)` syntax for source-driven pipelines
- ✅ Source registration and cursor spawning at wait() tasks
- ✅ Pipeline mode vs continuous mode (`pipeline=true/false`)
- ✅ Multiple sources per pipeline with independent polling

**Exception Handling:**
- ✅ Context-based exception propagation (exception, exception_source_task)
- ✅ Sequential flow: exceptions flow forward, tasks auto-skip
- ✅ Fork replication: exceptions copy to all branches
- ✅ Merge early exit: first exception terminates other branches
- ✅ Error handler tasks: LogErrorTask, ClearErrorTask, OnErrorTask, RethrowErrorTask
- ✅ Optional error handling via `handles_exceptions=True` flag
- ✅ Type-specific exception filtering in OnErrorTask

**Loop Control:**
- ✅ BreakOnCondition: exit loop on matching exit codes
- ✅ ContinueOnCondition: retry loop on matching exit codes
- ✅ DiagnosticCondition: max iterations and timeout limits
- ✅ BreakOnFailCondition: simple break on any non-zero exit code
- ✅ Stack-based loop state management for nested loops
- ✅ Loop iteration counting and timing metrics

**Task System:**
- ✅ BaseTask with input/output contracts
- ✅ Connector base class for structural tasks (Fork, Merge, Loop)
- ✅ Task configuration via `configure(params)` from DSL
- ✅ Task registry with `@register_task` decorator
- ✅ Parameter validation and type conversion
- ✅ Time budget support for cooperative scheduling

**Comprehensive Test Coverage:**
- ✅ Exception propagation (6 tests: sequential, fork, merge, handlers)
- ✅ Loop control (3 tests: break_on with exit codes)
- ✅ Parser functionality (DSL syntax validation)
- ✅ Integration tests (full pipeline execution)

### 🎯 Current Capabilities

The pipeline system now supports:

1. **Complex Control Flow**: Sequential, parallel, loops, and nested combinations
2. **Stream Processing**: Source-driven pipelines with continuous polling
3. **Error Recovery**: Exception propagation with handler tasks for graceful degradation
4. **Loop Control**: Exit code-based conditions for dynamic iteration
5. **Concurrent Execution**: ThreadPoolExecutor with cursor-based parallelism
6. **Type Safety**: Input/output contracts with static validation
7. **Observability**: Trace events and metrics for monitoring
8. **Flexible DSL**: Declarative syntax for complex workflows

## Core Concepts

### Three-Phase Pipeline Architecture

**Phase 1: Parser (DSL → AST)**
- Lexer tokenizes DSL text into operators, identifiers, literals
- Parser builds Abstract Syntax Tree (syntax structure)
- Nodes: TaskNode, SequenceNode, ParallelNode, LoopNode
- No task instantiation at this phase - pure syntax

**Phase 2: Builder (AST → Decorated AST)**
- First pass: Decorates AST nodes with task instances
- Wires `upstream_tasks` based on AST structure
- Second pass: Wires `downstream_tasks` (inverse of upstream)
- Result: Doubly-linked execution graph embedded in AST

**Phase 3: Runner (Execution)**
- Cursor-queue model navigates task graph
- Multiple cursors can traverse concurrently (pipelining)
- Merge coordination: multiple cursors wait at merge points
- Context flows through task graph, not AST

### Pipeline Components

**BaseTask** - Abstract base class for all pipeline tasks
- Defines `input_contract` and `output_contract` for static validation
- Implements `run(context) -> context` for task execution
- Supports `configure(params)` for parameter injection from DSL
- Includes `time_budget_ms` for cooperative timing control
- **Doubly-linked graph**: `upstream_tasks` and `downstream_tasks` lists

**Context** - Data container passed between tasks
- Holds typed data via `ContextDataType` enum (IMAGE, TEXT, DETECTIONS, EMBEDDINGS, etc.)
- Distinguishes mutable (DETECTIONS, CROPS) vs immutable (IMAGE, TEXT) data
- Supports `split()` for parallel branches with proper data handling
- Maintains reference to `pipeline_runner` for I/O queue access

**Cursor** - Execution pointer in task graph
- Holds `current_task` (not AST node) and `context` (data)
- Navigates via task's `downstream_tasks` list
- Multiple cursors enable concurrent pipeline traversal (pipelining)
- Readiness check: all `upstream_tasks` completed?

**PipelineRunner** - Orchestrates cursor-based execution
- Cursor-queue model: `task_queue` (waiting) and `ready_queue` (executable)
- Executes ready cursors in parallel via ThreadPoolExecutor
- **Merge coordination**: Tracks arriving cursors at merge points
- **Fork handling**: Copies context independently for each branch
- Integrates metrics collection and execution tracing
- **Provides I/O queues for environment-agnostic input/output**
- Thread-safe state management with Events

**DSL Parser** - Converts text → AST
- Lexer: Tokenizes DSL into operators, keywords, literals
- Parser: Recursive descent builds AST (SequenceNode, ParallelNode, etc.)
- No task instantiation - pure syntax tree
- Supports loops `{}`, parallel `[]`, control flow `:task():`

**Pipeline Builder** - Decorates AST with tasks
- Two-pass decoration:
  1. Create task instances, wire `upstream_tasks`
  2. Wire `downstream_tasks` (reverse links)
- Task registry maps DSL names → Python classes
- Parameter injection via `task.configure(**params)`
- Skips structural nodes (SequenceNode) in execution graph

## Execution Model Summary

**Cursor-Queue Navigation:**
- **Cursor**: Holds `current_task` (a BaseTask) + `context` (data)
- **ready_queue**: Cursors ready to execute (dependencies satisfied)
- **task_queue**: Cursors waiting on dependencies
- **completed_tasks**: Set of finished task IDs

**Navigation Algorithm:**
```python
1. Find root tasks from AST structure
2. Create initial cursors at root tasks
3. While cursors remain:
   a. Check which cursors are ready (all upstreams completed)
   b. Execute ready cursors in parallel
   c. Advance: spawn new cursors at downstream_tasks
   d. Handle fork (copy context) and merge (wait for all branches)
4. Return final context
```

**Key Behaviors:**
- **Fork**: Copies context independently for each branch
- **Merge**: Waits until all upstream branches complete, then executes merge with all contexts
- **Concurrent Pipelining**: Multiple cursors traverse graph simultaneously
- **No AST During Execution**: Runner navigates task graph directly via downstream_tasks

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                      DSL PIPELINE TEXT                         │
│  camera() -> detect() -> [clip_text(), clip_vision()] -> merge│
└───────────────────────────┬────────────────────────────────────┘
                            │
                   ┌────────▼─────────┐
                   │  PHASE 1: PARSER │
                   │  DSL → AST       │
                   └────────┬─────────┘
                            │
                ┌───────────▼──────────────┐
                │  Abstract Syntax Tree    │
                │  ┌──────────────────┐   │
                │  │  SequenceNode    │   │
                │  │   tasks: [...]   │   │
                │  │   ├─ TaskNode    │   │
                │  │   ├─ TaskNode    │   │
                │  │   └─ ParallelNode│   │
                │  │      ├─ TaskNode │   │
                │  │      └─ TaskNode │   │
                │  └──────────────────┘   │
                └───────────┬──────────────┘
                            │
                   ┌────────▼──────────┐
                   │ PHASE 2: BUILDER  │
                   │ Decorate with     │
                   │ task instances    │
                   └────────┬──────────┘
                            │
            ┌───────────────▼────────────────┐
            │  Doubly-Linked Task Graph      │
            │  ┌──────┐    ┌──────┐          │
            │  │camera│◄──►│detect│          │
            │  └──┬───┘    └──┬───┘          │
            │     │           │              │
            │     │       ┌───▼────┐         │
            │     │       │  fork  │         │
            │     │       └───┬────┘         │
            │     │           │              │
            │     │     ┌─────┴──────┐       │
            │     │     │            │       │
            │     │  ┌──▼───┐    ┌──▼───┐   │
            │     │  │clip_T│    │clip_V│   │
            │     │  └──┬───┘    └──┬───┘   │
            │     │     └─────┬─────┘       │
            │     │           │             │
            │     │       ┌───▼────┐        │
            │     │       │ merge  │        │
            │     │       └────────┘        │
            └────────────────┬───────────────┘
                             │
                    ┌────────▼──────────┐
                    │  PHASE 3: RUNNER  │
                    │  Cursor execution │
                    └────────┬──────────┘
                             │
        ┌────────────────────▼─────────────────────┐
        │          Cursor-Queue Execution          │
        │  ready_queue: [Cursor#1@camera]          │
        │  task_queue:  [Cursor#2@detect]          │
        │                                           │
        │  Cursors navigate via downstream_tasks   │
        │  Multiple cursors = pipelined execution  │
        │  Merge coordination: wait for all branches│
        └───────────────────────────────────────────┘
```

## Context Data Types

```python
class ContextDataType(Enum):
    TEXT = "text"               # Text data (user input, model output) - immutable
    IMAGE = "image"             # PIL Image (immutable)
    DETECTIONS = "detections"   # List of Detection objects (mutable)
    CROPS = "crops"             # List of image crops (mutable)
    EMBEDDINGS = "embeddings"   # List of embedding vectors (mutable)
    MATCHES = "matches"         # Semantic matches (mutable)
    PROMPT_EMBEDDINGS = "prompt_embeddings"  # Text embeddings (immutable)
    AUDIT = "audit"             # Execution metadata (mutable)
```

**Mutable vs Immutable:**
- **Immutable data** (IMAGE, TEXT, PROMPT_EMBEDDINGS): Shared by reference across branches
- **Mutable data** (DETECTIONS, CROPS, EMBEDDINGS, MATCHES): Deep-copied when splitting to prevent interference

## Environment-Agnostic I/O System

The pipeline uses **queue-based I/O coordination** to work seamlessly across console, web services, GUI, or test environments.

### Architecture

**Design Philosophy:**
- **PipelineRunner knows nothing about the environment** (console, web, GUI)
- **Tasks use queues instead of direct I/O** (`input()`, `print()`)
- **External environment feeds/drains queues** and displays results
- **Thread-safe by design** using `queue.Queue` and `threading.Event`

### Components

**PipelineRunner I/O Interface:**
```python
class PipelineRunner:
    def __init__(self, pipeline):
        self.input_queue = queue.Queue()   # External → Pipeline
        self.output_queue = queue.Queue()  # Pipeline → External
        self._running = threading.Event()  # State: pipeline executing
        self._stop_requested = threading.Event()  # Graceful shutdown
    
    def send_input(self, text: str) -> None:
        """External environment sends input to pipeline."""
        self.input_queue.put(text)
    
    def get_output(self, timeout: float = 0.1) -> Optional[str]:
        """External environment polls for output."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def is_running(self) -> bool:
        """Check if pipeline is executing."""
        return self._running.is_set()
    
    def request_stop(self) -> None:
        """Request graceful pipeline shutdown."""
        self._stop_requested.set()
```

**ConsoleInputTask:**
```python
class ConsoleInputTask(BaseTask):
    def run(self, context: Context) -> Context:
        # Get input from runner's queue (not stdin)
        runner = context.pipeline_runner
        user_input = runner.input_queue.get(timeout=60)
        
        # Store in TEXT context
        context.data[ContextDataType.TEXT].append(user_input)
        
        # Set exit code: 0=success, 1=empty input
        self.exit_code = 0 if user_input else 1
        return context
```

**ConsoleOutputTask:**
```python
class ConsoleOutputTask(BaseTask):
    def run(self, context: Context) -> Context:
        # Get TEXT from context
        text = context.data[ContextDataType.TEXT][-1]
        
        # Send to runner's queue (not stdout)
        runner = context.pipeline_runner
        runner.output_queue.put(text)
        
        return context
```

### Environment Integration

**Console Application:**
```python
# Console detects pipeline running and routes I/O
while True:
    runner = app._pipeline_runner
    is_pipeline_running = runner and runner.is_running()
    
    # Change prompt: > (command) vs % (pipeline)
    prompt = "\n% " if is_pipeline_running else "\n> "
    user_input = input(prompt).strip()
    
    if user_input.startswith('/'):
        # Handle commands normally
        process_command(user_input)
    elif is_pipeline_running:
        # Forward non-command input to pipeline
        runner.send_input(user_input)
        
        # Poll for output and display
        while True:
            output = runner.get_output(timeout=0.1)
            if output is None:
                break
            print(output)
    else:
        print("Type /help for commands")
```

**Web Service (Future):**
```python
@app.post("/pipeline/input")
def pipeline_input(text: str):
    runner.send_input(text)
    return {"status": "sent"}

@app.get("/pipeline/output")
def pipeline_output():
    outputs = []
    while True:
        output = runner.get_output(timeout=0.1)
        if output is None:
            break
        outputs.append(output)
    return {"outputs": outputs}
```

**GUI (Future):**
```python
# User types in text widget
def on_user_input(text):
    runner.send_input(text)

# Background thread drains output queue
def output_poller():
    while runner.is_running():
        output = runner.get_output(timeout=0.5)
        if output:
            display_widget.append_text(output)
```

### Benefits

✅ **Environment-Agnostic**: Same pipeline runs in console, web, GUI, tests
✅ **Thread-Safe**: `queue.Queue` handles concurrent access atomically
✅ **Clean Separation**: Tasks don't know about console, web, or GUI
✅ **No Coordination Code**: Queues handle synchronization naturally
✅ **Simpler Than Alternatives**: No pipes, /dev/tty, or complex IPC

### Thread Safety

**State Management:**
- `_running` Event: Set when pipeline executing, clear when done
- `_stop_requested` Event: Set by external `/stop` command
- Both use atomic operations (no locks needed)

**Queue Operations:**
- `put()` and `get()` are thread-safe by design
- Console thread puts to input_queue
- Pipeline thread gets from input_queue
- Pipeline thread puts to output_queue
- Console thread gets from output_queue

**No Race Conditions:**
- Queue operations are atomic
- Events use thread-safe primitives
- Context owned by single pipeline thread
- No shared mutable state between threads

### Example Pipeline

**SmolVLM Chat (pipelines/smolvlm_chat.dsl):**
```dsl
{
    [camera(type="none"), console_input(prompt="Command: ") -> break_on(code=1)]
    -> history_update(id="hist", prompt=true, format="simple")
    -> smolvlm(system_prompt="You are a helpful vision assistant.")
    -> history_update(id="hist", response=true)
    -> console_output()
}
```

**Execution Flow:**
1. Console displays `%` prompt (pipeline mode)
2. User types message, console calls `runner.send_input(message)`
3. `console_input_task` gets from `input_queue`, adds to TEXT context
4. History task formats TEXT for model
5. SmolVLM generates response, adds to TEXT context
6. History task stores conversation pair
7. `console_output_task` puts TEXT to `output_queue`
8. Console polls `get_output()` and prints response
9. Loop repeats until empty input (triggers `break_on`)

## Task Adapters

### CameraTask
Captures images from camera devices.

**Configuration:**
```python
camera = factory.create_task("camera", "cam0", {
    "type": "imx219",      # Camera type: imx219, imx500, image_library, none
    "device": "0",         # Device ID
    "resolution": "640x480" # Optional resolution
})
```

**Contracts:**
- Input: None (source task)
- Output: IMAGE (PIL.Image)

### DetectorTask
Runs object detection on images.

**Configuration:**
```python
detector = factory.create_task("detector", "yolo", {
    "type": "yolo_cpu",    # Detector type: yolo, yolo_cpu, imx500
    "model": "yolov8n.pt", # Model path
    "confidence": "0.25",  # Confidence threshold
    "device": "cuda:0"     # Optional: cuda device (for yolo type)
})
```

**Contracts:**
- Input: IMAGE (required), DETECTIONS (optional from previous stage)
- Output: DETECTIONS (list of Detection objects)

### ConsoleInputTask
Captures user text input via pipeline runner's input queue.

**Configuration:**
```python
console_input = factory.create_task("console_input", "input1", {
    "prompt": "You: "  # Text to display (informational only)
})
```

**Contracts:**
- Input: None (can be source or follow start task)
- Output: TEXT (str)

**Exit Codes:**
- 0: Non-empty input received (success)
- 1: Empty input received (can trigger break_on)

**I/O Model:**
- Gets input from `context.pipeline_runner.input_queue`
- External environment (console, web, GUI) feeds the queue
- Blocks with 60-second timeout waiting for input

### ConsoleOutputTask
Displays text via pipeline runner's output queue.

**Configuration:**
```python
console_output = factory.create_task("console_output", "output1", {
    "which": "last"  # Options: "all", "first", "last"
})
```

**Contracts:**
- Input: TEXT (required)
- Output: None (sink task)

**I/O Model:**
- Sends TEXT to `context.pipeline_runner.output_queue`
- External environment polls queue and displays output
- Supports outputting "all", "first", or "last" TEXT entries

### SmolVLMTask
Runs vision-language model inference.

**Self-Instantiation:**
- Creates own SmolVLMModel via lazy `@property`
- No dependency injection required

**Configuration:**
```python
smolvlm = factory.create_task("smolvlm", "vlm1", {
    "system_prompt": "You are a helpful assistant.",
    "runtime": "onnx"  # Options: "onnx", "transformers", "auto"
})
```

**Contracts:**
- Input: IMAGE, TEXT (expects TEXT[-2]=history, TEXT[-1]=user_input)
- Output: TEXT (model response appended to context)

**Operation:**
- Reads TEXT[-2] for conversation history
- Reads TEXT[-1] for current user input
- Builds messages list with system prompt
- Generates response and appends to TEXT

### HistoryUpdateTask
Updates conversation history with stateful prompt/response modes.

**State Sharing:**
- Use explicit `id` parameter to share state across invocations
- Same `id` reuses task instance (maintains history)

**Configuration:**
```python
# Pre-model: prepare history + user input
history_pre = factory.create_task("history_update", "hist", {
    "id": "hist",           # Required for state sharing
    "prompt": "true",       # Mode: prepare for model
    "format": "simple"      # Format: xml, simple, json, markdown
})

# Post-model: store response
history_post = factory.create_task("history_update", "hist", {
    "id": "hist",           # Same ID = same instance
    "response": "true",     # Mode: store response
    "max_pairs": "10"       # Optional: limit history
})
```

**Contracts:**
- Prompt mode: Reads TEXT[-1] (user input), formats history, appends both
- Response mode: Reads TEXT[-1] (model response), stores in history

**Modes:**
- **prompt=true**: Pops TEXT[-1], appends formatted_history, appends user_input
- **response=true**: Reads TEXT[-1], updates last conversation pair

### StartTask
Empty source task for split entry points.

```python
start = factory.create_task("start", "start")
```

**Contracts:**
- Input: None (source task)
- Output: None (passes through context unchanged)

## Connector Subclasses

Connectors are specialized task types that appear as nodes in the execution graph. They handle **merge strategies** - how to combine multiple input contexts into one output context.

### Connector (Base)
Default merge strategy: concatenates mutable data, takes first immutable data.

**How it works in the execution graph:**
```
Task A ─┐
        ├─> Connector (merge) ─> Next Task
Task B ─┘
```

The connector is a normal task node with multiple upstream_tasks. When all upstream tasks complete, the runner calls `run_connector(contexts)` with all incoming contexts.

**DSL Example:**
```
[split(): branch_a, branch_b :merge()]
```

### FirstCompleteConnector
Takes the first branch to complete, ignores others. Useful for race conditions.

**How it works:**
```
Fast Branch ──┐
              ├─> FirstComplete ──> (only uses first arrival)
Slow Branch ──┘
```

The connector executes as soon as the first upstream task completes. Other branches are ignored.

**DSL Example:**
```
[split(): fast_model, accurate_model :first_complete()]
```

### OrderedMergeConnector
Reorders branch results before merging based on configuration.

**How it works:**
```python
# Configuration specifies branch order
ordered_merge = factory.create_connector("ordered_merge", "merge1", {
    "order": "2,1"  # Process second branch first, then first branch
})
```

Useful when branch execution order doesn't match desired processing order.

**DSL Example:**
```
[split(): high_priority, low_priority :ordered_merge(order=2,1)]
```

### ForkConnector and MergeConnector

**ForkConnector** - Special task that marks the start of parallel branches:
- Appears in AST as `ParallelNode.fork_task`
- When cursor reaches fork, runner **copies context** for each downstream branch
- Ensures branches have independent data (no mutation interference)

**MergeConnector** - Special task that marks the end of parallel branches:
- Appears in AST as `ParallelNode.merge_task`
- Runner uses **merge coordination**: waits for all upstream branches to complete
- Executes once with all contexts: `run_connector([context1, context2, ...])`

**Execution model:**
```
Fork ────┬──> Branch 1 ────┐
         │                  ├──> Merge
         └──> Branch 2 ────┘
         
# Fork copies context independently for each branch
# Merge waits until both branches arrive, then combines contexts
```

## Usage Examples

### Example 1: Simple Linear Pipeline (DSL)

**Recommended approach:** Use DSL for pipeline definition.

```python
from src.pipeline.dsl_parser import parse_pipeline
from src.pipeline.pipeline_runner import PipelineRunner
from src.pipeline.task_base import Context

# Define pipeline in DSL
dsl_text = """
camera(type=none, device=0) -> 
detect(type=yolo_cpu, model=yolov8n.pt, confidence=0.3) -> 
viewer(type=cv)
"""

# Parse DSL (Parser creates AST)
ast_root = parse_pipeline(dsl_text)

# Build decorated AST (Builder wires task graph)
from src.pipeline.dsl_parser import PipelineBuilder
builder = PipelineBuilder()
decorated_ast = builder.build(ast_root)

# Execute pipeline (Runner uses cursor-queue model)
runner = PipelineRunner(decorated_ast)
context = Context()
result = runner.run(context)

# Access results
detections = result.data.get(ContextDataType.DETECTIONS, [])
print(f"Found {len(detections)} objects")

runner.shutdown()
```

**What happens:**
1. Parser: Tokenizes DSL → Builds AST (SequenceNode with 3 TaskNodes)
2. Builder: Creates task instances → Wires upstream/downstream links
3. Runner: Creates cursor at camera_task → Executes → Advances to detector → Advances to viewer

### Example 2: Parallel Branches with Fork/Merge (DSL)

```python
from src.pipeline.dsl_parser import parse_pipeline
from src.pipeline.pipeline_runner import PipelineRunner
from src.pipeline.task_base import Context

# Define parallel pipeline
dsl_text = """
[split(): 
    camera(type=none, device=0), 
    console_input(prompt="You: ")
:merge()] -> vlm(type=smolvlm) -> viewer(type=text)
"""

# Parse and build
ast_root = parse_pipeline(dsl_text)
builder = PipelineBuilder()
decorated_ast = builder.build(ast_root)

# Execute
runner = PipelineRunner(decorated_ast)
context = Context()
result = runner.run(context)

runner.shutdown()
```

**What happens:**
1. Parser: Creates ParallelNode with fork/merge tasks
2. Builder: Wires fork → [camera, console_input] → merge → vlm → viewer
3. Runner execution:
   - Cursor reaches fork → **copies context** → spawns 2 cursors
   - Camera cursor executes camera task
   - Console cursor executes console_input task
   - Both cursors reach merge → **merge coordination waits**
   - When both arrive → merge executes with both contexts → cursor continues to vlm

**Key insight:** Fork copies context independently, merge waits for all branches.

### Example 3: Legacy Factory API (Deprecated)

The old PipelineFactory API still works but is deprecated in favor of DSL:

```python
from src.pipeline.pipeline_factory import create_default_factory
from src.pipeline.pipeline_runner import PipelineRunner
from src.pipeline.task_base import Context

# Create factory
factory = create_default_factory()

# Create tasks with configuration
camera = factory.create_task("camera", "cam0", {
    "type": "none",
    "device": "0"
})
detector = factory.create_task("detector", "yolo", {
    "type": "yolo_cpu",
    "model": "yolov8n.pt",
    "confidence": "0.3"
})

# Build pipeline
pipeline = factory.create_connector("connector", "simple_pipeline")
pipeline.add_task(camera)
pipeline.add_task(detector)
pipeline.add_edge(camera, detector)

# Run pipeline
runner = PipelineRunner(pipeline)
context = Context()
result = runner.run(context)

# Access results
detections = result.data.get(ContextDataType.DETECTIONS, [])
print(f"Found {len(detections)} objects")

runner.shutdown()
```

**Note:** This API bypasses the three-phase model and directly manipulates the task graph. Use DSL for new development.

### Example 4: Race Condition with FirstComplete (DSL)

Use FirstCompleteConnector to take the fastest result:

```python
from src.pipeline.dsl_parser import parse_pipeline

# Define race between fast and accurate detectors
dsl_text = """
camera(type=none, device=0) -> 
[split(): 
    detect(type=yolo_cpu, model=yolov8n.pt, confidence=0.5),
    detect(type=yolo_cpu, model=yolov8x.pt, confidence=0.3)
:first_complete()] -> viewer(type=cv)
"""

ast_root = parse_pipeline(dsl_text)
# ... build and run as before
```

**What happens:**
- Fast detector (yolov8n) likely completes first
- FirstComplete connector returns first arrival's context
- Slow detector (yolov8x) result discarded
- Useful for latency-sensitive applications with backup computation



## Metrics Integration

The pipeline automatically collects metrics when a `Collector` is provided. Metrics are initialized from the AST structure during runner initialization.

```python
from src.metrics.metrics_collector import Collector, Session
from src.metrics.instruments import AverageDurationInstrument, CountInstrument
from src.pipeline.dsl_parser import parse_pipeline, PipelineBuilder
from src.pipeline.pipeline_runner import PipelineRunner

# Parse DSL
dsl_text = "camera() -> detect() -> viewer()"
ast_root = parse_pipeline(dsl_text)
builder = PipelineBuilder()
decorated_ast = builder.build(ast_root)

# Create collector and session
collector = Collector("pipeline_metrics")
session = Session(collector)

# Add instruments
session.add_instrument(
    AverageDurationInstrument("task.execution.duration", binding_keys=["task_id"]),
    "task.execution.duration"
)
session.add_instrument(
    CountInstrument("task.execution.count", binding_keys=["task_id", "status"]),
    "task.execution.count"
)

# Create runner with collector (passes AST root for initialization)
runner = PipelineRunner(decorated_ast, collector=collector)
result = runner.run(context)

# Export metrics
for ts_name, inst in session._instruments:
    exported = inst.export()
    print(exported)
```

**Collected Metrics:**
- `task.execution.duration` - Time spent in each task (per task_id)
- `task.execution.count` - Success/failure counts per task
- `task.failures` - Failure counts by error type
- `context.data.size` - Size of data in context (when tracked)

**How it works:**
- Runner initialization walks AST to discover all tasks
- Instruments bind to task_ids from AST structure
- Execution records metrics for each cursor advancement
- Merge coordination properly attributes execution time to merge task

## Execution Trace

The pipeline records detailed execution traces for debugging and analysis. Traces survive cursor navigation, fork copying, and merge coordination.

```python
from src.pipeline.diagnostic_task import print_trace
from src.pipeline.dsl_parser import parse_pipeline, PipelineBuilder
from src.pipeline.pipeline_runner import PipelineRunner

# Parse and build
dsl_text = "[split(): camera(), console_input() :merge()] -> vlm()"
ast_root = parse_pipeline(dsl_text)
builder = PipelineBuilder()
decorated_ast = builder.build(ast_root)

# Create runner with trace enabled
runner = PipelineRunner(decorated_ast, enable_trace=True)
result = runner.run(context)

# Display trace
print_trace(result)
```

**Output Example:**
```
======================================================================
EXECUTION TRACE
======================================================================
        Time |       Thread | Task                 | Upstreams            | Event
----------------------------------------------------------------------
        0.0ms |   8368922816 | fork                 | -                    | split
       12.3ms |   8368922817 | camera               | fork                 | execute
       15.2ms |   8368922818 | console_input        | fork                 | execute
       28.4ms |   8368922816 | merge                | camera, console_input | merge
       30.1ms |   8368922816 | vlm                  | merge                | execute
======================================================================
Total events: 5
======================================================================
```

**Trace Features:**
- **Cursor-aware**: Each cursor records its execution path
- **Fork copying**: When fork copies context, trace data is copied independently
- **Merge coordination**: Merge receives traces from all incoming branches
- **Chronological timeline**: Events sorted by timestamp
- **Thread tracking**: Identifies parallel execution (multiple cursors)
- **Dependency graph**: Shows task relationships via upstreams
- **Event types**: `execute` (normal task), `split` (fork), `merge` (join)

**Trace Backends** (see `src/pipeline/trace.py`):
- `InMemoryTrace`: Store events in context data (default)
- `LogTrace`: Emit events to logger for aggregation (future)
- `NoOpTrace`: Disable tracing for performance (future)

**Trace Data Format:**
```python
# Each event is a tuple:
(timestamp, thread_id, task_id, upstream_task_ids, event_type)
# Example:
(1234567890.123, 8368922816, 'camera', ['fork'], 'execute')
(1234567890.456, 8368922816, 'merge', ['camera', 'console_input'], 'merge')
```

**How It Works with Cursors:**
1. Initial cursor has empty trace
2. Each task execution adds trace event to cursor's context
3. Fork copies context → each branch cursor gets independent trace copy
4. Merge combines traces from all arriving cursors
5. Final result context contains complete execution history

**Use Cases:**
- Debug cursor navigation and execution order
- Verify fork/merge coordination working correctly
- Analyze performance bottlenecks (identify slow tasks)
- Verify pipeline structure at runtime
- Monitor production pipelines (with LogTrace)

## Logging

All pipeline components log at INFO level by default:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

**Log Events:**
- Pipeline initialization (worker count, collector)
- Graph building (task count, validation results)
- Batch execution (ready tasks, completion progress)
- Task start/completion (task ID, type)
- Connector operations (merge/split with context counts)
- Errors and warnings (failures, timeouts, validation issues)

## Contract Validation

The pipeline validates data contracts at graph build time:

```python
class MyTask(BaseTask):
    def __init__(self):
        super().__init__("my_task")
        # Define what data this task needs and produces
        self.input_contract = {
            ContextDataType.IMAGE: Image.Image,
            ContextDataType.PROMPT: str
        }
        self.output_contract = {
            ContextDataType.RESPONSE: str
        }
```

**Validation Rules:**
- Source tasks must have empty input contracts
- Sink tasks have no successors
- All task inputs must be satisfied by predecessor outputs
- Graph must be acyclic (no cycles detected)
- All tasks must be reachable from sources

**Validation Errors:**
```
ValueError: Task 'detector' requires IMAGE but no predecessor provides it
ValueError: Cycle detected: task_a -> task_b -> task_a
ValueError: Graph validation failed: 0 source tasks found (need at least 1)
```

## Time Budget Support

Tasks can specify execution time budgets:

```python
class MyTask(BaseTask):
    def __init__(self):
        super().__init__("my_task", time_budget_ms=5000)  # 5 second budget
    
    def run(self, context: Context) -> Context:
        self._record_start()  # Start timing
        
        while processing:
            if not self.should_continue():
                # Time budget exceeded, wrap up gracefully
                break
            # ... do work ...
        
        return context
```

**Current Status:** Infrastructure in place, enforcement pending.

## Planned DSL Support

The pipeline is designed to support DSL-based definitions:

```
# Linear chain
camera(type=imx219,device=0) -> 
detector(type=yolo_cpu,model=yolov8n.pt) -> 
viewer

# Parallel branches with merge
start -> [clone_split(): 
    camera(device=0), 
    console_input(prompt="You: ")
:merge()] -> 
smolvlm -> 
console_output

# Race condition
camera -> [clone_split():
    detector(model=yolov8n.pt,confidence=0.5),
    detector(model=yolov8x.pt,confidence=0.3)
:first_complete()] -> 
viewer
```

**Grammar (BNF):**
```bnf
<pipeline>     ::= <chain>
<chain>        ::= <stage> | <stage> "->" <chain>
<stage>        ::= <node> | <connector>
<node>         ::= <task_name> | <task_name> "(" <params> ")"
<connector>    ::= "[" <split_func> ":" <branches> ":" <merge_func> "]"
<split_func>   ::= "clone_split()" | "partition_split()" | "filter_split(" <params> ")"
<merge_func>   ::= "merge()" | "merge(" <params> ")" | "first_complete()" | "ordered_merge(" <params> ")"
<branches>     ::= <chain> | <chain> "," <branches>
<params>       ::= <param> | <param> "," <params>
<param>        ::= <key> "=" <value>
<task_name>    ::= <identifier>
<key>          ::= <identifier>
<value>        ::= <string> | <number> | <identifier>
```

## Testing

Run pipeline tests to verify the three-phase architecture:

```bash
cd /Users/patrick/Dev/VLMChat

# Test simple sequence (camera -> detector)
python test_camera_yolo_viewer.py

# Test parallel fork/merge
python test_integration_complete.py

# Test CLIP pipeline (complex merge coordination)
python test_clip_pipeline_build.py

# Run all DSL parser tests
pytest tests/pipeline/ -v
```

**Test Coverage:**
1. **Simple Sequence**: camera() -> detect() -> viewer()
   - Tests: cursor creation, sequential advancement, context flow
2. **Parallel Fork/Merge**: [split(): branch_a, branch_b :merge()]
   - Tests: fork context copying, merge coordination, cursor synchronization
3. **CLIP Pipeline**: Complex merge with 2 branches
   - Tests: merge coordination correctness, arrival tracking
4. **Loops**: {camera() -> detect() :break_on(exit_code=1)}
   - Tests: loop cursor reinjection, exit code handling
5. **DSL Parser**: Tokenization, AST construction, builder decoration
   - Tests: visitor pattern, downstream wiring, edge cases

**What Tests Verify:**
- ✅ Parser creates correct AST structure
- ✅ Builder wires upstream_tasks and downstream_tasks properly
- ✅ Runner creates cursors at correct root tasks (from AST)
- ✅ Fork copies context independently (no mutation interference)
- ✅ Merge waits for all branches (coordination works)
- ✅ Final context returned (not initial input)
- ✅ Trace events survive fork/merge operations
- ✅ Metrics collect correctly for all task types

**Running Specific Tests:**
```bash
# Test AST construction
pytest tests/pipeline/test_dsl_parser.py::test_ast_construction -v

# Test downstream wiring
pytest tests/pipeline/test_dsl_parser.py::test_downstream_wiring -v

# Test cursor execution
pytest tests/pipeline/test_runner.py::test_cursor_advancement -v

# Test merge coordination
pytest tests/pipeline/test_runner.py::test_merge_coordination -v
```

## File Organization

The pipeline system is organized into three main layers matching the architecture:

```
src/pipeline/
├── task_base.py              # Core abstractions (BaseTask, Context)
│                             # - BaseTask with upstream_tasks + downstream_tasks
│                             # - Context with data dictionary
├── dsl_parser.py            # Parser + Builder (DSL → Decorated AST)
│                             # - Lexer: tokenizes DSL
│                             # - Parser: builds AST (TaskNode, SequenceNode, etc.)
│                             # - PipelineBuilder: decorates AST with tasks
│                             # - Visitor pattern base classes
├── pipeline_runner.py        # Runner (Cursor-Queue Execution)
│                             # - Cursor dataclass (current_task + context)
│                             # - Cursor-queue execution loop
│                             # - Merge coordination (merge_arrivals)
│                             # - Fork context copying
│                             # - Metrics and trace integration
├── connectors.py            # Connector subclasses (merge strategies)
│                             # - Connector (base merge)
│                             # - FirstCompleteConnector
│                             # - OrderedMergeConnector
│                             # - ForkConnector (marks parallel start)
│                             # - MergeConnector (marks parallel end)
├── loop_connector.py        # Loop execution with exit codes
├── trace.py                 # Execution trace recording
│                             # - InMemoryTrace (default)
│                             # - LogTrace, NoOpTrace (future)
├── pipeline_factory.py      # Legacy factory API (deprecated)
└── tasks/
    ├── camera_task.py           # Camera adapter
    ├── detector_task.py         # Object detector adapter
    ├── console_input_task.py    # Console input (queue-based)
    ├── console_output_task.py   # Console output (queue-based)
    ├── smolvlm_task.py         # VLM inference adapter
    ├── history_update_task.py   # History update (dual-mode)
    ├── break_on_task.py         # Exit code condition
    └── start_task.py           # Empty source task
```

**Key Architectural Files:**
- `dsl_parser.py`: Implements Parser + Builder phases
- `pipeline_runner.py`: Implements Runner phase with cursor-queue model
- `task_base.py`: Defines doubly-linked task graph structure
- `connectors.py`: Merge strategies appear as tasks in execution graph

## Pipeline DSL Specification

The VLMChat Pipeline DSL provides a declarative syntax for defining complex vision processing workflows with loops, parallel execution, and conditional control flow.

### Design Philosophy

**Goals:**
- **Minimal**: Few operators, consistent semantics
- **Readable**: Visual structure matches execution flow
- **Composable**: Nesting works naturally
- **Secure**: No arbitrary code execution (parameterized conditions only)
- **Extensible**: Add tasks/conditions without syntax changes

**Non-Goals:**
- General-purpose programming language
- Dynamic code evaluation
- Turing-complete computation

### Core Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `->` | Sequential flow | `a -> b -> c` |
| `[]` | Parallel grouping | `[a, b, c]` |
| `{}` | Loop scope | `{tasks}` |
| `:` | Control flow marker | `:timeout(60)` |

### BNF Grammar

```bnf
<pipeline>         ::= <task_sequence> | <loop>

<task_sequence>    ::= <task> ( "->" <task> )*

<task>             ::= <regular_task> 
                     | <control_task>
                     | <parallel>
                     | <loop>

<regular_task>     ::= <identifier> "(" <params>? ")"

<control_task>     ::= ":"? <identifier> "(" <params>? ")" ":"?

<parallel>         ::= "[" <parallel_body> "]"

<parallel_body>    ::= <split_op>? <task_list> <merge_op>?

<split_op>         ::= <identifier> "():"

<merge_op>         ::= ":" <identifier> "()"

<task_list>        ::= <task> ( "," <task> )*

<loop>             ::= "{" <task_sequence> "}"

<params>           ::= <param> ( "," <param> )*

<param>            ::= <identifier> "=" <value>

<value>            ::= <string> | <number> | <boolean>

<identifier>       ::= [a-zA-Z_][a-zA-Z0-9_]*

<string>           ::= '"' [^"]* '"'

<number>           ::= [0-9]+ ( "." [0-9]+ )?

<boolean>          ::= "true" | "false"
```

**Lexical Notes:**
- Whitespace is ignored except within strings
- Spare arrows adjacent to colons are no-ops: `->:` = `:->` = `:`
- Comments: `#` to end of line

### Syntax Rules

#### 1. Sequential Flow
```dsl
a -> b -> c
```
Tasks execute in order: `a`, then `b`, then `c`.

#### 2. Parallel Execution
```dsl
# Implicit default split/merge
[a, b, c]

# Explicit split strategy
[round_robin(): a, b, c]

# Explicit merge strategy
[a, b, c :priority_merge()]

# Both explicit
[fanout(): a, b, c :union()]
```

**Semantics:**
- Tasks execute in parallel (implementation may be sequential)
- Split strategy determines how input context is divided
- Merge strategy determines how outputs are combined
- Position determines role: split after `[`, merge before `]`

#### 3. Loops
```dsl
# Infinite loop (no exit condition)
{camera -> process -> save}

# Loop with exit condition
{camera -> process ->:timeout(60)}

# Loop with entry guard
{:check_daylight() -> camera -> process}

# Loop with multiple controls
{
  :check_ready() ->
  camera ->
  detect ->
  :break_if_empty():continue_if_few(min=5) ->
  process ->
  :timeout(300):count(1000)
}
```

**Semantics:**
- Loop body executes repeatedly
- Control tasks (`:task()`) can break or continue
- Empty loops `{}` are syntax errors

#### 4. Control Flow

**Control Task Syntax:**
```dsl
:task()      # Leading colon
task():      # Trailing colon
:task():     # Both (style preference)
```

All forms are equivalent - the colon marks the task as a control task.

**Control Actions:**
- `PASS`: Continue to next task (default)
- `BREAK`: Exit loop immediately
- `CONTINUE`: Jump to loop start, skip remaining tasks

**Control Evaluation:**
```dsl
:a():b():c()
```
Sequential evaluation (C convention):
1. Evaluate `a()`, if not `PASS`, stop
2. Evaluate `b()`, if not `PASS`, stop
3. Evaluate `c()`

**Control Scope:**
- Control tasks only valid inside `{}` loops
- Using `:task()` outside loop is syntax error

#### 5. Nesting

```dsl
# Nested loops
{
  outer_setup() ->
  {
    inner_task() ->:break_if_done()
  } ->
  outer_cleanup() ->
  :timeout(300)
}

# Parallel with loops
[
  {continuous_capture() ->:timeout(60)},
  {process_batch() ->:count(10)}
]

# Loops with parallel
{
  camera() ->
  [detect(), track()] ->
  merge() ->
  :timeout(120)
}
```

All combinations of nesting are supported.

### Complete Examples

#### Example 1: Continuous Vision Pipeline
```dsl
{
  [camera(), prompt_injector()] ->
  detect(model=yolov8n, confidence=0.5) ->
  [
    cluster(max_clusters=3, threshold=0.5),
    pass()
  ] ->
  merge(target_count=8) ->
  clip_vision() ->
  clip_compare(min_similarity=0.25) ->
  context_cleanup(keep_types=prompt_embeddings,metrics) ->
  :timeout(300)
}
```

**Execution:**
1. Start loop
2. Parallel: capture camera + inject prompts
3. Run YOLO detection
4. Parallel: cluster detections + pass originals
5. Merge to 8 detections
6. Generate CLIP embeddings
7. Compare to prompts
8. Clean up context (keep only immutable data)
9. Check timeout (break if >= 300s)
10. If not broken, return to step 2

#### Example 2: Retry Pattern
```dsl
{
  camera() ->
  detect() ->
  :continue_if_empty():continue_if_few(min=5) ->
  process() ->
  save() ->
  :count(100)
}
```

**Execution:**
- If detection returns empty: restart loop (continue)
- If detection returns < 5: restart loop (continue)
- Otherwise: process and save
- Exit after 100 successful iterations

#### Example 3: Multiple Exit Conditions
```dsl
{
  :check_daylight():check_battery() ->
  camera() ->
  :break_if_dark() ->
  detect() ->
  :break_if_error() ->
  process() ->
  :timeout(60):count(1000):check_disk_full()
}
```

**Entry guards:** Must pass `check_daylight()` AND `check_battery()` to start

**Mid-loop breaks:** Stop if camera dark or processing error

**Exit conditions:** Stop if timeout OR count reached OR disk full

#### Example 4: Complex Nested Pipeline
```dsl
{
  :check_start() ->
  [init_models(), load_prompts()] ->
  camera() ->
  {
    :retry(max=3) ->
    enhance() ->
    validate() ->
    :break_if_acceptable(threshold=0.8)
  } ->
  [
    detect() -> cluster(),
    track() -> smooth()
  ] ->
  merge() ->
  clip_vision() ->
  save() ->
  :timeout(600):check_stop_signal()
}
```

### Parameterized Conditions

Control tasks use parameterized conditions (no arbitrary code):

```dsl
# Timeout
:timeout(seconds=60)

# Iteration count
:count(max=1000)

# Detection checks
:break_if_empty()
:continue_if_few(min=5)
:break_if_many(max=100)

# Confidence checks
:break_if_low_confidence(threshold=0.7)
:continue_if_uncertain(max=0.5)

# Time-based
:check_daylight()
:check_time_range(start=08:00, end=18:00)

# Resource checks
:check_battery(min=20)
:check_disk_space(min_gb=10)
:check_memory(max_mb=500)

# Custom conditions (user-defined)
:my_custom_check(param1=value1, param2=value2)
```

**Security:** No `eval()` or dynamic code execution. Conditions are registered Python classes with parameterized constructors.

### Implementation Notes

**Parser Complexity:**
- Lexer: Simple tokenization (operators, identifiers, literals)
- Parser: Recursive descent (straightforward)
- Validation: Single pass after parsing

**Runtime:**
- Control tasks set flags in loop stack
- LoopConnector checks flags after each task
- Short-circuit evaluation for control sequences

**Error Messages:**
```
Error: Control task ':timeout(60)' must be inside loop {}
  at line 5: camera ->:timeout(60)-> process

Error: Empty loop body not allowed
  at line 3: {}

Error: Unknown task 'invalid_task'
  at line 7: invalid_task() -> process

Error: Parameter 'confidence' expects number, got string
  at line 4: detect(confidence=high)
```

## Best Practices

**1. Task Design**
- Keep tasks focused on single responsibility
- Define clear input/output contracts
- Use configure() for DSL-compatible tasks
- Implement cooperative timing with should_continue()

**2. Pipeline Construction**
- Use factory for consistent task creation
- Validate contracts match between connected tasks
- Provide meaningful task IDs for debugging
- Use appropriate merge strategies for branches

**3. Error Handling**
- Check for required data in context before use
- Raise descriptive errors with task_id in message
- Use logging for operational visibility
- Handle exceptions gracefully in run() methods

**4. Performance**
- Minimize data copying in mutable types
- Use time budgets for long-running tasks
- Consider parallel execution opportunities
- Profile with metrics collection enabled

**5. Testing**
- Test tasks in isolation first
- Validate contracts match expectations
- Test both success and error paths
- Verify metrics collection works correctly

## Architectural Analysis & Design Patterns

### Problem Domain

The VLMChat pipeline system addresses a distinct problem space:

**Problem Domain:**
- Vision-heavy workflows (cameras, detections, embeddings)
- Edge-first deployment (Raspberry Pi, Jetson, resource-constrained devices)
- Continuous streaming pattern (real-time camera processing)
- Hardware abstraction (multiple camera types, platforms)

**Design Philosophy:**
- In-process execution (single process, shared memory)
- Subclass-based extensibility (type-safe, IDE-friendly)
- Domain-specific primitives (detections, clustering, vision models)
- Resource-aware optimization (immutable/mutable context tracking)

### Comparison with Other Frameworks

While LangChain excels at text-heavy, cloud-first, request/response workflows, VLMChat is optimized for edge vision applications:

| Aspect | VLMChat | LangChain | Rationale |
|--------|---------|-----------|-----------|
| **Process Model** | In-process | Multi-process/API | Lower latency, less overhead for edge |
| **Primary Domain** | Vision + VLM | Text + LLM | Different primitives needed |
| **Communication** | Context object | Message passing | No serialization overhead |
| **Deployment** | Single device | Distributed | Matches edge constraints |
| **State Management** | Immutable/mutable tracking | Memory windows | Memory-efficient for continuous streams |
| **Extensibility** | Subclassing | Plugin loading | Better IDE support, type safety |

### Software Engineering Patterns

The architecture employs several well-established patterns:

#### 1. Pipeline Pattern (Pipes & Filters)
```python
Task1 → Task2 → Task3
# Context flows through, each task transforms it
```
- **Implementation**: `BaseTask.run(context) → Context`
- **Benefits**: Composable, testable, debuggable

#### 2. Strategy Pattern (Interchangeable Components)
```python
BaseCamera ← IMX219Camera, IMX500Camera, NoneCamera
BaseRuntime ← OpenClipBackend, (Future: TensorRTBackend)
```
- **Benefits**: Swap implementations without changing pipeline logic

#### 3. Factory Pattern (Dynamic Creation)
```python
factory.register_task("camera", CameraTask)
factory.create_task("camera", "cam1", {...})
```
- **Benefits**: Decouple creation from usage, enable DSL

#### 4. Context Object Pattern
```python
Context.data[ContextDataType.X] = value
# Shared state between pipeline stages
```
- **Benefits**: Avoids parameter explosion, type-safe enum keys, immutable/mutable optimization

#### 5. Observer Pattern (Metrics)
```python
collector.data_point("task.duration", {...}, value)
```
- **Benefits**: Separation of concerns, tasks don't know about metrics details

#### 6. Decorator Pattern (Task Wrapping)
```python
DetectionViewer wraps ObjectClusterer
```
- **Benefits**: Composable enhancements without changing core logic

#### 7. Template Method Pattern
```python
class BaseTask:
    def run(self, context):  # Template
        # Common pre-processing
        result = self._run_impl(context)  # Hook for subclasses
        # Common post-processing
        return result
```
- **Benefits**: Common behavior with customizable steps

### Layered Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                      │
│  - Pipeline DSL (declarative definitions)               │
│  - Test runners and examples                            │
│  - Configuration files                                  │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                  PIPELINE FRAMEWORK                      │
│  - Task abstraction (BaseTask)                          │
│  - Connector abstraction (fan-out, merge)               │
│  - Context management (lifecycle, split)                │
│  - Factory (dynamic creation)                           │
│  - Runner (execution orchestration)                     │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                  DOMAIN LAYER (Tasks)                    │
│  - Camera tasks (hardware acquisition)                  │
│  - Detection tasks (YOLO, clusterer)                    │
│  - Vision tasks (CLIP, VLM)                             │
│  - Utility tasks (cleanup, pass, etc.)                  │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│               INFRASTRUCTURE LAYER                       │
│  - Hardware abstraction (cameras, platforms)            │
│  - Model backends (CLIP, SmolVLM)                       │
│  - Metrics collection and instrumentation               │
│  - Storage (file I/O, caching)                          │
└─────────────────────────────────────────────────────────┘
```

**Key Benefits:**
- Clean separation of concerns
- Layers can be swapped independently
- Easy to test (mock lower layers)
- Clear dependency direction (top → bottom)

### Novel Contributions

The VLMChat pipeline system introduces several innovations for edge vision:

1. **Semantic Object Clustering**
   - CLIP-based semantic grouping of detections
   - Cost matrices for pair and single algorithms
   - Hierarchical detection trees with children

2. **Immutable/Mutable Context Optimization**
   - Automatic deep copy for mutable data (detections)
   - Reference sharing for immutable data (images, embeddings)
   - Memory-efficient context splitting for fan-out patterns

3. **Context Lifecycle Management**
   - `ContextCleanupTask` for managing data persistence
   - Automatic removal of mutable data between iterations
   - Preservation of immutable data (e.g., prompt embeddings)

4. **Hot-Swappable Embeddings**
   - `PromptEmbeddingSourceTask` for dynamic prompt updates
   - File-based and onboard (CLIP text encoder) modes
   - Version tracking and automatic reload detection

5. **Hardware-Aware Task Execution**
   - Platform detection (Pi, Jetson, Mac)
   - Camera abstraction (IMX219, IMX500, library)
   - Device-specific optimizations

6. **Continuous Pipeline Operation**
   - Designed for long-running edge deployments
   - Context cleanup between iterations
   - Hot-swap configuration without restart

### Rapid Development & Production Scaffolding

The pipeline architecture supports **both rapid experimentation and production deployment**:

**Development Velocity:**
- **Decorator Pattern for Visualization**: Wrap any detector with `DetectionViewer` for instant visual debugging
  ```python
  # Rapid prototyping - add visualization without changing core logic
  clusterer = ObjectClusterer(detector, semantic_provider, ...)
  viewer = DetectionViewer(clusterer, display_time_ms=1000)
  
  # Same interface, visual feedback added
  detections = viewer.detect(image)
  ```
- **Factory Pattern for Experimentation**: Swap implementations by changing config strings
  ```python
  # Try different detectors without code changes
  detector = factory.create_task("detector", "d1", {"type": "yolo_cpu"})
  detector = factory.create_task("detector", "d1", {"type": "imx500"})
  ```
- **Metrics Collection**: Built-in instrumentation for performance analysis

**Production Readiness:**
- **DSL Configuration**: Production and development configs via parameter overrides
  ```python
  # Development: low confidence, CPU
  runner = PipelineRunner(ast, overrides={
      "detector": {"confidence": 0.3, "device": "cpu"}
  })
  
  # Production: high confidence, GPU
  runner = PipelineRunner(ast, overrides={
      "detector": {"confidence": 0.5, "device": "cuda"}
  })
  ```
- **Task Contracts**: Type-safe interfaces prevent runtime errors
- **Time Budgets**: Cooperative timing for real-time constraints
- **Metrics Export**: Production monitoring with same instrumentation

**Example: Development to Production Workflow**
1. **Prototype**: Define pipeline in DSL with development parameters
2. **Debug**: Use execution traces to verify cursor flow and merge coordination
3. **Optimize**: Profile with metrics, tune thresholds via overrides
4. **Deploy**: Apply production overrides, same AST structure
5. **Monitor**: Metrics continue tracking performance

This **configuration-based approach** means development and production use the same pipeline structure - only parameters change.

### Design Decisions & Trade-offs

| Decision | Chosen Approach | Alternative | Rationale |
|----------|-----------------|-------------|-----------|
| **Process Model** | In-process | Multi-process | Nanosecond latency, simpler debugging, lower overhead |
| **Task Communication** | Context object | Message passing | Zero serialization cost, type-safe |
| **Extensibility** | Subclassing | Dynamic plugins | Better IDE support, simpler for MVP |
| **Configuration** | DSL + JSON | Pure code | Balance flexibility and simplicity |
| **State Management** | Immutable/mutable | Always copy | Memory efficient, faster splits |
| **Deployment Model** | Single device | Distributed | Matches edge resource constraints |
| **Type Safety** | Enum-based keys | String keys | Compile-time checking, IDE autocomplete |

All choices are **domain-appropriate** - optimized for edge vision rather than cloud text processing.

### Ideas from Other Frameworks

Concepts worth considering for future enhancement:

**High Value:**
1. **Batch processing**: `pipeline.batch([ctx1, ctx2, ctx3])` for processing multiple inputs
2. **Streaming**: `for chunk in pipeline.stream(context)` for real-time results
3. **Callbacks/Hooks**: Extensible event system for logging, debugging, metrics
4. **Fallbacks**: Graceful degradation with backup tasks

**Medium Value:**
5. **Lambda tasks**: Inline transformations for quick prototyping
6. **Conditional routing**: If/else logic in pipeline flow
7. **Retry logic**: Automatic retry on transient failures

**Lower Priority:**
8. **LLM-specific features**: Prompt templates, output parsers (less relevant for vision-first use cases)

### System Composition

```
VLMChat Pipeline System = 
    Pipeline Framework (composition, execution)
  + Vision-First Primitives (detections, clustering, CLIP)
  + Edge Optimization (resource-aware, in-process)
  + Continuous Execution (streaming, cleanup)
  + Hardware Abstraction (cameras, platforms)
```

This combination creates a unique system optimized for **embodied AI and edge vision applications**.

## Multi-CPU Parallelization Strategy

### Design Principles

**No New Syntax Required**
- Multi-CPU execution is an **implementation detail**, not a workflow concern
- Existing parallel operator `[task_a, task_b, task_c]` remains unchanged
- PipelineRunner decides execution strategy based on system capabilities
- Maintains DSL minimalism and readability goals

### Current Architecture Foundation

The pipeline already has parallelization primitives:

1. **Context Data Model**
   - Immutable data (`IMAGE`, `PROMPT`, `RESPONSE`) shared by reference
   - Mutable data (`DETECTIONS`, `CROPS`, `EMBEDDINGS`) deep-copied on split
   - `ContextDataType.is_mutable` flag distinguishes behavior

2. **Parallel Grouping**
   ```dsl
   camera -> [detector_a, detector_b, detector_c] -> merge
   ```
   Current: Sequential execution
   Future: Multi-process/threaded based on task annotations

3. **Split/Merge Semantics**
   - `Context.split()` creates isolated branches for mutable data
   - Connectors define merge strategies for combining results

### Execution Modes

**Task Annotation Approach (Recommended)**

Tasks declare multiprocessing compatibility:

```python
class ClipVisionTask(BaseTask):
    supports_multiprocessing = True  # Task is picklable, has no unpicklable state
    
class DetectorTask(BaseTask):
    supports_multiprocessing = True
    
class ConsoleInputTask(BaseTask):
    supports_multiprocessing = False  # Interactive, not parallelizable
```

**PipelineRunner** selects execution automatically:
- Check task `supports_multiprocessing` flags
- Evaluate CPU count and data size
- Choose optimal execution: sequential, threaded, or multiprocess

**Factory-Based Hints (Optional Override)**

For explicit control, use connector parameters:

```python
# DSL remains unchanged
# camera -> [detector_a, detector_b] -> merge

# Factory configuration specifies execution mode
parallel_connector = factory.create_connector("parallel", "split", {
    "mode": "process"  # Options: "sequential", "thread", "process", "auto"
})
```

### Shared Memory Implementation

**Immutable Data Sharing**

Convert large immutable data (images) to shared memory:

```python
from multiprocessing import shared_memory
import numpy as np

class Context:
    def to_shared_memory(self) -> Dict[str, Any]:
        """Convert immutable data to shared memory references."""
        shm_refs = {}
        
        for data_type, data in self.data.items():
            if not data_type.is_mutable and data:
                # IMAGE: Convert PIL to numpy array in shared memory
                if data_type == ContextDataType.IMAGE:
                    img_array = np.array(data[0])
                    shm = shared_memory.SharedMemory(
                        create=True, 
                        size=img_array.nbytes
                    )
                    shm_array = np.ndarray(
                        img_array.shape, 
                        dtype=img_array.dtype, 
                        buffer=shm.buf
                    )
                    shm_array[:] = img_array[:]
                    
                    shm_refs[data_type] = {
                        'name': shm.name,
                        'shape': img_array.shape,
                        'dtype': str(img_array.dtype)
                    }
                    
                # PROMPT_EMBEDDINGS: Share pre-computed embeddings
                elif data_type == ContextDataType.PROMPT_EMBEDDINGS:
                    # Already numpy, put directly in shared memory
                    emb_array = np.array(data)
                    shm = shared_memory.SharedMemory(
                        create=True,
                        size=emb_array.nbytes
                    )
                    shm_array = np.ndarray(
                        emb_array.shape,
                        dtype=emb_array.dtype,
                        buffer=shm.buf
                    )
                    shm_array[:] = emb_array[:]
                    shm_refs[data_type] = {
                        'name': shm.name,
                        'shape': emb_array.shape,
                        'dtype': str(emb_array.dtype)
                    }
        
        return shm_refs
    
    @staticmethod
    def from_shared_memory(shm_refs: Dict, mutable_data: Dict) -> 'Context':
        """Reconstruct context from shared memory and copied mutable data."""
        ctx = Context()
        
        # Attach immutable data from shared memory
        for data_type, ref in shm_refs.items():
            shm = shared_memory.SharedMemory(name=ref['name'])
            array = np.ndarray(
                ref['shape'], 
                dtype=np.dtype(ref['dtype']), 
                buffer=shm.buf
            )
            
            if data_type == ContextDataType.IMAGE:
                ctx.data[data_type] = [Image.fromarray(array)]
            else:
                ctx.data[data_type] = array.copy()  # Copy to worker's memory
        
        # Attach mutable data (pre-copied before spawning)
        ctx.data.update(mutable_data)
        
        return ctx
```

**Mutable Data Handling**

Each worker process receives deep-copied mutable data:

```python
# In ParallelConnector.run()
mutable_copies = []
for _ in self.output_tasks:
    mutable_copy = {
        dt: copy.deepcopy(data) 
        for dt, data in context.data.items() 
        if dt.is_mutable
    }
    mutable_copies.append(mutable_copy)
```

### Multiprocess Connector Implementation

```python
from multiprocessing import Process, Queue
import pickle

class ParallelConnector(Connector):
    """Executes tasks in parallel using sequential, threaded, or multiprocess mode."""
    
    def __init__(self, task_id: str, mode: str = "auto"):
        super().__init__(task_id)
        self.mode = mode  # "sequential", "thread", "process", "auto"
    
    def configure(self, params: Dict[str, str]) -> None:
        """Configure execution mode from DSL factory params."""
        if "mode" in params:
            self.mode = params["mode"]
    
    def run(self, context: Context) -> Context:
        """Execute output_tasks with optimal parallelization strategy."""
        
        # Determine execution mode
        execution_mode = self._select_execution_mode()
        
        if execution_mode == "process":
            return self._run_multiprocess(context)
        elif execution_mode == "thread":
            return self._run_threaded(context)
        else:
            return self._run_sequential(context)
    
    def _select_execution_mode(self) -> str:
        """Auto-select execution mode based on task properties and system."""
        import multiprocessing
        
        if self.mode != "auto":
            return self.mode
        
        # Check if all tasks support multiprocessing
        all_support_mp = all(
            getattr(task, 'supports_multiprocessing', False) 
            for task in self.output_tasks
        )
        
        # Check CPU availability
        cpu_count = multiprocessing.cpu_count()
        
        if all_support_mp and cpu_count > 1 and len(self.output_tasks) > 1:
            return "process"
        elif len(self.output_tasks) > 1:
            return "thread"
        else:
            return "sequential"
    
    def _run_multiprocess(self, context: Context) -> Context:
        """Execute tasks in separate processes with shared memory."""
        
        # Convert immutables to shared memory
        shm_refs = context.to_shared_memory()
        
        # Deep copy mutable data for each worker
        mutable_copies = []
        for _ in self.output_tasks:
            mutable_copy = {
                dt: copy.deepcopy(data)
                for dt, data in context.data.items()
                if dt.is_mutable
            }
            mutable_copies.append(mutable_copy)
        
        # Create result queue
        result_queue = Queue()
        
        # Spawn worker processes
        processes = []
        for i, task in enumerate(self.output_tasks):
            # Serialize task (must be picklable)
            try:
                task_pickle = pickle.dumps(task)
            except Exception as e:
                logger.warning(f"Task {task.task_id} not picklable, falling back to sequential: {e}")
                self._cleanup_shared_memory(shm_refs)
                return self._run_sequential(context)
            
            p = Process(
                target=self._worker_process,
                args=(task_pickle, mutable_copies[i], shm_refs, result_queue, i)
            )
            p.start()
            processes.append(p)
        
        # Collect results
        results = [None] * len(processes)
        for _ in processes:
            worker_id, result_data = result_queue.get()
            results[worker_id] = result_data
        
        # Wait for completion
        for p in processes:
            p.join()
        
        # Cleanup shared memory
        self._cleanup_shared_memory(shm_refs)
        
        # Merge results back into context
        merged_context = self._merge_results(context, results)
        return merged_context
    
    @staticmethod
    def _worker_process(task_pickle, mutable_data, shm_refs, result_queue, worker_id):
        """Worker process: reconstruct context, run task, return results."""
        # Deserialize task
        task = pickle.loads(task_pickle)
        
        # Reconstruct context from shared memory + mutable data
        ctx = Context.from_shared_memory(shm_refs, mutable_data)
        
        # Run task
        result_ctx = task.run(ctx)
        
        # Send back only mutable results (immutables already shared)
        result_data = {
            dt: data 
            for dt, data in result_ctx.data.items() 
            if dt.is_mutable
        }
        result_queue.put((worker_id, result_data))
    
    def _cleanup_shared_memory(self, shm_refs: Dict):
        """Clean up shared memory allocations."""
        for ref in shm_refs.values():
            try:
                shm = shared_memory.SharedMemory(name=ref['name'])
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass  # Already cleaned up
    
    def _run_threaded(self, context: Context) -> Context:
        """Execute tasks using ThreadPoolExecutor (for I/O-bound tasks)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Split context for each task
        contexts = context.split(len(self.output_tasks), {})
        
        results = []
        with ThreadPoolExecutor(max_workers=len(self.output_tasks)) as executor:
            futures = {
                executor.submit(task.run, ctx): i 
                for i, (task, ctx) in enumerate(zip(self.output_tasks, contexts))
            }
            
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                results.append((idx, result))
        
        # Sort by original order and merge
        results.sort(key=lambda x: x[0])
        return self.merge_strategy([r[1] for r in results])
    
    def _run_sequential(self, context: Context) -> Context:
        """Execute tasks sequentially (current behavior)."""
        contexts = context.split(len(self.output_tasks), {})
        results = []
        for task, ctx in zip(self.output_tasks, contexts):
            result = task.run(ctx)
            results.append(result)
        return self.merge_strategy(results)
    
    def _merge_results(self, base_context: Context, results: List[Dict]) -> Context:
        """Merge worker results back into context."""
        # Reconstruct full contexts from results
        full_contexts = []
        for result_data in results:
            ctx = Context()
            ctx.data.update(base_context.data)  # Immutables
            ctx.data.update(result_data)  # Mutable results
            full_contexts.append(ctx)
        
        # Use connector's merge strategy
        return self.merge_strategy(full_contexts)
```

### Synchronization Primitives

**Standard Python Approaches (No New Syntax)**

```python
from multiprocessing import Lock, Barrier, Semaphore

# Explicit synchronization point (rarely needed)
class SyncTask(BaseTask):
    """Explicit barrier for coordinating parallel branches."""
    def __init__(self, task_id: str, num_tasks: int):
        super().__init__(task_id)
        self.barrier = Barrier(num_tasks)
    
    def run(self, context: Context) -> Context:
        # Wait for all parallel tasks to reach this point
        self.barrier.wait()
        return context

# Rate limiting for shared resources
class RateLimitedTask(BaseTask):
    _semaphore = Semaphore(2)  # Only 2 concurrent executions
    
    def run(self, context: Context) -> Context:
        with self._semaphore:
            # Protected execution
            return self._protected_run(context)
```

### Performance Considerations

**When to Use Multiprocessing:**
- CPU-bound tasks (YOLO inference, clustering, embeddings)
- Large batch processing (multiple images)
- Tasks with `supports_multiprocessing = True`

**When to Use Threading:**
- I/O-bound tasks (file loading, network requests)
- Tasks sharing large objects (avoid pickling overhead)
- Mixed CPU/I/O workloads

**When to Stay Sequential:**
- Interactive tasks (console input/output)
- Tasks with unpicklable state
- Small workloads (overhead > benefit)

**Memory Trade-offs:**
- Shared memory: Fast, but requires numpy-compatible data
- Deep copy: Simple, but expensive for large mutable data
- Hybrid: Share immutables, copy mutables (current approach)

### Migration Path

**Phase 1: Annotation (Current)**
- Add `supports_multiprocessing` flags to tasks
- No execution changes, documentation only

**Phase 2: Implementation**
- Implement `ParallelConnector` with mode selection
- Add `Context.to_shared_memory()` / `from_shared_memory()`
- Test with simple pipelines

**Phase 3: Optimization**
- Auto-detection based on profiling
- Adaptive strategy based on data size
- Metrics for execution mode decisions

**Phase 4: Advanced Features**
- GPU task support
- Distributed execution (multiple devices)
- Dynamic load balancing

### Example Usage

**Automatic (No Changes Required)**
```python
# Current DSL works unchanged
pipeline = """
camera -> [detector_a, detector_b, detector_c] -> merge -> clip_vision
"""

# PipelineRunner auto-selects:
# - Sequential: 1 CPU or tasks don't support MP
# - Threaded: Mixed tasks or I/O-bound
# - Multiprocess: 3+ CPUs and all tasks support MP
```

**Explicit Mode (Advanced)**
```python
# Factory configuration for specific execution mode
factory.create_connector("parallel", "detector_split", {
    "mode": "process"  # Force multiprocessing
})

# Or in task creation
detector_a = factory.create_task("yolo_detector", "det_a", {
    "model": "yolov8n.pt",
    "mode": "process"  # Task-level hint
})
```

**Monitoring**
```python
# Metrics track execution mode decisions
collector.data_point("pipeline.execution.mode", 
                     {"connector": "detector_split", "mode": "process"}, 
                     1)

collector.data_point("pipeline.shared_memory.bytes",
                     {"data_type": "IMAGE"},
                     image_bytes)
```

### Summary

**Key Design Decisions:**
1. **No new DSL syntax** - Keeps language minimal and readable
2. **Annotation-based** - Tasks declare capabilities, runner optimizes
3. **Shared memory for immutables** - Efficient for large images/embeddings
4. **Standard Python primitives** - Leverages multiprocessing, threading modules
5. **Automatic optimization** - System adapts to available resources
6. **Gradual migration** - Add annotations now, implement execution later

This approach maintains DSL simplicity while enabling true multi-CPU parallelization when beneficial.

## Future Enhancements

**Short Term:**
- ~~DSL parser implementation~~ ✅ Complete with timing syntax
- ~~Parameter override system~~ ✅ Complete (v1.0 - tested and validated)
- ~~DSL `id` parameter support~~ ✅ Complete - properly overrides auto-generated IDs
- Integration with chat_services.py
- Time budget enforcement in runner (~ and >= timing)
- Additional connector types (AllComplete, AnyComplete, ConditionalSplit)

**Medium Term:**
- **Hard timeout/kill syntax (`!`)**: Thread termination for stuck tasks
  - Syntax: `task()!500` kills thread after 500ms
  - Use case: Production resilience when tasks hang
  - Implementation: Kill thread, create new context, add replacement thread to pool
  - Example: `detect()~200!500` targets 200ms, kills at 500ms
  - Combines with other timing: `task()>=33!500` (min 33ms, max 500ms)
- Task annotations for multiprocessing support
- ParallelConnector with execution mode selection
- Shared memory implementation for immutable data
- Metrics for parallel execution monitoring
- **Parameter override enhancements**:
  - YAML configuration file support
  - Regex pattern matching for task IDs
  - Validation against task's accepted parameters
  - Override logging and audit trail

**Long Term:**
- Automatic execution strategy selection
- GPU task support and scheduling
- Distributed execution across devices
- Dynamic pipeline reconfiguration
- Pipeline visualization tools
- Performance optimization

## Parameter Override System

The pipeline runner supports runtime parameter overrides for testing, optimization, and configuration without modifying DSL definitions or code. **Status: ✅ Complete and validated (v1.0)**

### Basic Usage

```python
# Simple ID-based override
runner = PipelineRunner(pipeline, overrides={
    "detector": {"confidence": 0.3, "device": "cuda"}
})

# Type-qualified override (when IDs collide)
runner = PipelineRunner(pipeline, overrides={
    "YoloDetectorTask.detector": {"confidence": 0.3}
})

# Wildcard for bulk changes
runner = PipelineRunner(pipeline, overrides={
    "DiagnosticTask.*": {"delay_ms": 10}  # All diagnostic tasks
})
```

### Task ID Assignment

Tasks can be given explicit IDs via DSL or receive auto-generated IDs:

**Explicit ID (recommended):**
```python
# DSL: Use 'id' parameter to specify task ID
dsl = """
diagnostic(id="main", message="Hello") ->
detector(id="yolo", confidence=0.5)
"""
# Override: Use the explicit ID
overrides = {"main": {"delay_ms": 50}}
```

**Auto-generated ID (fallback):**
```python
# DSL: Without 'id' parameter, tasks get {taskname}_{counter}
dsl = """
diagnostic(message="Hello") ->
diagnostic(message="World")
"""
# Results in: diagnostic_1, diagnostic_2
# Override: Use auto-generated IDs
overrides = {"diagnostic_1": {"delay_ms": 50}}
```

### Override Priority

Overrides are applied in priority order (later overrides win):

1. **Type wildcard** (`TaskType.*`): Applies to all tasks of that type (lowest priority)
2. **Type-qualified** (`TaskType.task_id`): Specific task by type and ID
3. **Simple ID** (`task_id`): Specific task by ID (highest priority)

```python
overrides = {
    # 1. All YOLO detectors get confidence 0.2
    "YoloDetectorTask.*": {"confidence": 0.2},
    
    # 2. Main detector overridden to 0.3
    "YoloDetectorTask.main_detector": {"confidence": 0.3},
    
    # 3. Simple ID wins: confidence = 0.5
    "main_detector": {"confidence": 0.5}
}
```

**Priority Resolution Example:**
```python
# Given task: diagnostic(id="processor")
overrides = {
    "DiagnosticTask.*": {"delay_ms": 1},           # Applied first
    "DiagnosticTask.processor": {"delay_ms": 2},   # Overrides previous
    "processor": {"message": "Custom"}             # Highest priority
}
# Result: processor gets delay_ms=2, message="Custom"
```

### Use Cases

**Parameter Sweeps (Testing/Optimization):**
```python
# Test different confidence thresholds
results = []
for conf in [0.1, 0.2, 0.3, 0.4, 0.5]:
    runner = PipelineRunner(pipeline, overrides={
        "detector": {"confidence": conf}
    })
    result = runner.run(context)
    results.append((conf, evaluate_result(result)))
```

**Environment-Specific Configuration:**
```python
# Production vs Development
if env == "production":
    overrides = {
        "detector": {"confidence": 0.5, "device": "cuda"},
        "clusterer": {"max_clusters": 10}
    }
else:
    overrides = {
        "detector": {"confidence": 0.3, "device": "cpu"},
        "clusterer": {"max_clusters": 5}
    }

runner = PipelineRunner(pipeline, overrides=overrides)
```

**A/B Testing:**
```python
# Compare different expansion factors
test_a = PipelineRunner(pipeline, overrides={
    "expander": {"expansion_factor": 0.10}
})

test_b = PipelineRunner(pipeline, overrides={
    "expander": {"expansion_factor": 0.20}
})
```

### Implementation Details

**Timing:**
- Overrides are applied during `build_graph()` after graph construction
- Each task receives `task.configure(**merged_overrides)` with all matching overrides
- Tasks must implement `configure(**kwargs)` to accept overrides
- The `id` parameter in DSL is extracted before instantiation and not passed to `configure()`

**Logging:**
```
INFO:pipeline_runner:Applying parameter overrides to 8 tasks
DEBUG:pipeline_runner:Applied wildcard overrides from 'DiagnosticTask.*' to detector
DEBUG:pipeline_runner:Applied type-qualified overrides from 'YoloDetectorTask.main_detector'
INFO:pipeline_runner:Configuring task 'main_detector' with overrides: {'confidence': 0.5}
```

**Validation:**
- Tasks silently ignore unknown parameters (duck typing)
- Recommended: Tasks should validate parameters in `configure()` and log warnings
- Future: Schema validation against task's accepted parameters
- Future: Warning/error on unused overrides

**Testing:**
- All override patterns validated in Test 6 of `pipeline_runner.py`
- Wildcard, type-qualified, and simple ID patterns tested
- Priority chain verified: wildcard < type-qualified < simple ID

### DSL Integration

Overrides work with DSL-defined pipelines through the three-phase model:

```python
from src.pipeline.dsl_parser import parse_pipeline, PipelineBuilder
from src.pipeline.pipeline_runner import PipelineRunner

# Parse DSL (Phase 1: Parser)
dsl = """
diagnostic(id="task1", message="Hello", delay_ms=100) ->
diagnostic(id="task2", message="World", delay_ms=200)
"""
ast_root = parse_pipeline(dsl)

# Build decorated AST (Phase 2: Builder)
builder = PipelineBuilder()
decorated_ast = builder.build(ast_root)

# Run with overrides (Phase 3: Runner)
# Overrides applied during runner initialization via build_graph()
runner = PipelineRunner(decorated_ast, overrides={
    "DiagnosticTask.*": {"delay_ms": 50}  # Override all diagnostic tasks
})

# Or override specific task
runner = PipelineRunner(decorated_ast, overrides={
    "task1": {"message": "Overridden!"}
})

result = runner.run(context)
```

**How it works:**
1. **Parser**: Parses DSL parameters into AST (e.g., `message="Hello"`)
2. **Builder**: Creates task instances with DSL parameters
3. **Runner**: Applies overrides via `task.configure(**merged_overrides)` during `build_graph()`
4. **Execution**: Cursors execute tasks with final merged parameters

**Override precedence:**
- DSL parameters < Wildcard overrides < Type-qualified overrides < Simple ID overrides
- Example: `task1` overrides `DiagnosticTask.task1` overrides `DiagnosticTask.*`

### Best Practices

1. **Use explicit IDs in DSL** - `diagnostic(id="main")` is clearer than auto-generated `diagnostic_1`
2. **Use simple IDs for common cases** - clearest and most maintainable
3. **Use type qualification when IDs collide** - explicit disambiguation
4. **Use wildcards sparingly** - can have unintended effects across many tasks
5. **Document override expectations** - specify what parameters are safe to override
6. **Test with overrides** - verify behavior matches expectations
7. **Validate in configure()** - tasks should check parameter values and log warnings

### Future Enhancements

- **YAML configuration files**: Load overrides from external config
- **Regex pattern matching**: `"detector_.*": {params}` for flexible matching
- **Parameter validation**: Check against task's accepted parameters
- **Override audit trail**: Track what was overridden and why
- **Conditional overrides**: Apply based on environment variables or conditions
- GPU memory management for parallel detector branches

## Support

For questions or issues with the pipeline system:
1. Check test examples in `pipeline_runner.py` __main__ section
2. Review logging output at INFO level for execution traces
3. Verify contracts match between connected tasks
4. Ensure factory registration for custom tasks
5. Check metrics for performance bottlenecks

## Environment Singleton - Dictionary-based Key-Value Store

### Overview

The **Environment singleton** provides a general-purpose key-value store for sharing state between pipeline tasks and the chat application. Instead of hardcoded attributes, it uses a dictionary with namespaced keys following the pattern:

```
{taskType}+{taskId}+{key}
```

This design allows any task to store/retrieve arbitrary data without modifying the Environment class itself.

### Key Format Examples

- `Camera+cam1+current_image` - Camera task instance "cam1" stores its current image
- `ObjectDetector+det1+results` - Object detector instance "det1" stores detection results
- `App+main+history` - Chat application stores conversation history
- `ImageProcessor+proc1+threshold` - Image processor stores its threshold parameter

### Why Dictionary-based Design?

**✅ Advantages:**

1. **Fully Extensible**: Any task can store/retrieve data without modifying Environment
2. **Namespace Separation**: Multiple instances of the same task type can coexist
3. **No Coupling**: Tasks don't need to know about each other's data structures
4. **Easy Debugging**: `env.keys()` shows all active data
5. **General Purpose**: Works for any data type (images, configs, results, etc.)

**🎯 Use Cases:**

- **Pipeline Tasks**: Share intermediate results between tasks
- **State Management**: Store task configuration and runtime state
- **Cross-component Communication**: Chat app and pipeline system share data
- **Multi-instance Tasks**: Multiple cameras, detectors, etc. with separate state

### API Reference

#### Core Methods

```python
from pipeline.environment import Environment

# Get singleton instance
env = Environment.get_instance()

# Set a value
env.set("Camera+cam1+current_image", image)

# Get a value
image = env.get("Camera+cam1+current_image")

# Get with default
image = env.get("Camera+cam1+current_image", default_image)

# Check if key exists
if env.has("Camera+cam1+current_image"):
    image = env.get("Camera+cam1+current_image")

# Alternative syntax using 'in'
if "Camera+cam1+current_image" in env:
    image = env.get("Camera+cam1+current_image")

# Remove a key
removed = env.remove("Camera+cam1+current_image")  # Returns True if removed

# Get all keys
all_keys = env.keys()

# Get number of keys
count = len(env)

# Clear all data
env.clear()

# Reset singleton (mainly for testing)
Environment.reset()
```

#### Backward Compatibility

For existing code that used the old attribute-based API:

```python
# Old style - still works via properties
env.current_image = image
env.history = history_manager

# Or via methods
env.set_image(image)
env.get_image()
env.clear_image()

# These map to:
# "App+main+current_image"
# "App+main+history"
```

### Usage Examples

#### Example 1: Camera Task Storing Image

```python
from pipeline.environment import Environment

class CameraTask:
    def __init__(self, task_id="cam1"):
        self.task_id = task_id
        self.env = Environment.get_instance()
    
    def capture(self):
        image = self.hardware_capture()
        key = f"Camera+{self.task_id}+current_image"
        self.env.set(key, image)
        return image
```

#### Example 2: Detector Using Camera Image

```python
from pipeline.environment import Environment

class ObjectDetectorTask:
    def __init__(self, task_id="det1", camera_id="cam1"):
        self.task_id = task_id
        self.camera_id = camera_id
        self.env = Environment.get_instance()
    
    def execute(self):
        # Get image from camera task
        image_key = f"Camera+{self.camera_id}+current_image"
        image = self.env.get(image_key)
        
        if image is None:
            raise ValueError(f"No image available from camera {self.camera_id}")
        
        # Run detection
        results = self.detect_objects(image)
        
        # Store results for next task
        results_key = f"ObjectDetector+{self.task_id}+results"
        self.env.set(results_key, results)
        
        return results
```

#### Example 3: Multiple Task Instances

```python
from pipeline.environment import Environment

# Create multiple camera instances
env = Environment.get_instance()

# Camera 1
env.set("Camera+cam1+current_image", image1)
env.set("Camera+cam1+resolution", (1920, 1080))

# Camera 2
env.set("Camera+cam2+current_image", image2)
env.set("Camera+cam2+resolution", (640, 480))

# Each maintains separate state
cam1_image = env.get("Camera+cam1+current_image")
cam2_image = env.get("Camera+cam2+current_image")
```

#### Example 4: Configuration Storage

```python
from pipeline.environment import Environment

class ConfigurableTask:
    def __init__(self, task_id):
        self.task_id = task_id
        self.env = Environment.get_instance()
        
        # Store default configuration
        config_key = f"{self.__class__.__name__}+{task_id}+config"
        self.env.set(config_key, {
            "threshold": 0.5,
            "max_items": 10,
            "enabled": True
        })
    
    def get_config(self):
        config_key = f"{self.__class__.__name__}+{self.task_id}+config"
        return self.env.get(config_key, {})
    
    def update_config(self, updates):
        config = self.get_config()
        config.update(updates)
        config_key = f"{self.__class__.__name__}+{self.task_id}+config"
        self.env.set(config_key, config)
```

### Best Practices

#### 1. Use Consistent Key Format
```python
# Good - consistent format
key = f"{self.__class__.__name__}+{self.task_id}+{data_name}"

# Also good - explicit task type
key = f"Camera+{self.task_id}+current_image"
```

#### 2. Check Before Getting
```python
# Good - check existence first
if env.has(key):
    value = env.get(key)

# Or use default
value = env.get(key, default_value)
```

#### 3. Clean Up When Done
```python
# Remove data when no longer needed
env.remove(f"Task+{task_id}+temp_data")

# Or clear all task data
for key in env.keys():
    if key.startswith(f"Task+{task_id}+"):
        env.remove(key)
```

#### 4. Document Your Keys
```python
class MyTask:
    """
    Task that processes images.
    
    Environment Keys Used:
    - Camera+cam1+current_image (input): PIL.Image
    - MyTask+{task_id}+results (output): List[Dict]
    - MyTask+{task_id}+config (state): Dict
    """
```

### Debugging Tips

#### List All Active Keys
```python
env = Environment.get_instance()
print("Active environment keys:")
for key in env.keys():
    value = env.get(key)
    print(f"  {key}: {type(value).__name__}")
```

#### Check Data Flow
```python
# At task boundaries, log what's being stored/retrieved
logger.debug(f"Storing result: {key}")
env.set(key, result)

logger.debug(f"Retrieving input: {key}")
input_data = env.get(key)
```

#### Clear Between Pipeline Runs
```python
# Clear task-specific data between runs
def cleanup_task_data(task_id):
    env = Environment.get_instance()
    prefix = f"Task+{task_id}+"
    for key in list(env.keys()):  # Copy list to avoid modification during iteration
        if key.startswith(prefix):
            env.remove(key)
```

### Migration from Old API

If you have existing code using the old attribute-based API:

**Before (Hardcoded Attributes):**
```python
env = Environment.get_instance()
env.current_image = image
env.history = history

# Access
image = env.current_image
history = env.history
```

**After (Dictionary-based):**
```python
env = Environment.get_instance()
env.set("App+main+current_image", image)
env.set("App+main+history", history)

# Access
image = env.get("App+main+current_image")
history = env.get("App+main+history")
```

**Backward Compatibility:**
The old API still works via properties:
```python
# This still works!
env.current_image = image
image = env.current_image

# Maps to: env.set("App+main+current_image", image)
```

### Summary

The refactored Environment provides a **general-purpose, extensible key-value store** that:
- ✅ Works for any task type without code changes
- ✅ Supports multiple instances of the same task
- ✅ Maintains backward compatibility
- ✅ Enables clean namespace separation
- ✅ Makes debugging easier with `keys()` inspection

This design scales naturally as you add new task types and use cases!
