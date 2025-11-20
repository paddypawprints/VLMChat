# VLMChat Pipeline Architecture

## Overview

The VLMChat pipeline system provides a flexible, DAG-based architecture for composing vision and language processing workflows. It supports parallel execution, metrics collection, and dynamic configuration through a factory pattern with plans for DSL-based pipeline definitions.

## Core Concepts

### Pipeline Components

**BaseTask** - Abstract base class for all pipeline tasks
- Defines `input_contract` and `output_contract` for static validation
- Implements `run(context) -> context` for task execution
- Supports `configure(params)` for parameter injection from DSL
- Includes `time_budget_ms` for cooperative timing control

**Connector** - Special task that manages DAG structure
- Extends `BaseTask` with split/merge capabilities
- Contains `internal_tasks` and edges between them
- Implements `split_strategy()` and `merge_strategy()` for data flow
- Can be nested for complex pipelines

**Context** - Data container passed between tasks
- Holds typed data via `ContextDataType` enum (IMAGE, PROMPT, RESPONSE, DETECTIONS, etc.)
- Distinguishes mutable (DETECTIONS, CROPS) vs immutable (IMAGE, PROMPT) data
- Supports `split()` for parallel branches with proper data handling

**PipelineRunner** - Orchestrates pipeline execution
- Flattens connector structure into execution graph
- Performs topological sort for dependency resolution
- Executes tasks in parallel where possible (MVP: sequential)
- Integrates metrics collection and logging
- Validates contracts at graph build time

**PipelineFactory** - Dynamic task/connector instantiation
- Registry mapping string names → Python classes
- `create_task(name, task_id, params)` for dynamic creation
- `create_connector(name, connector_id, params)` for merge strategies
- `create_default_factory()` provides pre-registered standard tasks

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       PipelineRunner                        │
│  - Flattens DAG structure                                   │
│  - Validates contracts                                      │
│  - Executes tasks with dependency resolution                │
│  - Collects metrics                                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                        Connector                            │
│  ┌────────────┐      ┌────────────┐      ┌────────────┐   │
│  │   Task A   │─────→│   Task B   │─────→│   Task C   │   │
│  └────────────┘      └────────────┘      └────────────┘   │
│                            │                                │
│                     split_strategy()                        │
│                            │                                │
│              ┌─────────────┴─────────────┐                 │
│              ▼                           ▼                 │
│     ┌────────────────┐         ┌────────────────┐         │
│     │ Branch Task 1  │         │ Branch Task 2  │         │
│     └────────────────┘         └────────────────┘         │
│              │                           │                 │
│              └─────────────┬─────────────┘                 │
│                            │                                │
│                     merge_strategy()                        │
│                            ▼                                │
│                     ┌────────────┐                         │
│                     │   Task D   │                         │
│                     └────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## Context Data Types

```python
class ContextDataType(Enum):
    PROMPT = "prompt"           # User text input (immutable)
    IMAGE = "image"             # PIL Image (immutable)
    RESPONSE = "response"       # Model text output (immutable)
    DETECTIONS = "detections"   # List of Detection objects (mutable)
    CROPS = "crops"             # List of image crops (mutable)
    EMBEDDINGS = "embeddings"   # List of embedding vectors (mutable)
    MATCHES = "matches"         # Semantic matches (mutable)
    AUDIT = "audit"             # Execution metadata (mutable)
```

**Mutable vs Immutable:**
- **Immutable data** (IMAGE, PROMPT, RESPONSE): Shared by reference across branches
- **Mutable data** (DETECTIONS, CROPS, EMBEDDINGS): Deep-copied when splitting to prevent interference

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
Captures user text input from console.

**Configuration:**
```python
console_input = factory.create_task("console_input", "input1", {
    "prompt": "You: "  # Text to display
})
```

**Contracts:**
- Input: None (can be source or follow start task)
- Output: PROMPT (str)

### ConsoleOutputTask
Displays text to console.

**Configuration:**
```python
console_output = factory.create_task("console_output", "output1")
```

**Contracts:**
- Input: RESPONSE (str)
- Output: None (sink task)

### SmolVLMTask
Runs vision-language model inference.

**Note:** Requires pre-initialized model and prompt objects (dependency injection).

```python
smolvlm = SmolVLMTask(model, prompt, "smolvlm")
```

**Contracts:**
- Input: IMAGE, PROMPT
- Output: RESPONSE (str)

### HistoryUpdateTask
Updates conversation history.

**Note:** Requires pre-initialized history object (dependency injection).

```python
history_update = HistoryUpdateTask(history, "history")
```

**Contracts:**
- Input: PROMPT, RESPONSE
- Output: None (sink task)

### StartTask
Empty source task for split entry points.

```python
start = factory.create_task("start", "start")
```

**Contracts:**
- Input: None (source task)
- Output: None (passes through context unchanged)

## Connector Subclasses

### Connector (Base)
Default merge strategy: concatenates mutable data, takes first immutable data.

```python
connector = factory.create_connector("connector", "main_pipeline")
```

### FirstCompleteConnector
Takes the first branch to complete, ignores others. Useful for race conditions.

```python
first_complete = factory.create_connector("first_complete", "race")
```

**DSL Example:**
```
[clone_split(): fast_model, accurate_model :first_complete()]
```

### OrderedMergeConnector
Reorders branch results before merging based on configuration.

```python
ordered_merge = factory.create_connector("ordered_merge", "merge1", {
    "order": "2,1"  # Process second branch first, then first branch
})
```

**DSL Example:**
```
[clone_split(): high_priority, low_priority :ordered_merge(order=2,1)]
```

## Usage Examples

### Example 1: Simple Linear Pipeline

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

### Example 2: Parallel VLM Workflow

```python
from src.pipeline.pipeline_factory import create_default_factory
from src.pipeline.pipeline_runner import PipelineRunner
from src.pipeline.task_base import Context, ContextDataType

factory = create_default_factory()

# Create main pipeline
pipeline = factory.create_connector("connector", "vlm_pipeline")

# Create start task for split
start = factory.create_task("start", "start")

# Create split connector
split_connector = factory.create_connector("connector", "splitter")
split_connector.output_tasks = []  # Will add branches

# Create parallel tasks
camera = factory.create_task("camera", "cam0", {"type": "none", "device": "0"})
console_input = factory.create_task("console_input", "input1", {"prompt": "You: "})

# Create merge connector
merge_connector = factory.create_connector("connector", "merger")

# Create VLM task (requires pre-initialized model and prompt)
from src.models.SmolVLM.smol_vlm_model import SmolVLMModel
from src.prompt.prompt import Prompt

model = SmolVLMModel(config)
prompt_manager = Prompt(config)
smolvlm = SmolVLMTask(model, prompt_manager, "smolvlm")

# Create output task
console_output = factory.create_task("console_output", "output1")

# Build graph
pipeline.add_task(start)
pipeline.add_task(split_connector)
pipeline.add_task(camera)
pipeline.add_task(console_input)
pipeline.add_task(merge_connector)
pipeline.add_task(smolvlm)
pipeline.add_task(console_output)

# Wire edges
pipeline.add_edge(start, split_connector)

# Split to parallel capture
split_connector.output_tasks = [camera, console_input]
pipeline.add_edge(split_connector, camera)
pipeline.add_edge(split_connector, console_input)

# Merge from parallel branches
pipeline.add_edge(camera, merge_connector)
pipeline.add_edge(console_input, merge_connector)

# Continue to VLM and output
pipeline.add_edge(merge_connector, smolvlm)
pipeline.add_edge(smolvlm, console_output)

# Run pipeline
runner = PipelineRunner(pipeline)
context = Context()
result = runner.run(context)

runner.shutdown()
```

### Example 3: Race Condition with FirstComplete

```python
# Create split with two different detectors
split_connector = factory.create_connector("connector", "splitter")

detector_fast = factory.create_task("detector", "fast", {
    "type": "yolo_cpu",
    "model": "yolov8n.pt",
    "confidence": "0.5"
})

detector_accurate = factory.create_task("detector", "accurate", {
    "type": "yolo_cpu", 
    "model": "yolov8x.pt",
    "confidence": "0.3"
})

# Use FirstComplete to take fastest result
first_complete = factory.create_connector("first_complete", "race")

# Wire up
split_connector.output_tasks = [detector_fast, detector_accurate]
# ... add edges ...

# First detector to complete wins, other result discarded
```

## Metrics Integration

The pipeline automatically collects metrics when a `Collector` is provided:

```python
from src.metrics.metrics_collector import Collector, Session
from src.metrics.instruments import AverageDurationInstrument, CountInstrument

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

# Create runner with collector
runner = PipelineRunner(pipeline, collector=collector)
result = runner.run(context)

# Export metrics
for ts_name, inst in session._instruments:
    exported = inst.export()
    print(exported)
```

**Collected Metrics:**
- `task.execution.duration` - Time spent in each task
- `task.execution.count` - Success/failure counts per task
- `task.failures` - Failure counts by error type
- `context.data.size` - Size of data in context (when tracked)

## Execution Trace

The pipeline can record detailed execution traces for debugging and analysis:

```python
from src.pipeline.diagnostic_task import print_trace

# Create runner with trace enabled
runner = PipelineRunner(pipeline, enable_trace=True)
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
        0.0ms |   8368922816 | task1                | -                    | execute
       12.3ms |   8368922816 | fork                 | task1                | split
       15.1ms |   8368922817 | task2                | fork                 | execute
       15.2ms |   8368922818 | task3                | fork                 | execute
       28.4ms |   8368922816 | merge                | task2, task3         | merge
       30.1ms |   8368922816 | task4                | merge                | execute
======================================================================
Total events: 6
======================================================================
```

**Trace Features:**
- **Chronological timeline**: Events sorted by timestamp
- **Thread tracking**: Identifies parallel execution
- **Dependency graph**: Shows task relationships via upstreams
- **Event types**: `execute` (task), `split` (fork), `merge` (join)
- **Context preservation**: Traces survive splits and merges
- **Pluggable backends**: InMemoryTrace (default), LogTrace, NoOpTrace

**Trace Backends** (see `src/pipeline/trace.py`):
- `InMemoryTrace`: Store events in context data (default)
- `LogTrace`: Emit events to logger for aggregation (future)
- `NoOpTrace`: Disable tracing for performance (future)

**Trace Data Format:**
```python
# Each event is a tuple:
(timestamp, thread_id, task_id, upstream_task_ids, event_type)
# Example:
(1234567890.123, 8368922816, 'task1', [], 'execute')
(1234567890.456, 8368922816, 'fork', ['task1'], 'split')
```

**Use Cases:**
- Debug execution order and parallelism
- Analyze performance bottlenecks
- Verify pipeline structure
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

**Parser Status:** Not yet implemented. Factory and configure() support in place.

## Testing

Run all pipeline tests:

```bash
cd /Users/patrick/Dev/VLMChat
python -m src.pipeline.pipeline_runner
```

**Test Coverage:**
1. Linear pipeline (source → detector → embedder)
2. Branching pipeline with split/merge
3. FirstCompleteConnector
4. OrderedMergeConnector
5. PipelineFactory dynamic creation
6. Factory with configured OrderedMergeConnector
7. CameraTask + DetectorTask with configure() (requires dependencies)

## File Organization

```
src/pipeline/
├── task_base.py              # Core abstractions (BaseTask, Connector, Context)
├── pipeline_runner.py        # Execution engine
├── pipeline_factory.py       # Dynamic instantiation
├── connectors.py             # Connector subclasses
├── camera_task.py            # Camera adapter
├── detector_task.py          # Object detector adapter
├── console_input_task.py     # Console input adapter
├── console_output_task.py    # Console output adapter
├── smolvlm_task.py          # VLM inference adapter
├── history_update_task.py    # History update adapter
└── start_task.py            # Empty source task
```

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
- **Remove Decorators**: Unwrap visualization layers for deployment
  ```python
  # Development: with viewer
  pipeline.add_task(DetectionViewer(clusterer))
  
  # Production: direct detector
  pipeline.add_task(clusterer)
  ```
- **Task Contracts**: Type-safe interfaces prevent runtime errors
- **Time Budgets**: Cooperative timing for real-time constraints
- **Metrics Export**: Production monitoring with same instrumentation

**Example: DetectionViewer Workflow**
1. **Prototype**: Add viewer to see what's detected
2. **Debug**: Adjust clustering parameters visually
3. **Optimize**: Profile with metrics, tune thresholds
4. **Deploy**: Remove viewer, keep core detector
5. **Monitor**: Metrics continue tracking performance

This **decorator-based scaffolding** means development tools don't pollute production code - they wrap it temporarily and unwrap cleanly.

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

Overrides work with DSL-defined pipelines:

```python
dsl = """
diagnostic(id="task1", message="Hello", delay_ms=100) ->
diagnostic(id="task2", message="World", delay_ms=200)
"""

# Override delay for all tasks
runner = PipelineRunner.from_dsl(dsl, overrides={
    "DiagnosticTask.*": {"delay_ms": 50}
})

# Override specific task
runner = PipelineRunner.from_dsl(dsl, overrides={
    "task1": {"message": "Overridden!"}
})
```

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
