# VLMChat Pipeline Architecture

## Overview

The VLMChat pipeline system provides a flexible, DAG-based architecture for composing vision and language processing workflows using a **fluent API** with method chaining.

**Architecture Philosophy:**
- **Fluent API**: Type-safe pipeline construction with Python method chaining
- **Task Graph**: Doubly-linked task graph with upstream/downstream connections
- **Runner**: Cursor-based execution engine navigating the task graph

This design enables intuitive pipeline building, IDE autocomplete support, and efficient parallel execution.

## Quick Start

```python
from vlmchat.pipeline import Pipeline
from vlmchat.pipeline.tasks import DiagnosticTask

# Create pipeline with method chaining
pipeline = Pipeline("my_pipeline")

task1 = DiagnosticTask("task1")
task2 = DiagnosticTask("task2")
task3 = DiagnosticTask("task3")

pipeline.chain([task1, task2, task3])

# Or use method chaining
pipeline.add(task1).then(task2).then(task3)

# Execute
result = pipeline.run()
```

## Implementation Status

### ✅ Implemented Features

**Fluent API:**
- ✅ Pipeline builder class with method chaining
- ✅ `.add()` - Add root or branch tasks
- ✅ `.then()` - Sequential task connection
- ✅ `.chain()` - Multi-task sequential chains
- ✅ `.fork()` - Parallel branch creation
- ✅ `.merge()` - Branch merging
- ✅ `.parallel()` - Nested branches with builder functions
- ✅ `.run()` - Pipeline execution

**Execution Engine:**
- ✅ Cursor-queue execution model with ThreadPoolExecutor
- ✅ Doubly-linked task graph navigation (upstream/downstream tasks)
- ✅ Fork execution with independent context copies per branch
- ✅ Merge coordination with multi-cursor synchronization
- ✅ Exit code propagation through pipeline (0=success, 1=empty, 2=exception, 3+=custom)
- ✅ Trace event recording for debugging and metrics
- ✅ Metrics collection (execution time, task counts)

**Exception Handling:**
- ✅ Context-based exception propagation (exception, exception_source_task)
- ✅ Sequential flow: exceptions flow forward, tasks auto-skip
- ✅ Fork replication: exceptions copy to all branches
- ✅ Merge early exit: first exception terminates other branches
- ✅ Error handler tasks for graceful degradation
- ✅ Type-specific exception filtering

**Task System:**
- ✅ BaseTask with simplified requires/produces contracts
- ✅ Connector base class for structural tasks (Fork, Merge)
- ✅ Context get/set methods for data passing
- ✅ NullContext for missing data handling
- ✅ Task registry with `@register_task` decorator
- ✅ Time budget support for cooperative scheduling

**Test Coverage:**
- ✅ Exception propagation tests
- ✅ Integration tests
- ✅ Fluent API examples (5 working examples)

### 🎯 Current Capabilities

The pipeline system supports:

1. **Fluent API**: Type-safe pipeline building with Python
2. **Complex Control Flow**: Sequential, parallel branches, and nested combinations
3. **Error Recovery**: Exception propagation with graceful degradation
4. **Concurrent Execution**: ThreadPoolExecutor with cursor-based parallelism
5. **Simplified Contracts**: String-based requires/produces instead of enums
6. **Observability**: Trace events and metrics for monitoring

## Core Concepts

### Pipeline Building

**Pipeline Class** - Fluent interface for building task graphs
```python
from vlmchat.pipeline import Pipeline

pipeline = Pipeline("example")
```

**Methods:**
- `add(task)` - Add a task as root or branch point
- `then(task)` - Connect task sequentially to previous task
- `chain(tasks)` - Connect multiple tasks in sequence
- `fork(branches)` - Split into parallel branches
- `merge(connector, task)` - Merge branches back together
- `parallel(*builders)` - Create nested branches with builder functions
- `run(context, max_workers, enable_trace)` - Execute the pipeline

### Pipeline Components

**BaseTask** - Abstract base class for all pipeline tasks
- Defines `requires` and `produces` lists for contracts
- Implements `run(context) -> context` for task execution
- Supports `configure(**params)` for parameter injection
- **Doubly-linked graph**: `upstream_tasks` and `downstream_tasks` lists

**Context** - Data container passed between tasks
- Uses `get(key, default)` to retrieve data
- Uses `set(key, value)` to store data
- Supports `split(num_branches)` for parallel execution
- Maintains `exception` and `exception_source_task` for error propagation

**Cursor** - Execution pointer in task graph
- Holds `current_task` and `context` (data)
- Navigates via task's `downstream_tasks` list
- Multiple cursors enable concurrent pipeline traversal

**PipelineRunner** - Orchestrates cursor-based execution
- Cursor-queue model: `task_queue` (waiting) and `ready_queue` (executable)
- Executes ready cursors in parallel via ThreadPoolExecutor
- **Merge coordination**: Tracks arriving cursors at merge points
- **Fork handling**: Copies context independently for each branch
- Thread-safe state management with Events

## Building Pipelines

### Sequential Pipelines

```python
from vlmchat.pipeline import Pipeline
from vlmchat.pipeline.tasks import DiagnosticTask

# Method 1: Using chain()
pipeline = Pipeline("sequential")
tasks = [DiagnosticTask(f"task{i}") for i in range(3)]
pipeline.chain(tasks)

# Method 2: Using add() and then()
pipeline = Pipeline("sequential")
pipeline.add(DiagnosticTask("task1")) \
        .then(DiagnosticTask("task2")) \
        .then(DiagnosticTask("task3"))
```

### Parallel Pipelines

```python
from vlmchat.pipeline import Pipeline
from vlmchat.pipeline.tasks import DiagnosticTask

# Create parallel branches
pipeline = Pipeline("parallel")

input_task = DiagnosticTask("input")
branch1 = DiagnosticTask("branch1")
branch2 = DiagnosticTask("branch2")
branch3 = DiagnosticTask("branch3")
output_task = DiagnosticTask("output")

# Fork and merge
pipeline.add(input_task) \
        .fork([branch1, branch2, branch3]) \
        .merge(merge_task=output_task)
```

### Nested Pipelines

```python
from vlmchat.pipeline import Pipeline
from vlmchat.pipeline.tasks import DiagnosticTask

pipeline = Pipeline("nested")

start = DiagnosticTask("start")

pipeline.add(start).parallel(
    # Branch 1: two tasks in sequence
    lambda p: p.add(DiagnosticTask("b1_step1")).then(DiagnosticTask("b1_step2")),
    
    # Branch 2: three tasks in sequence
    lambda p: p.add(DiagnosticTask("b2_step1"))
               .then(DiagnosticTask("b2_step2"))
               .then(DiagnosticTask("b2_step3")),
).merge()
```

### Complex Multi-Stage

```python
from vlmchat.pipeline import Pipeline
from vlmchat.pipeline.tasks import DiagnosticTask

pipeline = Pipeline("complex")

# Stage 1: Input
input_task = DiagnosticTask("input")

# Stage 2: Parallel preprocessing
prep1 = DiagnosticTask("preprocess1")
prep2 = DiagnosticTask("preprocess2")

# Stage 3: Merge and process
merge_process = DiagnosticTask("merge_process")

# Stage 4: Parallel analysis
analyze1 = DiagnosticTask("analyze1")
analyze2 = DiagnosticTask("analyze2")

# Stage 5: Final output
output = DiagnosticTask("output")

# Build pipeline: fork -> merge -> fork -> merge
pipeline.add(input_task) \
        .fork([prep1, prep2]) \
        .merge(merge_task=merge_process) \
        .fork([analyze1, analyze2]) \
        .merge(merge_task=output)
```

## Task Development

### Creating Custom Tasks

```python
from vlmchat.pipeline import BaseTask, Context, register_task

@register_task('my_task')
class MyTask(BaseTask):
    def __init__(self, task_id: str = "my_task"):
        super().__init__(task_id)
        self.requires = ["input_data"]  # What this task needs
        self.produces = ["output_data"]  # What this task creates
    
    def run(self, context: Context) -> Context:
        """Execute the task."""
        # Get input data (returns NullContext if missing)
        input_data = context.get("input_data")
        
        # Check for null context
        if context.is_null():
            return context
        
        # Process data
        result = self.process(input_data)
        
        # Store output
        context.set("output_data", result)
        
        return context
    
    def process(self, data):
        """Your processing logic here."""
        return data
```

### Task Configuration

```python
task = MyTask("custom_task")
task.configure(
    param1="value1",
    param2=42,
    param3=True
)
```

### Using Context

```python
def run(self, context: Context) -> Context:
    # Get data with default
    value = context.get("key", default=None)
    
    # Set data
    context.set("key", value)
    
    # Check for exceptions
    if context.has_exception():
        exc = context.get_exception()
        source = context.get_exception_source()
        # Handle error...
    
    # Check for null context
    if context.is_null():
        return context  # Skip processing
    
    return context
```

## Execution Model

### Cursor-Based Execution

The pipeline uses a cursor-queue model:

1. **Initialization**: Create cursors at root tasks
2. **Ready Check**: Move cursors to ready queue when all upstream tasks complete
3. **Parallel Execution**: Execute ready cursors concurrently via ThreadPoolExecutor
4. **Navigation**: After execution, spawn new cursors for downstream tasks
5. **Merge Coordination**: Wait at merge points until all branches arrive
6. **Completion**: Pipeline finishes when all cursors complete

### Fork/Merge Behavior

**Fork (Parallel Branches):**
- Context is copied for each branch
- Each branch executes independently
- Branches can complete at different times

**Merge (Join Branches):**
- Waits for all upstream branches
- Merges contexts using connector's merge strategy
- Continues with single merged context

## Examples

See [examples/fluent_api_examples.py](../examples/fluent_api_examples.py) for working examples:
- Simple sequential chains
- Method chaining
- Parallel fork/merge
- Nested branches
- Complex multi-stage pipelines

## Advanced Features

### Metrics Collection

```python
from vlmchat.pipeline.services import Collector

collector = Collector()
result = pipeline.run(context, enable_trace=True)

# Access metrics
for name, timeseries in collector.timeseries.items():
    print(f"{name}: {timeseries.export()}")
```

### Exception Handling

Tasks can handle exceptions by checking context:

```python
def run(self, context: Context) -> Context:
    if context.has_exception():
        exc = context.get_exception()
        # Log, recover, or propagate
        context.clear_exception()  # If handling
    
    # Normal processing...
    return context
```

### Trace Events

Enable tracing for detailed execution logs:

```python
result = pipeline.run(enable_trace=True)

# Trace events are recorded in runner.trace_events
```

## Architecture Diagrams

### Task Graph Structure

```
┌─────────────┐
│  Task A     │
│ (root)      │
└─────┬───────┘
      │ downstream_tasks
      ▼
┌─────────────┐
│  Task B     │ ◄── upstream_tasks points to Task A
│             │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  Task C     │
│             │
└─────────────┘
```

### Fork/Merge Pattern

```
         ┌─────────────┐
         │   Input     │
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │ ForkConnector│
         └──┬─────┬────┘
            │     │
      ┌─────▼─┐ ┌▼─────┐
      │Branch1│ │Branch2│
      └───┬───┘ └───┬──┘
          │         │
      ┌───▼─────────▼───┐
      │ MergeConnector  │
      └────────┬─────────┘
               │
         ┌─────▼──────┐
         │   Output   │
         └────────────┘
```

### Cursor Execution Flow

```
1. [Cursor @ Input]
         │
         ▼
2. [Cursor @ Fork] ──► Context Copy
         │
    ┌────┴────┐
    │         │
3. [C1 @ B1] [C2 @ B2]  ◄── Parallel execution
    │         │
    └────┬────┘
         │
4. [Both @ Merge] ──► Wait & Merge
         │
         ▼
5. [Cursor @ Output]
```

## Performance Considerations

- **Parallelism**: Use `max_workers` to control ThreadPoolExecutor size
- **Context Copying**: Fork creates shallow copies of context data
- **Merge Waiting**: Blocked cursors wait for slowest branch
- **Memory**: NullContext avoids allocating data for missing inputs

## Future Enhancements

- Stream source integration
- Dynamic pipeline modification
- Pipeline visualization tools
- Advanced merge strategies
- Conditional branching
- Pipeline composition and reuse
