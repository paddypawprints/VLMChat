"""Duck-typed visitor for pipeline introspection and visualization.

Visitors implement methods like:
    - visit_source(task, task_id)
    - visit_task(task, task_id)
    - visit_sink(task, task_id)
    - visit_buffer(buffer, buffer_id)
    - visit_connection(from_task, to_task, buffer, from_port, to_port)
    - get_result() -> Any

No ABC - pure duck typing for flexibility.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runner import Runner


class PipelineTraverser:
    """Traverses pipeline graph and calls visitor methods.
    
    Distinguishes sources (no inputs), tasks (has inputs and outputs),
    and sinks (has inputs, no outputs or unused outputs).
    
    Usage:
        visitor = MermaidVisitor()
        traverser = PipelineTraverser(runner)
        traverser.traverse(visitor)
        result = visitor.get_result()
    """
    
    def __init__(self, runner: 'Runner'):
        """Initialize traverser.
        
        Args:
            runner: Runner containing tasks to traverse
        """
        self.runner = runner
    
    def traverse(self, visitor) -> None:
        """Traverse pipeline and call visitor methods (duck-typed).
        
        Calls visitor methods in order:
        1. visit_source/visit_task/visit_sink for each task
        2. visit_buffer for each unique buffer
        3. visit_connection for each edge
        4. visit_observer for each observer relationship (buffer → task)
        
        Args:
            visitor: Any object with visit_* methods
        """
        visited_buffers = {}  # buffer object -> buffer_id mapping
        task_ids = {}  # task object -> task_id mapping
        buffer_counter = 0
        task_counter = 0
        
        # Visit all tasks (categorize as source/task/sink)
        # First pass: collect tasks and check for observer relationships
        task_observers = set()  # Tasks that are observers
        for task in self.runner.tasks:
            for out_name, out_buf in task.outputs.items():
                for observer_task, _ in out_buf.observers:
                    task_observers.add(observer_task)
        
        for task in self.runner.tasks:
            task_id = f"task{task_counter}"
            task_ids[task] = task_id
            task_counter += 1
            
            # Categorize task type
            has_inputs = len(task.inputs) > 0
            has_outputs = len(task.outputs) > 0
            is_observer = task in task_observers
            
            if not has_inputs and not is_observer:
                # Source task (creates data)
                if hasattr(visitor, 'visit_source'):
                    visitor.visit_source(task, task_id)
            elif not has_outputs or is_observer:
                # Sink task (consumes/observes data)
                if hasattr(visitor, 'visit_sink'):
                    visitor.visit_sink(task, task_id)
            else:
                # Processing task (transforms data)
                if hasattr(visitor, 'visit_task'):
                    visitor.visit_task(task, task_id)
        
        # Visit all unique buffers
        for task in self.runner.tasks:
            for out_name, out_buf in task.outputs.items():
                if out_buf not in visited_buffers:
                    buffer_id = f"buf{buffer_counter}"
                    visited_buffers[out_buf] = buffer_id
                    buffer_counter += 1
                    
                    if hasattr(visitor, 'visit_buffer'):
                        visitor.visit_buffer(out_buf, buffer_id)
        
        # Visit all connections (edges)
        for task in self.runner.tasks:
            for out_name, out_buf in task.outputs.items():
                buffer_id = visited_buffers[out_buf]
                from_task_id = task_ids[task]
                
                # Find all tasks reading from this buffer
                for other_task in self.runner.tasks:
                    for in_name, in_buf in other_task.inputs.items():
                        if in_buf is out_buf:
                            to_task_id = task_ids[other_task]
                            
                            if hasattr(visitor, 'visit_connection'):
                                visitor.visit_connection(
                                    from_task=task,
                                    from_task_id=from_task_id,
                                    to_task=other_task,
                                    to_task_id=to_task_id,
                                    buffer=out_buf,
                                    buffer_id=buffer_id,
                                    from_port=out_name,
                                    to_port=in_name,
                                )
        
        # Visit observer relationships (buffer → task)
        if hasattr(visitor, 'visit_observer'):
            for buf, buf_id in visited_buffers.items():
                for observer_task, label in buf.observers:
                    if observer_task in task_ids:
                        visitor.visit_observer(
                            buffer=buf,
                            buffer_id=buf_id,
                            observer_task=observer_task,
                            observer_task_id=task_ids[observer_task],
                            label=label
                        )


class MermaidVisitor:
    """Generate Mermaid flowchart diagram from pipeline.
    
    Output format:
        flowchart TD
            task0[Camera]
            task1[YOLO]
            task2[AlertPublisher]
            buf0{{buffer: frames}}
            task0 -->|frames| buf0
            buf0 -->|in| task1
    """
    
    def __init__(self):
        self.lines = ["flowchart TD"]
        self.buffer_info = {}  # buffer_id -> (name, size, policy)
        self.sources = []  # Track sources
        self.tasks = []  # Track processing tasks
        self.sinks = []  # Track sinks
        self.buffers = []  # Track buffers
        self.connections = []  # Track connections
    
    def visit_source(self, task, task_id):
        """Visit source task (creates data)."""
        self.sources.append(f"    {task_id}([{task.name}])")
    
    def visit_task(self, task, task_id):
        """Visit processing task (transforms data)."""
        self.tasks.append(f"    {task_id}[{task.name}]")
    
    def visit_sink(self, task, task_id):
        """Visit sink task (consumes data)."""
        self.sinks.append(f"    {task_id}[/{task.name}/]")
    
    def visit_buffer(self, buffer, buffer_id):
        """Visit buffer (connection)."""
        policy_name = getattr(buffer.policy_func, '__name__', 'policy')
        # Simplify policy name (drop_oldest_policy -> drop_oldest)
        policy_name = policy_name.replace('_policy', '')
        
        name = buffer.name or "buffer"
        size = buffer.size
        
        # Store info for connection labels
        self.buffer_info[buffer_id] = (name, size, policy_name)
        
        # Render buffer as diamond node with metadata
        label = f"{name}<br/>size={size}<br/>{policy_name}"
        self.buffers.append(f"    {buffer_id}{{{{{label}}}}}")
    
    def visit_connection(self, from_task, from_task_id, to_task, to_task_id, 
                        buffer, buffer_id, from_port, to_port):
        """Visit connection between tasks through buffer."""
        # Task → Buffer
        self.connections.append(f"    {from_task_id} -->|{from_port}| {buffer_id}")
        
        # Buffer → Task
        self.connections.append(f"    {buffer_id} -->|{to_port}| {to_task_id}")
    
    def visit_observer(self, buffer, buffer_id, observer_task, observer_task_id, label):
        """Visit observer relationship (buffer -.-> task)."""
        # Use dashed line to show observation (not consumption)
        self.connections.append(f"    {buffer_id} -.->|{label}| {observer_task_id}")
    
    def get_result(self) -> str:
        """Get Mermaid diagram as string."""
        # Build diagram with subgraphs for better organization
        result = [self.lines[0]]  # flowchart TD
        result.append("")
        
        # Sources subgraph
        if self.sources:
            result.append("    subgraph Sources")
            result.extend(self.sources)
            result.append("    end")
            result.append("")
        
        # Processing tasks subgraph
        if self.tasks:
            result.append("    subgraph Processing")
            result.extend(self.tasks)
            result.append("    end")
            result.append("")
        
        # Sinks subgraph
        if self.sinks:
            result.append("    subgraph Sinks")
            result.extend(self.sinks)
            result.append("    end")
            result.append("")
        
        # Buffers (no subgraph, shown between tasks)
        if self.buffers:
            result.extend(self.buffers)
            result.append("")
        
        # Connections
        if self.connections:
            result.extend(self.connections)
        
        return "\n".join(result)
