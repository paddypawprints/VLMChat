# Pipeline Visualization System

## Duck-Typed Visitor Pattern

Self-documenting pipeline visualization using duck-typed visitor pattern.

### Architecture

**No ABCs** - Pure duck typing for maximum flexibility:
```python
# Visitor just needs these methods (all optional):
class MyVisitor:
    def visit_source(task, task_id): ...
    def visit_task(task, task_id): ...
    def visit_sink(task, task_id): ...
    def visit_buffer(buffer, buffer_id): ...
    def visit_connection(from_task, to_task, buffer, ...): ...
    def get_result(): ...
```

### Task Classification

Traverser automatically categorizes tasks:
- **Sources** (no inputs) → Stadium shape `([Camera])`
- **Tasks** (has inputs + outputs) → Rectangle `[YOLO]`
- **Sinks** (has inputs, no/unused outputs) → Trapezoid `[/Display/]`

### Buffers

Rendered as diamond nodes with metadata:
```
{{buffer_name<br/>size=30<br/>drop_oldest}}
```

### Usage

**Generate diagram from code:**
```bash
# Run from macos-device
python -m macos_device --diagram

# Or use justfile
just diagram
```

**Outputs:**
- `web-platform/diagrams/pipeline-topology.mmd` - Mermaid source
- `web-platform/client/public/diagrams/pipeline-topology.svg` - SVG for UI

**Programmatic use:**
```python
from camera_framework import PipelineTraverser, MermaidVisitor

# Build pipeline
runner = Runner([...])

# Generate diagram
visitor = MermaidVisitor()
traverser = PipelineTraverser(runner)
traverser.traverse(visitor)
mermaid_code = visitor.get_result()
```

### Custom Visitors

Create any visitor - no inheritance required:

```python
class JSONVisitor:
    def __init__(self):
        self.nodes = []
        self.edges = []
    
    def visit_source(self, task, task_id):
        self.nodes.append({'id': task_id, 'type': 'source', 'name': task.name})
    
    def visit_task(self, task, task_id):
        self.nodes.append({'id': task_id, 'type': 'task', 'name': task.name})
    
    def visit_connection(self, from_task, from_task_id, to_task, to_task_id, **kwargs):
        self.edges.append({'from': from_task_id, 'to': to_task_id})
    
    def get_result(self):
        return {'nodes': self.nodes, 'edges': self.edges}
```

### Benefits

1. **Decoupled** - Pipeline has zero knowledge of visualization
2. **Flexible** - Create visitors for any format (Mermaid, D3, Cytoscape, JSON, etc.)
3. **Pythonic** - Duck typing, no ABCs, simple interfaces
4. **Named outputs** - Connections show semantic port names
5. **Auto-categorization** - Sources/tasks/sinks distinguished automatically

### Integration

Visitor integrates with existing tooling:
- Uses `npx @mermaid-js/mermaid-cli` (same as other diagrams)
- Outputs to same directory structure
- Can be viewed in web UI at `/diagrams/pipeline-topology.svg`
