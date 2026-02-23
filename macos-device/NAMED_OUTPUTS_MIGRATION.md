# Camera Framework API Changes - Named Outputs Migration

## Summary of Breaking Changes

### 1. Buffer API Changes

**`can_accept()` now returns actual capacity status:**
```python
# OLD: Always returned True
buffer.can_accept()  # Always True

# NEW: Returns actual capacity (except for drop policies)
buffer.can_accept()  # Returns has_capacity() for blocking_policy
```

**`blocking_policy` no longer blocks threads:**
```python
# OLD: Blocked in while loop
def blocking_policy(buffer, item):
    while True:  # ← Blocked worker thread
        if len(buffer.data) < size:
            return True
        sleep(0.001)

# NEW: Returns False when full
def blocking_policy(buffer, item):
    if len(buffer.data) < size:
        return True
    elif buffer.strict:
        raise BufferFullError(...)  # Crash in strict mode
    else:
        return False  # Let is_ready() prevent this
```

**New strict mode:**
```python
# Raises exception if put() called when full
buffer = Buffer(size=1, policy=blocking_policy, strict=True)
```

### 2. BaseTask API Changes

**Outputs changed from list to dict:**
```python
# OLD: List of buffers
class BaseTask:
    self.outputs = []  # List[Buffer]
    self.inputs = []   # List[Buffer]

# NEW: Dict of named buffers
class BaseTask:
    self.outputs = {}  # Dict[str, Buffer]
    self.inputs = {}   # Dict[str, Buffer]
```

**New methods for wiring:**
```python
# OLD
task.outputs.append(buffer)
task.inputs.append(buffer)

# NEW
task.add_output('default', buffer)
task.add_input('default', buffer)
```

**is_ready() default changed:**
```python
# OLD: ANY input, ALL outputs
has_input = any(buf.has_data() for buf in self.inputs)

# NEW: ALL inputs, ALL outputs
has_input = all(buf.has_data() for buf in self.inputs.values())
```

### 3. Task Implementation Changes

**Reading from inputs:**
```python
# OLD
ctx = self.inputs[0].get()

# NEW
ctx = self.inputs['default'].get()
```

**Writing to outputs:**
```python
# OLD
for buf in self.outputs:
    buf.put(ctx)

# NEW
self.outputs['default'].put(ctx)

# Or for routers with multiple outputs:
if condition:
    self.outputs['path_a'].put(ctx)
else:
    self.outputs['path_b'].put(ctx)
```

## Migration Guide

### Step 1: Update All Task Implementations

**Camera tasks:**
```python
# OLD
def process(self):
    ctx = Context()
    frame = capture()
    ctx.append("frame", frame)
    for buf in self.outputs:
        buf.put(ctx)

# NEW
def process(self):
    ctx = Context()
    frame = capture()
    ctx.append("frame", frame)
    self.outputs['default'].put(ctx)
```

**Processing tasks:**
```python
# OLD
def process(self):
    ctx = self.inputs[0].get()
    if ctx:
        result = process_data(ctx)
        for buf in self.outputs:
            buf.put(ctx)

# NEW
def process(self):
    ctx = self.inputs['default'].get()
    if ctx:
        result = process_data(ctx)
        self.outputs['default'].put(ctx)
```

**Router tasks:**
```python
# NEW (routers benefit from named outputs)
def process(self):
    ctx = self.inputs['default'].get()
    if ctx:
        if should_route_to_a():
            self.outputs['path_a'].put(ctx)
        elif should_route_to_b():
            self.outputs['path_b'].put(ctx)
```

### Step 2: Update Pipeline Wiring

**OLD wiring (append to lists):**
```python
camera.outputs.append(camera_buffer)
yolo.inputs.append(camera_buffer)
yolo.outputs.append(yolo_buffer)
tracker.inputs.append(yolo_buffer)
tracker.outputs.append(alert_buffer)
```

**NEW wiring (named connections):**
```python
# Simple linear pipeline - use 'default' for standard flow
camera.add_output('default', camera_buffer)
yolo.add_input('default', camera_buffer)
yolo.add_output('default', yolo_buffer)
tracker.add_input('default', yolo_buffer)
tracker.add_output('default', alert_buffer)

# Router with multiple outputs - use semantic names
vlm_queue.add_input('default', tracker_buffer)
vlm_queue.add_output('alerts', alert_buffer)
vlm_queue.add_output('vlm', smolvlm_buffer)
```

### Step 3: Update Custom is_ready() Implementations

**For tasks with optional inputs:**
```python
# OLD
def is_ready(self):
    has_input = any(buf.has_data() for buf in self.inputs)
    return has_input and all(buf.has_capacity() for buf in self.outputs)

# NEW  
def is_ready(self):
    # Check specific input by name
    has_primary = self.inputs['primary'].has_data()
    has_optional = self.inputs.get('optional', None)
    has_optional_data = has_optional.has_data() if has_optional else False
    
    return has_primary and all(buf.has_capacity() for buf in self.outputs.values())
```

**For router tasks:**
```python
# NEW - VLMQueue only checks alert output capacity
def is_ready(self):
    has_input = self.inputs['default'].has_data()
    
    # Only check alert buffer capacity (VLM managed internally)
    alert_ready = self.outputs['alerts'].has_capacity()
    
    return has_input and alert_ready
```

## Files That Need Updates

### camera-framework/
- ✅ `camera_framework/buffer.py` - DONE
- ✅ `camera_framework/task.py` - DONE
- ⚠️ `camera_framework/cameras/image_library_camera.py` - Needs update
- ⚠️ `camera_framework/bridges/*.py` - All bridge tasks need updates

### macos-device/
- ⚠️ `macos_device/__main__.py` - Update all wiring
- ⚠️ `macos_device/camera.py` - Update outputs
- ⚠️ `macos_device/yolo_task.py` - Update inputs/outputs
- ⚠️ `macos_device/detection_tracker.py` - Update inputs/outputs
- ⚠️ `macos_device/alert_publisher.py` - Update inputs
- ⚠️ `macos_device/category_router.py` - Update (router task!)
- ⚠️ `macos_device/attribute_task.py` - Update inputs/outputs
- ⚠️ `macos_device/attribute_color_filter.py` - Update inputs/outputs
- ⚠️ `macos_device/clusterer.py` - Update inputs/outputs
- ✅ `macos_device/vlm_queue.py` - Create NEW
- ✅ `macos_device/smolvlm_worker.py` - Create NEW

### Examples and Tests
- ⚠️ `camera-framework/examples/*.py` - All examples need updates
- ⚠️ `camera-framework/tests/*.py` - All tests need updates

## Benefits of This Change

1. **Self-Documenting Pipelines**: Named outputs make dataflow explicit
   ```python
   vlm_queue.outputs['alerts']  # Clear what this is
   vlm_queue.outputs['vlm']     # vs outputs[0], outputs[1]
   ```

2. **Type-Safe Wiring**: IDE autocomplete helps prevent mistakes
   ```python
   # Typo caught immediately
   vlm_queue.add_output('alert', buf)  # Should be 'alerts'
   ```

3. **Flexible Routing**: Router tasks can manage multiple outputs semantically
   ```python
   # Easy to understand routing logic
   if needs_immediate_alert:
       self.outputs['high_priority'].put(ctx)
   elif needs_vlm:
       self.outputs['vlm_queue'].put(ctx)
   else:
       self.outputs['standard'].put(ctx)
   ```

4. **Better is_ready() Logic**: Tasks can check specific outputs
   ```python
   # Check only critical output
   return self.inputs['default'].has_data() and \
          self.outputs['critical'].has_capacity()
   ```

5. **No Thread Blocking**: blocking_policy cooperates with Runner scheduling
   - Task won't run if output full (is_ready() returns False)
   - No worker threads hanging in sleep loops
   - Clean failure with strict mode (BufferFullError)

## Next Steps

1. Implement VLMQueue with internal queue management
2. Update all existing tasks to use named inputs/outputs
3. Update __main__.py wiring to use add_input/add_output
4. Test the complete pipeline with new architecture
