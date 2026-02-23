# Runner Scheduling - Current Implementation

## Pseudocode

```python
# ==================== Runner.run_once() ====================
def run_once():
    now = current_time()
    futures = []
    
    # 1. Process one-off queued tasks (always ready)
    while task_queue has tasks:
        task = dequeue()
        submit_to_threadpool(task)
    
    # 2. Check each pipeline task
    for task in pipeline_tasks:
        
        # A. Check interval timing
        if task.interval is set:
            if (now - task.last_run) < task.interval:
                skip_task()  # Too soon since last run
                continue
        
        # B. Check if already running
        if task._background_busy:
            skip_task()  # Prevent double-submission
            continue
        
        # C. Check task readiness
        if not task.is_ready():
            skip_task()  # Not ready to run
            continue
        
        # D. Submit to thread pool
        task._background_busy = True  # Mark BEFORE submit
        submit_to_threadpool(task)
    
    # 3. Wait for all submitted tasks to complete
    for future in futures:
        future.result()  # Blocks until task finishes


# ==================== BaseTask.is_ready() ====================
def is_ready():
    # Source tasks (no inputs) always ready
    if no inputs:
        return True
    
    # Check input availability
    has_input = any(buffer.has_data() for buffer in inputs)
    
    # Check output capacity
    if has outputs:
        can_output = all(buffer.can_accept() for buffer in outputs)
    else:
        can_output = True  # No outputs = always ready
    
    return has_input AND can_output


# ==================== Buffer.has_data() ====================
def has_data():
    with lock:
        return len(data) > 0  # True if queue has items


# ==================== Buffer.can_accept() ====================
def can_accept():
    return True  # Always returns True for all policies
    # Note: Policies handle full buffers internally (drop/block)


# ==================== Buffer.put() ====================
def put(item):
    return policy_func(buffer, item)

# Policy examples:
def drop_oldest_policy(buffer, item):
    with lock:
        buffer.data.append(item)  # deque with maxlen auto-drops oldest
        return True

def drop_newest_policy(buffer, item):
    with lock:
        if len(data) < size:
            buffer.data.append(item)
            return True
        # Silently drop new item
        return True  # Still returns True!

def blocking_policy(buffer, item):
    while True:
        with lock:
            if len(data) < size:
                buffer.data.append(item)
                return True
        sleep(0.001)  # Wait for space
```

## Key Observations

### 1. **Buffer.can_accept() Always Returns True**
```python
def can_accept(self) -> bool:
    """Always returns True since all policies accept (may discard internally)."""
    return True
```

**Implication:** The `is_ready()` check for output capacity is **currently useless**!

```python
can_output = all(buf.can_accept() for buf in self.outputs)
# Always evaluates to True because can_accept() always returns True
```

This means tasks will **always** try to write, and policies handle overflow:
- `drop_oldest_policy` - Overwrites oldest silently
- `drop_newest_policy` - Drops new item silently
- `blocking_policy` - **BLOCKS inside put()** ⚠️

### 2. **Blocking Policy Problem**

With blocking_policy:
```python
smolvlm_buffer = Buffer(size=1, policy=blocking_policy)

# Task executes in threadpool
def process():
    ctx = inputs[0].get()
    for buf in outputs:
        buf.put(ctx)  # ← BLOCKS HERE if buffer full
```

**This blocks the worker thread**, not just the task scheduling!

### 3. **Multiple Outputs Problem**

```python
class VLMQueue(BaseTask):
    def process(self):
        detection = inputs[0].get()
        
        # Try to write to BOTH outputs
        alert_buffer.put(detection)      # ← Might drop
        smolvlm_buffer.put(detection)    # ← Might block!
```

If smolvlm_buffer uses `blocking_policy`, the worker thread **hangs** waiting for SmolVLM to consume.

## Solutions for VLMQueue

### Option 1: Override is_ready() - Ignore Outputs ✅

```python
class VLMQueue(BaseTask):
    def is_ready(self) -> bool:
        """Router ignores downstream capacity - manages queues internally."""
        # Only check inputs
        has_input = any(buf.has_data() for buf in self.inputs)
        return has_input  # ← Don't check outputs
    
    def process(self):
        detection = inputs[0].get()
        
        # Route based on vlm_required
        if vlm_required and not vlm_queue_full:
            vlm_queue.append(detection)
        else:
            alert_queue.append(detection)
        
        # Drain internal queues to output buffers
        if alert_queue and alert_buffer.can_accept():
            alert_buffer.put(alert_queue.pop(0))
        
        if vlm_queue and smolvlm_buffer.can_accept():
            smolvlm_buffer.put(vlm_queue.pop(0))
```

**Pro:** 
- VLMQueue never blocks
- Internal queues provide buffering
- Clean separation of routing logic

**Con:**
- Still need to handle `smolvlm_buffer.put()` blocking if using blocking_policy

### Option 2: Non-Blocking Write Pattern ✅

```python
class VLMQueue(BaseTask):
    def process(self):
        detection = inputs[0].get()
        
        # Manage internal queues
        route_detection(detection)
        
        # Non-blocking drain to outputs
        self._try_drain_alert_queue()
        self._try_drain_vlm_queue()
    
    def _try_drain_vlm_queue(self):
        """Try to drain VLM queue without blocking."""
        if not vlm_queue:
            return
        
        # Check SmolVLM worker state directly (not buffer)
        if smolvlm_worker.is_busy():
            return  # Don't try to write
        
        # Try non-blocking write
        detection = vlm_queue[0]
        if smolvlm_buffer.put(detection):  # Returns immediately
            vlm_queue.pop(0)
```

### Option 3: Change smolvlm_buffer Policy ✅

Instead of `blocking_policy`, use `drop_newest_policy`:

```python
# SmolVLM buffer drops new items if full (busy)
smolvlm_buffer = Buffer(size=1, policy=drop_newest_policy)

class VLMQueue:
    def _try_drain_vlm_queue(self):
        if not vlm_queue:
            return
        
        detection = vlm_queue[0]
        accepted = smolvlm_buffer.put(detection)
        
        if accepted:
            vlm_queue.pop(0)  # Success - remove from queue
        else:
            pass  # Buffer full - try again next iteration
```

**Pro:**
- No blocking
- VLMQueue can check return value
- Keeps VLM queue intact if write fails

## Recommended Architecture

```python
# 1. VLMQueue with internal queues + custom is_ready()
class VLMQueue(BaseTask):
    def __init__(self, max_vlm_queue=10, max_alert_queue=50):
        self.vlm_queue = []       # Internal queue for VLM processing
        self.alert_queue = []     # Internal queue for alerts
        self.max_vlm_queue = max_vlm_queue
        self.max_alert_queue = max_alert_queue
    
    def is_ready(self) -> bool:
        """Override: ignore output capacity, manage internally."""
        return any(buf.has_data() for buf in self.inputs)
    
    def process(self):
        # 1. Route incoming detection
        detection = self.inputs[0].get()
        if detection:
            self._route_detection(detection)
        
        # 2. Drain internal queues to output buffers
        self._drain_queues()
    
    def _route_detection(self, detection):
        """Route to VLM or alert queue based on vlm_required."""
        vlm_required = detection.get('vlm_required', False)
        
        if vlm_required and not self._is_smolvlm_busy():
            # Route to VLM queue
            if len(self.vlm_queue) >= self.max_vlm_queue:
                # Eject oldest from VLM queue to alert queue
                oldest = self.vlm_queue.pop(0)
                oldest['vlm_timeout'] = True
                self.alert_queue.append(oldest)
            
            self.vlm_queue.append(detection)
        else:
            # Route to alert queue
            if len(self.alert_queue) >= self.max_alert_queue:
                # Drop oldest alert (for now)
                self.alert_queue.pop(0)
            
            self.alert_queue.append(detection)
    
    def _drain_queues(self):
        """Try to move items from internal queues to output buffers."""
        # Drain alert queue to alert_buffer
        if self.alert_queue:
            self.outputs[0].put(self.alert_queue.pop(0))
        
        # Drain VLM queue to smolvlm_buffer (only if not busy)
        if self.vlm_queue and not self._is_smolvlm_busy():
            # Try to write (non-blocking with drop_newest_policy)
            detection = self.vlm_queue[0]
            if self.outputs[1].put(detection):
                self.vlm_queue.pop(0)  # Success
    
    def _is_smolvlm_busy(self):
        """Check if SmolVLM worker is processing."""
        return self.smolvlm_worker.is_busy()


# 2. Buffer configuration
alert_buffer = Buffer(size=50, policy=drop_oldest_policy, name="alerts")
smolvlm_buffer = Buffer(size=1, policy=drop_newest_policy, name="smolvlm")
```

## Questions

1. **Buffer.can_accept() returning True always** - Should we change this to actually check capacity?
   - Pro: More accurate scheduling
   - Con: Breaks existing behavior, may cause unexpected blocking

2. **SmolVLM buffer size** - You said "1 element blocking buffer", but blocking seems problematic. Should we:
   - Use `drop_newest_policy` (size=1, non-blocking)? ✅
   - Keep blocking but handle in worker thread?
   
3. **Alert queue overflow** - You said "just drop for now". Should alert_queue use:
   - Fixed size with drop_oldest?
   - Unbounded (risk OOM)?
   - What size? (10? 50? 100?)

Let me know your thoughts!
