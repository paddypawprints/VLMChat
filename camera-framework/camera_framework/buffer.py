"""Ring buffer for inter-task communication with configurable policies."""

import threading
import time
from collections import deque
from typing import Any, Optional, Callable


class BufferFullError(Exception):
    """Raised when put() is called on a full buffer in strict mode."""
    pass


# Policy functions - passed to Buffer.__init__
def blocking_policy(buffer: 'Buffer', item: Any) -> bool:
    """Refuse writes when full (backpressure via is_ready()).
    
    This policy returns False when buffer is full, signaling that
    the task should not have called put(). The Runner's is_ready()
    check should prevent this scenario.
    
    If strict mode is enabled, raises BufferFullError instead.
    """
    with buffer.lock:
        if len(buffer.data) < buffer.size:
            buffer.data.append(item)
            if buffer.keep_latest:
                buffer._latest = item
            return True
        elif buffer.strict:
            raise BufferFullError(f"Buffer '{buffer.name}' is full (size={buffer.size}). Task violated is_ready() contract.")
        else:
            return False


def drop_oldest_policy(buffer: 'Buffer', item: Any) -> bool:
    """Drop oldest item if full (ring buffer behavior)."""
    with buffer.lock:
        buffer.data.append(item)  # deque with maxlen handles overflow
        if buffer.keep_latest:
            buffer._latest = item  # Cache latest value
        return True


def drop_newest_policy(buffer: 'Buffer', item: Any) -> bool:
    """Drop new item if buffer full (silently discard)."""
    with buffer.lock:
        if len(buffer.data) < buffer.size:
            buffer.data.append(item)
            if buffer.keep_latest:
                buffer._latest = item  # Cache latest value if accepted
        # Always return True - item either buffered or silently discarded
        # Producer never blocks, item auto-GC'd if not added
        return True


def decimate_policy(n: int) -> Callable[['Buffer', Any], bool]:
    """Create decimation policy (only accept every Nth item).
    
    Args:
        n: Accept every Nth item (2 = half rate, 3 = third rate, etc.)
        
    Returns:
        Policy function for rate control
    """
    def _decimate(buffer: 'Buffer', item: Any) -> bool:
        with buffer.lock:
            buffer.counter += 1
            if buffer.counter % n == 0:
                buffer.data.append(item)
                if buffer.keep_latest:
                    buffer._latest = item  # Cache latest value
                return True
            # Even if not accepted into queue, update latest (if enabled)
            if buffer.keep_latest:
                buffer._latest = item
            return False
    return _decimate


class Buffer:
    """Thread-safe ring buffer connecting pipeline tasks.
    
    Architecture: Buffers are 1-producer:1-consumer channels.
    - For fan-out (1:N): Upstream writes to N separate buffers
    - For fan-in (N:1): Downstream reads from N buffers (multiple inputs)
    - Multiple consumers on one buffer = competing consumers (queue pattern)
    
    Ownership and reference counting:
    - put(item): Buffer acquires reference, caller's ref released after return
    - get(): Transfers reference from buffer to caller (caller must release)
    - Dropped items: Python GC releases automatically when not added to buffer
    - Multiple downstreams: Use separate buffers (each holds its own reference)
    
    Policy behavior for dropped items:
    - drop_newest: Returns False, item not added → caller's ref goes out of scope
    - decimate: Returns False for skipped items → automatic GC
    - drop_oldest: deque.append() releases oldest atomically when full
    
    Buffers implement backpressure and rate control through policy functions.
    
    Examples:
        # Fast path with backpressure
        camera_to_yolo = Buffer(size=60, policy=blocking_policy)
        
        # Slow consumer (always get latest)
        yolo_to_smolvlm = Buffer(size=1, policy=drop_oldest_policy)
        
        # Rate control (30fps → 15fps)
        clusterer_to_encoder = Buffer(size=30, policy=decimate_policy(2))
        
        # Fan-out: camera writes to both buffers
        camera.outputs = [buffer_to_yolo, buffer_to_display]
        
        # Fan-in: fusion reads from both buffers
        fusion.inputs = [buffer_from_camera, buffer_from_lidar]
    """
    
    def __init__(
        self,
        size: int = 60,
        policy: Callable[['Buffer', Any], bool] = drop_oldest_policy,
        name: str = "",
        strict: bool = False,
        keep_latest: bool = False,
    ):
        """Initialize buffer.
        
        Args:
            size: Maximum number of items to hold
            policy: Function(buffer, item) -> bool that handles put logic
            name: Optional name for debugging
            strict: If True, raise BufferFullError when put() called on full buffer with blocking_policy
            keep_latest: If True, cache latest item for peek() (needed for snapshot tasks)
        """
        self.size = size
        self.policy_func = policy
        self.name = name
        self.strict = strict
        self.keep_latest = keep_latest
        self.data = deque(maxlen=size)
        self.lock = threading.Lock()
        
        # Counter for decimate policy
        self.counter = 0
        
        # Latest value cache - only populated if keep_latest=True
        # This allows snapshot-like tasks to always get the latest value
        # even when the queue has been consumed
        self._latest = None
        
        # Observers - tasks that peek/sample but don't consume (e.g., SnapshotTask)
        self.observers = []  # List of (task, label) tuples
        
        # Writers - tasks that produce data to this buffer
        # Used to validate strict mode doesn't have multiple producers
        self.writers = []  # List of tasks
    
    def has_data(self) -> bool:
        """Check if buffer has data available to consume.
        
        Returns True only if there's data in the queue that can be consumed with get().
        The cached _latest value is NOT counted since it's only for peek().
        
        Use in is_ready(): Check if task has input data to process.
        """
        with self.lock:
            return len(self.data) > 0
    
    def has_capacity(self) -> bool:
        """Check if buffer has space for more data.
        
        Returns True if buffer is not full. This reflects actual buffer state,
        not policy behavior. Policies may still accept data when full (by dropping).
        
        Use in is_ready(): Check if downstream can accept output without dropping.
        For tasks that want backpressure, check this before writing.
        
        Returns:
            True if buffer has space, False if at capacity
        """
        with self.lock:
            return len(self.data) < self.size
    
    def can_accept(self) -> bool:
        """Check if buffer will accept a put() call without dropping or refusing.
        
        Returns the same as has_capacity() - True if buffer has space.
        
        Policy-specific behavior:
        - drop_oldest_policy: Always returns True (auto-drops oldest)
        - drop_newest_policy: Returns True (silently drops when full)
        - blocking_policy: Returns has_capacity() (refuses when full)
        - decimate_policy: Returns True (drops based on decimation)
        
        For strict mode with blocking_policy, calling put() when this
        returns False will raise BufferFullError.
        
        Returns:
            True if buffer can accept without dropping/refusing
        """
        # For drop policies, always accept (they handle overflow)
        if self.policy_func in (drop_oldest_policy, drop_newest_policy):
            return True
        
        # For blocking and other policies, check actual capacity
        return self.has_capacity()
    
    def put(self, item: Any) -> bool:
        """Add item to buffer according to policy.
        
        Thread-safe reference transfer:
        1. Caller invokes put(item) - both hold references
        2. Policy function executes under lock:
           - If accepted: buffer.data.append(item) acquires reference
           - If dropped: no append, caller's ref auto-released when out of scope
        3. put() returns True/False
        4. Caller's local reference goes out of scope after return
        
        For dropped items (returns False):
        - Item is NOT added to buffer
        - Caller's reference goes out of scope
        - Python GC releases the item (no memory leak)
        
        For multiple downstream tasks (fan-out):
        - Don't use one buffer for multiple consumers!
        - Instead: upstream task writes to multiple buffers
        - Each buffer holds its own reference to a Context
        
        Args:
            item: Data to add
            
        Returns:
            True if item was added, False if dropped
        """
        # Apply policy function
        return self.policy_func(self, item)
    
    def get(self) -> Optional[Any]:
        """Remove and return oldest item from buffer.
        
        Transfers ownership from buffer to caller:
        - Buffer releases its reference
        - Caller now owns the item
        - Item will be GC'd when caller is done with it
        
        Returns:
            Item or None if buffer empty
        """
        with self.lock:
            if self.data:
                return self.data.popleft()
            return None
    
    def peek(self) -> Optional[Any]:
        """View latest item without removing it.
        
        Returns the most recent item added to the buffer, regardless of
        whether the queue has been consumed. This allows snapshot-like
        tasks to always get the latest value.
        
        Note: Returns a reference but does NOT transfer ownership.
        Buffer still holds the item.
        
        Returns:
            Most recent item or None if buffer never received data
        """
        with self.lock:
            return self._latest
    
    def clear(self) -> None:
        """Remove all items from buffer.
        
        Releases all references - items will be GC'd if no other refs exist.
        """
        with self.lock:
            self.data.clear()
            self.counter = 0
            self._latest = None
    
    def add_observer(self, task, label: str = "observe"):
        """Register a task as an observer (peeks/samples but doesn't consume).
        
        Args:
            task: Task that observes this buffer (e.g., SnapshotTask)
            label: Relationship label (e.g., "snapshot", "monitor")
        """
        self.observers.append((task, label))
    
    def add_writer(self, task):
        """Register a task as a writer (produces data to this buffer).
        
        Validates that strict buffers only have a single writer to prevent
        race conditions in multi-threaded execution.
        
        Args:
            task: Task that writes to this buffer
            
        Raises:
            ValueError: If buffer is strict and already has a writer
        """
        self.writers.append(task)
        
        # Strict mode only works with single producer
        if self.strict and len(self.writers) > 1:
            writer_names = [t.name for t in self.writers]
            raise ValueError(
                f"Buffer '{self.name}' is in strict mode but has {len(self.writers)} writers: {writer_names}. "
                f"Strict mode only works with a single producer. "
                f"Use drop_oldest_policy or drop_newest_policy for multiple producers."
            )
    
    def __len__(self) -> int:
        """Get current number of items in buffer."""
        with self.lock:
            return len(self.data)
    
    def __repr__(self) -> str:
        name_str = f" '{self.name}'" if self.name else ""
        policy_name = getattr(self.policy_func, '__name__', repr(self.policy_func))
        return f"Buffer{name_str}(size={self.size}, policy={policy_name}, items={len(self)})"
