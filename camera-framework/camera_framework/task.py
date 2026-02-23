"""Base task abstraction."""

import time
import threading
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .buffer import Buffer


class BaseTask(ABC):
    """Abstract base class for pipeline tasks.
    
    Tasks process Context objects, reading inputs and writing outputs.
    Context flows through buffers between tasks like data through pipes.
    
    Execution model:
    - Source tasks (no inputs): Create and populate Context, write to output buffers
    - Processing tasks (has inputs): Read Context from input buffer, modify, write to outputs
    - One-off tasks (queued): Get temporary Context for command processing
    
    Args:
        name: Optional task name (defaults to class name)
        fields: Optional field name mappings for inputs/outputs
        interval: Optional interval in seconds for periodic tasks (None = run every iteration)
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        fields: Optional[Dict[str, str]] = None,
        interval: Optional[float] = None,
    ):
        self.name = name or self.__class__.__name__
        self.fields = fields or {}
        self.interval = interval
        self.last_run = 0.0  # Track last execution time for periodic tasks
        self._background_busy = False  # Track if background execution is running
        self._busy_lock = threading.Lock()  # Protect _background_busy flag
        
        # Named buffer connections (set by pipeline builder)
        # For most tasks: single input/output with semantic names
        # For routers: multiple named outputs (e.g., 'alerts', 'vlm')
        self.inputs: Dict[str, 'Buffer'] = {}
        self.outputs: Dict[str, 'Buffer'] = {}
    
    def add_output(self, name: str, buffer: 'Buffer') -> None:
        """Add a named output buffer.
        
        Args:
            name: Semantic name for this output (e.g., 'default', 'alerts', 'vlm')
            buffer: Buffer to connect
        
        Raises:
            ValueError: If buffer is strict and already has a writer
        
        Example:
            camera.add_output('frame', camera_buffer)
            vlm_queue.add_output('alerts', alert_buffer)
            vlm_queue.add_output('vlm', smolvlm_buffer)
        """
        self.outputs[name] = buffer
        buffer.add_writer(self)  # Register and validate
    
    def add_input(self, name: str, buffer: 'Buffer') -> None:
        """Add a named input buffer.
        
        Args:
            name: Semantic name for this input (e.g., 'default', 'detections')
            buffer: Buffer to connect
        """
        self.inputs[name] = buffer
    
    def field(self, name: str) -> str:
        """Get mapped field name, or original if no mapping exists."""
        return self.fields.get(name, name)
    
    def is_ready(self) -> bool:
        """Check if task can run (has input data and output capacity).
        
        Default policy: Ready if ALL inputs have data and ALL outputs have capacity.
        
        Minimal Buffer Interface (safe to use in is_ready):
        - buffer.has_data() -> bool: Returns True if data available to consume
        - buffer.has_capacity() -> bool: Returns True if buffer not full
        - buffer.can_accept() -> bool: Depends on policy (check before put())
        
        Override for custom scheduling policies:
        - Router tasks: Check specific outputs, ignore others
        - Source tasks: Always ready (no inputs)
        - Optional inputs: Use any() instead of all()
        - Conditional tasks: Check specific buffer patterns
        
        Important:
        - Do NOT call get() or put() in is_ready() (read-only check)
        - Do NOT access buffer internals (policy, data, lock)
        - Only use documented interface methods
        
        Returns:
            True if task should execute, False if should wait
        """
        # Source tasks (no inputs) are always ready
        if not self.inputs:
            return True
        
        # Default: ready if ALL inputs have data and ALL outputs have capacity
        # Most tasks have single input/output, so this is straightforward
        has_input = all(buf.has_data() for buf in self.inputs.values())
        has_output_capacity = all(buf.has_capacity() for buf in self.outputs.values()) if self.outputs else True
        
        return has_input and has_output_capacity
    
    def warmup(self) -> None:
        """Optional eager-initialisation called before the pipeline starts.

        Override in tasks that load heavy resources (ML models, ONNX runtimes,
        GPU allocations) so the main pipeline loop is never blocked on first use.
        The default implementation does nothing.
        """
        pass

    def start(self) -> None:
        """Optional startup called after warmup, before the pipeline loop begins.

        Override in tasks that need background threads or persistent connections
        (e.g. cameras, MQTT subscribers). The default implementation does nothing.
        """
        pass

    def stop(self) -> None:
        """Optional teardown called when the pipeline is shutting down.

        Override in tasks that hold resources started in start() or warmup().
        The default implementation does nothing.
        """
        pass

    @abstractmethod
    def process(self) -> None:
        """Process data by reading from input buffers and writing to output buffers.
        
        Implementation pattern:
        - Processing tasks: Read from self.inputs['default'].get(), modify, write to outputs
        - Source tasks: Create dict, populate, write to self.outputs
        - One-off tasks: Create temporary dict, process, no buffer I/O
        
        Example (processing task with single input/output):
            message = self.inputs['default'].get()
            if message:
                frame = message.get("frame", [[]])[0]
                result = self.process_frame(frame)
                message.setdefault("result", []).append(result)
                self.outputs['default'].put(message)
        
        Example (source task):
            message = {}
            frame = self.capture_frame()
            message.setdefault("frame", []).append(frame)
            self.outputs['default'].put(message)
        
        Example (router task with multiple outputs):
            message = self.inputs['default'].get()
            if needs_vlm:
                self.outputs['vlm'].put(message)
            else:
                self.outputs['alerts'].put(message)
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
