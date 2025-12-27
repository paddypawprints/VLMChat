"""
Base class for all tasks in the pipeline.

Attributes:
    task_id (str): Unique identifier for the task.
    input_contract (Dict[str, Any]): Input data for the task.
    output_contract (Dict[str, Any]): Output data for the task.
    status (str): Current status of the task (e.g., "pending", "running", "completed").
"""

import logging
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
import enum
from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
import copy
import time

if TYPE_CHECKING:
    from ..cache.item import CachedItem
    from ..cache.types import CachedItemType

logger = logging.getLogger(__name__)


# Task Registry
_task_registry: Dict[str, type] = {}


def register_task(name: str):
    """
    Decorator to register a task class by name.
    
    Usage:
        @register_task('my_task')
        class MyTask(BaseTask):
            ...
    
    Args:
        name: The name used in DSL to reference this task
    
    Returns:
        The decorator function
    """
    def decorator(cls):
        _task_registry[name] = cls
        return cls
    return decorator


def get_task_registry() -> Dict[str, type]:
    """
    Get a copy of the task registry.
    
    Returns:
        Dictionary mapping task names to task classes
    """
    return _task_registry.copy()


class LoopControlAction(enum.Enum):
    """
    Actions that loop control conditions can return.
    
    Used by LoopCondition tasks to control loop execution flow.
    """
    PASS = "pass"           # Continue to next task (default)
    CONTINUE = "continue"   # Jump back to loop start, skip remaining tasks
    BREAK = "break"         # Exit loop, skip remaining tasks


class ContextDataType(enum.Enum):
    TEXT = "text"
    IMAGE = "image"
    DETECTIONS = "detections"
    MATCHES = "matches"
    CROPS = "crops"
    EMBEDDINGS = "embeddings"
    SIMILARITY_SCORES = "similarity_scores"
    PROMPT_EMBEDDINGS = "prompt_embeddings"  # Deprecated: use EMBEDDINGS instead
    PROMPT_SIMILARITY = "prompt_similarity"
    AUDIT = "audit"
    LOOP_STACK = "loop_stack"
    DIAGNOSTIC = "diagnostic"
    
    def __init__(self, type_name: str):
        self.type_name = type_name


class Context:
    def __init__(self) -> None:
        # Label-based data structure: data[data_type][label] = [items]
        # Default label is "default" when no label specified
        self.data: Dict[ContextDataType, Dict[str, List['CachedItem']]] = {}
        self.collector = None  # Set by runner if metrics enabled
        self.config = None  # Application config (VLMChatConfig)
        self.pipeline_runner = None  # Set by runner for queue-based I/O
        self.exception: Optional[Exception] = None  # Carried exception for error handling
        self.exception_source_task: Optional[str] = None  # Task that raised exception
        self._cache = None  # Set by runner - ItemCache instance
    
    def split(self, num_branches: int, immutable_cache: Dict[ContextDataType, List[Any]]) -> List['Context']:
        """
        Split context into multiple branches with shallow-copied lists.
        
        Each branch gets its own list that can be independently modified (add/remove items),
        but the list items themselves are shared references. This is memory-efficient for
        heavy objects (images, embeddings) while providing list isolation across branches.
        
        To modify an item: remove from list, create modified copy, add back to list.
        """
        contexts = []
        for _ in range(num_branches):
            ctx = Context()
            ctx.collector = self.collector
            ctx.config = self.config  # Share config across branches
            ctx.pipeline_runner = self.pipeline_runner  # Share runner reference
            ctx._cache = self._cache  # Share cache reference
            
            # Shallow copy all label dicts and lists - each branch gets own copies
            for data_type, label_dict in self.data.items():
                ctx.data[data_type] = {}
                for label, items in label_dict.items():
                    if self.collector:
                        start = time.time()
                        ctx.data[data_type][label] = list(items)
                        duration_ms = (time.time() - start) * 1000.0
                        self.collector.data_point("context.split.copy.duration", 
                                                 {"data_type": data_type.type_name, "label": label, "copy_type": "shallow"}, 
                                                 duration_ms)
                    else:
                        ctx.data[data_type][label] = list(items)
            
            # Copy exception state to all branches (exception replicates at forks)
            ctx.exception = self.exception
            ctx.exception_source_task = self.exception_source_task
            
            contexts.append(ctx)
        return contexts
    
    def has_exception(self) -> bool:
        """Check if context carries an exception."""
        return self.exception is not None
    
    def get_exception(self) -> Optional[Exception]:
        """Get the carried exception, if any."""
        return self.exception
    
    def get_exception_source(self) -> Optional[str]:
        """Get the task_id that raised the exception."""
        return self.exception_source_task
    
    def clear_exception(self) -> None:
        """Clear the carried exception (for error handler tasks)."""
        self.exception = None
        self.exception_source_task = None
    
    def set_exception(self, exc: Exception, source_task: str) -> None:
        """Set exception on context (called by runner when task fails)."""
        self.exception = exc
        self.exception_source_task = source_task
    
    def add_data(self, data_type: ContextDataType, item: 'CachedItem', label: str = "default") -> None:
        """
        Add cached item to context under specified label.
        
        Args:
            data_type: Type of data (IMAGE, TEXT, etc.)
            item: CachedItem to add (ImageContainer, Detection, etc.)
            label: Label to store under (e.g., "frame", "detections", "clusters")
        """
        if data_type not in self.data:
            self.data[data_type] = {}
        if label not in self.data[data_type]:
            self.data[data_type][label] = []
        self.data[data_type][label].append(item)
    
    def clear(self, 
              data_type: Optional[ContextDataType] = None, 
              label: Optional[str] = None, 
              items: int = 0) -> None:
        """
        Remove data from context.
        
        Follows Python slice semantics for items parameter:
        - Positive: remove first N items
        - Negative: remove last N items
        - Zero: remove all items
        
        Args:
            data_type: Type to clear (None = all types)
            label: Label to clear (None = all labels within data_type)
            items: Number of items to remove (0 = all, positive = first N, negative = last N)
                   Only used when label is specified.
        
        Examples:
            clear()                                    # Clear all context
            clear(data_type=IMAGE)                     # Remove all IMAGE data
            clear(label="frame")                       # Remove all "frame" labels across all types
            clear(data_type=IMAGE, label="frame")      # Remove all IMAGE["frame"] items
            clear(data_type=IMAGE, label="frame", items=-1)  # Remove last IMAGE["frame"] item
            clear(data_type=IMAGE, label="frame", items=-3)  # Remove last 3 IMAGE["frame"] items
            clear(data_type=IMAGE, label="frame", items=2)   # Remove first 2 IMAGE["frame"] items
        """
        # Case 1: clear() - remove everything
        if data_type is None and label is None:
            self.data.clear()
            return
        
        # Case 2: clear(data_type=X) - remove all data of that type
        if data_type is not None and label is None:
            if data_type in self.data:
                del self.data[data_type]
            return
        
        # Case 3: clear(label="x") - remove label from all types
        if data_type is None and label is not None:
            for dtype in list(self.data.keys()):
                if label in self.data[dtype]:
                    del self.data[dtype][label]
                    # Clean up empty type dicts
                    if not self.data[dtype]:
                        del self.data[dtype]
            return
        
        # Case 4: clear(data_type=X, label="y") - remove specific label
        if data_type not in self.data or label not in self.data[data_type]:
            return  # Nothing to clear
        
        item_list = self.data[data_type][label]
        
        # items=0: remove all
        if items == 0:
            del self.data[data_type][label]
            # Clean up empty type dict
            if not self.data[data_type]:
                del self.data[data_type]
        # items > 0: remove first N items
        elif items > 0:
            del item_list[:items]
            # Clean up if list is now empty
            if not item_list:
                del self.data[data_type][label]
                if not self.data[data_type]:
                    del self.data[data_type]
        # items < 0: remove last N items
        else:  # items < 0
            del item_list[items:]
            # Clean up if list is now empty
            if not item_list:
                del self.data[data_type][label]
                if not self.data[data_type]:
                    del self.data[data_type]
    
    @property
    def cache(self):
        """Get ItemCache instance for adding cached items."""
        if self._cache is None:
            # Lazy import to avoid circular dependency
            from ..cache.cache import ItemCache
            self._cache = ItemCache.global_instance()
        return self._cache

class BaseTask(ABC): 
    def __init__(self, task_id: str, time_budget_ms: Optional[int] = None) -> None:
        self.task_id = task_id
        # Contract format with labels: {ContextDataType: {label: (CachedItemType, format)}}
        # Example: {ContextDataType.IMAGE: {"roadview": (CachedItemType.IMAGE, "pil"), "cabview": (CachedItemType.IMAGE, "pil")}}
        # Special label "default" used when task doesn't specify labels
        # The CachedItemType determines which container class (ImageContainer, etc.)
        # The format is a string like "pil", "numpy", "torch_cpu" etc.
        from ..cache.types import CachedItemType
        self.input_contract: Dict[ContextDataType, Dict[str, tuple['CachedItemType', Optional[str]]]] = {}
        self.output_contract: Dict[ContextDataType, Dict[str, tuple['CachedItemType', Optional[str]]]] = {}
        self.readiness = False
        self.upstream_tasks: List['BaseTask'] = []
        self.downstream_tasks: List['BaseTask'] = []  # Doubly-linked execution graph
        self.collector = None  # Set by runner if metrics enabled
        self._trace_recorder = None  # Set by runner if tracing enabled
        self.exit_code = 0  # Task exit code: 0=success, non-zero=failure (unix convention)
        
        # Cooperative timing support
        self.time_budget_ms = time_budget_ms  # Milliseconds allowed for execution
        self._start_time: Optional[float] = None  # Set when task starts
        
        # Image format declarations (Phase 1: minimal GPU optimization)
        self.native_input_format: Optional['ImageFormat'] = None   # Format task expects
        self.native_output_format: Optional['ImageFormat'] = None  # Format task produces

    @abstractmethod
    def run(self, context: Context) -> Context:
        """Run the task and update the output contract."""
        return context
    
    def configure(self, **kwargs) -> None:
        """
        Apply DSL parameters to task configuration.
        
        Override in subclasses that need runtime configuration from DSL.
        Default implementation is a no-op.
        
        Args:
            **kwargs: Key-value parameters from DSL (e.g., object="person", confidence="0.8")
        
        Example:
            # DSL: filter(object=person, confidence=0.8)
            # Calls: task.configure(object="person", confidence="0.8")
        """
        pass  # Default: no-op
    
    def should_continue(self) -> bool:
        """
        Check if task should continue executing (cooperative timing).
        
        Tasks with long-running operations (loops, iterations) should
        periodically call this method and stop gracefully if it returns False.
        
        Returns:
            True if task should continue, False if time budget exceeded
        
        Example:
            for item in large_list:
                if not self.should_continue():
                    break  # Time budget exceeded, return partial results
                process(item)
        """
        if self.time_budget_ms is None:
            return True  # No time budget, always continue
        
        if self._start_time is None:
            return True  # Not started yet
        
        elapsed_ms = (time.time() - self._start_time) * 1000.0
        return elapsed_ms < self.time_budget_ms
    
    def env_set(self, key: str, value: Any) -> None:
        """
        Store value in Environment with automatic taskType and taskId.
        
        This is a convenience wrapper around Environment.set() that automatically
        populates the taskType (from class name) and taskId (from self.task_id).
        
        Args:
            key: The key for this data (e.g., "current_image", "results")
            value: The value to store
        
        Example:
            # Instead of:
            env = Environment.get_instance()
            env.set("Camera", self.task_id, "current_image", image)
            
            # Use:
            self.env_set("current_image", image)
        """
        from .environment import Environment
        env = Environment.get_instance()
        env.set(self.__class__.__name__, self.task_id, key, value)
    
    def env_get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve value from Environment with automatic taskType and taskId.
        
        This is a convenience wrapper around Environment.get() that automatically
        populates the taskType (from class name) and taskId (from self.task_id).
        
        """
        from .environment import Environment
        env = Environment.get_instance()
        return env.get(self.__class__.__name__, self.task_id, key, default)
    
    def env_has(self, key: str) -> bool:
        """
        Check if key exists in Environment with automatic taskType and taskId.
        
        Args:
            key: The key to check
        
        Returns:
            True if key exists, False otherwise
        
        Example:
            if self.env_has("cached_results"):
                results = self.env_get("cached_results")
        """
        from .environment import Environment
        env = Environment.get_instance()
        return env.has(self.__class__.__name__, self.task_id, key)
    
    def env_remove(self, key: str) -> bool:
        """
        Remove key from Environment with automatic taskType and taskId.
        
        Args:
            key: The key to remove
        
        Returns:
            True if key was removed, False if it didn't exist
        
        Example:
            self.env_remove("temporary_data")
        """
        from .environment import Environment
        env = Environment.get_instance()
        return env.remove(self.__class__.__name__, self.task_id, key)
    
    def _record_start(self) -> None:
        """
        Record task start time for cooperative timing.
        
        Called by PipelineRunner before task execution.
        Should not be called directly by task implementations.
        """
        self._start_time = time.time()
    
    def _record_trace(self, event_type: str = 'execute', submission_time: float = None, exit_code: int = 0) -> None:
        """
        Record execution trace event via runner's trace recorder.
        
        Trace format: (timestamp, thread_id, task_id, upstream_task_ids, event_type, submission_time, exit_code)
        - timestamp: float, time.time() - when execution started
        - thread_id: int, threading.get_ident()
        - task_id: str, self.task_id
        - upstream_task_ids: List[str], task IDs this task depends on
        - event_type: str, 'execute', 'split', 'merge', etc.
        - submission_time: float, time.time() - when submitted to thread pool (None if not applicable)
        - exit_code: int, 0=success, non-zero=failure (unix convention)
        
        Args:
            event_type: Type of event ('execute', 'split', 'merge')
            submission_time: When task was submitted to thread pool (for wait time calculation)
            exit_code: Task exit code (0=success, non-zero=failure)
        """
        # Only record if trace recorder is set (runner manages trace)
        if not self._trace_recorder:
            return
        
        import threading
        
        upstream_ids = [task.task_id for task in self.upstream_tasks]
        trace_event = (
            time.time(),
            threading.get_ident(),
            self.task_id,
            upstream_ids,
            event_type,
            submission_time if submission_time is not None else time.time(),
            exit_code
        )
        
        self._trace_recorder(trace_event)
    
    def validate_input_contract(self, upstream_task: 'BaseTask') -> bool:
        """
        Validate if this task can accept outputs from upstream task.
        Override for custom validation logic.
        
        With label-based contracts, validates that required labels are present
        and types match.
        
        Args:
            upstream_task: The upstream task to validate against
            
        Returns:
            True if compatible, raises ValueError if not
        """
        for input_type, required_labels in self.input_contract.items():
            # Check if upstream produces this data type
            if input_type not in upstream_task.output_contract:
                raise ValueError(
                    f"Cannot connect {upstream_task.task_id} to {self.task_id}: "
                    f"{upstream_task.task_id} doesn't produce required {input_type.type_name}"
                )
            
            produced_labels = upstream_task.output_contract[input_type]
            
            # Check each required label
            for label, (required_item_type, required_format) in required_labels.items():
                # Check if upstream produces this label
                if label not in produced_labels:
                    raise ValueError(
                        f"Cannot connect {upstream_task.task_id} to {self.task_id}: "
                        f"{upstream_task.task_id} doesn't produce required label '{label}' for {input_type.type_name}"
                    )
                
                # Check if cached item types match
                produced_item_type, produced_format = produced_labels[label]
                if produced_item_type != required_item_type:
                    raise ValueError(
                        f"Cannot connect {upstream_task.task_id} to {self.task_id}: "
                        f"Type mismatch for {input_type.type_name}[{label}] - "
                        f"upstream produces {produced_item_type.value}, downstream requires {required_item_type.value}"
                    )
                
                # Warn if formats don't match (conversion will happen at runtime)
                if required_format is not None and produced_format is not None:
                    if required_format != produced_format:
                        logger.warning(
                            f"Format mismatch connecting {upstream_task.task_id} -> {self.task_id}: "
                            f"{input_type.type_name}[{label}] - "
                            f"upstream produces '{produced_format}', downstream requires '{required_format}'. "
                            f"Runtime conversion will be performed."
                        )
        
        return True
    
    # ============================================================
    # Task Documentation Methods
    # ============================================================
    
    def describe(self) -> str:
        """
        Return a human-readable description of what this task does.
        
        Subclasses should override to provide specific descriptions.
        
        Returns:
            A one or two sentence description of the task's purpose
        """
        return f"{self.__class__.__name__} task - no description provided."
    
    def describe_contracts(self) -> Dict[str, Dict[ContextDataType, Dict[str, tuple['CachedItemType', Optional[str]]]]]:
        """
        Return input and output contracts for this task.
        
        This method extracts the contract information from the task's
        input_contract and output_contract dictionaries.
        
        Returns:
            Dict with 'inputs' and 'outputs' keys, each containing
            a Dict[ContextDataType, Dict[label, (CachedItemType, format)]] mapping
        """
        return {
            'inputs': self.input_contract.copy() if self.input_contract else {},
            'outputs': self.output_contract.copy() if self.output_contract else {}
        }
    
    def describe_parameters(self) -> Dict[str, Union[str, Dict[str, Any]]]:
        """
        Return parameter descriptions for this task.
        
        Subclasses should override to provide parameter documentation.
        
        Each parameter can be described as:
        - Simple string: "Parameter description with type info, defaults, etc."
        - Structured dict: {
            'description': "What this parameter does",
            'type': "str" | "int" | "float" | "bool" | "list" | etc.,
            'default': value,
            'choices': [option1, option2, ...],
            'required': True/False,
            'example': "example value",
            'format': "Additional format info"
          }
        
        Returns:
            Dict[str, Union[str, Dict[str, Any]]] - Parameter name to description mapping
        """
        return {}
    
    def describe_exit_codes(self) -> Dict[int, str]:
        """
        Return exit code descriptions for this task.
        
        Subclasses should override to document their exit codes.
        
        Exit code convention (unix-style):
        - 0: Success
        - 1: Generic failure (empty data, no results, etc.)
        - 2: Exception/error during execution (set automatically by runner)
        - 3+: Task-specific error codes
        
        Returns:
            Dict[int, str] - Exit code to description mapping
        
        Example:
            ```python
            def describe_exit_codes(self) -> Dict[int, str]:
                return {
                    0: "Success: valid input received",
                    1: "Failure: empty input",
                    3: "Failure: invalid format"
                }
            ```
        """
        return {
            0: "Success",
            2: "Exception during execution"
        }


class Connector(BaseTask):
    def __init__(self, task_id: str, time_budget_ms: Optional[int] = None) -> None:
        super().__init__(task_id, time_budget_ms)
        self.input_tasks: List[BaseTask] = []
        self.output_tasks: List[BaseTask] = []
        self.internal_tasks: List[BaseTask] = []
        self.edges: List[tuple[BaseTask, BaseTask]] = []
        self.immutable_cache: Dict[ContextDataType, List[Any]] = {}
        self.split_contexts: List[Context] = []
        self.collector = None  # Set by runner if metrics enabled
    
    def configure(self, **kwargs) -> None:
        """
        Configure connector behavior from DSL parameters.
        
        Override in subclasses to implement custom split/merge behavior.
        Default connector has no configurable parameters.
        
        Args:
            **kwargs: Key-value parameters from DSL
        
        Example:
            # Subclass implementation:
            class OrderedMergeConnector(Connector):
                def configure(self, **kwargs):
                    if "order" in kwargs:
                        self.order = parse_order(kwargs["order"])
        """
        pass  # Default: no-op
    
    def add_input_task(self, task: BaseTask) -> None:
        self.input_tasks.append(task)
    
    def add_output_task(self, task: BaseTask) -> None:
        self.output_tasks.append(task)
    
    def add_task(self, task: BaseTask) -> None:
        self.internal_tasks.append(task)
    
    def add_edge(self, from_task: BaseTask, to_task: BaseTask) -> None:
        if not isinstance(to_task, Connector) and to_task.upstream_tasks:
            raise ValueError(
                f"Task {to_task.task_id} already has upstream task {to_task.upstream_tasks[0].task_id}. "
                f"Only Connectors can have multiple upstream tasks."
            )
        
        to_task.validate_input_contract(from_task)
        
        self.edges.append((from_task, to_task))
        to_task.upstream_tasks.append(from_task)
    
    def build_graph(self) -> List[BaseTask]:
        """Build flattened execution graph with dependencies."""
        graph = []
        for task in self.internal_tasks:
            if isinstance(task, Connector):
                if task.internal_tasks:
                    graph.extend(task.build_graph())
                else:
                    graph.append(task)
            else:
                graph.append(task)
        return graph
    
    def split_strategy(self, context: Context, num_branches: int) -> List[Context]:
        """
        Default split: deep copy mutable data, share immutable refs.
        Override for custom split logic.
        """
        return context.split(num_branches, self.immutable_cache)
    
    def merge_strategy(self, contexts: List[Context]) -> Context:
        """
        Default merge: concatenate all lists by label with ID-based deduplication.
        
        With label-based data structure, merges each label separately to maintain
        semantic separation (e.g., "roadview" and "cabview" remain distinct).
        
        Deduplication uses item.id attribute (or object identity as fallback)
        to prevent the same object from appearing multiple times when branches
        share references.
        
        Override for custom merge logic (e.g., ordered_merge preserves structure).
        """
        logger.debug(f"Connector {self.task_id}: merging {len(contexts)} contexts")
        merged = Context()
        
        # Preserve context attributes from first context
        if contexts:
            merged.pipeline_runner = contexts[0].pipeline_runner
            merged.collector = contexts[0].collector
            merged.config = contexts[0].config
            merged._cache = contexts[0]._cache
        
        # Structure: data_by_type[data_type][label] = [items]
        data_by_type: Dict[ContextDataType, Dict[str, List[Any]]] = {}
        
        # Concatenate all items from all contexts, preserving labels
        for ctx in contexts:
            for data_type, label_dict in ctx.data.items():
                if data_type not in data_by_type:
                    data_by_type[data_type] = {}
                
                for label, items in label_dict.items():
                    if label not in data_by_type[data_type]:
                        data_by_type[data_type][label] = []
                    data_by_type[data_type][label].extend(items)
        
        # Deduplicate by ID within each label to prevent duplicate references
        for data_type, label_dict in data_by_type.items():
            for label, items in label_dict.items():
                if items:
                    seen_ids = set()
                    deduped = []
                    for item in items:
                        # Try to get stable ID first, fall back to object identity
                        item_id = getattr(item, 'id', None) or id(item)
                        if item_id not in seen_ids:
                            seen_ids.add(item_id)
                            deduped.append(item)
                    data_by_type[data_type][label] = deduped
                    
                    if len(deduped) < len(items):
                        logger.debug(f"Connector {self.task_id}: deduplicated {data_type.type_name}[{label}] "
                                   f"from {len(items)} to {len(deduped)} items")
        
        merged.data = data_by_type
        
        # Record merge trace event
        self._record_trace('merge')
        
        logger.debug(f"Connector {self.task_id}: merge complete, {len(data_by_type)} data types")
        return merged
    
    def run_connector(self, contexts: List[Context]) -> Context:
        """
        Execute connector: merge upstream contexts, split for downstream branches.
        
        Args:
            contexts: List of contexts from all upstream tasks
            
        Returns:
            Merged context (also stores split contexts for downstreams)
        """
        logger.info(f"Connector {self.task_id}: starting run_connector with {len(contexts)} input contexts")
        
        # Merge with metrics
        if self.collector:
            with self.collector.duration_timer("connector.merge.duration", 
                                              {"connector_id": self.task_id, "num_inputs": str(len(contexts))}):
                merged = self.merge_strategy(contexts)
        else:
            merged = self.merge_strategy(contexts)
        
        # Track merged context size
        if self.collector:
            for data_type, data in merged.data.items():
                self.collector.data_point("connector.context.size", 
                                         {"connector_id": self.task_id, "data_type": data_type.type_name}, 
                                         len(data) if data else 0)
        
        # Split with metrics
        if len(self.output_tasks) > 1:
            logger.info(f"Connector {self.task_id}: splitting to {len(self.output_tasks)} branches")
            if self.collector:
                with self.collector.duration_timer("connector.split.duration", 
                                                  {"connector_id": self.task_id, "num_outputs": str(len(self.output_tasks))}):
                    self.split_contexts = self.split_strategy(merged, len(self.output_tasks))
            else:
                self.split_contexts = self.split_strategy(merged, len(self.output_tasks))
            
            # Record split trace events (one per output)
            for _ in self.split_contexts:
                self._record_trace('split')
            
            logger.debug(f"Connector {self.task_id}: split complete, created {len(self.split_contexts)} contexts")
        else:
            self.split_contexts = []
        
        logger.info(f"Connector {self.task_id}: run_connector complete")
        return merged
    
    def run(self, context: Context) -> Context:
        """
        BaseTask interface - should not be called directly for connectors.
        Use run_connector() instead.
        """
        return context


class LoopCondition(BaseTask):
    """
    Base class for loop control conditions.
    
    Loop conditions are tasks marked with ':' in the DSL that control
    loop execution flow by returning LoopControlAction values.
    
    Subclasses must implement evaluate() to return:
    - LoopControlAction.PASS: Continue to next task
    - LoopControlAction.CONTINUE: Jump to loop start
    - LoopControlAction.BREAK: Exit loop
    
    The run() method handles setting the control action in the loop stack.
    
    Example:
        ```python
        class TimeoutCondition(LoopCondition):
            def __init__(self, timeout_seconds: float):
                super().__init__()
                self.timeout = timeout_seconds
            
            def evaluate(self, context: Context) -> LoopControlAction:
                stack = context.data.get(ContextDataType.LOOP_STACK, [])
                if not stack:
                    return LoopControlAction.PASS
                
                elapsed = time.time() - stack[-1]['start_time']
                if elapsed >= self.timeout:
                    return LoopControlAction.BREAK
                return LoopControlAction.PASS
        ```
        
        DSL:
        ```
        {camera -> process ->:timeout(60)}
        ```
    """
    
    @abstractmethod
    def evaluate(self, context: Context) -> LoopControlAction:
        """
        Evaluate condition and return control action.
        
        Args:
            context: Pipeline context with LOOP_STACK
        
        Returns:
            LoopControlAction: PASS, CONTINUE, or BREAK
        """
        pass
    
    def run(self, context: Context) -> Context:
        """
        Evaluate condition and set control action in loop stack.
        
        Args:
            context: Pipeline context
        
        Returns:
            Context with updated loop stack control action
        """
        # Evaluate condition
        action = self.evaluate(context)
        
        # Set control action in loop stack
        stack = context.data.get(ContextDataType.LOOP_STACK, [])
        if stack:
            stack[-1]['control'] = action
            if action != LoopControlAction.PASS:
                logger.info(f"[{self.task_id}] Control action: {action.value}")
        else:
            # Control task outside loop - log warning but don't error
            # (allows testing conditions in isolation)
            logger.warning(f"[{self.task_id}] Control task evaluated outside loop context")
        
        return context

