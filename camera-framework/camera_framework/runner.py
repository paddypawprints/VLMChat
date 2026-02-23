"""Pipeline runner with sequential task execution."""

import time
import threading
from typing import List
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from .task import BaseTask


class Runner:
    """Pipeline runner with sequential task execution.
    
    Tasks execute in the order they were added.
    One-off tasks can be queued (e.g., from MQTT handlers).
    All tasks run in thread pool for parallelism.
    """
    
    def __init__(self, tasks: List[BaseTask] = None, max_workers: int = 4, collector=None):
        self.tasks: List[BaseTask] = tasks or []
        self.task_queue: deque = deque()  # One-off tasks from MQTT, etc.
        self._queue_lock = threading.Lock()  # Protect task_queue
        self.collector = collector  # Optional metrics collector
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="task")
    
    def add_task(self, task: BaseTask) -> 'Runner':
        """Add task to pipeline. Returns self for fluent chaining."""
        self.tasks.append(task)
        return self
    
    def queue_task(self, task: BaseTask) -> None:
        """Queue a one-off task (e.g., from MQTT handler). Thread-safe."""
        with self._queue_lock:
            self.task_queue.append(task)
    
    def run_once(self) -> None:
        """Execute ready tasks once and return.
        
        Submits ready tasks to thread pool and waits for them to complete.
        No callbacks - explicit scheduling by caller in main loop.
        """
        now = time.time()
        futures = []
        
        # Collect one-off queued tasks (always ready)
        with self._queue_lock:
            while self.task_queue:
                task = self.task_queue.popleft()
                futures.append(self._submit_task(task, now))
        
        # Collect ready pipeline tasks
        for task in self.tasks:
            # Check interval timing
            if task.interval is not None and (now - task.last_run) < task.interval:
                continue
            
            # Skip if already running
            if task._background_busy:
                continue
            
            # Check readiness (has input, can output)
            if not task.is_ready():
                continue
            
            # Mark as busy BEFORE submitting to prevent double-submission
            task._background_busy = True
            futures.append(self._submit_task(task, now))
        
        # Wait for all tasks to complete
        # DO NOT swallow exceptions - they should propagate to caller
        for future in futures:
            future.result()  # Will raise if task failed
    
    def _submit_task(self, task: BaseTask, submit_time: float):
        """Submit a single task to thread pool."""
        def run_task():
            try:
                task._background_busy = True
                task.last_run = submit_time
                task.process()
            except Exception as e:
                import logging
                logging.error(f"Task {task.name} failed: {e}", exc_info=True)
                raise
            finally:
                task._background_busy = False
        
        return self.executor.submit(run_task)
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool."""
        self.executor.shutdown(wait=wait)
    
    def stats(self) -> dict:
        """Get runner statistics."""
        return {
            "tasks": len(self.tasks),
        }
