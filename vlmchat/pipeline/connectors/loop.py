"""
Loop connector for repeating pipeline execution.

Implements loop control with stack-based state management for nested loops.
Control flow is managed by LoopCondition tasks embedded in the body.
"""

import time
import logging
from typing import List
from .base import Connector
from ..core.task_base import BaseTask, Context, ContextDataType, LoopControlAction

logger = logging.getLogger(__name__)


class LoopConnector(Connector):
    """
    Executes body tasks repeatedly until a LoopCondition breaks the loop.
    
    Uses stack-based state management to support nested loops. Each loop
    maintains its own state (iteration count, start time, control action) on
    the LOOP_STACK in context.
    
    Loop execution:
    1. Push new loop state onto stack
    2. Execute body tasks sequentially
    3. Check if any task returned BREAK or CONTINUE
    4. If BREAK: exit loop
    5. If CONTINUE: restart from step 2
    6. If all tasks PASS: continue to next iteration
    7. Pop loop state from stack
    
    Control flow is managed by LoopCondition tasks embedded in the body.
    There is no separate "exit task" - control tasks can appear anywhere.
    
    Attributes:
        body_tasks: List of tasks to execute in loop body (includes control tasks)
    
    Example:
        ```python
        # Create loop with embedded control conditions
        body = [camera, detector, processor, TimeoutCondition(60.0)]
        loop = LoopConnector(body)
        
        # Execute loop
        result_context = loop.run(input_context)
        ```
        
        DSL:
        ```
        {camera -> detector -> processor ->:timeout(60)}
        ```
    
    Nested loops:
        ```python
        # Outer loop with inner loop in body
        inner_loop = LoopConnector([process, CountCondition(10)])
        outer_body = [setup, inner_loop, cleanup, TimeoutCondition(60.0)]
        outer_loop = LoopConnector(outer_body)
        ```
        
        DSL:
        ```
        {
            setup -> 
            {process ->:count(10)} ->
            cleanup ->
            :timeout(60)
        }
        ```
    """
    
    def __init__(self, body_tasks: List[BaseTask], task_id: str = "loop"):
        """
        Initialize loop connector.
        
        Args:
            body_tasks: Tasks to execute in loop body (sequential), 
                       including any LoopCondition tasks for control
            task_id: Unique identifier for this loop (default: "loop")
        """
        super().__init__(task_id)
        self.body_tasks = body_tasks
        self.internal_tasks = body_tasks  # For nested connectors to work
        logger.info(f"LoopConnector initialized with {len(body_tasks)} body tasks")
    
    def build_graph(self) -> List[BaseTask]:
        """
        Return self as a single task (don't flatten body).
        
        The loop manages its body tasks internally and shouldn't be 
        expanded into the top-level graph. This allows the loop to
        control iteration and re-execution of its body.
        """
        return [self]
    
    def run(self, context: Context) -> Context:
        """
        Execute loop until a LoopCondition breaks or continues.
        
        Args:
            context: Pipeline context (may have LOOP_STACK from parent loops)
        
        Returns:
            Context after loop completion with this loop's state removed
        """
        # Initialize or get loop stack
        if ContextDataType.LOOP_STACK not in context.data:
            context.data[ContextDataType.LOOP_STACK] = []
        
        stack = context.data[ContextDataType.LOOP_STACK]
        
        # Push new loop state onto stack
        loop_state = {
            'iteration': 0,
            'start_time': time.time(),
            'control': LoopControlAction.PASS,
            'last_exit_code': 0  # Track last executed task's exit code
        }
        stack.append(loop_state)
        
        depth = len(stack)
        logger.info(f"[Loop depth={depth}] Starting loop")
        
        # Execute loop
        while True:
            loop_state['iteration'] += 1
            loop_state['control'] = LoopControlAction.PASS
            iteration = loop_state['iteration']
            
            logger.debug(f"[Loop depth={depth}] Iteration {iteration} starting")
            
            # Execute body tasks sequentially
            for i, task in enumerate(self.body_tasks):
                try:
                    # Record trace event for task execution
                    if hasattr(task, '_record_trace') and task._record_trace:
                        task._record_trace('execute')
                    
                    logger.debug(f"[Loop depth={depth}] Running task {i} ({task.task_id}) in iteration {iteration}")
                    context = task.run(context)
                    logger.debug(f"[Loop depth={depth}] Task {i} ({task.task_id}) completed in iteration {iteration}")
                    
                    # Store task's exit code in loop state for conditions to access
                    loop_state['last_exit_code'] = task.exit_code
                    
                except Exception as e:
                    import traceback
                    logger.error(f"[Loop depth={depth}] Task {i} ({task.task_id}) failed in iteration {iteration}: {e}")
                    logger.debug(f"[Loop depth={depth}] Traceback:\n{traceback.format_exc()}")
                    # Break loop on error
                    loop_state['control'] = LoopControlAction.BREAK
                    break
                
                # Check control action (set by LoopCondition tasks)
                control = loop_state.get('control', LoopControlAction.PASS)
                
                if control == LoopControlAction.BREAK:
                    logger.info(f"[Loop depth={depth}] BREAK signal from task {i} ({task.task_id}) in iteration {iteration}")
                    break
                elif control == LoopControlAction.CONTINUE:
                    logger.info(f"[Loop depth={depth}] CONTINUE signal from task {i} ({task.task_id}) in iteration {iteration}")
                    break
                # PASS: continue to next task
            
            # Handle control action
            control = loop_state.get('control', LoopControlAction.PASS)
            
            if control == LoopControlAction.BREAK:
                logger.info(f"[Loop depth={depth}] Exiting loop after {iteration} iterations")
                break  # Exit while loop
            elif control == LoopControlAction.CONTINUE:
                logger.debug(f"[Loop depth={depth}] Continuing to iteration {iteration + 1}")
                continue  # Restart while loop (next iteration)
            # PASS: continue to next iteration naturally
            
            logger.debug(f"[Loop depth={depth}] Iteration {iteration} complete, continuing")
        
        # Pop loop state from stack
        stack.pop()
        
        # Clean up stack if empty
        if len(stack) == 0:
            context.data.pop(ContextDataType.LOOP_STACK, None)
            logger.debug("[Loop] Stack empty, removed from context")
        
        elapsed = time.time() - loop_state['start_time']
        logger.info(f"[Loop depth={depth}] Completed {loop_state['iteration']} iterations in {elapsed:.1f}s")
        
        return context
    
    def configure(self, **kwargs) -> None:
        """
        Configure loop connector from parameters.
        
        Args:
            **kwargs: Configuration parameters (currently unused, body tasks set in __init__)
        """
        # Body tasks are set via constructor
        # This method exists for interface compatibility
        pass

