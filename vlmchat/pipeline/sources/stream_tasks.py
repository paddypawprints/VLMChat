"""
Stream consumption tasks for event-driven pipelines.

These tasks interact with StreamSources to consume data.
Actual logic is implemented in PipelineRunner for visibility into pipeline state.
"""

import logging
from typing import Optional

from .task_base import BaseTask, Context

logger = logging.getLogger(__name__)


class WaitTask(BaseTask):
    """
    Wait for new data from a stream source and control cursor spawning.
    
    Must be the first task in a pipeline. Only one wait() per pipeline allowed.
    
    Behavior:
    - Blocks cursor until source has new data
    - pipeline=True: spawns new cursor immediately when data arrives (pipelining)
    - pipeline=False: waits until pipeline completes before spawning next cursor
    
    Actual logic implemented in PipelineRunner._handle_wait_task() for
    visibility into cursor state and spawning.
    
    Examples:
        wait(camera)  # Default pipelining
        wait(camera, pipeline=False)  # Serialize processing
        wait(mqtt, topic="commands", qos=1)  # Pass params to source
    """
    
    def __init__(self, source_name: str = "", pipeline: bool = True, **kwargs):
        """
        Initialize wait task.
        
        Args:
            source_name: Name of the stream source to wait for
            pipeline: If True, spawn new cursor immediately (default). 
                     If False, wait until pipeline completes.
            **kwargs: Additional parameters (stored for source-specific use)
        """
        # Extract task_id if provided, otherwise generate
        task_id = kwargs.pop('task_id', f"wait_{source_name}" if source_name else "wait")
        super().__init__(task_id=task_id, **kwargs)
        
        self.source_name = source_name
        self.pipeline = pipeline
        self.source_params = kwargs  # Store extra params for source
        
        logger.debug(f"WaitTask created: source={source_name}, pipeline={pipeline}, params={kwargs}")
    
    def run(self, context: Context) -> Context:
        """
        Run method - actual logic in PipelineRunner._handle_wait_task().
        
        This is a marker method. The runner intercepts WaitTask and handles
        cursor blocking/spawning before calling this.
        """
        # By the time we get here, runner has:
        # 1. Checked for new data
        # 2. Blocked cursor if no data (we won't reach here)
        # 3. Spawned new cursor if pipeline=True
        # 4. Set respawn flag if pipeline=False
        
        # Just pass context through
        return context
    
    def configure(self, **kwargs) -> None:
        """
        Configure task parameters.
        
        Overrides base implementation to handle source_name and pipeline parameters.
        """
        if 'source_name' in kwargs:
            self.source_name = kwargs.pop('source_name')
        
        if 'pipeline' in kwargs:
            self.pipeline = kwargs.pop('pipeline')
        
        # Store remaining as source params
        self.source_params.update(kwargs)
        
        # Call base configure (though it might not do anything useful here)
        if kwargs:
            super().configure(**kwargs)


class LatestTask(BaseTask):
    """
    Get the latest data from a stream source, skipping old items.
    
    Never blocks - if no data available, returns empty context.
    Can be used after wait() or standalone.
    
    Actual logic implemented in PipelineRunner._handle_latest_task() for
    access to source registry.
    
    Examples:
        latest(camera)  # Get latest frame
        latest(detections, confidence=0.5)  # Pass params to source
    """
    
    def __init__(self, source_name: str = "", **kwargs):
        """
        Initialize latest task.
        
        Args:
            source_name: Name of the stream source to read from
            **kwargs: Additional parameters (stored for source-specific use)
        """
        # Extract task_id if provided, otherwise generate
        task_id = kwargs.pop('task_id', f"latest_{source_name}" if source_name else "latest")
        super().__init__(task_id=task_id, **kwargs)
        
        self.source_name = source_name
        self.source_params = kwargs  # Store extra params for source
        
        logger.debug(f"LatestTask created: source={source_name}, params={kwargs}")
    
    def run(self, context: Context) -> Context:
        """
        Run method - actual logic in PipelineRunner._handle_latest_task().
        
        This is a marker method. The runner intercepts LatestTask and fetches
        data from the source before calling this.
        """
        # By the time we get here, runner has:
        # 1. Fetched latest data from source
        # 2. Added data to context
        
        # Just pass context through
        return context
    
    def configure(self, **kwargs) -> None:
        """Configure task parameters."""
        if 'source_name' in kwargs:
            self.source_name = kwargs.pop('source_name')
        
        # Store remaining as source params
        self.source_params.update(kwargs)
        
        # Call base configure (though it might not do anything useful here)
        if kwargs:
            super().configure(**kwargs)


class NextTask(BaseTask):
    """
    Get the next sequential data item from a stream source.
    
    Maintains cursor position to ensure no frames are skipped.
    Blocks if next item not yet available.
    
    Actual logic implemented in PipelineRunner._handle_next_task() for
    cursor position tracking.
    
    Examples:
        next(camera)  # Get next frame in sequence
        next(mqtt, timeout=5000)  # Pass params to source
    """
    
    def __init__(self, source_name: str = "", **kwargs):
        """
        Initialize next task.
        
        Args:
            source_name: Name of the stream source to read from
            **kwargs: Additional parameters (stored for source-specific use)
        """
        # Extract task_id if provided, otherwise generate
        task_id = kwargs.pop('task_id', f"next_{source_name}" if source_name else "next")
        super().__init__(task_id=task_id, **kwargs)
        
        self.source_name = source_name
        self.source_params = kwargs  # Store extra params for source
        
        logger.debug(f"NextTask created: source={source_name}, params={kwargs}")
    
    def run(self, context: Context) -> Context:
        """
        Run method - actual logic in PipelineRunner._handle_next_task().
        
        This is a marker method. The runner intercepts NextTask and fetches
        next sequential data from the source before calling this.
        """
        # By the time we get here, runner has:
        # 1. Fetched next data from source (or blocked if not ready)
        # 2. Added data to context
        
        # Just pass context through
        return context
    
    def configure(self, **kwargs) -> None:
        """Configure task parameters."""
        if 'source_name' in kwargs:
            self.source_name = kwargs.pop('source_name')
        
        # Store remaining as source params
        self.source_params.update(kwargs)
        
        # Call base configure (though it might not do anything useful here)
        if kwargs:
            super().configure(**kwargs)
