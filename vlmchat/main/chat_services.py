# main/chat_application.py
"""
Main chat application orchestrating all components.

This module contains the SmolVLMChatApplication class which serves as the main
coordinator for the chat application. It integrates the SmolVLM model, prompt
handling, image processing, camera capture, and user interface components.
"""

import logging
import threading
from typing import Optional, Dict, Any
import os

from vlmchat.metrics.metrics_collector import Collector
from vlmchat.utils.config import VLMChatConfig
from .service_response import ServiceResponse
from .service_response import ServiceResponse as SR

logger = logging.getLogger(__name__)
# ServiceResponse codes are defined in `src/main/service_response.py`.
# For quick reference:
#   SR.Code.OK (0)                - Success
#   SR.Code.EXIT (1)              - Exit interactive loop
#   SR.Code.IMAGE_LOAD_FAILED (2) - Image URL/file load failed
#   SR.Code.INVALID_FORMAT (3)    - Invalid /format argument
#   SR.Code.CAMERA_FAILED (4)     - Camera capture failed
#   SR.Code.NO_METRICS_SESSION (5)- No metrics session available
#   SR.Code.BACKEND_FAILED (6)    - Backend query or switch failed
#   SR.Code.UNKNOWN_COMMAND (7)   - Unrecognized command

class VLMChatServices:
    """Pipeline execution service for loading and running DSL pipelines."""
    
    def __init__(self, config: VLMChatConfig, collector: Collector):
        """
        Initialize the pipeline execution service.
        
        Args:
            config: VLMChatConfig object with application settings
            collector: Metrics Collector for telemetry
        """
        # Configure logging from global configuration
        logging.basicConfig(
            level=getattr(logging, config.logging.level),
            format=config.logging.format
        )
        
        # Store configuration and collector
        self._config = config
        self._collector = collector
        
        # Initialize Environment singleton
        from vlmchat.pipeline.environment import Environment
        self._environment = Environment.get_instance()
        
        # Pipeline execution state
        self._current_pipeline = None
        self._pipeline_runner = None
        self._pipeline_thread: Optional[threading.Thread] = None
        self._pipeline_stop_flag = threading.Event()
        
        logger.info("Pipeline execution service initialized successfully")

    # --- Service methods (business logic) ---------------------------------

    def _service_clear_environment(self) -> ServiceResponse:
        """Clear all environment data."""
        self._environment.clear()
        return ServiceResponse(ServiceResponse.Code.OK, "Environment cleared.")

    def _service_show_environment(self) -> ServiceResponse:
        """Show all environment keys and their types."""
        keys = self._environment.keys()
        if not keys:
            return ServiceResponse(ServiceResponse.Code.OK, "Environment is empty.")
        
        lines = ["Environment Keys:"]
        for key in sorted(keys):
            # Parse the key to get value
            parts = key.split("+")
            if len(parts) == 3:
                value = self._environment.get(parts[0], parts[1], parts[2])
                value_type = type(value).__name__
                lines.append(f"  {key}: {value_type}")
            else:
                lines.append(f"  {key}: (invalid key format)")
        return ServiceResponse(ServiceResponse.Code.OK, "\n".join(lines))

    def _service_pipeline(self, dsl_or_file: str) -> ServiceResponse:
        """Load a pipeline from DSL or file."""
        from vlmchat.pipeline.dsl_parser import DSLParser, create_task_registry
        
        # Determine if it's a file or inline DSL
        is_file = dsl_or_file.endswith('.dsl') or os.path.isfile(dsl_or_file)
        
        try:
            # Get pipeline directories from config
            pipeline_dirs = [os.path.expanduser(d) for d in self._config.paths.pipeline_dirs]
            
            # Create parser with pipeline directories
            registry = create_task_registry()
            parser = DSLParser(registry, pipeline_dirs=pipeline_dirs)
            
            if is_file:
                # Load DSL from file
                file_path = dsl_or_file
                if not os.path.isabs(file_path):
                    # Try to find in pipeline directories
                    for pdir in pipeline_dirs:
                        candidate = os.path.join(pdir, file_path)
                        if os.path.isfile(candidate):
                            file_path = candidate
                            break
                
                if not os.path.isfile(file_path):
                    return ServiceResponse(ServiceResponse.Code.IMAGE_LOAD_FAILED, f"Pipeline file not found: {dsl_or_file}")
                
                with open(file_path, 'r') as f:
                    dsl_text = f.read()
                logger.info(f"Loaded pipeline from file: {file_path}")
            else:
                # Use as inline DSL
                dsl_text = dsl_or_file
                logger.info("Loaded inline pipeline DSL")
            
            # Parse the DSL
            self._current_pipeline = parser.parse(dsl_text)
            return ServiceResponse(ServiceResponse.Code.OK, f"Pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            return ServiceResponse(ServiceResponse.Code.IMAGE_LOAD_FAILED, f"Failed to load pipeline: {e}")
    
    def _service_run(self, overrides_str: str) -> ServiceResponse:
        """Run the currently loaded pipeline with optional overrides."""
        if self._current_pipeline is None:
            return ServiceResponse(ServiceResponse.Code.IMAGE_LOAD_FAILED, "No pipeline loaded. Use /pipeline first")
        
        if self._pipeline_thread is not None and self._pipeline_thread.is_alive():
            return ServiceResponse(ServiceResponse.Code.IMAGE_LOAD_FAILED, "Pipeline already running. Use /stop first")
        
        # Parse overrides (simple key=value pairs)
        overrides = {}
        if overrides_str.strip():
            for pair in overrides_str.split():
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    # Simple parsing - treat as string unless it looks like a number
                    try:
                        if '.' in value:
                            overrides[key.strip()] = float(value)
                        else:
                            overrides[key.strip()] = int(value)
                    except ValueError:
                        overrides[key.strip()] = value.strip()
        
        # Start pipeline in background thread
        self._pipeline_stop_flag.clear()
        
        def run_pipeline():
            try:
                from vlmchat.pipeline.pipeline_runner import PipelineRunner
                from vlmchat.pipeline.task_base import Context
                
                logger.info(f"Starting pipeline execution with overrides: {overrides}")
                
                # Create fresh context with config injected
                context = Context()
                context.config = self._config
                context.collector = self._collector
                
                # Create runner and execute
                runner = PipelineRunner(
                    self._current_pipeline,
                    collector=self._collector,
                    enable_trace=True,  # Enable execution tracing
                    overrides=overrides
                )
                
                # Store runner so console can access queues
                self._pipeline_runner = runner
                
                result = runner.run(context)
                logger.info("Pipeline execution completed")
                
            except Exception as e:
                logger.error(f"Pipeline execution failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Clear runner reference when done
                self._pipeline_runner = None
        
        self._pipeline_thread = threading.Thread(target=run_pipeline, daemon=True)
        self._pipeline_thread.start()
        
        return ServiceResponse(ServiceResponse.Code.OK, "Pipeline started")
    
    def _service_stop(self) -> ServiceResponse:
        """Stop the currently running pipeline."""
        if self._pipeline_thread is None or not self._pipeline_thread.is_alive():
            return ServiceResponse(ServiceResponse.Code.OK, "No pipeline running")
        
        # Request graceful stop via runner if available
        if self._pipeline_runner:
            self._pipeline_runner.request_stop()
        
        self._pipeline_stop_flag.set()
        logger.info("Pipeline stop requested")
        return ServiceResponse(ServiceResponse.Code.OK, "Pipeline stop requested")
    
    def _service_status(self) -> ServiceResponse:
        """Get pipeline execution status."""
        if self._pipeline_thread is None or not self._pipeline_thread.is_alive():
            status = "No pipeline running"
        else:
            status = "Pipeline is running"
        
        if self._current_pipeline is not None:
            status += "\nPipeline loaded: Yes"
        else:
            status += "\nPipeline loaded: No"
        
        return ServiceResponse(ServiceResponse.Code.OK, status)
    
    def _service_trace(self) -> ServiceResponse:
        """
        Display execution trace of last pipeline run.
        
        Returns:
            ServiceResponse with formatted trace output
        """
        from vlmchat.pipeline.trace import print_trace_events
        from io import StringIO
        import sys
        
        runner = getattr(self, '_pipeline_runner', None)
        if runner is None:
            return ServiceResponse(ServiceResponse.Code.ERROR, "No pipeline has been run yet")
        
        if not runner.trace_events:
            return ServiceResponse(ServiceResponse.Code.OK, "No trace events recorded (tracing may be disabled)")
        
        # Capture print_trace_events output
        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()
        try:
            print_trace_events(runner.trace_events)
            output = buffer.getvalue()
        finally:
            sys.stdout = old_stdout
        
        return ServiceResponse(ServiceResponse.Code.OK, output)
    
    def _service_describe(self, task_name: str) -> ServiceResponse:
        """
        Get help documentation for a task.
        
        Args:
            task_name: Name of the task to describe (as registered in task registry)
            
        Returns:
            ServiceResponse with formatted task documentation
        """
        from vlmchat.pipeline.dsl_parser import create_task_registry
        from vlmchat.pipeline.task_help_formatter import TaskHelpFormatter
        
        # Get task registry
        registry = create_task_registry()
        
        if task_name not in registry:
            available = ", ".join(sorted(registry.keys()))
            return ServiceResponse(
                ServiceResponse.Code.UNKNOWN_COMMAND,
                f"Unknown task '{task_name}'. Available tasks: {available}"
            )
        
        # Create a temporary instance of the task
        task_class = registry[task_name]
        try:
            task = task_class(task_id=f"{task_name}_temp")
        except TypeError:
            # Some tasks may require constructor arguments
            task = task_class()
            task.task_id = f"{task_name}_temp"
        
        # Format help
        formatter = TaskHelpFormatter()
        help_text = formatter.format_console(task)
        
        return ServiceResponse(ServiceResponse.Code.OK, help_text)

