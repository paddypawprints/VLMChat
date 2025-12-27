"""
Error handler tasks for pipeline exception management.

These tasks can inspect, log, clear, or re-raise exceptions in the pipeline.
They set handles_exceptions=True to opt-in to execution when exceptions are present.
"""

import logging
from typing import Dict, Any, Optional, Set
from ..task_base import BaseTask, Context, register_task

logger = logging.getLogger(__name__)


@register_task('log_error')
class LogErrorTask(BaseTask):
    """
    Logs exception to logger but doesn't clear it.
    
    Exception continues propagating after logging.
    Always executes (logs even if no exception).
    
    Parameters:
    - level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - message: Optional custom message prefix
    
    Example DSL:
        task_a -> task_b -> log_error(level="ERROR", message="Pipeline failed") -> task_c
    """
    
    def __init__(self, task_id: str = "log_error"):
        super().__init__(task_id)
        self.handles_exceptions = True  # Opt-in to see exceptions
        self.level = "ERROR"
        self.message = "Exception in pipeline"
    
    def configure(self, **params) -> None:
        """Configure logging parameters."""
        if "level" in params:
            self.level = params["level"].upper()
        if "message" in params:
            self.message = params["message"]
    
    def run(self, context: Context) -> Context:
        """Log exception if present, don't clear it."""
        exception = context.get_exception()
        if exception:
            log_func = getattr(logger, self.level.lower(), logger.error)
            source = context.get_exception_source()
            log_func(f"{self.message}: {type(exception).__name__} from task '{source}': {exception}")
        
        return context
    
    def describe(self) -> str:
        return "Logs exception to logger but allows it to continue propagating."
    
    def describe_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "level": {
                "description": "Logging level",
                "type": "str",
                "default": "ERROR",
                "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            },
            "message": {
                "description": "Custom log message prefix",
                "type": "str",
                "default": "Exception in pipeline"
            }
        }


@register_task('clear_error')
class ClearErrorTask(BaseTask):
    """
    Clears exception silently, allowing pipeline to continue.
    
    Use this to recover from expected errors.
    No logging - silent recovery.
    
    Example DSL:
        task_a -> task_b -> clear_error() -> task_c
    """
    
    def __init__(self, task_id: str = "clear_error"):
        super().__init__(task_id)
        self.handles_exceptions = True  # Opt-in to see exceptions
    
    def run(self, context: Context) -> Context:
        """Clear exception silently."""
        if context.has_exception():
            context.clear_exception()
        
        return context
    
    def describe(self) -> str:
        return "Clears exception silently, allowing pipeline to continue normally."


@register_task('on_error')
class OnErrorTask(BaseTask):
    """
    Handles specific exception types, optionally clearing them.
    
    Parameters:
    - type: Exception type name(s) to handle (comma-separated)
    - clear: Whether to clear exception after handling (default: True)
    - log: Whether to log the exception (default: True)
    - log_level: Logging level if log=True (default: ERROR)
    
    Only handles exceptions matching the specified type(s).
    Other exceptions pass through untouched.
    
    Example DSL:
        task_a -> on_error(type="ValueError,TypeError", clear=true) -> task_b
        task_c -> on_error(type="TimeoutError", clear=false, log=true) -> task_d
    """
    
    def __init__(self, task_id: str = "on_error"):
        super().__init__(task_id)
        self.handles_exceptions = True  # Opt-in to see exceptions
        self.exception_types: Optional[Set[str]] = None  # None = handle all
        self.should_clear = True
        self.should_log = True
        self.log_level = "ERROR"
    
    def configure(self, **params) -> None:
        """Configure exception handling."""
        if "type" in params:
            type_param = params["type"]
            if isinstance(type_param, str):
                # Parse comma-separated list
                self.exception_types = set(t.strip() for t in type_param.split(","))
            else:
                self.exception_types = {str(type_param)}
        
        if "clear" in params:
            self.should_clear = self._parse_bool(params["clear"])
        
        if "log" in params:
            self.should_log = self._parse_bool(params["log"])
        
        if "log_level" in params:
            self.log_level = params["log_level"].upper()
    
    def _parse_bool(self, value) -> bool:
        """Parse boolean from various formats."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1", "on")
        return bool(value)
    
    def run(self, context: Context) -> Context:
        """Handle exception if it matches type criteria."""
        exception = context.get_exception()
        if not exception:
            return context
        
        # Check if this exception type should be handled
        exception_type_name = type(exception).__name__
        
        if self.exception_types is None or exception_type_name in self.exception_types:
            # This exception matches our criteria
            if self.should_log:
                log_func = getattr(logger, self.log_level.lower(), logger.error)
                source = context.get_exception_source()
                log_func(f"Handling {exception_type_name} from task '{source}': {exception}")
            
            if self.should_clear:
                context.clear_exception()
                logger.debug(f"Cleared exception {exception_type_name}")
        
        return context
    
    def describe(self) -> str:
        return "Handles specific exception types, optionally clearing and/or logging them."
    
    def describe_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "type": {
                "description": "Exception type name(s) to handle (comma-separated). Omit to handle all.",
                "type": "str",
                "example": "ValueError,TypeError"
            },
            "clear": {
                "description": "Whether to clear the exception",
                "type": "bool",
                "default": True
            },
            "log": {
                "description": "Whether to log the exception",
                "type": "bool",
                "default": True
            },
            "log_level": {
                "description": "Logging level if log=true",
                "type": "str",
                "default": "ERROR",
                "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            }
        }


@register_task('rethrow_error')
class RethrowErrorTask(BaseTask):
    """
    Logs exception and re-raises it (for testing/debugging).
    
    Parameters:
    - message: Optional custom message
    
    Example DSL:
        task_a -> rethrow_error(message="Debug point") -> task_b
    """
    
    def __init__(self, task_id: str = "rethrow_error"):
        super().__init__(task_id)
        self.handles_exceptions = True
        self.message = "Re-raising exception"
    
    def configure(self, **params) -> None:
        if "message" in params:
            self.message = params["message"]
    
    def run(self, context: Context) -> Context:
        """Log and re-raise exception."""
        exception = context.get_exception()
        if exception:
            source = context.get_exception_source()
            logger.error(f"{self.message}: {type(exception).__name__} from '{source}': {exception}")
            # Re-raise by keeping exception in context (it will propagate)
            # Or actually raise it to stop pipeline
            raise exception
        
        return context
    
    def describe(self) -> str:
        return "Logs exception and re-raises it (stops pipeline execution)."
