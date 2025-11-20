"""
PipelineFactory - Dynamic task and connector instantiation.

Provides a registry-based factory for creating pipeline tasks and connectors
from string names with parameter injection. Used by the DSL parser to build
pipelines from text definitions.

Example:
    factory = PipelineFactory()
    factory.register_task("camera", CameraTask)
    factory.register_connector("first_complete", FirstCompleteConnector)
    
    # Create from DSL
    camera = factory.create_task("camera", {"device": "0", "resolution": "640x480"})
    connector = factory.create_connector("first_complete")
"""

import logging
from typing import Dict, Type, Optional, Any
from .task_base import BaseTask, Connector

logger = logging.getLogger(__name__)


class PipelineFactory:
    """
    Factory for creating pipeline tasks and connectors from string names.
    
    Maintains registries mapping names to classes and provides methods
    for dynamic instantiation with parameter injection.
    """
    
    def __init__(self):
        """Initialize empty registries."""
        self._task_registry: Dict[str, Type[BaseTask]] = {}
        self._connector_registry: Dict[str, Type[Connector]] = {}
        logger.info("PipelineFactory initialized")
    
    def register_task(self, name: str, task_class: Type[BaseTask]) -> None:
        """
        Register a task class with a string name.
        
        Args:
            name: String identifier for the task (used in DSL)
            task_class: Class that extends BaseTask
            
        Example:
            factory.register_task("camera", CameraTask)
            factory.register_task("yolo", DetectorTask)
        """
        if name in self._task_registry:
            logger.warning(f"Overwriting existing task registration: {name}")
        
        self._task_registry[name] = task_class
        logger.info(f"Registered task: {name} -> {task_class.__name__}")
    
    def register_connector(self, name: str, connector_class: Type[Connector]) -> None:
        """
        Register a connector class with a string name.
        
        Args:
            name: String identifier for the connector (used in DSL)
            connector_class: Class that extends Connector
            
        Example:
            factory.register_connector("first_complete", FirstCompleteConnector)
            factory.register_connector("ordered_merge", OrderedMergeConnector)
        """
        if name in self._connector_registry:
            logger.warning(f"Overwriting existing connector registration: {name}")
        
        self._connector_registry[name] = connector_class
        logger.info(f"Registered connector: {name} -> {connector_class.__name__}")
    
    def create_task(self, name: str, task_id: str, params: Optional[Dict[str, str]] = None) -> BaseTask:
        """
        Create a task instance from registered name.
        
        Args:
            name: Registered task name (e.g., "camera", "yolo")
            task_id: Unique identifier for this task instance
            params: Optional parameters to pass to configure()
            
        Returns:
            Instantiated task with parameters configured
            
        Raises:
            ValueError: If name is not registered
            
        Example:
            camera = factory.create_task("camera", "cam0", {"device": "0", "type": "imx219"})
        """
        if name not in self._task_registry:
            logger.error(f"Attempted to create unregistered task: {name}")
            raise ValueError(f"Unknown task name: {name}. Available: {list(self._task_registry.keys())}")
        
        task_class = self._task_registry[name]
        
        # Try different constructor patterns
        task = None
        
        # Pattern 1: Try task_id as keyword argument (most flexible)
        try:
            task = task_class(task_id=task_id)
        except TypeError:
            pass
        
        # Pattern 2: Try task_id as positional argument
        if task is None:
            try:
                task = task_class(task_id)
            except TypeError:
                pass
        
        # Pattern 3: Try no arguments, set task_id after
        if task is None:
            try:
                task = task_class()
                task.task_id = task_id
            except Exception as e:
                logger.error(f"Failed to instantiate task {name}: {e}")
                raise ValueError(f"Could not instantiate task '{name}'. "
                               f"Constructor must accept task_id parameter or no parameters.")
        
        # Configure with parameters if provided
        if params:
            try:
                task.configure(params)
                logger.info(f"Created task '{task_id}' ({name}) with params: {params}")
            except Exception as e:
                logger.error(f"Failed to configure task {name} with params {params}: {e}")
                raise
        else:
            logger.info(f"Created task '{task_id}' ({name})")
        
        return task
    
    def create_connector(self, name: str, connector_id: str, 
                        params: Optional[Dict[str, str]] = None,
                        time_budget_ms: Optional[int] = None) -> Connector:
        """
        Create a connector instance from registered name.
        
        Args:
            name: Registered connector name (e.g., "first_complete", "ordered_merge")
            connector_id: Unique identifier for this connector instance
            params: Optional parameters to pass to configure()
            time_budget_ms: Optional time budget for connector execution
            
        Returns:
            Instantiated connector with parameters configured
            
        Raises:
            ValueError: If name is not registered
            
        Example:
            merge = factory.create_connector("ordered_merge", "merge1", 
                                            {"order": "2,1"})
        """
        if name not in self._connector_registry:
            # Try default Connector for unregistered names
            logger.warning(f"Unknown connector name: {name}, using default Connector")
            connector = Connector(connector_id, time_budget_ms)
        else:
            connector_class = self._connector_registry[name]
            connector = connector_class(connector_id, time_budget_ms)
        
        # Configure with parameters if provided
        if params:
            try:
                connector.configure(params)
                logger.info(f"Created connector '{connector_id}' ({name}) with params: {params}")
            except Exception as e:
                logger.error(f"Failed to configure connector {name} with params {params}: {e}")
                raise
        else:
            logger.info(f"Created connector '{connector_id}' ({name})")
        
        return connector
    
    def list_tasks(self) -> list[str]:
        """Return list of registered task names."""
        return sorted(self._task_registry.keys())
    
    def list_connectors(self) -> list[str]:
        """Return list of registered connector names."""
        return sorted(self._connector_registry.keys())
    
    def is_task_registered(self, name: str) -> bool:
        """Check if a task name is registered."""
        return name in self._task_registry
    
    def is_connector_registered(self, name: str) -> bool:
        """Check if a connector name is registered."""
        return name in self._connector_registry


def create_default_factory() -> PipelineFactory:
    """
    Create a factory with all built-in tasks and connectors registered.
    
    Returns:
        PipelineFactory with standard tasks/connectors pre-registered
        
    Example:
        factory = create_default_factory()
        camera = factory.create_task("camera", "cam0")
    """
    factory = PipelineFactory()
    
    # Register connectors
    from .connectors import FirstCompleteConnector, OrderedMergeConnector
    factory.register_connector("connector", Connector)
    factory.register_connector("first_complete", FirstCompleteConnector)
    factory.register_connector("ordered_merge", OrderedMergeConnector)
    
    # Register tasks
    try:
        from .tasks.camera_task import CameraTask
        factory.register_task("camera", CameraTask)
    except ImportError:
        logger.warning("CameraTask not available for registration")
    
    try:
        from .detector_task import DetectorTask
        factory.register_task("detector", DetectorTask)
    except ImportError:
        logger.warning("DetectorTask not available for registration")
    
    try:
        from .tasks.console_input_task import ConsoleInputTask
        factory.register_task("console_input", ConsoleInputTask)
    except ImportError:
        logger.warning("ConsoleInputTask not available for registration")
    
    try:
        from .tasks.console_output_task import ConsoleOutputTask
        factory.register_task("console_output", ConsoleOutputTask)
    except ImportError:
        logger.warning("ConsoleOutputTask not available for registration")
    
    try:
        from .smolvlm_task import SmolVLMTask
        factory.register_task("smolvlm", SmolVLMTask)
    except ImportError:
        logger.warning("SmolVLMTask not available for registration")
    
    try:
        from .tasks.history_update_task import HistoryUpdateTask
        factory.register_task("history_update", HistoryUpdateTask)
    except ImportError:
        logger.warning("HistoryUpdateTask not available for registration")
    
    try:
        from .start_task import StartTask
        factory.register_task("start", StartTask)
    except ImportError:
        logger.warning("StartTask not available for registration")
    
    logger.info(f"Default factory created with {len(factory.list_tasks())} tasks "
                f"and {len(factory.list_connectors())} connectors")
    
    return factory
