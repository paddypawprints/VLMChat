"""
Pipeline Environment - Singleton providing general-purpose key-value store for pipeline tasks.

This module provides a global singleton that tasks can access to store and retrieve
arbitrary data using namespaced keys. The recommended key format is:
    {taskType}+{taskId}+{key}

This pattern allows:
- Tasks to store intermediate results for other tasks
- Multiple instances of the same task type to maintain separate state
- Chat application to share state with pipeline system
- No hardcoded attributes - fully extensible

Examples:
    "Camera+cam1+current_image"    - Camera task stores current image
    "ObjectDetector+det1+results"  - Detector stores detection results
    "App+main+history"             - Chat app stores conversation history
"""

from typing import Optional, Any, List
import logging

logger = logging.getLogger(__name__)


class Environment:
    """
    Singleton environment providing a general-purpose key-value store.
    
    Tasks can access this via Environment.get_instance() to store and retrieve
    arbitrary data without modifying the Environment class itself. Uses a
    dictionary-based design where keys follow the pattern: {taskType}+{taskId}+{key}
    
    This design allows any task to store/retrieve data while maintaining
    namespace separation between different task types and instances.
    """
    
    _instance: Optional['Environment'] = None
    
    def __init__(self):
        """
        Private constructor. Use get_instance() instead.
        
        Raises:
            RuntimeError: If called directly instead of via get_instance()
        """
        if Environment._instance is not None:
            raise RuntimeError("Environment is a singleton. Use Environment.get_instance()")
        
        self._store: dict[str, Any] = {}
        logger.info("Environment singleton initialized with empty store")

    @classmethod
    def get_instance(cls) -> 'Environment':
        """
        Get the singleton Environment instance.
        
        Creates the instance on first call, returns existing instance thereafter.
        
        Returns:
            Environment: The singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance. Primarily for testing.
        """
        if cls._instance is not None:
            cls._instance._store.clear()
        cls._instance = None
        logger.info("Environment singleton reset")
    
    def set(self, taskType: str, taskId: str, key: str, value: Any) -> None:
        """
        Set a value in the environment store.
        
        Args:
            key: Namespaced key (recommended: {taskType}+{taskId}+{key})
            value: Any value to store
        """
        self._store[self._env_key(taskType, taskId, key)] = value
        logger.debug(f"Environment: set '{key}' = {type(value).__name__}")
    
    def get(self, taskType: str, taskId: str, key: str, default: Any = None) -> Any:
        """
        Get a value from the environment store.
        
        Args:
            key: Namespaced key to retrieve
            default: Default value if key not found
        
        Returns:
            The stored value or default if key doesn't exist
        """
        value = self._store.get(self._env_key(taskType, taskId, key), default)
        if value is not None:
            logger.debug(f"Environment: get '{self._env_key(taskType, taskId, key)}' -> {type(value).__name__}")
        return value
    
    def has(self, taskType: str, taskId: str, key: str) -> bool:
        """
        Check if a key exists in the environment store.
        
        Args:
            key: Key to check
        
        Returns:
            bool: True if key exists, False otherwise
        """
        return self._env_key(taskType, taskId, key) in self._store
    
    def remove(self, taskType: str, taskId: str, key: str) -> bool:
        """
        Remove a key from the environment store.
        
        Args:
            key: Key to remove
        
        Returns:
            bool: True if key was removed, False if it didn't exist
        """
        env_key = self._env_key(taskType, taskId, key)
        if env_key in self._store:
            del self._store[env_key]
            logger.debug(f"Environment: removed '{env_key}'")
            return True
        return False
    
    def keys(self) -> List[str]:
        """
        Get all keys currently in the environment store.
        
        Returns:
            List[str]: List of all keys
        """
        return list(self._store.keys())
    
    def clear(self) -> None:
        """Clear all data from the environment store."""
        count = len(self._store)
        self._store.clear()
        logger.info(f"Environment: cleared {count} keys")
    
    def __len__(self) -> int:
        """Return the number of keys in the store."""
        return len(self._store)
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for checking key existence."""
        return key in self._store
    
    def _env_key(self, taskType: str, taskId: str, key: str) -> str:
        """
        Construct a namespaced environment key.
        
        Args:
            taskType: Type of the task (e.g., "Camera", "ObjectDetector")
            taskId: Unique identifier for the task instance (e.g., "cam1", "det1")
            key: Specific key name (e.g., "current_image", "results")
        
        Returns:
            str: Namespaced key in the format "{taskType}+{taskId}+{key}"
        """
        return f"{taskType}+{taskId}+{key}"
    

