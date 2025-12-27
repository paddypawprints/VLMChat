"""
Global cache for immutable pipeline items with reference counting.

Supports pluggable storage backends:
- InProcessBackend: Direct references (current)
- SharedMemoryBackend: Serialized shared memory (future)
"""
import threading
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Set, Optional
import logging

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract storage backend for cached items."""
    
    @abstractmethod
    def store(self, key: str, item: Any, format: str, metadata: Dict[str, Any]) -> None:
        """Store item in cache."""
        pass
    
    @abstractmethod
    def retrieve(self, key: str) -> tuple:
        """Retrieve item and metadata. Returns (item, format, metadata)."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete item from cache."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass


class InProcessBackend(StorageBackend):
    """Direct reference storage - no serialization."""
    
    def __init__(self):
        self._items: Dict[str, tuple] = {}  # key -> (item, format, metadata)
    
    def store(self, key: str, item: Any, format: str, metadata: Dict[str, Any]) -> None:
        self._items[key] = (item, format, metadata)
    
    def retrieve(self, key: str) -> tuple:
        return self._items[key]
    
    def delete(self, key: str) -> None:
        del self._items[key]
    
    def exists(self, key: str) -> bool:
        return key in self._items


class ItemCache:
    """
    Global cache for immutable pipeline items with reference counting.
    
    Items are never mutated - only created, referenced, and released.
    Automatic cleanup when reference count reaches zero.
    """
    
    _instance: Optional['ItemCache'] = None
    _lock = threading.Lock()
    
    def __init__(self, backend: Optional[StorageBackend] = None):
        self._backend = backend or InProcessBackend()
        self._refcounts: Dict[str, int] = {}
        self._refcount_lock = threading.Lock()
    
    @classmethod
    def global_instance(cls) -> 'ItemCache':
        """Get or create global cache instance (singleton)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_global_instance(cls) -> None:
        """Reset global instance (for testing)."""
        with cls._lock:
            cls._instance = None
    
    def add(self, item: Any, format: str = 'native', metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add item to cache and return cache key.
        
        Args:
            item: The data to cache (image, embedding, etc.)
            format: Primary format identifier
            metadata: Optional metadata (dimensions, timestamps, etc.)
        
        Returns:
            cache_key: Unique identifier for this cached item
        """
        cache_key = str(uuid.uuid4())
        metadata = metadata or {}
        
        with self._refcount_lock:
            self._backend.store(cache_key, item, format, metadata)
            self._refcounts[cache_key] = 1
        
        logger.debug(f"Cache ADD: {cache_key[:8]}... format={format} refcount=1")
        return cache_key
    
    def retain(self, cache_key: str) -> None:
        """Increment reference count."""
        with self._refcount_lock:
            if cache_key in self._refcounts:
                self._refcounts[cache_key] += 1
                logger.debug(f"Cache RETAIN: {cache_key[:8]}... refcount={self._refcounts[cache_key]}")
    
    def release(self, cache_key: str) -> None:
        """
        Decrement reference count, cleanup if zero.
        
        Args:
            cache_key: Key to release
        """
        with self._refcount_lock:
            if cache_key not in self._refcounts:
                logger.warning(f"Cache RELEASE: {cache_key[:8]}... not found")
                return
            
            self._refcounts[cache_key] -= 1
            refcount = self._refcounts[cache_key]
            
            logger.debug(f"Cache RELEASE: {cache_key[:8]}... refcount={refcount}")
            
            if refcount == 0:
                # Cleanup
                self._backend.delete(cache_key)
                del self._refcounts[cache_key]
                logger.debug(f"Cache DELETE: {cache_key[:8]}... (refcount=0)")
    
    def retrieve(self, cache_key: str) -> tuple:
        """
        Retrieve item and metadata.
        
        Returns:
            (item, format, metadata) tuple
        """
        return self._backend.retrieve(cache_key)
    
    def get_refcount(self, cache_key: str) -> int:
        """Get current reference count (for debugging/testing)."""
        with self._refcount_lock:
            return self._refcounts.get(cache_key, 0)
    
    def collect_unreferenced(self, active_keys: Set[str]) -> int:
        """
        Garbage collect items not in active set.
        Called by PipelineRunner at strategic points.
        
        This sets refcount to 0 for items not in active set and removes them.
        
        Args:
            active_keys: Set of cache keys currently referenced
        
        Returns:
            Number of items collected
        """
        collected = 0
        
        with self._refcount_lock:
            # Find keys that are not in active set
            all_keys = set(self._refcounts.keys())
            unreferenced = all_keys - active_keys
            
            for cache_key in unreferenced:
                # Set refcount to 0 and cleanup
                self._refcounts[cache_key] = 0
                self._backend.delete(cache_key)
                del self._refcounts[cache_key]
                collected += 1
                logger.debug(f"Cache GC: {cache_key[:8]}... collected")
        
        if collected > 0:
            logger.info(f"Cache GC: Collected {collected} unreferenced items")
        
        return collected
    
    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._refcount_lock:
            return {
                'total_items': len(self._refcounts),
                'total_refcount': sum(self._refcounts.values())
            }
