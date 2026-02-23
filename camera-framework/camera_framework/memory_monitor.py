"""Production memory monitoring using weakref tracking.

Tracks large objects (images, tensors) to detect memory leaks in production.
"""

import weakref
import threading
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ObjectLifetime:
    """Track lifetime of a monitored object."""
    obj_type: str
    created_at: float
    size_bytes: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age_seconds(self) -> float:
        """Age of object in seconds."""
        return time.monotonic() - self.created_at
    
    @property
    def age_ms(self) -> float:
        """Age of object in milliseconds."""
        return self.age_seconds * 1000


class MemoryMonitor:
    """Monitor object lifetimes using weakref for leak detection.
    
    Usage:
        monitor = MemoryMonitor()
        
        # Track large objects
        message = {}
        monitor.track(ctx, "Context", size_bytes=1024, metadata={"source": "camera"})
        
        # Periodically check for leaks
        monitor.report()
    """
    
    def __init__(self, report_interval: float = 60.0):
        """Initialize memory monitor.
        
        Args:
            report_interval: Seconds between automatic reports
        """
        self.tracked: Dict[int, ObjectLifetime] = {}
        self.lock = threading.Lock()
        self.report_interval = report_interval
        self._last_report = time.monotonic()
        
        # Statistics
        self.total_tracked = 0
        self.total_cleaned = 0
    
    def track(self, obj: Any, obj_type: str, size_bytes: Optional[int] = None, 
              metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track an object's lifetime using weakref.
        
        Args:
            obj: Object to track (must support weakref)
            obj_type: Type name for reporting (e.g., "Context", "ImageContainer")
            size_bytes: Optional size in bytes
            metadata: Optional metadata dict
        """
        try:
            obj_id = id(obj)
            lifetime = ObjectLifetime(
                obj_type=obj_type,
                created_at=time.monotonic(),
                size_bytes=size_bytes,
                metadata=metadata or {}
            )
            
            # Create weakref with cleanup callback
            def on_cleanup(ref):
                self._on_object_deleted(obj_id)
            
            weak = weakref.ref(obj, on_cleanup)
            
            with self.lock:
                self.tracked[obj_id] = lifetime
                self.total_tracked += 1
                
        except TypeError:
            logger.warning(f"Cannot track {obj_type}: weakref not supported")
    
    def _on_object_deleted(self, obj_id: int) -> None:
        """Callback when tracked object is GC'd."""
        with self.lock:
            if obj_id in self.tracked:
                del self.tracked[obj_id]
                self.total_cleaned += 1
    
    def check(self) -> Dict[str, Any]:
        """Check for potential memory leaks.
        
        Returns:
            Dict with leak detection info
        """
        with self.lock:
            # Group by type
            by_type: Dict[str, List[ObjectLifetime]] = {}
            for lifetime in self.tracked.values():
                if lifetime.obj_type not in by_type:
                    by_type[lifetime.obj_type] = []
                by_type[lifetime.obj_type].append(lifetime)
            
            # Analyze each type
            leaks = []
            for obj_type, lifetimes in by_type.items():
                count = len(lifetimes)
                if count == 0:
                    continue
                
                ages = [lt.age_seconds for lt in lifetimes]
                avg_age = sum(ages) / count
                max_age = max(ages)
                
                # Flag potential leaks (objects alive > 60 seconds)
                if max_age > 60.0:
                    leaks.append({
                        "type": obj_type,
                        "count": count,
                        "avg_age_sec": avg_age,
                        "max_age_sec": max_age,
                        "oldest": max(lifetimes, key=lambda x: x.age_seconds)
                    })
            
            return {
                "timestamp": datetime.now().isoformat(),
                "total_alive": len(self.tracked),
                "total_tracked": self.total_tracked,
                "total_cleaned": self.total_cleaned,
                "by_type": {
                    obj_type: len(lifetimes) 
                    for obj_type, lifetimes in by_type.items()
                },
                "potential_leaks": leaks
            }
    
    def report(self) -> None:
        """Log memory status report."""
        stats = self.check()
        
        logger.info(f"Memory Monitor: {stats['total_alive']} objects alive, "
                   f"{stats['total_cleaned']} cleaned")
        
        if stats['potential_leaks']:
            logger.warning(f"Potential leaks detected: {len(stats['potential_leaks'])} types")
            for leak in stats['potential_leaks']:
                logger.warning(f"  {leak['type']}: {leak['count']} objects, "
                             f"max age {leak['max_age_sec']:.1f}s")
        
        self._last_report = time.monotonic()
    
    def maybe_report(self) -> None:
        """Report if report_interval has elapsed."""
        if time.monotonic() - self._last_report >= self.report_interval:
            self.report()
    
    def reset(self) -> None:
        """Reset all tracking (for testing)."""
        with self.lock:
            self.tracked.clear()
            self.total_tracked = 0
            self.total_cleaned = 0


# Global instance for easy import
memory_monitor = MemoryMonitor()


def track_context(ctx: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to track a Context object.
    
    Usage:
        from camera_framework.memory_monitor import track_context
        
        message = {}
        track_context(ctx, metadata={"source": "camera", "frame": 123})
    """
    memory_monitor.track(ctx, "Context", metadata=metadata)


def track_image(img: Any, size_bytes: Optional[int] = None, 
                metadata: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to track an image/tensor.
    
    Usage:
        from camera_framework.memory_monitor import track_image
        
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        track_image(frame, size_bytes=frame.nbytes, metadata={"source": "camera"})
    """
    memory_monitor.track(img, "Image", size_bytes=size_bytes, metadata=metadata)
