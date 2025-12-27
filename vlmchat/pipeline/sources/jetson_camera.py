"""
Jetson-optimized camera source with zero-copy buffer pool.

This module provides a memory-efficient camera implementation for Jetson
platforms using a reference-counted buffer pool with automatic promotion.
"""

import logging
import threading
import time
from collections import defaultdict
from typing import List, Optional
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .camera import CameraSource
from ..cache.image import ImageContainer
from ..image.formats import ImageFormat

logger = logging.getLogger(__name__)


class PooledBuffer:
    """
    A reference-counted buffer in the pool.
    
    Tracks which ImageContainers reference this buffer and supports
    automatic promotion (copy to owned memory) when buffer is needed.
    """
    
    def __init__(self, data: np.ndarray, index: int):
        """
        Initialize pooled buffer.
        
        Args:
            data: Preallocated numpy array
            index: Buffer index in pool
        """
        self.data = data
        self.index = index
        self.refcount = 0
        self.capture_time = 0.0
        self.valid = True
        self.containers: List['ImageContainer'] = []
    
    def add_ref(self, container: 'ImageContainer') -> None:
        """Register an ImageContainer that references this buffer."""
        self.refcount += 1
        if container not in self.containers:
            self.containers.append(container)
    
    def remove_ref(self, container: 'ImageContainer') -> None:
        """Unregister an ImageContainer."""
        self.refcount -= 1
        if container in self.containers:
            self.containers.remove(container)
        assert self.refcount >= 0, f"Negative refcount on buffer {self.index}"
    
    def invalidate(self) -> None:
        """Mark buffer as invalid (being recycled)."""
        self.valid = False


class BufferPool:
    """
    Fixed-size pool of preallocated numpy arrays.
    
    Provides reference-counted buffers that are automatically promoted
    (copied to owned memory) when the pool needs to reclaim them.
    
    Features:
    - Zero allocation during capture (buffers preallocated)
    - Automatic promotion prevents data loss
    - Thread-safe for camera callback use
    - Bounded memory (fixed pool size)
    """
    
    def __init__(self, num_buffers: int, width: int, height: int, channels: int = 3):
        """
        Initialize buffer pool.
        
        Args:
            num_buffers: Number of buffers to preallocate
            width: Frame width in pixels
            height: Frame height in pixels
            channels: Number of color channels (default: 3 for BGR)
        """
        self.num_buffers = num_buffers
        self.width = width
        self.height = height
        self.channels = channels
        
        # Preallocate all buffers
        self.buffers = [
            PooledBuffer(
                data=np.empty((height, width, channels), dtype=np.uint8),
                index=i
            )
            for i in range(num_buffers)
        ]
        
        self.lock = threading.Lock()
        self.metrics = defaultdict(int)
        
        logger.info(f"BufferPool initialized: {num_buffers} buffers of {width}x{height}x{channels} "
                   f"({width * height * channels * num_buffers / 1024 / 1024:.1f} MB)")
    
    def acquire(self) -> Optional[PooledBuffer]:
        """
        Acquire a buffer from the pool.
        
        If no free buffers available, promotes the oldest borrowed buffer
        to owned memory and reclaims it.
        
        Returns:
            PooledBuffer or None if pool exhausted after promotion
        """
        with self.lock:
            # Try to find free buffer
            for buf in self.buffers:
                if buf.refcount == 0:
                    buf.refcount = 1  # Mark as in-use by camera
                    buf.capture_time = time.monotonic()
                    buf.valid = True
                    self.metrics['buffers_acquired'] += 1
                    return buf
            
            # No free buffers - need to promote oldest
            oldest = self._find_oldest_borrowed()
            if oldest is None:
                # Should not happen - all buffers are free (refcount=0)
                logger.error("Pool exhausted but no borrowed buffers found")
                self.metrics['frames_dropped_pool_error'] += 1
                return None
            
            # Promote all containers using this buffer
            self._promote_buffer(oldest)
            
            # Reclaim buffer
            oldest.refcount = 0
            oldest.valid = False
            oldest.containers.clear()
            
            # Now acquire it
            oldest.refcount = 1
            oldest.capture_time = time.monotonic()
            oldest.valid = True
            
            self.metrics['buffers_evicted'] += 1
            return oldest
    
    def release(self, buffer: PooledBuffer) -> None:
        """
        Release a buffer reference (called by camera after writing frame).
        
        Args:
            buffer: Buffer to release
        """
        with self.lock:
            buffer.refcount -= 1
            self.metrics['buffers_released'] += 1
    
    def get_pressure(self) -> float:
        """
        Get current pool utilization.
        
        Returns:
            Float from 0.0 (empty) to 1.0 (full)
        """
        with self.lock:
            borrowed = sum(1 for buf in self.buffers if buf.refcount > 0)
            return borrowed / self.num_buffers
    
    def get_metrics(self) -> dict:
        """Get pool statistics."""
        with self.lock:
            borrowed = sum(1 for buf in self.buffers if buf.refcount > 0)
            return {
                'total_buffers': self.num_buffers,
                'borrowed_buffers': borrowed,
                'free_buffers': self.num_buffers - borrowed,
                'pressure': borrowed / self.num_buffers,
                'buffers_acquired': self.metrics['buffers_acquired'],
                'buffers_released': self.metrics['buffers_released'],
                'buffers_evicted': self.metrics['buffers_evicted'],
                'frames_promoted': self.metrics['frames_promoted'],
                'frames_dropped': self.metrics['frames_dropped_pool_error'],
            }
    
    def _find_oldest_borrowed(self) -> Optional[PooledBuffer]:
        """Find the oldest buffer with references (candidate for promotion)."""
        # Look for buffers that are in use (refcount > 0)
        # Prefer those with containers, but if none, take any borrowed buffer
        with_containers = [buf for buf in self.buffers if buf.refcount > 0 and buf.containers]
        if with_containers:
            return min(with_containers, key=lambda b: b.capture_time)
        
        # No containers, but still borrowed - just take oldest
        borrowed = [buf for buf in self.buffers if buf.refcount > 0]
        if borrowed:
            return min(borrowed, key=lambda b: b.capture_time)
        
        return None
    
    def _promote_buffer(self, buffer: PooledBuffer) -> None:
        """
        Promote all ImageContainers using this buffer to owned memory.
        
        Copies buffer data to each container so they become independent
        of the pooled buffer.
        
        Args:
            buffer: Buffer to promote
        """
        for container in list(buffer.containers):
            # Call container's promotion method
            if hasattr(container, '_promote_to_owned'):
                container._promote_to_owned()
                self.metrics['frames_promoted'] += 1


class JetsonCameraSource(CameraSource):
    """
    Jetson camera source with GStreamer and zero-copy buffer pool.
    
    Uses GStreamer nvarguscamerasrc for hardware-accelerated camera capture
    and a buffer pool to minimize memory allocations/copies. Supports 
    automatic promotion to owned memory when buffers are needed.
    
    Continuous capture thread ensures frames are always being read from
    GStreamer, regardless of pipeline processing speed.
    
    Examples:
        @camera: jetson_camera(device=0, fps=30, pool_size=60)
        
        wait(camera) -> latest(camera) -> yolo() -> display()
    """
    
    def __init__(self, name: str, device: int = 0, fps: float = 30.0,
                 buffer_size: int = 300, width: Optional[int] = None,
                 height: Optional[int] = None, use_gstreamer: bool = True,
                 flip_method: int = 2, pool_size: int = 60, output_label: str = "frame"):
        """
        Initialize Jetson camera source with buffer pool.
        
        Args:
            name: Unique identifier for this source
            device: Camera device index (0 for default camera)
            fps: Target frames per second
            buffer_size: Number of frames to keep in ring buffer
            width: Optional frame width (None = camera default)
            height: Optional frame height (None = camera default)
            use_gstreamer: If True, use GStreamer pipeline (ignored, always True on Jetson)
            flip_method: Flip method for GStreamer (0=none, 2=rotate-180)
            pool_size: Number of buffers in pool (default: 60 = 2 sec @ 30fps)
            output_label: Label to use when adding frames to context (default: "frame")
        """
        # Initialize base camera
        super().__init__(name, device, fps, buffer_size, width, height, output_label)
        
        # GStreamer settings
        self.use_gstreamer = use_gstreamer
        self.flip_method = flip_method
        
        # Create buffer pool
        self._pool = BufferPool(
            num_buffers=pool_size,
            width=self.target_width,
            height=self.target_height,
            channels=3  # BGR
        )
        
        # Continuous capture thread
        self._capture_thread = None
        self._capture_running = False
        
        logger.info(f"JetsonCameraSource '{name}' initialized with {pool_size} buffer pool, GStreamer={use_gstreamer}")
    
    def _gstreamer_pipeline(self) -> str:
        """Build GStreamer pipeline string for Jetson camera."""
        return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                self.target_width,
                self.target_height,
                int(self.fps),
                self.flip_method,
                self.target_width,
                self.target_height,
            )
        )
    
    def _continuous_capture_loop(self):
        """
        Continuous capture loop running in background thread.
        
        Continuously reads frames from GStreamer and adds to ring buffer.
        This ensures frames are captured even if poll() isn't called fast enough.
        """
        logger.info(f"Camera '{self.name}' capture thread started")
        
        while self._capture_running:
            if not self.capture or not self.capture.isOpened():
                time.sleep(0.01)
                continue
            
            # Get buffer from pool
            pool_buffer = self._pool.acquire()
            if pool_buffer is None:
                # Pool exhausted - log and continue (frame dropped)
                logger.debug(f"Camera '{self.name}' dropped frame: pool exhausted")
                time.sleep(0.001)
                continue
            
            # Capture frame directly into pool buffer
            ret = self.capture.grab()
            if ret:
                ret = self.capture.retrieve(pool_buffer.data)
            
            if not ret:
                # Capture failed - release buffer and continue
                self._pool.release(pool_buffer)
                time.sleep(0.001)
                continue
            
            # Release camera's reference - ImageContainer will hold its own
            self._pool.release(pool_buffer)
            
            # Create ImageContainer in pooled mode
            current_time = time.time()
            cache_key = f"{self.name}_frame_{int(current_time * 1000)}"
            container = ImageContainer(
                cache_key=cache_key,
                pooled_buffer=pool_buffer
            )
            
            # Add to ring buffer (this is thread-safe in base class)
            with self.lock:
                self.ring_buffer.append((self.sequence_counter, container))
                self.latest_data = container
                self.latest_sequence = self.sequence_counter
                self.sequence_counter += 1
            
            # FPS throttling
            if self.frame_interval > 0:
                time.sleep(self.frame_interval)
        
        logger.info(f"Camera '{self.name}' capture thread stopped")
    
    def start(self) -> None:
        """Start camera with GStreamer and continuous capture thread."""
        if not self.running:
            # Open camera with GStreamer
            if self.use_gstreamer:
                pipeline = self._gstreamer_pipeline()
                logger.info(f"Camera '{self.name}' using GStreamer pipeline")
                self.capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)  # type: ignore[attr-defined]
            else:
                # Fallback to standard V4L2
                self.capture = cv2.VideoCapture(self.device)
            
            if not self.capture.isOpened():
                raise RuntimeError(f"Failed to open camera device {self.device}")
            
            # Get actual resolution
            actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # type: ignore[attr-defined]
            actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # type: ignore[attr-defined]
            logger.info(f"Camera '{self.name}' opened: {actual_width}x{actual_height} @ {self.fps} fps")
            
            # Mark as running
            self.running = True
        
        # Start continuous capture thread
        if self._capture_thread is None or not self._capture_thread.is_alive():
            self._capture_running = True
            self._capture_thread = threading.Thread(
                target=self._continuous_capture_loop,
                name=f"Camera_{self.name}_Capture",
                daemon=True
            )
            self._capture_thread.start()
            logger.info(f"Camera '{self.name}' started continuous capture")
    
    def stop(self) -> None:
        """Stop continuous capture thread and release camera."""
        # Stop capture thread
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_running = False
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
            logger.info(f"Camera '{self.name}' stopped continuous capture")
        
        # Mark as stopped and release camera
        self.running = False
        if self.capture:
            self.capture.release()
            self.capture = None
            logger.info(f"Camera '{self.name}' released")
    
    def _capture_data(self) -> Optional[ImageContainer]:
        """
        Get latest captured frame from ring buffer.
        
        NOTE: With continuous capture thread, this method is not used.
        Frames are captured continuously and added directly to ring buffer.
        poll() in base class just checks if new data is available.
        
        Returns:
            None (continuous capture bypasses this method)
        """
        # Continuous capture thread handles frame capture
        # poll() in base class checks ring buffer directly
        return None
    
    def poll(self) -> bool:
        """
        Check if new frame available from continuous capture.
        
        Overrides base class to just check ring buffer.
        Actual capture happens in background thread.
        
        Returns:
            True if new data available in ring buffer
        """
        if not self.running:
            return False
        
        # Check if ring buffer has new data
        with self.lock:
            return len(self.ring_buffer) > 0
    
    def get_pool_metrics(self) -> dict:
        """
        Get buffer pool statistics.
        
        Returns:
            Dict with pool metrics (pressure, buffers, promotions, etc.)
        """
        return self._pool.get_metrics()
