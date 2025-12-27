"""
ImageLibrary Camera implementation.

This module provides a camera implementation that traverses a directory tree
of images and presents them according to camera configuration.
"""

import os
import time
import threading
import logging
from pathlib import Path
from typing import Tuple, List, Optional
from PIL import Image
import cv2
import numpy as np

from .camera_base import BaseCamera, CameraModel, Platform, Device
from ..metrics.metrics_collector import Collector, ValueType

logger = logging.getLogger(__name__)


class ImageLibraryCamera(BaseCamera):
    """
    Camera implementation that reads images from a directory tree.
    
    Traverses directories in depth-first order (sorted by name) and presents
    images according to the configured frame rate. Images are resized/cropped/padded
    to match the specified resolution.
    """

    def __init__(
        self,
        image_base_dir: str,
        width: int = 640,
        height: int = 480,
        framerate: int = 5,
        manual_mode: bool = False,
        loop_once: bool = False,
        platform: Platform = Platform.RPI,
        device: Device = Device.CAMERA0,
        save_dir: str = "./captures",
        metrics_collector: Optional[Collector] = None,
    ):
        """
        Initialize ImageLibrary camera.

        Args:
            image_base_dir: Base directory containing images to traverse
            width: Target image width in pixels
            height: Target image height in pixels
            framerate: Frame rate (images per second)
            manual_mode: If True, images advance only on capture_single_image() calls (no background thread)
            loop_once: If True, stop after processing all images once (don't loop). Sets exhausted flag.
            platform: Platform type (defaults to RPI)
            device: Device identifier (defaults to CAMERA0)
            save_dir: Directory for saving captured images
            metrics_collector: Optional metrics collector for instrumentation
        """
        super().__init__(CameraModel.IMAGE_LIBRARY, platform, device)
        
        self.image_base_dir = Path(image_base_dir).resolve()
        self.width = width
        self.height = height
        self.framerate = framerate
        self.frame_duration = 1.0 / framerate if framerate > 0 else 0.2
        self.manual_mode = manual_mode
        self.loop_once = loop_once
        self.save_dir = save_dir
        
        # Validate base directory
        if not self.image_base_dir.exists():
            raise ValueError(f"Image base directory does not exist: {image_base_dir}")
        if not self.image_base_dir.is_dir():
            raise ValueError(f"Image base directory is not a directory: {image_base_dir}")
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize metrics
        self._metrics_collector = metrics_collector
        self._setup_metrics()
        
        # Image list and current state
        self._image_paths: List[Path] = []
        self._current_index: int = 0
        self._current_image: Optional[Image.Image] = None
        self._current_path: Optional[str] = None
        self._exhausted: bool = False  # Set to True when all images processed (loop_once mode)
        self._lock = threading.Lock()
        
        # Thread control
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False
        
        # Scan directory for images
        self._scan_directory()
        
        if not self._image_paths:
            logger.warning(f"No images found in {image_base_dir}")
    
    def _setup_metrics(self):
        """Setup metrics instruments for tracking camera operations."""
        if self._metrics_collector is None:
            return
        
        # Register timeseries
        self._metrics_collector.register_timeseries(
            "image_library_operations",
            registered_attribute_keys=["operation"],
            max_count=1000
        )
        self._metrics_collector.register_timeseries(
            "image_library_framerate",
            registered_attribute_keys=[],
            max_count=100
        )
    
    def _record_metric(self, operation: str, value: float = 1.0):
        """Record a metric datapoint."""
        if self._metrics_collector is None:
            return
        
        self._metrics_collector.add_datapoint(
            "image_library_operations",
            ValueType.INT,
            int(value),
            attributes={"operation": operation}
        )
    
    def _record_frame_duration(self, duration: float):
        """Record frame duration for framerate calculation."""
        if self._metrics_collector is None:
            return
        
        self._metrics_collector.add_datapoint(
            "image_library_framerate",
            ValueType.DURATION,
            duration,
            attributes={}
        )
    
    def _scan_directory(self):
        """
        Scan the directory tree for images in depth-first order.
        
        Traverses directories sorted by name and collects all image files.
        """
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        
        def walk_depth_first(directory: Path) -> List[Path]:
            """Recursively walk directory tree in depth-first order."""
            images = []
            
            # Get all entries sorted by name
            try:
                entries = sorted(directory.iterdir(), key=lambda p: p.name)
            except (PermissionError, OSError) as e:
                logger.warning(f"Cannot access directory {directory}: {e}")
                return images
            
            # Separate files and directories
            files = [e for e in entries if e.is_file()]
            directories = [e for e in entries if e.is_dir()]
            
            # Add image files from current directory
            for file_path in files:
                if file_path.suffix.lower() in supported_extensions:
                    images.append(file_path)
            
            # Recursively process subdirectories
            for subdir in directories:
                images.extend(walk_depth_first(subdir))
            
            return images
        
        self._image_paths = walk_depth_first(self.image_base_dir)
        logger.info(f"Found {len(self._image_paths)} images in {self.image_base_dir}")
    
    def _process_image(self, image_path: Path) -> Image.Image:
        """
        Load and process an image to match target dimensions.
        
        Processing steps:
        1. Load image
        2. Resize to match target resolution (aspect-ratio aware)
        3. Center crop if dimensions are too large
        4. Pad with black if dimensions are too small
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Processed PIL Image
        """
        # Load image with opencv
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        orig_height, orig_width = img_rgb.shape[:2]
        target_width, target_height = self.width, self.height
        
        # Calculate scaling factor to match target resolution
        # We want to resize so that one dimension matches exactly
        scale_w = target_width / orig_width
        scale_h = target_height / orig_height
        
        # Use the larger scale to ensure we can crop (not pad on that dimension)
        scale = max(scale_w, scale_h)
        
        # Resize image
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        if scale != 1.0:
            img_rgb = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            self._record_metric("resize")
        
        # Now we need to crop/pad to exact dimensions
        current_height, current_width = img_rgb.shape[:2]
        
        # Center crop if too large
        if current_width > target_width or current_height > target_height:
            # Calculate crop coordinates
            start_x = (current_width - target_width) // 2
            start_y = (current_height - target_height) // 2
            
            end_x = start_x + target_width
            end_y = start_y + target_height
            
            img_rgb = img_rgb[start_y:end_y, start_x:end_x]
            self._record_metric("crop")
        
        # Pad with black if too small
        current_height, current_width = img_rgb.shape[:2]
        if current_width < target_width or current_height < target_height:
            # Calculate padding
            pad_top = (target_height - current_height) // 2
            pad_bottom = target_height - current_height - pad_top
            pad_left = (target_width - current_width) // 2
            pad_right = target_width - current_width - pad_left
            
            img_rgb = cv2.copyMakeBorder(
                img_rgb,
                pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)  # Black padding
            )
            self._record_metric("fill")
        
        # Convert to PIL Image
        pil_image = Image.fromarray(img_rgb)
        return pil_image
    
    def _update_thread(self):
        """Thread function that updates the current image based on frame rate."""
        while not self._stop_event.is_set():
            frame_start = time.time()
            
            if self._image_paths:
                # Get the next image
                with self._lock:
                    if self._current_index >= len(self._image_paths):
                        if self.loop_once:
                            self._exhausted = True
                            logger.info("ImageLibraryCamera: All images processed (loop_once=True)")
                            break  # Exit thread
                        else:
                            self._current_index = 0  # Loop back to start
                    
                    current_path = self._image_paths[self._current_index]
                    self._current_index += 1
                
                try:
                    # Process the image
                    logger.debug(f"Loading image: {current_path}")
                    processed_image = self._process_image(current_path)
                    
                    # Update current image
                    with self._lock:
                        self._current_image = processed_image
                        self._current_path = str(current_path)
                    
                except Exception as e:
                    logger.error(f"Failed to process image {current_path}: {e}")
            
            # Calculate how long to sleep
            frame_elapsed = time.time() - frame_start
            self._record_frame_duration(frame_elapsed)
            
            sleep_time = self.frame_duration - frame_elapsed
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)
    
    def start(self):
        """Start the camera thread."""
        if self.manual_mode:
            logger.info("ImageLibraryCamera in manual mode, skipping thread start")
            self._started = True
            return
        
        with self._lock:
            if self._started:
                logger.warning("ImageLibraryCamera already started")
                return
            
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._update_thread, daemon=True)
            self._thread.start()
            self._started = True
            logger.info("ImageLibraryCamera started")
    
    def stop(self):
        """Stop the camera thread."""
        with self._lock:
            if not self._started:
                return
            
            self._stop_event.set()
            self._started = False
        
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        logger.info("ImageLibraryCamera stopped")
    
    def capture_single_image(self) -> Tuple[str, Image.Image]:
        """
        Capture the currently available image.
        
        In manual mode, this advances to the next image on each call.
        In threaded mode, this returns whatever the background thread has loaded.
        
        Returns:
            Tuple[str, Image.Image]: File path and PIL Image object
        """
        with self._lock:
            # Manual mode: advance to next image on each call
            if self.manual_mode:
                if self._image_paths:
                    if self._current_index >= len(self._image_paths):
                        if self.loop_once:
                            self._exhausted = True
                            logger.info("ImageLibraryCamera: All images processed (loop_once=True)")
                            # Return last image when exhausted
                        else:
                            self._current_index = 0  # Loop back to start
                    
                    if not self._exhausted:
                        current_path = self._image_paths[self._current_index]
                        self._current_index += 1
                        
                        try:
                            logger.debug(f"Loading image: {current_path}")
                            self._current_image = self._process_image(current_path)
                            self._current_path = str(current_path)
                        except Exception as e:
                            logger.error(f"Failed to process image {current_path}: {e}")
                            # Fall through to return current image or blank
                else:
                    # No images available, create a blank image
                    if self._current_image is None:
                        blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                        self._current_image = Image.fromarray(blank)
                        self._current_path = "blank"
            
            # Threaded mode or fallback: return current image
            else:
                if self._current_image is None:
                    # If no image is loaded yet, load the first one
                    if self._image_paths:
                        current_path = self._image_paths[0]
                        self._current_image = self._process_image(current_path)
                        self._current_path = str(current_path)
                    else:
                        # No images available, create a blank image
                        blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                        self._current_image = Image.fromarray(blank)
                        self._current_path = "blank"
            
            return self._current_path, self._current_image.copy()
    
    @property
    def save_path(self) -> str:
        """Get the directory path for saving captured images."""
        return self.save_dir
    
    def is_exhausted(self) -> bool:
        """Check if all images have been processed (loop_once mode only)."""
        with self._lock:
            return self._exhausted
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, '_stop_event'):
            self.stop()
