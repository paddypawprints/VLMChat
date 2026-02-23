"""
ImageLibraryCamera - Mock camera that cycles through images from a directory.

This module provides a camera implementation that traverses a directory tree
of images and presents them like a live camera feed.
"""

import logging
import time
import threading
from pathlib import Path
from typing import Optional
from PIL import Image
import cv2
import numpy as np

from camera_framework import BaseTask

logger = logging.getLogger(__name__)


class ImageLibraryCamera(BaseTask):
    """
    Camera task that reads images from a directory tree.
    
    Traverses directories in depth-first order (sorted by name) and cycles
    through images. Images are resized/cropped/padded to match the specified 
    resolution.
    
    Args:
        image_dir: Base directory containing images to traverse
        width: Target image width in pixels (default: 1920)
        height: Target image height in pixels (default: 1080)
        framerate: Frame rate in images per second (default: 1)
        name: Task name (default: "ImageLibraryCamera")
        fields: Field mappings (default output: "frame")
    
    Field mappings:
        frame: Output field for captured frame (default: "frame")
    """
    
    def __init__(
        self,
        image_dir: str,
        width: int = 1920,
        height: int = 1080,
        framerate: float = 5.0,
        name: str = "ImageLibraryCamera",
        fields: dict = None,
        collector=None,  # Optional metrics collector
    ):
        super().__init__(name, fields)  # No interval - runner controls schedule
        
        self.image_dir = Path(image_dir).resolve()
        self.width = width
        self.height = height
        self.framerate = framerate
        self.frame_duration = 1.0 / framerate if framerate > 0 else 1.0
        self.collector = collector  # For memory tracking
        
        # Validate directory
        if not self.image_dir.exists():
            raise ValueError(f"Image directory does not exist: {image_dir}")
        if not self.image_dir.is_dir():
            raise ValueError(f"Image path is not a directory: {image_dir}")
        
        # Image list and current state
        self._image_paths: list[Path] = []
        self._current_index: int = 0
        self._current_image: Optional[Image.Image] = None
        self._current_path: Optional[str] = None
        self._lock = threading.Lock()
        
        # Thread control
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_ready = False  # Flag: new frame available from background thread
        self._started = False
        
        # Scan directory for images
        self._scan_directory()
        
        if not self._image_paths:
            logger.warning(f"No images found in {image_dir}")
    
    def _scan_directory(self):
        """
        Scan the directory tree for images in depth-first order.
        
        Traverses directories sorted by name and collects all image files.
        Only processes jpg, jpeg, and png files to avoid metadata files.
        """
        supported_extensions = {'.jpg', '.jpeg', '.png'}
        
        def walk_depth_first(directory: Path) -> list[Path]:
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
        
        self._image_paths = walk_depth_first(self.image_dir)
        logger.info(f"Found {len(self._image_paths)} images in {self.image_dir}")
    
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
                        self._current_index = 0  # Loop back to start
                    
                    current_path = self._image_paths[self._current_index]
                    self._current_index += 1
                
                try:
                    # Process the image
                    logger.info(f"📷 Loading image: {current_path.name} (path: {current_path})")
                    processed_image = self._process_image(current_path)
                    
                    # Update current image
                    with self._lock:
                        self._current_image = processed_image
                        self._current_path = str(current_path)
                        self._frame_ready = True  # Flag that new frame is available
                    
                    logger.info(f"✓ Image loaded: {current_path.name}")
                except Exception as e:
                    logger.error(f"Failed to process image {current_path}: {e}")
            
            # Calculate how long to sleep
            frame_elapsed = time.time() - frame_start
            sleep_time = self.frame_duration - frame_elapsed
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)
    
    def is_ready(self) -> bool:
        """Camera is ready only when a new frame is available.
        
        This prevents runner from calling process() until background thread
        has produced a new frame at the configured framerate.
        """
        with self._lock:
            return self._frame_ready
    
    def start(self):
        """Start the camera thread."""
        with self._lock:
            if self._started:
                logger.warning("ImageLibraryCamera already started")
                return
            
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._update_thread, daemon=True)
            self._thread.start()
            self._started = True
            logger.info(f"ImageLibraryCamera started: {len(self._image_paths)} images at {self.framerate} fps")
    
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
    
    def process(self) -> None:
        """Write current frame to output buffers if new frame is available.
        
        Runner calls this on its own schedule. Only writes to buffer when
        background thread has produced a new frame (at framerate).
        """
        with self._lock:
            # Only write if new frame is ready
            if not self._frame_ready:
                return  # No new frame, return early
            
            # Clear flag - this frame will now be written
            self._frame_ready = False
            
            current_file = self._current_path
            current_file = self._current_path
            if self._current_image is None:
                # If no image is loaded yet, load the first one
                if self._image_paths:
                    current_path = self._image_paths[0]
                    try:
                        self._current_image = self._process_image(current_path)
                        self._current_path = str(current_path)
                        current_file = self._current_path
                    except Exception as e:
                        logger.error(f"Failed to load first image {current_path}: {e}")
                        # Create blank image as fallback
                        blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                        self._current_image = Image.fromarray(blank)
                        self._current_path = "blank"
                        current_file = "blank"
                else:
                    # No images available, create a blank image
                    blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    self._current_image = Image.fromarray(blank)
                    self._current_path = "blank"
                    current_file = "blank"
            
            # Copy frame while holding lock
            frame = self._current_image.copy()
        
        # Log which image is being processed
        logger.info(f"🎬 Processing frame from: {Path(current_file).name if current_file else 'unknown'}")
        
        # Create message dict
        message = {}
        
        # Add frame to message
        output_field = self.field("frame")
        message.setdefault(output_field, []).append(frame)
        
        # Only track memory and write to buffers if we have downstream consumers
        if not self.outputs:
            return
        
        # Track memory if collector available
        if self.collector and hasattr(self.collector, '_mem_instrument'):
            mem_inst = self.collector._mem_instrument
            
            # Track PIL Image
            img_id = id(frame)
            try:
                # Estimate size (width * height * channels * bytes_per_pixel)
                size_bytes = frame.width * frame.height * len(frame.getbands())
                self.collector.record("memory.track", size_bytes, attributes={
                    "type": "PIL.Image",
                    "obj_id": str(img_id),
                    "task": self.name
                })
                mem_inst.track_object(frame, img_id)
            except Exception as e:
                logger.debug(f"Failed to track frame memory: {e}")
            
            # Track message dict
            msg_id = id(message)
            self.collector.record("memory.track", 1024, attributes={  # Estimate
                "type": "dict",
                "obj_id": str(msg_id),
                "task": self.name
            })
            mem_inst.track_object(message, msg_id)
        
        # Write to output buffers
        for output_buf in self.outputs.values():
            output_buf.put(message)
    
    def capture(self) -> Image.Image:
        """
        Synchronous capture method for direct image access.
        
        Returns:
            PIL Image object
        """
        with self._lock:
            if self._current_image is None:
                # Load first image if none available
                if self._image_paths:
                    current_path = self._image_paths[0]
                    self._current_image = self._process_image(current_path)
                    self._current_path = str(current_path)
                else:
                    # No images available, create a blank image
                    blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    self._current_image = Image.fromarray(blank)
                    self._current_path = "blank"
            
            return self._current_image.copy()
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, '_stop_event'):
            self.stop()
