"""
Unit tests for ImageLibraryCamera.

These tests validate the ImageLibraryCamera functionality including:
- Directory traversal in depth-first order
- Image processing (resize, crop, fill)
- Frame rate timing
- Metrics tracking
- Thread management
"""

import os
import sys
import tempfile
import shutil
import time
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

# Ensure repo root and src are on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from utils.image_library_camera import ImageLibraryCamera
from utils.camera_base import CameraModel, Platform, Device
from utils.camera_factory import CameraFactory
from utils.metrics_collector import Collector, ValueType


class TestImageLibraryCamera:
    """Test suite for ImageLibraryCamera."""
    
    @pytest.fixture
    def temp_image_dir(self):
        """Create a temporary directory with test images."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def temp_save_dir(self):
        """Create a temporary directory for saved images."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def create_test_image(self, path: Path, width: int = 10, height: int = 10, color=(255, 0, 0)):
        """Create a simple test image."""
        img_array = np.full((height, width, 3), color, dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(str(path))
    
    def test_initialization(self, temp_image_dir, temp_save_dir):
        """Test camera initialization."""
        # Create a test image
        img_path = Path(temp_image_dir) / "test.jpg"
        self.create_test_image(img_path)
        
        camera = ImageLibraryCamera(
            image_base_dir=temp_image_dir,
            width=100,
            height=100,
            framerate=5,
            save_dir=temp_save_dir
        )
        
        assert camera.model == CameraModel.IMAGE_LIBRARY
        assert camera.width == 100
        assert camera.height == 100
        assert camera.framerate == 5
        assert camera.save_path == temp_save_dir
        assert len(camera._image_paths) == 1
    
    def test_invalid_directory(self, temp_save_dir):
        """Test initialization with invalid directory."""
        with pytest.raises(ValueError, match="does not exist"):
            ImageLibraryCamera(
                image_base_dir="/nonexistent/directory",
                save_dir=temp_save_dir
            )
    
    def test_directory_traversal_depth_first(self, temp_image_dir, temp_save_dir):
        """Test that directory traversal is depth-first and sorted."""
        # Create directory structure:
        # temp_image_dir/
        #   ├── a_dir/
        #   │   ├── a1.jpg
        #   │   └── b_subdir/
        #   │       └── b1.jpg
        #   ├── b_dir/
        #   │   └── c1.jpg
        #   └── root.jpg
        
        base = Path(temp_image_dir)
        
        # Create directories
        a_dir = base / "a_dir"
        a_dir.mkdir()
        b_subdir = a_dir / "b_subdir"
        b_subdir.mkdir()
        b_dir = base / "b_dir"
        b_dir.mkdir()
        
        # Create images
        self.create_test_image(base / "root.jpg", color=(255, 0, 0))
        self.create_test_image(a_dir / "a1.jpg", color=(0, 255, 0))
        self.create_test_image(b_subdir / "b1.jpg", color=(0, 0, 255))
        self.create_test_image(b_dir / "c1.jpg", color=(255, 255, 0))
        
        camera = ImageLibraryCamera(
            image_base_dir=temp_image_dir,
            save_dir=temp_save_dir
        )
        
        # Verify images are found
        assert len(camera._image_paths) == 4
        
        # Verify depth-first, sorted order
        # Expected: root.jpg, a_dir/a1.jpg, a_dir/b_subdir/b1.jpg, b_dir/c1.jpg
        paths_str = [str(p.relative_to(base)) for p in camera._image_paths]
        
        assert paths_str[0] == "root.jpg"
        assert paths_str[1] == str(Path("a_dir") / "a1.jpg")
        assert paths_str[2] == str(Path("a_dir") / "b_subdir" / "b1.jpg")
        assert paths_str[3] == str(Path("b_dir") / "c1.jpg")
    
    def test_image_resize_larger(self, temp_image_dir, temp_save_dir):
        """Test image resize when source is smaller than target."""
        # Create a 10x10 image
        img_path = Path(temp_image_dir) / "small.jpg"
        self.create_test_image(img_path, width=10, height=10)
        
        camera = ImageLibraryCamera(
            image_base_dir=temp_image_dir,
            width=100,
            height=100,
            save_dir=temp_save_dir
        )
        
        # Process the image
        processed = camera._process_image(camera._image_paths[0])
        
        # Verify dimensions
        assert processed.size == (100, 100)
    
    def test_image_resize_smaller(self, temp_image_dir, temp_save_dir):
        """Test image resize when source is larger than target."""
        # Create a 200x200 image
        img_path = Path(temp_image_dir) / "large.jpg"
        self.create_test_image(img_path, width=200, height=200)
        
        camera = ImageLibraryCamera(
            image_base_dir=temp_image_dir,
            width=100,
            height=100,
            save_dir=temp_save_dir
        )
        
        # Process the image
        processed = camera._process_image(camera._image_paths[0])
        
        # Verify dimensions
        assert processed.size == (100, 100)
    
    def test_image_crop(self, temp_image_dir, temp_save_dir):
        """Test image cropping when aspect ratio doesn't match."""
        # Create a 200x100 image (wide)
        img_path = Path(temp_image_dir) / "wide.jpg"
        self.create_test_image(img_path, width=200, height=100)
        
        camera = ImageLibraryCamera(
            image_base_dir=temp_image_dir,
            width=100,
            height=100,
            save_dir=temp_save_dir
        )
        
        # Process the image
        processed = camera._process_image(camera._image_paths[0])
        
        # Verify dimensions
        assert processed.size == (100, 100)
    
    def test_image_fill(self, temp_image_dir, temp_save_dir):
        """Test image padding when source is smaller."""
        # Create a 50x50 image
        img_path = Path(temp_image_dir) / "tiny.jpg"
        self.create_test_image(img_path, width=50, height=50, color=(255, 255, 255))
        
        camera = ImageLibraryCamera(
            image_base_dir=temp_image_dir,
            width=100,
            height=100,
            save_dir=temp_save_dir
        )
        
        # Process the image
        processed = camera._process_image(camera._image_paths[0])
        
        # Verify dimensions
        assert processed.size == (100, 100)
        
        # Verify that image has been padded
        # The white image should be in the center, with black padding around it
        img_array = np.array(processed)
        # Check that center area is white
        center_y, center_x = 50, 50
        assert img_array[center_y, center_x].tolist() == [255, 255, 255]
    
    def test_capture_single_image_without_start(self, temp_image_dir, temp_save_dir):
        """Test capturing image without starting the thread."""
        img_path = Path(temp_image_dir) / "test.jpg"
        self.create_test_image(img_path)
        
        camera = ImageLibraryCamera(
            image_base_dir=temp_image_dir,
            save_dir=temp_save_dir
        )
        
        # Capture without starting thread
        path, image = camera.capture_single_image()
        
        assert isinstance(path, str)
        assert isinstance(image, Image.Image)
    
    def test_capture_single_image_with_thread(self, temp_image_dir, temp_save_dir):
        """Test capturing images with thread running."""
        # Create multiple test images
        for i in range(3):
            img_path = Path(temp_image_dir) / f"test_{i}.jpg"
            self.create_test_image(img_path)
        
        camera = ImageLibraryCamera(
            image_base_dir=temp_image_dir,
            framerate=10,  # Fast framerate for testing
            save_dir=temp_save_dir
        )
        
        try:
            camera.start()
            
            # Give thread time to load first image
            time.sleep(0.15)
            
            # Capture should return current image
            path1, image1 = camera.capture_single_image()
            assert isinstance(path1, str)
            assert isinstance(image1, Image.Image)
            
            # Wait for next frame
            time.sleep(0.15)
            
            # Should get a different image (or same if looped)
            path2, image2 = camera.capture_single_image()
            assert isinstance(path2, str)
            
        finally:
            camera.stop()
    
    def test_thread_start_stop(self, temp_image_dir, temp_save_dir):
        """Test thread start and stop functionality."""
        img_path = Path(temp_image_dir) / "test.jpg"
        self.create_test_image(img_path)
        
        camera = ImageLibraryCamera(
            image_base_dir=temp_image_dir,
            save_dir=temp_save_dir
        )
        
        # Start thread
        camera.start()
        assert camera._started is True
        assert camera._thread is not None
        
        # Stop thread
        camera.stop()
        assert camera._started is False
    
    def test_image_looping(self, temp_image_dir, temp_save_dir):
        """Test that images loop back to the start."""
        # Create 2 test images
        for i in range(2):
            img_path = Path(temp_image_dir) / f"test_{i}.jpg"
            self.create_test_image(img_path, color=(i * 100, 0, 0))
        
        camera = ImageLibraryCamera(
            image_base_dir=temp_image_dir,
            framerate=20,  # Very fast for testing
            save_dir=temp_save_dir
        )
        
        try:
            camera.start()
            
            # Give enough time to loop through images
            time.sleep(0.4)
            
            # Camera should have looped (index reset)
            # This is a basic check - the exact behavior depends on timing
            assert camera._current_index <= len(camera._image_paths)
            
        finally:
            camera.stop()
    
    def test_metrics_collection(self, temp_image_dir, temp_save_dir):
        """Test that metrics are collected properly."""
        # Create test images of different sizes
        img_path1 = Path(temp_image_dir) / "small.jpg"
        self.create_test_image(img_path1, width=50, height=50)
        
        img_path2 = Path(temp_image_dir) / "large.jpg"
        self.create_test_image(img_path2, width=200, height=200)
        
        # Create metrics collector
        collector = Collector()
        
        camera = ImageLibraryCamera(
            image_base_dir=temp_image_dir,
            width=100,
            height=100,
            save_dir=temp_save_dir,
            metrics_collector=collector
        )
        
        # Process images to trigger metrics
        camera._process_image(camera._image_paths[0])
        camera._process_image(camera._image_paths[1])
        
        # Verify metrics were recorded
        # Check that operations timeseries has datapoints
        ts_data = collector.snapshot_timeseries("image_library_operations")
        assert len(ts_data) > 0
    
    def test_factory_creation(self, temp_image_dir, temp_save_dir):
        """Test creating ImageLibraryCamera through factory."""
        img_path = Path(temp_image_dir) / "test.jpg"
        self.create_test_image(img_path)
        
        args = {
            'image_base_dir': temp_image_dir,
            'width': 100,
            'height': 100,
            'framerate': 5,
            'save_dir': temp_save_dir,
        }
        
        camera = CameraFactory.create_camera(
            model=CameraModel.IMAGE_LIBRARY,
            platform=Platform.RPI,
            device=Device.CAMERA0,
            args=args
        )
        
        assert isinstance(camera, ImageLibraryCamera)
        assert camera.model == CameraModel.IMAGE_LIBRARY
    
    def test_factory_missing_args(self):
        """Test factory creation with missing required args."""
        with pytest.raises(ValueError, match="requires 'args'"):
            CameraFactory.create_camera(
                model=CameraModel.IMAGE_LIBRARY,
                platform=Platform.RPI,
            )
    
    def test_factory_missing_image_base_dir(self, temp_save_dir):
        """Test factory creation with missing image_base_dir."""
        args = {
            'width': 100,
            'save_dir': temp_save_dir,
        }
        
        with pytest.raises(ValueError, match="requires 'image_base_dir'"):
            CameraFactory.create_camera(
                model=CameraModel.IMAGE_LIBRARY,
                platform=Platform.RPI,
                args=args
            )
    
    def test_no_images_in_directory(self, temp_image_dir, temp_save_dir):
        """Test behavior when directory has no images."""
        # Don't create any images
        camera = ImageLibraryCamera(
            image_base_dir=temp_image_dir,
            save_dir=temp_save_dir
        )
        
        assert len(camera._image_paths) == 0
        
        # Capture should return blank image
        path, image = camera.capture_single_image()
        assert path == "blank"
        assert isinstance(image, Image.Image)
    
    def test_supported_image_formats(self, temp_image_dir, temp_save_dir):
        """Test that various image formats are supported."""
        # Create images with different extensions
        formats = ['jpg', 'png', 'bmp']
        
        for fmt in formats:
            img_path = Path(temp_image_dir) / f"test.{fmt}"
            self.create_test_image(img_path)
        
        camera = ImageLibraryCamera(
            image_base_dir=temp_image_dir,
            save_dir=temp_save_dir
        )
        
        # Should find all 3 images
        assert len(camera._image_paths) >= 3
    
    def test_frame_rate_timing(self, temp_image_dir, temp_save_dir):
        """Test that frame rate timing is approximately correct."""
        # Create test images
        for i in range(3):
            img_path = Path(temp_image_dir) / f"test_{i}.jpg"
            self.create_test_image(img_path)
        
        framerate = 10
        camera = ImageLibraryCamera(
            image_base_dir=temp_image_dir,
            framerate=framerate,
            save_dir=temp_save_dir
        )
        
        # Expected frame duration
        expected_duration = 1.0 / framerate
        assert abs(camera.frame_duration - expected_duration) < 0.001
    
    def test_1x1_image(self, temp_image_dir, temp_save_dir):
        """Test with minimal 1x1 pixel images as specified in requirements."""
        # Create 1x1 test images
        for i in range(5):
            img_path = Path(temp_image_dir) / f"tiny_{i}.jpg"
            self.create_test_image(img_path, width=1, height=1)
        
        camera = ImageLibraryCamera(
            image_base_dir=temp_image_dir,
            width=10,
            height=10,
            framerate=5,
            save_dir=temp_save_dir
        )
        
        try:
            camera.start()
            time.sleep(0.3)
            
            # Should be able to capture
            path, image = camera.capture_single_image()
            assert isinstance(image, Image.Image)
            assert image.size == (10, 10)
            
        finally:
            camera.stop()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
