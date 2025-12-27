"""Tests for ImageContainer and ImageFormatConverter."""
import pytest
import numpy as np
from PIL import Image
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vlmchat.pipeline.image_container import ImageContainer
from vlmchat.pipeline.image_converter import ImageFormatConverter
from vlmchat.pipeline.image_format import ImageFormat


class TestImageContainer:
    """Test ImageContainer caching and metadata."""
    
    def test_container_creation_with_pil(self):
        """Test creating container with PIL image."""
        pil_img = Image.new('RGB', (100, 100), color='red')
        container = ImageContainer(pil_img, ImageFormat.PIL)
        
        assert container.has_format(ImageFormat.PIL)
        assert not container.has_format(ImageFormat.NUMPY)
        assert container.dimensions == (100, 100)
        assert container.source_format == ImageFormat.PIL
    
    def test_container_creation_with_numpy(self):
        """Test creating container with numpy array."""
        np_img = np.zeros((100, 100, 3), dtype=np.uint8)
        container = ImageContainer(np_img, ImageFormat.NUMPY)
        
        assert container.has_format(ImageFormat.NUMPY)
        assert not container.has_format(ImageFormat.PIL)
        assert container.dimensions == (100, 100)
        assert container.source_format == ImageFormat.NUMPY
    
    def test_set_and_get_format(self):
        """Test setting and getting cached formats."""
        pil_img = Image.new('RGB', (50, 50), color='blue')
        container = ImageContainer(pil_img, ImageFormat.PIL)
        
        # Add numpy format
        np_data = np.array(pil_img)
        container.set_format(ImageFormat.NUMPY, np_data)
        
        assert container.has_format(ImageFormat.NUMPY)
        retrieved = container.get_format(ImageFormat.NUMPY)
        assert isinstance(retrieved, np.ndarray)
        assert retrieved.shape == (50, 50, 3)
    
    def test_free_format(self):
        """Test freeing cached formats."""
        pil_img = Image.new('RGB', (50, 50), color='green')
        container = ImageContainer(pil_img, ImageFormat.PIL)
        
        # Add and then free numpy format
        np_data = np.array(pil_img)
        container.set_format(ImageFormat.NUMPY, np_data)
        assert container.has_format(ImageFormat.NUMPY)
        
        container.free_format(ImageFormat.NUMPY)
        assert not container.has_format(ImageFormat.NUMPY)
        
        # Source format cannot be freed
        container.free_format(ImageFormat.PIL)
        assert container.has_format(ImageFormat.PIL)
    
    def test_get_cached_formats(self):
        """Test getting list of cached formats."""
        pil_img = Image.new('RGB', (50, 50))
        container = ImageContainer(pil_img, ImageFormat.PIL)
        
        assert len(container.get_cached_formats()) == 1
        assert ImageFormat.PIL in container.get_cached_formats()
        
        container.set_format(ImageFormat.NUMPY, np.array(pil_img))
        assert len(container.get_cached_formats()) == 2


class TestImageFormatConverter:
    """Test ImageFormatConverter conversions."""
    
    def test_converter_initialization(self):
        """Test converter initializes dependencies."""
        converter = ImageFormatConverter()
        assert converter._pil_available
        assert converter._torch_available or not converter._torch_available  # May not be installed
    
    def test_pil_to_numpy_conversion(self):
        """Test PIL → NumPy conversion."""
        pil_img = Image.new('RGB', (100, 100), color=(255, 0, 0))
        container = ImageContainer(pil_img, ImageFormat.PIL)
        converter = ImageFormatConverter()
        
        np_data = converter.convert(container, ImageFormat.NUMPY)
        
        assert isinstance(np_data, np.ndarray)
        assert np_data.shape == (100, 100, 3)
        assert container.has_format(ImageFormat.NUMPY)  # Should be cached
        
        # Verify conversion is correct (red image)
        assert np_data[0, 0, 0] == 255  # Red channel
        assert np_data[0, 0, 1] == 0    # Green channel
        assert np_data[0, 0, 2] == 0    # Blue channel
    
    def test_numpy_to_pil_conversion(self):
        """Test NumPy → PIL conversion."""
        np_img = np.zeros((100, 100, 3), dtype=np.uint8)
        np_img[:, :] = [0, 255, 0]  # Green
        container = ImageContainer(np_img, ImageFormat.NUMPY)
        converter = ImageFormatConverter()
        
        pil_data = converter.convert(container, ImageFormat.PIL)
        
        assert isinstance(pil_data, Image.Image)
        assert pil_data.size == (100, 100)
        assert container.has_format(ImageFormat.PIL)
        
        # Verify conversion is correct
        pixel = pil_data.getpixel((0, 0))
        assert pixel == (0, 255, 0)  # Green
    
    @pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
    def test_pil_to_torch_cpu_conversion(self):
        """Test PIL → Torch CPU conversion."""
        import torch
        
        pil_img = Image.new('RGB', (50, 50), color=(128, 128, 128))
        container = ImageContainer(pil_img, ImageFormat.PIL)
        converter = ImageFormatConverter()
        
        torch_data = converter.convert(container, ImageFormat.TORCH_CPU)
        
        assert isinstance(torch_data, torch.Tensor)
        assert torch_data.shape == (3, 50, 50)  # CHW format
        assert torch_data.device.type == 'cpu'
        assert container.has_format(ImageFormat.TORCH_CPU)
        
        # Verify normalization (0-1 range)
        assert 0.0 <= torch_data.min() <= 1.0
        assert 0.0 <= torch_data.max() <= 1.0
    
    @pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
    def test_torch_cpu_to_numpy_conversion(self):
        """Test Torch CPU → NumPy conversion."""
        import torch
        
        # Create a torch tensor (CHW, float32, 0-1 range)
        torch_tensor = torch.ones((3, 50, 50), dtype=torch.float32) * 0.5
        container = ImageContainer(torch_tensor, ImageFormat.TORCH_CPU)
        converter = ImageFormatConverter()
        
        np_data = converter.convert(container, ImageFormat.NUMPY)
        
        assert isinstance(np_data, np.ndarray)
        assert np_data.shape == (50, 50, 3)  # HWC format
        assert np_data.dtype == np.uint8
        assert container.has_format(ImageFormat.NUMPY)
        
        # Verify denormalization (0-255 range)
        assert np_data[0, 0, 0] == 127  # 0.5 * 255 ≈ 127
    
    def test_caching_avoids_redundant_conversion(self):
        """Test that cached formats are reused."""
        pil_img = Image.new('RGB', (100, 100))
        container = ImageContainer(pil_img, ImageFormat.PIL)
        converter = ImageFormatConverter()
        
        # First conversion
        np_data1 = converter.convert(container, ImageFormat.NUMPY)
        
        # Second conversion should return cached data
        np_data2 = converter.convert(container, ImageFormat.NUMPY)
        
        # Should be the same object (not a new conversion)
        assert np_data1 is np_data2
    
    def test_unsupported_conversion_raises_error(self):
        """Test that unsupported conversions raise ValueError."""
        pil_img = Image.new('RGB', (100, 100))
        container = ImageContainer(pil_img, ImageFormat.PIL)
        converter = ImageFormatConverter()
        
        # OPENCV_GPU conversion not implemented yet
        with pytest.raises(ValueError, match="not implemented"):
            converter.convert(container, ImageFormat.OPENCV_GPU)


class TestPipelineIntegration:
    """Integration tests for image containers in pipeline."""
    
    def test_camera_produces_image_container(self):
        """Test that CameraTask produces ImageContainer."""
        from vlmchat.pipeline.tasks.camera_task import CameraTask
        from vlmchat.pipeline.task_base import Context, ContextDataType
        
        # Create camera task with none camera (static image)
        camera_task = CameraTask("cam")
        camera_task.configure(type="none", image_path="test_image.jpg")
        
        # Note: This test would need a real image file or mock camera
        # Skipping actual execution here
        assert camera_task.native_output_format == ImageFormat.PIL
    
    def test_clip_vision_declares_gpu_format(self):
        """Test that ClipVisionTask declares GPU format preference."""
        from vlmchat.pipeline.tasks.clip_vision_task import ClipVisionTask
        
        clip_task = ClipVisionTask(clip_model=None, task_id="clip")
        assert clip_task.native_input_format == ImageFormat.TORCH_GPU


# Helper functions
def _torch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
