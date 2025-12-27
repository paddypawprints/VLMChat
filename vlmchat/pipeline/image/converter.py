"""Central image format conversion service."""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
from .formats import ImageFormat
import logging

if TYPE_CHECKING:
    from ..cache.image import ImageContainer

logger = logging.getLogger(__name__)


class ImageFormatConverter:
    """
    Central service for converting between image formats.
    
    This is the ONLY place where format conversions happen.
    """
    
    def __init__(self):
        """Initialize converter (loads conversion dependencies lazily)."""
        self._pil_available = False
        self._torch_available = False
        self._cv2_available = False
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check which conversion libraries are available."""
        try:
            import PIL
            self._pil_available = True
        except ImportError:
            pass
        
        try:
            import torch
            self._torch_available = True
        except ImportError:
            pass
        
        try:
            import cv2
            self._cv2_available = True
        except ImportError:
            pass
    
    def convert(self, container: ImageContainer, target_format: ImageFormat) -> Any:
        """
        Convert container to target format (uses cache if available).
        
        Args:
            container: ImageContainer to convert
            target_format: Desired output format
        
        Returns:
            Image data in target format
        
        Raises:
            ValueError: If conversion not supported
        """
        # Check cache first
        if container.has_format(target_format):
            logger.debug(f"Using cached {target_format.value}")
            return container.get_format(target_format)
        
        # Find source format to convert from (prefer formats already cached)
        source_format = self._select_source_format(container, target_format)
        source_data = container.get_format(source_format)
        
        # Perform conversion
        converted = self._convert_direct(source_data, source_format, target_format)
        
        # Cache result
        container.set_format(target_format, converted)
        logger.debug(f"Converted {source_format.value} → {target_format.value}")
        
        return converted
    
    def _select_source_format(self, container: ImageContainer, target: ImageFormat) -> ImageFormat:
        """Select best source format for conversion (minimize conversion steps)."""
        cached = container.get_cached_formats()
        
        # Prefer GPU→GPU, CPU→CPU to avoid transfers
        if target in (ImageFormat.TORCH_GPU, ImageFormat.OPENCV_GPU):
            if ImageFormat.TORCH_GPU in cached:
                return ImageFormat.TORCH_GPU
            if ImageFormat.OPENCV_GPU in cached:
                return ImageFormat.OPENCV_GPU
        
        if target in (ImageFormat.PIL, ImageFormat.NUMPY, ImageFormat.TORCH_CPU):
            if ImageFormat.TORCH_CPU in cached:
                return ImageFormat.TORCH_CPU
            if ImageFormat.NUMPY in cached:
                return ImageFormat.NUMPY
            if ImageFormat.PIL in cached:
                return ImageFormat.PIL
        
        # Fall back to source format
        return container.source_format
    
    def _convert_direct(self, data: Any, from_fmt: ImageFormat, to_fmt: ImageFormat) -> Any:
        """Perform actual conversion between formats."""
        # PIL conversions
        if from_fmt == ImageFormat.PIL:
            if to_fmt == ImageFormat.NUMPY:
                import numpy as np
                return np.array(data)
            elif to_fmt == ImageFormat.TORCH_CPU:
                import torch
                import numpy as np
                arr = np.array(data)
                tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                return tensor
            elif to_fmt == ImageFormat.TORCH_GPU:
                import torch
                import numpy as np
                arr = np.array(data)
                tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                return tensor.cuda()
        
        # NumPy conversions
        elif from_fmt == ImageFormat.NUMPY:
            if to_fmt == ImageFormat.PIL:
                from PIL import Image
                return Image.fromarray(data)
            elif to_fmt == ImageFormat.TORCH_CPU:
                import torch
                tensor = torch.from_numpy(data).permute(2, 0, 1).float() / 255.0
                return tensor
            elif to_fmt == ImageFormat.TORCH_GPU:
                import torch
                tensor = torch.from_numpy(data).permute(2, 0, 1).float() / 255.0
                return tensor.cuda()
        
        # Torch CPU conversions
        elif from_fmt == ImageFormat.TORCH_CPU:
            if to_fmt == ImageFormat.NUMPY:
                arr = (data.permute(1, 2, 0) * 255.0).byte().numpy()
                return arr
            elif to_fmt == ImageFormat.PIL:
                from PIL import Image
                arr = (data.permute(1, 2, 0) * 255.0).byte().numpy()
                return Image.fromarray(arr)
            elif to_fmt == ImageFormat.TORCH_GPU:
                return data.cuda()
        
        # Torch GPU conversions
        elif from_fmt == ImageFormat.TORCH_GPU:
            if to_fmt == ImageFormat.TORCH_CPU:
                return data.cpu()
            elif to_fmt == ImageFormat.NUMPY:
                arr = (data.cpu().permute(1, 2, 0) * 255.0).byte().numpy()
                return arr
            elif to_fmt == ImageFormat.PIL:
                from PIL import Image
                arr = (data.cpu().permute(1, 2, 0) * 255.0).byte().numpy()
                return Image.fromarray(arr)
        
        raise ValueError(f"Conversion {from_fmt.value} → {to_fmt.value} not implemented")
