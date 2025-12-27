"""Embedding container with multi-format caching (numpy, torch)."""
from typing import Optional, Dict, Any, List
from .item import CachedItem
import logging

logger = logging.getLogger(__name__)


class EmbeddingContainer(CachedItem):
    """
    Container for embeddings with format conversion support.
    
    Supports numpy, torch_cpu, and torch_gpu formats.
    Converts on-demand and caches results.
    """
    
    def __init__(self, cache_key: str, source_data: Any, source_format: str = 'numpy'):
        """
        Create container with source embedding data.
        
        Args:
            cache_key: Unique cache identifier
            source_data: Embedding data (numpy array or torch tensor)
            source_format: Format of source_data ('numpy', 'torch_cpu', 'torch_gpu')
        """
        super().__init__(cache_key)
        self._formats: Dict[str, Any] = {source_format: source_data}
        self.source_format = source_format
        self._shape = self._extract_shape(source_data, source_format)
    
    def _extract_shape(self, data: Any, fmt: str) -> tuple:
        """Extract shape from data."""
        try:
            if fmt == 'numpy':
                return data.shape
            elif fmt.startswith('torch'):
                return tuple(data.shape)
        except:
            return None
    
    def has_format(self, format: str) -> bool:
        """Check if format is cached."""
        return format in self._formats
    
    def get(self, format: Optional[str] = None) -> Any:
        """
        Get embedding in specified format.
        
        Args:
            format: Desired format ('numpy', 'torch_cpu', 'torch_gpu')
                   None returns source format
        
        Returns:
            Embedding data in requested format
        """
        fmt = format or self.source_format
        
        if fmt in self._formats:
            return self._formats[fmt]
        
        # Convert on-demand
        converted = self._convert_format(fmt)
        self._formats[fmt] = converted
        return converted
    
    def _convert_format(self, target: str) -> Any:
        """Convert embedding format."""
        source_data = self._formats[self.source_format]
        
        if self.source_format == 'numpy' and target.startswith('torch'):
            import torch
            tensor = torch.from_numpy(source_data)
            if target == 'torch_gpu':
                return tensor.cuda()
            return tensor
        
        elif self.source_format.startswith('torch') and target == 'numpy':
            return source_data.cpu().numpy()
        
        elif self.source_format == 'torch_cpu' and target == 'torch_gpu':
            return source_data.cuda()
        
        elif self.source_format == 'torch_gpu' and target == 'torch_cpu':
            return source_data.cpu()
        
        else:
            raise ValueError(f"Cannot convert {self.source_format} -> {target}")
    
    def get_cached_formats(self) -> List[str]:
        """Get list of currently cached formats."""
        return list(self._formats.keys())
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get embedding metadata without loading data."""
        return {
            'shape': self._shape,
            'source_format': self.source_format,
            'cached_formats': list(self._formats.keys())
        }
    
    def free_format(self, fmt: str) -> None:
        """Free cached format to save memory (especially GPU)."""
        if fmt in self._formats and fmt != self.source_format:
            del self._formats[fmt]
            logger.debug(f"Freed format {fmt}")
    
    @property
    def shape(self) -> tuple:
        """Get embedding shape."""
        return self._shape
    
    def __repr__(self) -> str:
        formats = ', '.join(self._formats.keys())
        return f"EmbeddingContainer({self._cache_key[:8]}..., shape={self._shape}, formats=[{formats}])"
