"""Image container with multi-format caching."""
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from ..image.formats import ImageFormat
from .item import CachedItem
import logging
from readerwriterlock import rwlock

if TYPE_CHECKING:
    from ..sources.jetson_camera import PooledBuffer

logger = logging.getLogger(__name__)


class ImageContainer(CachedItem):
    """
    Container for an image with multiple cached format representations.
    
    Stores source image and caches conversions on-demand. Runtime (PipelineRunner)
    decides when to convert and when to free formats.
    
    Supports two modes:
    - Owned mode: Container owns the image data (normal operation)
    - Pooled mode: Container references pooled buffer (zero-copy, Jetson optimization)
    """
    
    def __init__(self, cache_key: str, source_data: Any = None, source_format: ImageFormat = ImageFormat.NUMPY,
                 pooled_buffer: Optional['PooledBuffer'] = None):
        """
        Create container with source image data.
        
        Args:
            cache_key: Unique cache identifier
            source_data: Image data in source format (if owned mode)
            source_format: Format of source_data
            pooled_buffer: Optional pooled buffer reference (if pooled mode)
        """
        super().__init__(cache_key)
        
        # Pooled buffer support (zero-copy mode)
        self._pooled: Optional['PooledBuffer'] = pooled_buffer
        self._rwlock = rwlock.RWLockWrite()  # Writer-preference reader-writer lock
        
        if pooled_buffer:
            # Pooled mode - reference buffer, no owned data yet
            self._formats: Dict[ImageFormat, Any] = {}
            self.source_format = ImageFormat.NUMPY  # Pool buffers are always numpy
            pooled_buffer.add_ref(self)
            self._extract_dimensions(pooled_buffer.data, ImageFormat.NUMPY)
        elif source_data is not None:
            # Owned mode - store data directly
            self._formats: Dict[ImageFormat, Any] = {source_format: source_data}
            self.source_format = source_format
            self._extract_dimensions(source_data, source_format)
        else:
            # No source data (e.g., Detection subclass will populate lazily)
            self._formats: Dict[ImageFormat, Any] = {}
            self.source_format = source_format
        
        self._width: Optional[int] = None
        self._height: Optional[int] = None
    
    def _extract_dimensions(self, data: Any, fmt: ImageFormat) -> None:
        """Extract width/height from data for metadata."""
        try:
            if fmt == ImageFormat.PIL:
                self._width, self._height = data.size
            elif fmt == ImageFormat.NUMPY:
                self._height, self._width = data.shape[:2]
            elif fmt in (ImageFormat.TORCH_CPU, ImageFormat.TORCH_GPU):
                self._height, self._width = data.shape[1:3]
        except:
            pass  # Dimensions unknown
    
    def has_format(self, format: ImageFormat) -> bool:
        """Check if format is cached."""
        # Optimistic read - dict lookup is atomic in Python
        return format in self._formats
    
    def get(self, format: Optional[ImageFormat] = None) -> Any:
        """
        Get image in specified format (CachedItem interface).
        Uses optimistic locking for performance.
        
        Args:
            format: Desired ImageFormat (None = source format)
        
        Returns:
            Image data in requested format
        """
        fmt = format or self.source_format
        
        # Optimistic read - check if cached (no lock)
        if fmt in self._formats:
            return self._formats[fmt]
        
        # Not cached - check pooled buffer with read lock
        if self._pooled and fmt == ImageFormat.NUMPY:
            with self._rwlock.gen_rlock():
                if not self._pooled.valid:
                    raise RuntimeError("Pooled buffer was recycled - frame no longer available")
                return self._pooled.data
        
        # Need to convert - call conversion method
        return self._convert_format(fmt)
    
    def get_format(self, fmt: ImageFormat) -> Optional[Any]:
        """Get cached format data (None if not cached) - legacy method."""
        return self._formats.get(fmt)
    
    def set_format(self, fmt: ImageFormat, data: Any) -> None:
        """Cache converted format data."""
        self._formats[fmt] = data
        if self._width is None:
            self._extract_dimensions(data, fmt)
    
    def free_format(self, fmt: ImageFormat) -> None:
        """Free cached format to save memory."""
        if fmt in self._formats and fmt != self.source_format:
            del self._formats[fmt]
            logger.debug(f"Freed format {fmt.value}")
    
    def get_cached_formats(self) -> List[str]:
        """Get list of currently cached formats (CachedItem interface)."""
        return [f.value for f in self._formats.keys()]
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get image metadata without loading data."""
        return {
            'width': self._width,
            'height': self._height,
            'source_format': self.source_format.value,
            'cached_formats': [f.value for f in self._formats.keys()]
        }
    
    def _convert_format(self, target: ImageFormat) -> Any:
        """
        Convert from available format to target format.
        Thread-safe with reader-writer lock and optimistic pattern.
        """
        from ..image.converter import ImageFormatConverter
        converter = ImageFormatConverter()
        
        # Step 1: Read source data with read lock
        with self._rwlock.gen_rlock():
            # Double-check cache (another thread may have done conversion)
            if target in self._formats:
                return self._formats[target]
            
            if self._pooled:
                if not self._pooled.valid:
                    raise RuntimeError("Pooled buffer was recycled during conversion")
                source_data = self._pooled.data
                source_fmt = ImageFormat.NUMPY
            else:
                # Get source format data
                source_data = self._formats[self.source_format]
                source_fmt = self.source_format
        
        # Step 2: Perform conversion (outside lock, working on data copy)
        converted = converter._convert_direct(source_data, source_fmt, target)
        logger.debug(f"Converted {source_fmt} -> {target}: {type(converted)}")
        
        # Step 3: Write result with write lock
        with self._rwlock.gen_wlock():
            # Triple-check (another thread may have just done this)
            if target in self._formats:
                return self._formats[target]
            self._formats[target] = converted
            return converted
    
    def _promote_to_owned(self) -> None:
        """
        Promote from pooled mode to owned mode (automatic on pool eviction).
        
        Called by BufferPool when it needs to reclaim a buffer. Copies the
        buffer data so this container becomes independent of the pool.
        """
        with self._rwlock.gen_wlock():
            if not self._pooled:
                return  # Already owned
            
            if not self._pooled.valid:
                # Buffer already recycled - too late
                logger.warning("Cannot promote - pooled buffer already recycled")
                return
            
            # Copy data from pool buffer
            owned_data = self._pooled.data.copy()
            
            # Switch to owned mode
            self._formats[ImageFormat.NUMPY] = owned_data
            self._pooled.remove_ref(self)
            self._pooled = None
    
    def is_pooled(self) -> bool:
        """Check if container is in pooled mode (zero-copy)."""
        # Simple check - optimistic read
        return self._pooled is not None
    
    @property
    def dimensions(self) -> tuple:
        """Get (width, height) if known."""
        return (self._width, self._height) if self._width else (None, None)
    
    def __repr__(self) -> str:
        with self._rwlock.gen_rlock():
            formats = [f.value for f in self._formats.keys()]
            dims = f"{self._width}x{self._height}" if self._width else "unknown"
            mode = "pooled" if self._pooled else "owned"
            return f"ImageContainer({dims}, mode={mode}, cached={formats})"
    
    def __del__(self):
        """Release pooled buffer reference on container destruction."""
        if self._pooled:
            try:
                self._pooled.remove_ref(self)
            except:
                pass  # Ignore errors during cleanup
