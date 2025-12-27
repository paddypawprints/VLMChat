"""
OpenCLIP backend implementation for FashionClip.

Provides FashionClip runtime using the open_clip library with Marqo's FashionSigLIP model.
"""

import logging
import torch
from typing import List, TYPE_CHECKING
from PIL import Image

# --- Third-party imports ---
try:
    import open_clip  # type: ignore[import-untyped]
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    open_clip = None  # type: ignore[assignment]
# --- End third-party imports ---

from utils.config import VLMChatConfig

# Import FashionClipRuntimeBase
if TYPE_CHECKING:
    from models.FashionClip.fashion_clip_model import FashionClipRuntimeBase
else:
    from models.FashionClip.fashion_clip_model import FashionClipRuntimeBase

logger = logging.getLogger(__name__)


class OpenClipBackend(FashionClipRuntimeBase):
    """
    FashionClip runtime implementation using the 'open_clip' library with Marqo's FashionSigLIP model.
    """
    
    def __init__(self, config: VLMChatConfig):
        super().__init__(config)
        self.model = None
        self.preprocess_val = None
        self.tokenizer = None
        self._is_ready = False
        
        if not OPEN_CLIP_AVAILABLE:
            logger.error("OpenClipBackend Error: 'open_clip' library not installed.")
            return

        try:
            model_name = config.model.fashion_clip_model_name
            pretrained = config.model.fashion_clip_pretrained if config.model.fashion_clip_pretrained else None
            model_kwargs = config.model.fashion_clip_model_kwargs if config.model.fashion_clip_model_kwargs else {}
            
            logger.info(f"Loading FashionClip model: {model_name}")
            self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(  # type: ignore[union-attr]
                model_name,
                pretrained=pretrained,
                **model_kwargs
            )
            
            self.tokenizer = open_clip.get_tokenizer(model_name)  # type: ignore[union-attr]
            
            # Model needs to be in eval mode
            self.model.eval()
            
            self._is_ready = True
            logger.info("OpenClipBackend loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load FashionClip model: {e}", exc_info=True)
            self._is_ready = False

    @property
    def native_image_format(self) -> str:
        """FashionClip OpenCLIP backend prefers PIL images."""
        return "pil"

    @property
    def is_available(self) -> bool:
        """Returns True if the model loaded successfully."""
        return self._is_ready and self.model is not None

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encodes a single PIL Image into normalized feature tensor.
        
        Args:
            image: PIL Image to encode
            
        Returns:
            Normalized image feature tensor with shape (1, feature_dim)
        """
        if not self.is_available or self.preprocess_val is None or self.model is None:
            raise RuntimeError("OpenClipBackend is not ready.")
            
        preprocessed_image = self.preprocess_val(image.convert("RGB")).unsqueeze(0)  # type: ignore[union-attr]
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(preprocessed_image, normalize=True)  # type: ignore[union-attr]
            
        return image_features

    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """
        Encodes a list of text strings into normalized feature tensor.
        
        Args:
            text_prompts: List of text strings to encode
            
        Returns:
            Normalized text feature tensor with shape (len(text_prompts), feature_dim)
        """
        if not self.is_available or self.tokenizer is None or self.model is None:
            raise RuntimeError("OpenClipBackend is not ready.")
            
        text = self.tokenizer(text_prompts)  # type: ignore[misc]
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.model.encode_text(text, normalize=True)  # type: ignore[union-attr]
            
        return text_features
