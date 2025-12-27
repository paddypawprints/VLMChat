"""
CLIP Text Encoder Task

Encodes text prompts using CLIP text encoder and stores embeddings in context.

This task is reusable and can be used for:
- Generating text embeddings for comparison
- Prompt similarity analysis
- Zero-shot classification
- Text-to-image matching
"""

import numpy as np
import logging
from typing import List, Optional

from ..task_base import BaseTask, Context, ContextDataType, register_task

logger = logging.getLogger(__name__)


@register_task('clip_text_encoder')
class ClipTextEncoderTask(BaseTask):
    """
    Encodes text prompts using CLIP text encoder.
    
    This task:
    1. Takes a list of text prompts (from config or context)
    2. Encodes them using CLIP text encoder
    3. Normalizes embeddings to unit vectors
    4. Stores results in context for downstream tasks
    
    Output stored in ContextDataType.EMBEDDINGS as list of [label, embedding] pairs:
    [
        ['red shirt', np.ndarray],
        ['blue pants', np.ndarray],
        ...
    ]
    
    Example:
        encoder_task = ClipTextEncoderTask(
            prompts=[
                "a red shirt",
                "a blue shirt", 
                "a black shirt"
            ],
            clip_model=clip_model,
            task_id="text_encoder"
        )
        
        # Or read prompts from context TEXT
        encoder_task = ClipTextEncoderTask(
            clip_model=clip_model
        )
    """
    
    def __init__(self, 
                 prompts: Optional[List[str]] = None,
                 clip_model = None,
                 normalize: bool = True,
                 task_id: str = "clip_text_encoder"):
        """
        Initialize CLIP text encoder task.
        
        Args:
            prompts: List of text prompts to encode (optional if reading from context TEXT)
            clip_model: CLIPModel instance for encoding
            normalize: Whether to normalize embeddings to unit vectors (default: True)
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.prompts = prompts or []
        self.clip_model = clip_model
        self.normalize = normalize
        
        # Define contracts
        self.input_contract = {
            ContextDataType.TEXT: (list, str)  # Optional: read from TEXT
        }
        
        self.output_contract = {
            ContextDataType.EMBEDDINGS: list  # List of [label, embedding] pairs
        }
        
        logger.info(f"ClipTextEncoderTask '{task_id}' initialized with "
                   f"{len(self.prompts)} configured prompts, normalize={normalize}")
    
    def _get_prompts_from_context(self, context: Context) -> List[str]:
        """
        Extract prompts from context TEXT.
        
        Args:
            context: Pipeline context
            
        Returns:
            List of text prompts
        """
        if ContextDataType.TEXT not in context.data:
            return []
        
        data = context.data[ContextDataType.TEXT]
        
        # Handle list of strings
        if isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                return data
            else:
                logger.warning(f"Task '{self.task_id}': Unexpected list format in TEXT")
                return []
        else:
            logger.warning(f"Task '{self.task_id}': Unexpected data type {type(data)} for TEXT")
            return []
    
    def _encode_prompts(self, prompts: List[str]) -> np.ndarray:
        """
        Encode prompts using CLIP text encoder.
        
        Args:
            prompts: List of text prompts
            
        Returns:
            np.ndarray of shape (N, embedding_dim), optionally normalized
        """
        # Get CLIP model from instance or environment
        clip_model = self.clip_model
        if not clip_model:
            from ..environment import Environment
            env = Environment.get_instance()
            clip_model = env.get("services", "clip", "model")
            
            # If still not found, try to initialize it
            if not clip_model:
                logger.info(f"Task '{self.task_id}': Initializing CLIP text model...")
                from ...models.MobileClip.clip_text_model import ClipTextModel
                from ...utils.config import VLMChatConfig
                
                # Get config from context if available
                config = getattr(self, '_context', None) and getattr(self._context, 'config', None)
                if not config:
                    config = VLMChatConfig()
                
                clip_model = ClipTextModel(config=config, backend="auto")
                env.set("services", "clip", "model", clip_model)
                logger.info(f"Task '{self.task_id}': CLIP model initialized and cached")
        
        if not clip_model:
            raise ValueError(f"Task '{self.task_id}': clip_model required to encode prompts")
        
        embeddings = []
        for i, prompt in enumerate(prompts):
            try:
                # Encode single prompt using ClipTextModel.encode()
                emb = clip_model.encode([prompt])
                # Convert to numpy and squeeze to get (D,) shape
                emb_np = emb.cpu().numpy().squeeze()
                
                # Check if already normalized (some models return normalized embeddings)
                current_norm = np.linalg.norm(emb_np)
                
                # Normalize to unit vector if requested and not already normalized
                if self.normalize:
                    if current_norm > 1e-6:  # Avoid division by zero
                        # Only normalize if not already close to unit norm
                        if abs(current_norm - 1.0) > 0.01:
                            emb_np = emb_np / current_norm
                    else:
                        logger.warning(f"Task '{self.task_id}': Zero norm for prompt '{prompt}'")
                
                embeddings.append(emb_np)
                logger.debug(f"Task '{self.task_id}': Encoded prompt [{i}]: '{prompt[:50]}...'")
                
            except Exception as e:
                logger.error(f"Task '{self.task_id}': Failed to encode prompt '{prompt}': {e}")
                raise
        
        return np.array(embeddings)
    
    def run(self, context: Context) -> Context:
        """
        Encode text prompts and store embeddings in context.
        
        Args:
            context: Pipeline context
            
        Returns:
            Context with EMBEDDINGS data added (list of [text, embedding] pairs)
        """
        # Save context reference for lazy model loading
        self._context = context
        
        # Gather prompts from config and context
        prompts_to_encode = self.prompts.copy() if self.prompts else []
        
        # Add prompts from context TEXT
        context_prompts = self._get_prompts_from_context(context)
        if context_prompts:
            logger.info(f"Task '{self.task_id}': Found {len(context_prompts)} prompts in context TEXT")
            prompts_to_encode.extend(context_prompts)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_prompts = []
        for prompt in prompts_to_encode:
            if prompt not in seen:
                seen.add(prompt)
                unique_prompts.append(prompt)
        
        prompts_to_encode = unique_prompts
        
        if not prompts_to_encode:
            logger.warning(f"Task '{self.task_id}': No prompts to encode")
            context.data[ContextDataType.EMBEDDINGS] = []
            return context
        
        logger.info(f"Task '{self.task_id}': Encoding {len(prompts_to_encode)} prompts")
        
        # Encode prompts
        embeddings = self._encode_prompts(prompts_to_encode)
        
        logger.info(f"Task '{self.task_id}': Generated embeddings with shape {embeddings.shape}")
        
        # Store as list of [text, embedding] pairs
        embedding_pairs = [[text, emb] for text, emb in zip(prompts_to_encode, embeddings)]
        context.data[ContextDataType.EMBEDDINGS] = embedding_pairs
        
        return context
    
    def configure(self, **kwargs) -> None:
        """
        Configure task from parameters.
        
        Args:
            **kwargs: Configuration parameters with optional keys:
                - prompts: List[str] or comma-separated string of text prompts
                - normalize: bool whether to normalize embeddings
        """
        if 'prompts' in kwargs:
            if isinstance(kwargs['prompts'], str):
                # Split comma-separated string
                self.prompts = [p.strip() for p in kwargs['prompts'].split(',')]
            else:
                self.prompts = kwargs['prompts']
            logger.info(f"Task '{self.task_id}': Updated prompts to {len(self.prompts)} items")
        
        if 'normalize' in kwargs:
            self.normalize = bool(kwargs['normalize'])
            logger.info(f"Task '{self.task_id}': Updated normalize={self.normalize}")
