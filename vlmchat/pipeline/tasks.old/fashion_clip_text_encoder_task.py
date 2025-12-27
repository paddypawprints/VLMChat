"""
FashionClip Text Encoder Task

Encodes text prompts using FashionClip text encoder and stores embeddings in context.

This task is specifically designed for FashionClip's 768-dimensional embeddings,
optimized for fashion domain text-to-image matching.
"""

import numpy as np
import logging
from typing import List, Optional

from ..task_base import BaseTask, Context, ContextDataType, register_task

logger = logging.getLogger(__name__)


@register_task('fashion_clip_text_encoder')
class FashionClipTextEncoderTask(BaseTask):
    """
    Encodes text prompts using FashionClip text encoder.
    
    This task:
    1. Takes a list of text prompts (from config or context)
    2. Encodes them using FashionClip text encoder (768-dim embeddings)
    3. Normalizes embeddings to unit vectors
    4. Stores results in context for downstream tasks
    
    Output stored in ContextDataType.PROMPT_EMBEDDINGS:
    {
        'prompts': List[str],
        'embeddings': np.ndarray,  # Shape: (N, 768), normalized
    }
    
    Example:
        encoder_task = FashionClipTextEncoderTask(
            prompts=[
                "a red dress",
                "blue jeans", 
                "leather jacket"
            ],
            fashion_clip_model=fashion_clip_model,
            task_id="fashion_text_encoder"
        )
        
        # Or read prompts from context
        encoder_task = FashionClipTextEncoderTask(
            prompts_key=ContextDataType.DETECTION_PROMPTS,
            fashion_clip_model=fashion_clip_model
        )
    """
    
    def __init__(self, 
                 prompts: Optional[List[str]] = None,
                 prompts_key: Optional[ContextDataType] = None,
                 fashion_clip_model = None,
                 normalize: bool = True,
                 task_id: str = "fashion_clip_text_encoder"):
        """
        Initialize FashionClip text encoder task.
        
        Args:
            prompts: List of text prompts to encode (optional if reading from context)
            prompts_key: Context key to read prompts from (defaults to TEXT)
            fashion_clip_model: FashionClipModel instance for encoding
            normalize: Whether to normalize embeddings to unit vectors (default: True)
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.prompts = prompts or []
        # Default to TEXT like clip_text_encoder for compatibility
        self.prompts_key = prompts_key if prompts_key is not None else ContextDataType.TEXT
        self.fashion_clip_model = fashion_clip_model
        self.normalize = normalize
        
        # Define contracts
        self.input_contract = {
            ContextDataType.TEXT: (list, str)  # Read prompts from TEXT by default
        }
        
        self.output_contract = {
            ContextDataType.EMBEDDINGS: list  # List of [text, embedding] pairs
        }
        
        logger.info(f"FashionClipTextEncoderTask '{task_id}' initialized with "
                   f"{len(self.prompts)} configured prompts, "
                   f"prompts_key={prompts_key}, normalize={normalize}")
    
    def _get_prompts_from_context(self, context: Context) -> List[str]:
        """
        Extract prompts from context.
        
        Args:
            context: Pipeline context
            
        Returns:
            List of text prompts
        """
        if not self.prompts_key:
            return []
        
        if self.prompts_key not in context.data:
            logger.warning(f"Task '{self.task_id}': prompts_key '{self.prompts_key}' "
                         f"not found in context")
            return []
        
        data = context.data[self.prompts_key]
        
        # Handle different data formats
        if isinstance(data, list):
            # List of strings
            if all(isinstance(item, str) for item in data):
                return data
            # List of dicts with 'prompt' key
            elif all(isinstance(item, dict) and 'prompt' in item for item in data):
                return [item['prompt'] for item in data]
            else:
                logger.warning(f"Task '{self.task_id}': Unexpected list format in context")
                return []
        elif isinstance(data, str):
            # Single string - split by comma or return as single item
            if ',' in data:
                return [p.strip() for p in data.split(',')]
            else:
                return [data]
        else:
            logger.warning(f"Task '{self.task_id}': Unexpected data type {type(data)} "
                         f"for prompts_key '{self.prompts_key}'")
            return []
    
    def _encode_prompts(self, prompts: List[str]) -> np.ndarray:
        """
        Encode prompts using FashionClip text encoder.
        
        Args:
            prompts: List of text prompts
            
        Returns:
            np.ndarray of shape (N, 768), optionally normalized
        """
        # Get FashionClip model from instance or environment
        fashion_clip_model = self.fashion_clip_model
        if not fashion_clip_model:
            from ..environment import Environment
            env = Environment.get_instance()
            fashion_clip_model = env.get("services", "fashion_clip", "model")
            
            # If still not found, try to initialize it
            if not fashion_clip_model:
                logger.info(f"Task '{self.task_id}': Initializing FashionClip model...")
                from ...models.FashionClip.fashion_clip_model import FashionClipModel
                from ...utils.config import VLMChatConfig
                
                # Get config from context if available
                config = getattr(self, '_context', None) and getattr(self._context, 'config', None)
                if not config:
                    config = VLMChatConfig()
                
                fashion_clip_model = FashionClipModel(config=config)
                env.set("services", "fashion_clip", "model", fashion_clip_model)
                logger.info(f"Task '{self.task_id}': FashionClip model initialized and cached")
        
        if not fashion_clip_model:
            raise ValueError(f"Task '{self.task_id}': fashion_clip_model required to encode prompts")
        
        # Update instance variable
        self.fashion_clip_model = fashion_clip_model
        
        # Access the FashionClip runtime
        try:
            if hasattr(fashion_clip_model, '_runtime_as_fashion_clip'):
                runtime = fashion_clip_model._runtime_as_fashion_clip()
            elif hasattr(fashion_clip_model, '_runtime'):
                runtime = fashion_clip_model._runtime
            else:
                raise ValueError("Could not access FashionClip runtime - model must be FashionClipModel")
        except Exception as e:
            raise ValueError(f"Task '{self.task_id}': Failed to access FashionClip runtime: {e}")
        
        embeddings = []
        for i, prompt in enumerate(prompts):
            try:
                # Encode single prompt as a list
                emb = runtime.encode_text([prompt])
                # Convert to numpy and squeeze to get (D,) shape
                emb_np = emb.cpu().numpy().squeeze()
                
                # Check if already normalized (FashionClip returns normalized embeddings)
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
                logger.debug(f"Task '{self.task_id}': Encoded fashion prompt [{i}]: '{prompt[:50]}...'")
                
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
            Context with PROMPT_EMBEDDINGS data added
        """
        # Save context reference for model initialization
        self._context = context
        
        # Gather prompts from config and context
        prompts_to_encode = self.prompts.copy() if self.prompts else []
        
        # Add prompts from context if prompts_key specified
        if self.prompts_key:
            logger.debug(f"Task '{self.task_id}': Checking context for prompts_key={self.prompts_key}")
            logger.debug(f"Task '{self.task_id}': Context data keys: {list(context.data.keys())}")
            context_prompts = self._get_prompts_from_context(context)
            if context_prompts:
                logger.info(f"Task '{self.task_id}': Found {len(context_prompts)} prompts "
                           f"in context[{self.prompts_key}]")
                prompts_to_encode.extend(context_prompts)
            else:
                logger.warning(f"Task '{self.task_id}': No prompts found in context[{self.prompts_key}]")
        
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
        
        logger.info(f"Task '{self.task_id}': Encoding {len(prompts_to_encode)} fashion prompts")
        
        # Encode prompts
        embeddings = self._encode_prompts(prompts_to_encode)
        
        logger.info(f"Task '{self.task_id}': Generated FashionClip embeddings with shape {embeddings.shape}")
        
        # Store as list of [text, embedding] pairs (matching clip_text_encoder format)
        embedding_pairs = [[text, emb] for text, emb in zip(prompts_to_encode, embeddings)]
        context.data[ContextDataType.EMBEDDINGS] = embedding_pairs
        
        return context
    
    def configure(self, **kwargs) -> None:
        """
        Configure task from parameters.
        
        Args:
            **kwargs: Configuration parameters with optional keys:
                - prompts: List[str] or comma-separated string of text prompts
                - prompts_key: str name of ContextDataType to read prompts from
                - normalize: bool whether to normalize embeddings
        """
        if 'prompts' in kwargs:
            if isinstance(kwargs['prompts'], str):
                # Split comma-separated string
                self.prompts = [p.strip() for p in kwargs['prompts'].split(',')]
            else:
                self.prompts = kwargs['prompts']
            logger.info(f"Task '{self.task_id}': Updated prompts to {len(self.prompts)} items")
        
        if 'prompts_key' in kwargs:
            prompts_key_name = kwargs['prompts_key']
            # Convert string to ContextDataType
            try:
                self.prompts_key = ContextDataType[prompts_key_name]
                logger.info(f"Task '{self.task_id}': Updated prompts_key={self.prompts_key}")
            except KeyError:
                logger.error(f"Task '{self.task_id}': Invalid prompts_key '{prompts_key_name}'")
        
        if 'normalize' in kwargs:
            self.normalize = bool(kwargs['normalize'])
            logger.info(f"Task '{self.task_id}': Updated normalize={self.normalize}")
