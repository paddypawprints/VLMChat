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

from ..task_base import BaseTask, Context, ContextDataType

logger = logging.getLogger(__name__)


class ClipTextEncoderTask(BaseTask):
    """
    Encodes text prompts using CLIP text encoder.
    
    This task:
    1. Takes a list of text prompts (from config or context)
    2. Encodes them using CLIP text encoder
    3. Normalizes embeddings to unit vectors
    4. Stores results in context for downstream tasks
    
    Output stored in ContextDataType.PROMPT_EMBEDDINGS:
    {
        'prompts': List[str],
        'embeddings': np.ndarray,  # Shape: (N, embedding_dim), normalized
    }
    
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
        
        # Or read prompts from context
        encoder_task = ClipTextEncoderTask(
            prompts_key=ContextDataType.DETECTION_PROMPTS,
            clip_model=clip_model
        )
    """
    
    def __init__(self, 
                 prompts: Optional[List[str]] = None,
                 prompts_key: Optional[ContextDataType] = None,
                 clip_model = None,
                 normalize: bool = True,
                 task_id: str = "clip_text_encoder"):
        """
        Initialize CLIP text encoder task.
        
        Args:
            prompts: List of text prompts to encode (optional if reading from context)
            prompts_key: Context key to read prompts from (e.g., DETECTION_PROMPTS)
            clip_model: CLIPModel instance for encoding
            normalize: Whether to normalize embeddings to unit vectors (default: True)
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.prompts = prompts or []
        self.prompts_key = prompts_key
        self.clip_model = clip_model
        self.normalize = normalize
        
        # Define contracts
        self.input_contract = {}
        if prompts_key:
            self.input_contract[prompts_key] = (list, str)
        
        self.output_contract = {
            ContextDataType.PROMPT_EMBEDDINGS: dict
        }
        
        logger.info(f"ClipTextEncoderTask '{task_id}' initialized with "
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
        Encode prompts using CLIP text encoder.
        
        Args:
            prompts: List of text prompts
            
        Returns:
            np.ndarray of shape (N, embedding_dim), optionally normalized
        """
        if not self.clip_model:
            raise ValueError(f"Task '{self.task_id}': clip_model required to encode prompts")
        
        # Access the CLIP runtime - try different model types
        try:
            # Try MobileCLIP style first
            if hasattr(self.clip_model, '_runtime_as_clip'):
                runtime = self.clip_model._runtime_as_clip()
            # Try FashionCLIP style
            elif hasattr(self.clip_model, '_runtime_as_fashion_clip'):
                runtime = self.clip_model._runtime_as_fashion_clip()
            # Try direct runtime access
            elif hasattr(self.clip_model, '_runtime'):
                runtime = self.clip_model._runtime
            else:
                raise ValueError("Could not access CLIP runtime")
        except Exception as e:
            raise ValueError(f"Task '{self.task_id}': Failed to access CLIP runtime: {e}")
        
        embeddings = []
        for i, prompt in enumerate(prompts):
            try:
                # Encode single prompt as a list
                emb = runtime.encode_text([prompt])
                # Convert to numpy
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
            Context with PROMPT_EMBEDDINGS data added
        """
        # Gather prompts from config and context
        prompts_to_encode = self.prompts.copy() if self.prompts else []
        
        # Add prompts from context if prompts_key specified
        if self.prompts_key:
            context_prompts = self._get_prompts_from_context(context)
            if context_prompts:
                logger.info(f"Task '{self.task_id}': Found {len(context_prompts)} prompts "
                           f"in context[{self.prompts_key}]")
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
            context.data[ContextDataType.PROMPT_EMBEDDINGS] = {
                'prompts': [],
                'embeddings': np.array([]),
            }
            return context
        
        logger.info(f"Task '{self.task_id}': Encoding {len(prompts_to_encode)} prompts")
        
        # Encode prompts
        embeddings = self._encode_prompts(prompts_to_encode)
        
        logger.info(f"Task '{self.task_id}': Generated embeddings with shape {embeddings.shape}")
        
        # Store in context
        context.data[ContextDataType.PROMPT_EMBEDDINGS] = {
            'prompts': prompts_to_encode,
            'embeddings': embeddings,
            'normalized': self.normalize,
        }
        
        return context
    
    def configure(self, params: dict) -> None:
        """
        Configure task from parameters.
        
        Args:
            params: Configuration dict with optional keys:
                - prompts: List[str] or comma-separated string of text prompts
                - prompts_key: str name of ContextDataType to read prompts from
                - normalize: bool whether to normalize embeddings
        """
        if 'prompts' in params:
            if isinstance(params['prompts'], str):
                # Split comma-separated string
                self.prompts = [p.strip() for p in params['prompts'].split(',')]
            else:
                self.prompts = params['prompts']
            logger.info(f"Task '{self.task_id}': Updated prompts to {len(self.prompts)} items")
        
        if 'prompts_key' in params:
            prompts_key_name = params['prompts_key']
            # Convert string to ContextDataType
            try:
                self.prompts_key = ContextDataType[prompts_key_name]
                logger.info(f"Task '{self.task_id}': Updated prompts_key={self.prompts_key}")
            except KeyError:
                logger.error(f"Task '{self.task_id}': Invalid prompts_key '{prompts_key_name}'")
        
        if 'normalize' in params:
            self.normalize = bool(params['normalize'])
            logger.info(f"Task '{self.task_id}': Updated normalize={self.normalize}")
