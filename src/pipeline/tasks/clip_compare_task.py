"""
CLIP comparison task for semantic matching of detections.

Compares CLIP vision embeddings from detections against prompt embeddings
to compute similarity scores and find best matches.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple

from ..task_base import BaseTask, Context, ContextDataType, register_task
from ...object_detector.detection_base import Detection

logger = logging.getLogger(__name__)


@register_task('clip_compare')
class ClipCompareTask(BaseTask):
    """
    Compares vision embeddings to prompt embeddings for semantic matching.
    
    This task:
    1. Takes EMBEDDINGS (from ClipVisionTask) and PROMPT_EMBEDDINGS from context
    2. Computes cosine similarity between each detection embedding and all prompts
    3. Finds best matching prompt for each detection
    4. Stores similarity scores and matches in context
    5. Optionally filters detections by minimum similarity threshold
    
    The output MATCHES contains:
    - detection_index: Index into DETECTIONS list
    - prompt_index: Index into prompts list
    - prompt_text: Matched prompt string
    - similarity: Cosine similarity score (0-1)
    - embedding: Original detection embedding
    
    Example:
        compare_task = ClipCompareTask(
            task_id="clip_compare",
            min_similarity=0.3  # Filter out weak matches
        )
        
        # After execution, context contains:
        # - ContextDataType.MATCHES: List of match dicts
        # - ContextDataType.DETECTIONS: Optionally filtered detections
    """
    
    def __init__(self, task_id: str = "clip_compare", 
                 min_similarity: float = 0.0,
                 filter_detections: bool = False):
        """
        Initialize CLIP comparison task.
        
        Args:
            task_id: Unique identifier for this task
            min_similarity: Minimum similarity threshold (0-1)
            filter_detections: If True, remove detections below min_similarity
        """
        super().__init__(task_id)
        self.min_similarity = min_similarity
        self.filter_detections = filter_detections
        
        # Define contracts
        self.input_contract = {
            ContextDataType.EMBEDDINGS: list,
            ContextDataType.PROMPT_EMBEDDINGS: dict,
            ContextDataType.DETECTIONS: list
        }
        self.output_contract = {
            ContextDataType.MATCHES: list,
            ContextDataType.DETECTIONS: list  # May be filtered
        }
        
        logger.info(f"ClipCompareTask '{task_id}' initialized: "
                   f"min_similarity={min_similarity}, filter={filter_detections}")
    
    def run(self, context: Context) -> Context:
        """
        Compare embeddings and find best prompt matches.
        
        Args:
            context: Pipeline context with EMBEDDINGS, PROMPT_EMBEDDINGS, DETECTIONS
        
        Returns:
            Context with MATCHES added (and possibly filtered DETECTIONS)
        """
        # Get inputs from context
        embeddings = context.data.get(ContextDataType.EMBEDDINGS, [])
        prompt_data = context.data.get(ContextDataType.PROMPT_EMBEDDINGS, {})
        detections = context.data.get(ContextDataType.DETECTIONS, [])
        
        if not embeddings:
            logger.warning(f"Task '{self.task_id}': No embeddings in context")
            context.data[ContextDataType.MATCHES] = []
            return context
        
        if not prompt_data:
            logger.warning(f"Task '{self.task_id}': No prompt embeddings in context")
            context.data[ContextDataType.MATCHES] = []
            return context
        
        # Extract prompt embeddings and texts
        prompt_embeddings = prompt_data.get('embeddings', [])
        prompt_texts = prompt_data.get('prompts', [])
        
        if len(prompt_embeddings) == 0 or len(prompt_texts) == 0:
            logger.warning(f"Task '{self.task_id}': Empty prompt embeddings or texts")
            context.data[ContextDataType.MATCHES] = []
            return context
        
        logger.info(f"Task '{self.task_id}': Comparing {len(embeddings)} detections "
                   f"against {len(prompt_texts)} prompts")
        
        # Convert to numpy arrays for vectorized operations
        # Handle both flat arrays (512,) and shaped arrays (1, 512)
        vision_arrays = []
        for emb in embeddings:
            if isinstance(emb, np.ndarray):
                # Flatten if needed
                emb_flat = emb.flatten() if emb.ndim > 1 else emb
                vision_arrays.append(emb_flat)
            else:
                vision_arrays.append(np.array(emb).flatten())
        
        vision_embs = np.array(vision_arrays)  # Shape: (N_detections, embed_dim)
        prompt_embs = np.array(prompt_embeddings)  # Shape: (N_prompts, embed_dim)
        
        # Flatten prompt embeddings if needed
        if prompt_embs.ndim > 2:
            prompt_embs = prompt_embs.reshape(len(prompt_embeddings), -1)
        
        logger.debug(f"Vision embeddings shape: {vision_embs.shape}")
        logger.debug(f"Prompt embeddings shape: {prompt_embs.shape}")
        
        # Normalize embeddings for cosine similarity
        vision_norm = vision_embs / (np.linalg.norm(vision_embs, axis=1, keepdims=True) + 1e-8)
        prompt_norm = prompt_embs / (np.linalg.norm(prompt_embs, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarity matrix: (N_detections, N_prompts)
        similarity_matrix = np.dot(vision_norm, prompt_norm.T)
        
        # Find best match for each detection
        matches = []
        filtered_detections = []
        
        for det_idx in range(len(embeddings)):
            # Get similarity scores for this detection
            scores = similarity_matrix[det_idx]
            best_prompt_idx = int(np.argmax(scores))
            best_similarity = float(np.max(scores))  # Use np.max instead of indexing
            
            # Check threshold
            if best_similarity >= self.min_similarity:
                match = {
                    'detection_index': det_idx,
                    'prompt_index': best_prompt_idx,
                    'prompt_text': prompt_texts[best_prompt_idx],
                    'similarity': best_similarity,
                    'embedding': embeddings[det_idx],
                    'all_scores': scores.tolist()  # All prompt scores for this detection
                }
                matches.append(match)
                
                # Keep detection if filtering
                if self.filter_detections and det_idx < len(detections):
                    filtered_detections.append(detections[det_idx])
            else:
                logger.debug(f"Detection {det_idx} filtered: similarity={best_similarity:.3f} "
                           f"< threshold={self.min_similarity}")
        
        logger.info(f"Task '{self.task_id}': Found {len(matches)} matches "
                   f"(threshold={self.min_similarity})")
        
        # Log top matches
        for i, match in enumerate(matches[:5]):  # Show top 5
            logger.info(f"  Match {i+1}: '{match['prompt_text']}' "
                       f"(similarity={match['similarity']:.3f})")
        
        # Store results in context
        context.data[ContextDataType.MATCHES] = matches
        
        # Update detections if filtering
        if self.filter_detections:
            context.data[ContextDataType.DETECTIONS] = filtered_detections
            logger.info(f"Task '{self.task_id}': Filtered detections from "
                       f"{len(detections)} to {len(filtered_detections)}")
        
        return context
    
    def configure(self, **kwargs) -> None:
        """
        Configure task from parameters.
        
        Args:
            **kwargs: Configuration parameters with optional keys:
                - min_similarity: float (0-1)
                - filter_detections: bool
        """
        if 'min_similarity' in kwargs:
            self.min_similarity = float(kwargs['min_similarity'])
            logger.info(f"Task '{self.task_id}': Set min_similarity={self.min_similarity}")
        
        if 'filter_detections' in kwargs:
            filter_str = kwargs['filter_detections'].lower()
            self.filter_detections = filter_str in ('true', '1', 'yes')
            logger.info(f"Task '{self.task_id}': Set filter_detections={self.filter_detections}")
