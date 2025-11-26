"""
CLIP Comparator Task

Compares text and image embeddings to compute similarity scores.
Takes embeddings from parallel branches (text encoder + image encoder)
and computes cosine similarity matrix.
"""

import numpy as np
import logging
from typing import List, Dict, Optional

from ..task_base import BaseTask, Context, ContextDataType, register_task

logger = logging.getLogger(__name__)


@register_task('clip_comparator')
class ClipComparatorTask(BaseTask):
    """
    Compares text and image embeddings using cosine similarity.
    
    This task:
    1. Takes EMBEDDINGS from context (expects nested list from parallel branches)
    2. Separates text embeddings (branch 0) from image embeddings (branch 1)
    3. Computes cosine similarity matrix (text × image)
    4. Stores results in SIMILARITY_SCORES
    
    Input format (from ordered merge):
    EMBEDDINGS = [
        [  # Branch 0: Text embeddings
            ['red shirt', np.array(...)],
            ['blue pants', np.array(...)]
        ],
        [  # Branch 1: Image embeddings
            ['detection_0', np.array(...)],
            ['detection_1', np.array(...)]
        ]
    ]
    
    Output format in ContextDataType.SIMILARITY_SCORES:
    {
        'matrix': np.ndarray,        # Shape: (n_texts, n_detections)
        'texts': List[str],          # Row labels
        'detection_ids': List[str],  # Column labels
        'metric': 'cosine_similarity'
    }
    
    Example:
        comparator = ClipComparatorTask(
            text_branch=0,
            image_branch=1,
            task_id="clip_comparator"
        )
    """
    
    def __init__(self, 
                 text_branch: int = 0,
                 image_branch: int = 1,
                 temperature: float = 0.07,
                 probability_mode: str = "softmax",
                 task_id: str = "clip_comparator"):
        """
        Initialize CLIP comparator task.
        
        Args:
            text_branch: Index of branch containing text embeddings (default: 0)
            image_branch: Index of branch containing image embeddings (default: 1)
            temperature: Temperature for softmax (default: 0.07, lower = sharper)
            probability_mode: Method to convert similarities to probabilities:
                - "softmax": Apply softmax with temperature (sums to 1.0 per detection)
                - "similarity": Use raw cosine similarities as probabilities
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.text_branch = text_branch
        self.image_branch = image_branch
        self.temperature = temperature
        self.probability_mode = probability_mode
        
        # Validate probability_mode
        if self.probability_mode not in ["softmax", "similarity"]:
            logger.warning(f"Invalid probability_mode '{self.probability_mode}', "
                         f"defaulting to 'softmax'")
            self.probability_mode = "softmax"
        
        # Define contracts
        self.input_contract = {
            ContextDataType.EMBEDDINGS: list  # Nested list from merge
        }
        
        self.output_contract = {
            ContextDataType.SIMILARITY_SCORES: dict
        }
        
        logger.info(f"ClipComparatorTask '{task_id}' initialized with "
                   f"text_branch={text_branch}, image_branch={image_branch}, "
                   f"temperature={temperature}, probability_mode={probability_mode}")
    
    def run(self, context: Context) -> Context:
        """
        Compute similarity scores between text and image embeddings.
        
        Args:
            context: Pipeline context with EMBEDDINGS
            
        Returns:
            Context with SIMILARITY_SCORES added
        """
        # Get embeddings from context
        embeddings = context.data.get(ContextDataType.EMBEDDINGS)
        if not embeddings:
            logger.warning(f"Task '{self.task_id}': No EMBEDDINGS found in context")
            context.data[ContextDataType.SIMILARITY_SCORES] = {
                'matrix': np.array([]),
                'texts': [],
                'detection_ids': [],
                'metric': 'cosine_similarity'
            }
            return context
        
        # Check if embeddings are nested (from merge)
        if not isinstance(embeddings, list) or len(embeddings) == 0:
            logger.warning(f"Task '{self.task_id}': EMBEDDINGS is not a valid list")
            context.data[ContextDataType.SIMILARITY_SCORES] = {
                'matrix': np.array([]),
                'texts': [],
                'detection_ids': [],
                'metric': 'cosine_similarity'
            }
            return context
        
        # Check if first element is a list (nested structure from merge)
        if not isinstance(embeddings[0], list):
            logger.warning(f"Task '{self.task_id}': EMBEDDINGS is not nested. "
                         f"Expected format from ordered_merge with two branches.")
            context.data[ContextDataType.SIMILARITY_SCORES] = {
                'matrix': np.array([]),
                'texts': [],
                'detection_ids': [],
                'metric': 'cosine_similarity'
            }
            return context
        
        # Extract branches
        if len(embeddings) <= max(self.text_branch, self.image_branch):
            logger.error(f"Task '{self.task_id}': Not enough branches. "
                        f"Expected at least {max(self.text_branch, self.image_branch) + 1}, "
                        f"got {len(embeddings)}")
            context.data[ContextDataType.SIMILARITY_SCORES] = {
                'matrix': np.array([]),
                'texts': [],
                'detection_ids': [],
                'metric': 'cosine_similarity'
            }
            return context
        
        text_embs = embeddings[self.text_branch]
        image_embs = embeddings[self.image_branch]
        
        if not text_embs or not image_embs:
            logger.warning(f"Task '{self.task_id}': Empty embeddings. "
                         f"Text: {len(text_embs)}, Image: {len(image_embs)}")
            context.data[ContextDataType.SIMILARITY_SCORES] = {
                'matrix': np.array([]),
                'texts': [],
                'detection_ids': [],
                'metric': 'cosine_similarity'
            }
            return context
        
        logger.info(f"Task '{self.task_id}': Comparing {len(text_embs)} text × "
                   f"{len(image_embs)} image embeddings")
        
        # Extract labels and embeddings - format is [[label, embedding], ...]
        texts = [item[0] for item in text_embs]
        text_embeddings_list = [item[1] for item in text_embs]
        
        detection_ids = [item[0] for item in image_embs]
        image_embeddings_list = [item[1] for item in image_embs]
        
        # Convert to numpy arrays with consistent shape
        text_matrix = self._safe_convert_to_matrix(text_embeddings_list, "text")
        image_matrix = self._safe_convert_to_matrix(image_embeddings_list, "image")
        
        # Check embedding properties before normalization (INFO level for debugging)
        text_norms_before = np.linalg.norm(text_matrix, axis=1)
        image_norms_before = np.linalg.norm(image_matrix, axis=1)
        logger.info(f"Task '{self.task_id}': Before normalization - "
                   f"Text norms: mean={np.mean(text_norms_before):.3f}, "
                   f"Image norms: mean={np.mean(image_norms_before):.3f}")
        
        # UNIVERSAL NORMALIZATION - ensures cosine similarity works correctly
        text_matrix_norm = self._safe_normalize(text_matrix)
        image_matrix_norm = self._safe_normalize(image_matrix)
        
        # Check after normalization
        text_norms_after = np.linalg.norm(text_matrix_norm, axis=1)
        image_norms_after = np.linalg.norm(image_matrix_norm, axis=1)
        logger.info(f"Task '{self.task_id}': After normalization - "
                   f"Text norms: mean={np.mean(text_norms_after):.3f}, "
                   f"Image norms: mean={np.mean(image_norms_after):.3f}")
        
        # Compute cosine similarity: text_matrix @ image_matrix.T
        # After normalization, dot product = cosine similarity
        similarity_matrix = text_matrix_norm @ image_matrix_norm.T
        
        logger.info(f"Task '{self.task_id}': Computed similarity matrix "
                   f"with shape {similarity_matrix.shape}, "
                   f"range: [{np.min(similarity_matrix):.3f}, {np.max(similarity_matrix):.3f}]")
        
        # Compute probabilities based on mode
        if self.probability_mode == "similarity":
            # Use raw similarities as probabilities
            probabilities = similarity_matrix
            logger.debug(f"Task '{self.task_id}': Using raw similarities as probabilities")
        else:
            # Apply softmax with temperature
            probabilities = self._compute_softmax_probabilities(similarity_matrix)
            logger.debug(f"Task '{self.task_id}': Applied softmax with temperature={self.temperature}")
        
        # Store results
        context.data[ContextDataType.SIMILARITY_SCORES] = {
            'matrix': similarity_matrix,
            'probabilities': probabilities,
            'texts': texts,
            'detection_ids': detection_ids,
            'metric': 'cosine_similarity',
            'temperature': self.temperature,
            'probability_mode': self.probability_mode
        }
        
        return context
    
    def _compute_softmax_probabilities(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Compute softmax probabilities with temperature scaling.
        
        Applies softmax per detection (column-wise), so each detection gets
        a probability distribution over all prompts that sums to 1.0.
        
        Formula: P(prompt_i | detection) = exp(s_i/T) / sum_j(exp(s_j/T))
        
        Args:
            similarity_matrix: Cosine similarities, shape (n_texts, n_detections)
            
        Returns:
            Probability matrix, same shape as input
        """
        # Scale by temperature
        scaled = similarity_matrix / self.temperature
        
        # Apply softmax column-wise (axis=0)
        # Subtract max for numerical stability
        scaled_shifted = scaled - np.max(scaled, axis=0, keepdims=True)
        exp_scaled = np.exp(scaled_shifted)
        probabilities = exp_scaled / np.sum(exp_scaled, axis=0, keepdims=True)
        
        return probabilities
    
    def _safe_convert_to_matrix(self, embedding_list: List, name: str) -> np.ndarray:
        """
        Convert list of embeddings to consistent 2D matrix.
        
        Handles various input shapes from different encoder outputs:
        - (D,) single embedding → (1, D)
        - (N, D) batch → (N, D)
        - (N, 1, D) with extra dim → (N, D)
        
        Args:
            embedding_list: List of numpy arrays with potentially inconsistent shapes
            name: Name for debug logging (e.g., "text", "image")
            
        Returns:
            Consistent 2D matrix of shape (N, D)
        """
        if not embedding_list:
            return np.array([])
        
        # Convert to numpy array
        matrix = np.array(embedding_list, dtype=np.float32)
        
        # Handle various input shapes
        if len(matrix.shape) == 1:
            # Single embedding: (D,) → (1, D)
            matrix = matrix.reshape(1, -1)
        elif len(matrix.shape) == 2:
            # Already 2D, check if it's (N, D) or needs reshaping
            if matrix.shape[0] > 1 and matrix.shape[1] == 1:
                # Unlikely case: (N, 1) → need more context
                logger.warning(f"{name} matrix has shape {matrix.shape}, might be problematic")
        elif len(matrix.shape) > 2:
            # (N, 1, D) or (N, D, 1) → flatten to (N, D)
            matrix = matrix.reshape(matrix.shape[0], -1)
        
        logger.debug(f"{name} matrix converted to shape: {matrix.shape}")
        return matrix
    
    def _safe_normalize(self, matrix: np.ndarray) -> np.ndarray:
        """
        Safely normalize matrix rows to unit length (L2 normalization).
        
        This is CRITICAL for cosine similarity to work correctly. Without
        normalization, similarity scores depend on both angle AND magnitude,
        leading to artificially low scores.
        
        Args:
            matrix: Input matrix of shape (N, D)
            
        Returns:
            Normalized matrix where each row has length 1.0
        """
        if matrix.size == 0:
            return matrix
        
        # Ensure 2D
        if len(matrix.shape) == 1:
            matrix = matrix.reshape(1, -1)
        
        # L2 normalization: v_norm = v / ||v||
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        normalized = matrix / norms
        
        return normalized
    
    def _debug_embedding_properties(self, matrix: np.ndarray, name: str):
        """
        Debug helper to log embedding properties.
        
        Useful for validating that embeddings are properly normalized
        and identifying issues with embedding generation.
        
        Args:
            matrix: Embedding matrix of shape (N, D)
            name: Name for logging (e.g., "Raw Text", "Normalized Image")
        """
        if matrix.size == 0:
            logger.debug(f"=== {name} === Empty matrix")
            return
        
        # Calculate L2 norms of each embedding
        norms = np.linalg.norm(matrix, axis=1)
        is_normalized = np.allclose(norms, 1.0, atol=0.01)
        
        logger.debug(f"=== {name} Embeddings ===")
        logger.debug(f"  Shape: {matrix.shape}")
        logger.debug(f"  Normalized: {is_normalized}")
        logger.debug(f"  Norms - mean: {np.mean(norms):.3f}, "
                    f"range: [{np.min(norms):.3f}, {np.max(norms):.3f}]")
        logger.debug(f"  Values - range: [{np.min(matrix):.3f}, {np.max(matrix):.3f}]")
        
        if not is_normalized and np.max(norms) > 1.1:
            logger.warning(f"  {name} embeddings are NOT normalized! "
                         f"This will cause incorrect similarity scores.")
    
    def configure(self, **kwargs) -> None:
        """
        Configure task from parameters.
        
        Args:
            **kwargs: Configuration parameters with optional keys:
                - text_branch: int index of text embeddings branch
                - image_branch: int index of image embeddings branch
                - temperature: float temperature for softmax scaling
                - probability_mode: str "softmax" or "similarity"
        """
        if 'text_branch' in kwargs:
            self.text_branch = int(kwargs['text_branch'])
            logger.info(f"Task '{self.task_id}': Updated text_branch={self.text_branch}")
        
        if 'image_branch' in kwargs:
            self.image_branch = int(kwargs['image_branch'])
            logger.info(f"Task '{self.task_id}': Updated image_branch={self.image_branch}")
        
        if 'temperature' in kwargs:
            self.temperature = float(kwargs['temperature'])
            logger.info(f"Task '{self.task_id}': Updated temperature={self.temperature}")
        
        if 'probability_mode' in kwargs:
            mode = kwargs['probability_mode']
            if mode in ["softmax", "similarity"]:
                self.probability_mode = mode
                logger.info(f"Task '{self.task_id}': Updated probability_mode={self.probability_mode}")
            else:
                logger.warning(f"Task '{self.task_id}': Invalid probability_mode '{mode}', "
                             f"keeping current value '{self.probability_mode}'")
    
    def describe(self) -> str:
        """Task description for help system."""
        return ("Compares text and image embeddings using cosine similarity. "
                "Expects nested EMBEDDINGS from ordered_merge with text and image branches.")
    
    def describe_parameters(self) -> Dict[str, Dict[str, str]]:
        """Parameter descriptions for help system."""
        return {
            'text_branch': {
                'type': 'int',
                'default': '0',
                'description': 'Index of branch containing text embeddings'
            },
            'image_branch': {
                'type': 'int',
                'default': '1',
                'description': 'Index of branch containing image embeddings'
            },
            'temperature': {
                'type': 'float',
                'default': '0.07',
                'description': 'Temperature for softmax (lower = sharper distribution, only used in softmax mode)'
            },
            'probability_mode': {
                'type': 'str',
                'default': 'softmax',
                'description': 'Probability conversion: "softmax" (normalized, sums to 1.0) or "similarity" (raw cosine similarities)'
            }
        }


if __name__ == "__main__":
    # Example usage
    print("\n--- ClipComparatorTask Example ---\n")
    
    # Create test data
    ctx = Context()
    
    # Simulate embeddings from ordered merge
    text_embs = [
        ['red shirt', np.random.randn(512).astype(np.float32)],
        ['blue pants', np.random.randn(512).astype(np.float32)]
    ]
    
    image_embs = [
        ['detection_0', np.random.randn(512).astype(np.float32)],
        ['detection_1', np.random.randn(512).astype(np.float32)],
        ['detection_2', np.random.randn(512).astype(np.float32)]
    ]
    
    # Nested structure from merge
    ctx.data[ContextDataType.EMBEDDINGS] = [text_embs, image_embs]
    
    print(f"Input EMBEDDINGS:")
    print(f"  Branch 0 (text): {len(text_embs)} items")
    print(f"  Branch 1 (image): {len(image_embs)} items")
    
    # Create and run comparator
    comparator = ClipComparatorTask(task_id="test_comparator")
    output_ctx = comparator.run(ctx)
    
    # Check results
    scores = output_ctx.data.get(ContextDataType.SIMILARITY_SCORES)
    if scores:
        matrix = scores['matrix']
        texts = scores['texts']
        det_ids = scores['detection_ids']
        
        print(f"\nSimilarity matrix shape: {matrix.shape}")
        print(f"Texts: {texts}")
        print(f"Detection IDs: {det_ids}")
        print(f"\nMatrix:")
        print(matrix)
        
        # Show ranked results per text
        print(f"\nRanked matches per text:")
        for i, text in enumerate(texts):
            scores_for_text = matrix[i, :]
            sorted_indices = np.argsort(scores_for_text)[::-1]
            print(f"\n'{text}':")
            for j in sorted_indices:
                print(f"  {det_ids[j]}: {scores_for_text[j]:.3f}")
    
    print("\n✓ ClipComparatorTask computed similarity scores")
