"""
Prompt Similarity Analysis Task

This task computes pairwise similarity between text prompts in CLIP embedding space
to analyze semantic relationships and distinguishability of concepts.

Useful for:
- Understanding which prompts are too similar (will confuse CLIP)
- Identifying clusters of related concepts
- Debugging poor detection performance
- Guiding prompt engineering
"""

import numpy as np
import logging
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity

from ..task_base import BaseTask, Context, ContextDataType

logger = logging.getLogger(__name__)


class PromptSimilarityTask(BaseTask):
    """
    Analyzes similarity between text prompts in CLIP embedding space.
    
    This task:
    1. Takes a list of text prompts
    2. Encodes them using CLIP text encoder
    3. Computes pairwise cosine similarities
    4. Stores results in context for analysis
    
    The task expects PROMPT_EMBEDDINGS in context (from PromptEmbeddingSourceTask)
    or can encode prompts directly if clip_model is provided.
    
    Output stored in ContextDataType.PROMPT_SIMILARITY:
    {
        'prompts': List[str],
        'embeddings': np.ndarray,  # Shape: (N, embedding_dim)
        'similarity_matrix': np.ndarray,  # Shape: (N, N)
        'pairwise_scores': List[Dict],  # Detailed pairwise comparisons
    }
    
    Example:
        similarity_task = PromptSimilarityTask(
            prompts=[
                "a red shirt",
                "a blue shirt", 
                "a black shirt"
            ],
            clip_model=clip_model,
            threshold=0.85,  # Warn if similarity exceeds this
            task_id="similarity_analysis"
        )
    """
    
    def __init__(self, 
                 prompts: Optional[List[str]] = None,
                 clip_model = None,
                 threshold: float = 0.85,
                 task_id: str = "prompt_similarity"):
        """
        Initialize prompt similarity analysis task.
        
        Args:
            prompts: List of text prompts to analyze (optional if using existing embeddings)
            clip_model: CLIPModel instance for encoding (optional if embeddings in context)
            threshold: Similarity threshold for warnings (0.0-1.0)
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.prompts = prompts or []
        self.clip_model = clip_model
        self.threshold = threshold
        
        # Define contracts
        self.input_contract = {}  # Optional PROMPT_EMBEDDINGS
        self.output_contract = {
            ContextDataType.PROMPT_SIMILARITY: dict
        }
        
        logger.info(f"PromptSimilarityTask '{task_id}' initialized with "
                   f"{len(self.prompts)} prompts, threshold={threshold:.2f}")
    
    def _encode_prompts(self, prompts: List[str]) -> np.ndarray:
        """
        Encode prompts using CLIP text encoder.
        
        Args:
            prompts: List of text prompts
            
        Returns:
            np.ndarray of shape (N, embedding_dim)
        """
        if not self.clip_model:
            raise ValueError(f"Task '{self.task_id}': clip_model required to encode prompts")
        
        embeddings = []
        for prompt in prompts:
            emb = self.clip_model.encode([prompt])
            # Convert to numpy and normalize
            emb_np = emb.cpu().numpy().squeeze()
            # Normalize to unit vector for cosine similarity
            norm = np.linalg.norm(emb_np)
            if norm > 0:
                emb_np = emb_np / norm
            embeddings.append(emb_np)
        
        return np.array(embeddings)
    
    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix.
        
        Args:
            embeddings: np.ndarray of shape (N, embedding_dim)
            
        Returns:
            np.ndarray of shape (N, N) with cosine similarities
        """
        # If embeddings are already normalized, dot product = cosine similarity
        # Otherwise use sklearn's cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix
    
    def _analyze_similarities(self, 
                             prompts: List[str], 
                             similarity_matrix: np.ndarray) -> List[Dict]:
        """
        Generate detailed pairwise comparison data.
        
        Args:
            prompts: List of text prompts
            similarity_matrix: Pairwise similarity matrix
            
        Returns:
            List of dicts with pairwise comparison details
        """
        pairwise_scores = []
        
        for i, prompt1 in enumerate(prompts):
            for j, prompt2 in enumerate(prompts):
                if i < j:  # Only upper triangle (avoid duplicates)
                    similarity = float(similarity_matrix[i, j])
                    
                    # Categorize similarity level
                    if similarity > 0.95:
                        level = "VERY_HIGH"
                        warning = "Nearly identical - CLIP will struggle to distinguish"
                    elif similarity > 0.90:
                        level = "HIGH"
                        warning = "Very similar - May cause confusion"
                    elif similarity > self.threshold:
                        level = "MODERATE_HIGH"
                        warning = "Moderately similar - Watch for confusion"
                    elif similarity > 0.70:
                        level = "MODERATE"
                        warning = None
                    else:
                        level = "LOW"
                        warning = None
                    
                    pairwise_scores.append({
                        'prompt1': prompt1,
                        'prompt2': prompt2,
                        'prompt1_idx': i,
                        'prompt2_idx': j,
                        'similarity': similarity,
                        'level': level,
                        'warning': warning
                    })
        
        # Sort by similarity (highest first)
        pairwise_scores.sort(key=lambda x: x['similarity'], reverse=True)
        
        return pairwise_scores
    
    def run(self, context: Context) -> Context:
        """
        Compute and analyze prompt similarities.
        
        Args:
            context: Pipeline context
            
        Returns:
            Context with PROMPT_SIMILARITY data added
        """
        # Determine prompts to use
        prompts_to_analyze = self.prompts
        embeddings = None
        
        # Check if embeddings already in context
        if ContextDataType.PROMPT_EMBEDDINGS in context.data:
            prompt_data = context.data[ContextDataType.PROMPT_EMBEDDINGS]
            if 'prompts' in prompt_data and 'embeddings' in prompt_data:
                prompts_to_analyze = prompt_data['prompts']
                embeddings = prompt_data['embeddings']
                logger.info(f"Task '{self.task_id}': Using {len(prompts_to_analyze)} "
                           f"prompts from existing PROMPT_EMBEDDINGS")
        
        # If no prompts available, use configured prompts
        if not prompts_to_analyze:
            prompts_to_analyze = self.prompts
            if not prompts_to_analyze:
                logger.warning(f"Task '{self.task_id}': No prompts to analyze")
                context.data[ContextDataType.PROMPT_SIMILARITY] = {
                    'prompts': [],
                    'embeddings': np.array([]),
                    'similarity_matrix': np.array([]),
                    'pairwise_scores': []
                }
                return context
        
        # Encode prompts if needed
        if embeddings is None:
            logger.info(f"Task '{self.task_id}': Encoding {len(prompts_to_analyze)} prompts")
            embeddings = self._encode_prompts(prompts_to_analyze)
        else:
            # Convert to numpy if needed
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
        
        # Ensure 2D array
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        logger.info(f"Task '{self.task_id}': Embeddings shape: {embeddings.shape}")
        
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(embeddings)
        
        # Analyze pairwise similarities
        pairwise_scores = self._analyze_similarities(prompts_to_analyze, similarity_matrix)
        
        # Log warnings for high similarities
        high_sim_count = sum(1 for score in pairwise_scores 
                           if score['similarity'] > self.threshold)
        if high_sim_count > 0:
            logger.warning(f"Task '{self.task_id}': Found {high_sim_count} prompt pairs "
                         f"with similarity > {self.threshold:.2f}")
            for score in pairwise_scores[:5]:  # Log top 5
                if score['similarity'] > self.threshold:
                    logger.warning(f"  {score['similarity']:.3f}: "
                                 f"'{score['prompt1']}' <-> '{score['prompt2']}'")
        
        # Store results in context
        context.data[ContextDataType.PROMPT_SIMILARITY] = {
            'prompts': prompts_to_analyze,
            'embeddings': embeddings,
            'similarity_matrix': similarity_matrix,
            'pairwise_scores': pairwise_scores,
            'threshold': self.threshold,
            'high_similarity_count': high_sim_count
        }
        
        logger.info(f"Task '{self.task_id}': Analyzed {len(prompts_to_analyze)} prompts, "
                   f"computed {len(pairwise_scores)} pairwise comparisons")
        
        return context
    
    def configure(self, **kwargs) -> None:
        """
        Configure task from parameters.
        
        Args:
            **kwargs: Configuration parameters with optional keys:
                - prompts: List[str] of text prompts
                - threshold: float similarity threshold
                - clip_model: str reference to clip model (ignored, set directly)
        """
        if 'prompts' in kwargs:
            if isinstance(kwargs['prompts'], str):
                # Split comma-separated string
                self.prompts = [p.strip() for p in kwargs['prompts'].split(',')]
            else:
                self.prompts = kwargs['prompts']
            logger.info(f"Task '{self.task_id}': Updated prompts to {len(self.prompts)} items")
        
        if 'threshold' in kwargs:
            self.threshold = float(kwargs['threshold'])
            logger.info(f"Task '{self.task_id}': Updated threshold={self.threshold:.2f}")


def print_similarity_analysis(similarity_data: Dict, 
                             top_n: int = 10,
                             show_matrix: bool = True,
                             color_coded: bool = True):
    """
    Pretty-print similarity analysis results.
    
    Args:
        similarity_data: Dict from ContextDataType.PROMPT_SIMILARITY
        top_n: Number of top similar pairs to show
        show_matrix: Whether to print full similarity matrix
        color_coded: Use emoji color coding for similarity levels
    """
    prompts = similarity_data.get('prompts', [])
    similarity_matrix = similarity_data.get('similarity_matrix', np.array([]))
    pairwise_scores = similarity_data.get('pairwise_scores', [])
    threshold = similarity_data.get('threshold', 0.85)
    high_sim_count = similarity_data.get('high_similarity_count', 0)
    
    print("\n" + "=" * 80)
    print("PROMPT SIMILARITY ANALYSIS")
    print("=" * 80)
    print(f"\nAnalyzed {len(prompts)} prompts")
    print(f"Threshold: {threshold:.2f}")
    print(f"High similarity pairs (>{threshold:.2f}): {high_sim_count}")
    
    # Print prompts
    print(f"\nPrompts:")
    for i, prompt in enumerate(prompts):
        print(f"  [{i}] {prompt}")
    
    # Print top similar pairs
    if pairwise_scores:
        print(f"\n{'-' * 80}")
        print(f"TOP {top_n} MOST SIMILAR PAIRS:")
        print(f"{'-' * 80}")
        
        for i, score in enumerate(pairwise_scores[:top_n], 1):
            sim = score['similarity']
            level = score['level']
            warning = score['warning']
            
            # Color coding
            if color_coded:
                if sim > 0.95:
                    marker = "🔴"  # Red - Very high
                elif sim > 0.90:
                    marker = "🟠"  # Orange - High
                elif sim > threshold:
                    marker = "🟡"  # Yellow - Moderate-high
                elif sim > 0.70:
                    marker = "🟢"  # Green - Moderate
                else:
                    marker = "🔵"  # Blue - Low
            else:
                marker = f"[{level}]"
            
            print(f"\n{i}. {marker} Similarity: {sim:.3f}")
            print(f"   '{score['prompt1']}'")
            print(f"   '{score['prompt2']}'")
            if warning:
                print(f"   ⚠️  {warning}")
    
    # Print similarity matrix
    if show_matrix and len(similarity_matrix) > 0:
        print(f"\n{'-' * 80}")
        print("FULL SIMILARITY MATRIX:")
        print(f"{'-' * 80}")
        
        # Header
        print("\n     ", end="")
        for i in range(len(prompts)):
            print(f"[{i:2d}] ", end="")
        print()
        
        # Matrix rows
        for i in range(len(prompts)):
            print(f"[{i:2d}]  ", end="")
            for j in range(len(prompts)):
                sim = similarity_matrix[i, j]
                if i == j:
                    print("---- ", end="")  # Diagonal
                else:
                    print(f"{sim:.2f} ", end="")
            print()
    
    # Summary statistics
    if len(pairwise_scores) > 0:
        similarities = [s['similarity'] for s in pairwise_scores]
        print(f"\n{'-' * 80}")
        print("STATISTICS:")
        print(f"{'-' * 80}")
        print(f"  Mean similarity: {np.mean(similarities):.3f}")
        print(f"  Median similarity: {np.median(similarities):.3f}")
        print(f"  Std deviation: {np.std(similarities):.3f}")
        print(f"  Min similarity: {np.min(similarities):.3f}")
        print(f"  Max similarity: {np.max(similarities):.3f}")
        
        # Distribution
        very_high = sum(1 for s in similarities if s > 0.95)
        high = sum(1 for s in similarities if 0.90 < s <= 0.95)
        mod_high = sum(1 for s in similarities if threshold < s <= 0.90)
        moderate = sum(1 for s in similarities if 0.70 < s <= threshold)
        low = sum(1 for s in similarities if s <= 0.70)
        
        print(f"\nDistribution:")
        print(f"  Very high (>0.95): {very_high} pairs")
        print(f"  High (0.90-0.95): {high} pairs")
        print(f"  Moderate-high ({threshold:.2f}-0.90): {mod_high} pairs")
        print(f"  Moderate (0.70-{threshold:.2f}): {moderate} pairs")
        print(f"  Low (≤0.70): {low} pairs")
    
    print("\n" + "=" * 80)
