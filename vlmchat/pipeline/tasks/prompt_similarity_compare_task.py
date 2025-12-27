"""
Prompt Similarity Comparison Task

Computes pairwise similarity between prompt embeddings to analyze semantic relationships.

This task takes encoded prompts from context (via ClipTextEncoderTask) and computes
all pairwise similarities, identifying which prompts are too similar (will confuse CLIP)
and which are well-separated.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict

from ..task_base import BaseTask, Context, ContextDataType, register_task

logger = logging.getLogger(__name__)


@register_task('prompt_similarity')
class PromptSimilarityCompareTask(BaseTask):
    """
    Analyzes similarity between prompt embeddings.
    
    This task:
    1. Reads prompt embeddings from context (from ClipTextEncoderTask)
    2. Computes pairwise cosine similarities
    3. Categorizes similarity levels
    4. Stores detailed analysis in context
    
    Expects in context: ContextDataType.PROMPT_EMBEDDINGS
    {
        'prompts': List[str],
        'embeddings': np.ndarray,  # Shape: (N, embedding_dim)
    }
    
    Output stored in ContextDataType.PROMPT_SIMILARITY:
    {
        'prompts': List[str],
        'similarity_matrix': np.ndarray,  # Shape: (N, N)
        'pairwise_scores': List[Dict],  # Detailed pairwise comparisons
        'threshold': float,
        'high_similarity_count': int,
    }
    
    Example:
        compare_task = PromptSimilarityCompareTask(
            threshold=0.85,  # Warn if similarity exceeds this
            task_id="similarity_compare"
        )
    """
    
    def __init__(self, 
                 threshold: float = 0.85,
                 task_id: str = "prompt_similarity_compare"):
        """
        Initialize prompt similarity comparison task.
        
        Args:
            threshold: Similarity threshold for warnings (0.0-1.0)
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.threshold = threshold
        
        # Define contracts
        self.input_contract = {
            ContextDataType.PROMPT_EMBEDDINGS: dict
        }
        self.output_contract = {
            ContextDataType.PROMPT_SIMILARITY: dict
        }
        
        logger.info(f"PromptSimilarityCompareTask '{task_id}' initialized with "
                   f"threshold={threshold:.2f}")
    
    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix.
        
        Args:
            embeddings: np.ndarray of shape (N, embedding_dim)
            
        Returns:
            np.ndarray of shape (N, N) with cosine similarities
        """
        # Normalize embeddings to unit vectors if not already
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = embeddings / norms
        
        # Cosine similarity = dot product of normalized vectors
        similarity_matrix = normalized @ normalized.T
        
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
        # Get embeddings from context
        if ContextDataType.PROMPT_EMBEDDINGS not in context.data:
            raise ValueError(f"Task '{self.task_id}': PROMPT_EMBEDDINGS not found in context. "
                           f"Run ClipTextEncoderTask first.")
        
        prompt_data = context.data[ContextDataType.PROMPT_EMBEDDINGS]
        
        if 'prompts' not in prompt_data or 'embeddings' not in prompt_data:
            raise ValueError(f"Task '{self.task_id}': Invalid PROMPT_EMBEDDINGS format. "
                           f"Expected 'prompts' and 'embeddings' keys.")
        
        prompts = prompt_data['prompts']
        embeddings = prompt_data['embeddings']
        
        if not prompts or len(embeddings) == 0:
            logger.warning(f"Task '{self.task_id}': No prompts to analyze")
            context.data[ContextDataType.PROMPT_SIMILARITY] = {
                'prompts': [],
                'similarity_matrix': np.array([]),
                'pairwise_scores': [],
                'threshold': self.threshold,
                'high_similarity_count': 0
            }
            return context
        
        # Convert to numpy if needed
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        # Ensure 2D array
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        logger.info(f"Task '{self.task_id}': Analyzing {len(prompts)} prompts, "
                   f"embeddings shape: {embeddings.shape}")
        
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(embeddings)
        
        # Analyze pairwise similarities
        pairwise_scores = self._analyze_similarities(prompts, similarity_matrix)
        
        # Count high similarities
        high_sim_count = sum(1 for score in pairwise_scores 
                           if score['similarity'] > self.threshold)
        
        if high_sim_count > 0:
            logger.warning(f"Task '{self.task_id}': Found {high_sim_count} prompt pairs "
                         f"with similarity > {self.threshold:.2f}")
            for score in pairwise_scores[:5]:  # Log top 5
                if score['similarity'] > self.threshold:
                    logger.warning(f"  {score['similarity']:.3f}: "
                                 f"'{score['prompt1']}' <-> '{score['prompt2']}'")
        else:
            logger.info(f"Task '{self.task_id}': All prompt pairs below threshold "
                       f"({self.threshold:.2f})")
        
        # Store results in context
        context.data[ContextDataType.PROMPT_SIMILARITY] = {
            'prompts': prompts,
            'similarity_matrix': similarity_matrix,
            'pairwise_scores': pairwise_scores,
            'threshold': self.threshold,
            'high_similarity_count': high_sim_count
        }
        
        logger.info(f"Task '{self.task_id}': Computed {len(pairwise_scores)} "
                   f"pairwise comparisons")
        
        return context
    
    def configure(self, **kwargs) -> None:
        """
        Configure task from parameters.
        
        Args:
            **kwargs: Configuration parameters with optional keys:
                - threshold: float similarity threshold
        """
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
