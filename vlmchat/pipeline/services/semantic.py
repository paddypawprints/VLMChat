"""
Semantic similarity service for clustering.

Provides CLIP-based semantic similarity queries for category pairs.
Uses lazy initialization and caches category pair embeddings.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import torch

from ..models.clip_text import ClipTextOpenClip, ClipTextTensorRT
from ..categories import CocoCategory

logger = logging.getLogger(__name__)


class ClipSemanticService:
    """
    CLIP-based semantic similarity service.
    
    Provides query interface for semantic costs between category pairs.
    Lazy-loads category pair embeddings and caches results.
    """
    
    def __init__(self,
                 clip_text_model,  # ClipTextOpenClip or ClipTextTensorRT
                 cache_path: str = "~/.cache/vlmchat/category_pair_embeddings.npz"):
        """
        Initialize semantic service.
        
        Args:
            clip_text_model: CLIP text encoder (OpenClip or TensorRT)
            cache_path: Path to category pair embeddings cache (.npz format)
        """
        self.clip_model = clip_text_model
        self.cache_path = Path(cache_path).expanduser()
        
        # Lazy-loaded data
        self._pair_embeddings: Optional[Dict[str, np.ndarray]] = None
        self._categories = [cat.label for cat in CocoCategory]
        
        # Cache for query_all_pairs to avoid recomputing when prompts unchanged
        self._last_prompts: Optional[Tuple[str, ...]] = None
        self._last_as_probabilities: Optional[bool] = None
        self._cached_results: Optional[Dict[str, List[Tuple[str, str, float]]]] = None
        
        logger.info(f"ClipSemanticService initialized (cache: {self.cache_path})")
    
    def _ensure_cache_loaded(self) -> None:
        """Lazy-load category pair embeddings from cache."""
        if self._pair_embeddings is not None:
            return
        
        if self.cache_path.exists():
            logger.info(f"Loading category pair embeddings from {self.cache_path}")
            data = np.load(str(self.cache_path), allow_pickle=True)
            # Reconstruct dict from keys and embeddings arrays
            keys = data['keys']
            embeddings = data['embeddings'].astype(np.float32)  # Convert back to float32
            self._pair_embeddings = {str(k): emb for k, emb in zip(keys, embeddings)}
            logger.info(f"Loaded {len(self._pair_embeddings)} category pair embeddings")
        else:
            logger.info(f"Cache not found, building category pair embeddings...")
            self._pair_embeddings = self._build_pair_embeddings()
            self._save_cache()
    
    def _build_pair_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Build embeddings for all category pairs.
        
        Returns:
            Dictionary mapping "cat_a and cat_b" to embedding vector (numpy array)
        """
        pair_texts = []
        
        # Generate all pair combinations (including same-category)
        for i, cat_a in enumerate(self._categories):
            for j in range(i, len(self._categories)):
                cat_b = self._categories[j]
                if cat_a == cat_b:
                    pair_texts.append(cat_a)
                else:
                    pair_texts.append(f"{cat_a} and {cat_b}")
        
        logger.info(f"Encoding {len(pair_texts)} category pair texts...")
        
        # Encode all pairs
        embeddings_tensor = self.clip_model.encode(pair_texts, use_cache=False)
        
        # Convert to dict with numpy arrays
        pair_embeddings = {}
        for text, embedding in zip(pair_texts, embeddings_tensor):
            pair_embeddings[text] = embedding.cpu().numpy()
        
        logger.info(f"Built {len(pair_embeddings)} category pair embeddings")
        return pair_embeddings
    
    def _save_cache(self) -> None:
        """Save category pair embeddings to cache file (.npz format)."""
        if self._pair_embeddings is None:
            return
        
        # Create cache directory if it doesn't exist
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving category pair embeddings to {self.cache_path}")
        
        # Convert dict to arrays for compact storage
        keys = np.array(list(self._pair_embeddings.keys()))
        embeddings = np.array(list(self._pair_embeddings.values()), dtype=np.float16)
        
        np.savez_compressed(str(self.cache_path), keys=keys, embeddings=embeddings)
        
        # Calculate size
        size_mb = self.cache_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {len(self._pair_embeddings)} embeddings ({size_mb:.1f} MB)")
    
    
    def query(self, 
              cat_a: str, 
              cat_b: str, 
              prompts: List[str]) -> List[Tuple[str, float]]:
        """
        Query semantic similarity for a category pair against prompts.
        
        Args:
            cat_a: First category label (e.g., "person")
            cat_b: Second category label (e.g., "horse")
            prompts: List of text prompts to match against
            
        Returns:
            List of (prompt, similarity) tuples sorted by similarity (best first)
            Similarity is in [0, 1] where 1.0 = perfect match
        """
        # Ensure cache is loaded
        self._ensure_cache_loaded()
        assert self._pair_embeddings is not None  # Type narrowing
        
        # Handle empty prompts
        if not prompts:
            return []
        
        # Get pair text (same format as cache)
        if cat_a == cat_b:
            pair_text = cat_a
        else:
            # Try both orderings (cache might have either)
            pair_text = f"{cat_a} and {cat_b}"
            if pair_text not in self._pair_embeddings:
                pair_text = f"{cat_b} and {cat_a}"
        
        # Check if pair exists in cache
        if pair_text not in self._pair_embeddings:
            logger.warning(f"Category pair not in cache: {cat_a}, {cat_b}")
            return [(p, 0.0) for p in prompts]
        
        # Get pair embedding (convert numpy to tensor)
        pair_embedding = torch.from_numpy(self._pair_embeddings[pair_text]).float()
        
        # Encode prompts
        prompt_embeddings = self.clip_model.encode(prompts, use_cache=True)
        
        # Compute similarities
        similarities = (pair_embedding @ prompt_embeddings.T).cpu().numpy()
        
        # Sort by similarity (descending)
        results = [(prompts[i], float(similarities[i])) for i in range(len(prompts))]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def query_all_pairs(self, 
                       prompts: List[str],
                       as_probabilities: bool = False,
                       prob_center: Optional[float] = None,
                       prob_gain: float = 10.0) -> Dict[str, List[Tuple[str, str, float]]]:
        """
        Find best matching category pairs for each prompt across all cached pairs.
        
        Args:
            prompts: List of text prompts to match
            as_probabilities: If True, convert similarities to probabilities using sigmoid
            prob_center: Center point for sigmoid. If None, uses median of top similarities.
                        Recommended: None (auto) or 0.65 (fixed)
            prob_gain: Steepness of sigmoid curve (default 10.0)
            
        Returns:
            Dictionary mapping prompt -> list of (cat_a, cat_b, score) tuples
            sorted by score (best first). Score is either raw similarity [0.4-0.9]
            or probability [0-1] if as_probabilities=True.
            
        Note:
            Results are cached based on prompts. If the same prompts are queried
            again, cached results are returned without recomputing similarities.
            
            Probability conversion uses sigmoid: 1 / (1 + exp(-gain * (sim - center)))
            With auto center (median) and gain=10:
              - Similarities above median → prob > 0.50
              - Similarities at median → prob = 0.50
              - Similarities below median → prob < 0.50
              - Creates robust threshold around median, compresses extremes
            
        Example:
            >>> service.query_all_pairs(["person riding horse"], as_probabilities=True)
            {
                "person riding horse": [
                    ("person", "horse", 0.8954),  # probability, not similarity
                    ("person", "person", 0.7420),
                    ...
                ]
            }
        """
        if not prompts:
            return {}
        
        # Check if we can use cached results (including as_probabilities setting)
        prompts_tuple = tuple(prompts)
        if (self._last_prompts == prompts_tuple and 
            self._last_as_probabilities == as_probabilities and
            self._cached_results is not None):
            logger.debug(f"Using cached results for {len(prompts)} prompts")
            return self._cached_results
        
        # Ensure cache is loaded
        self._ensure_cache_loaded()
        assert self._pair_embeddings is not None
        
        logger.debug(f"Computing similarities for {len(prompts)} prompts vs 3,240 pairs")
        
        # Encode all prompts once
        prompt_embeddings = self.clip_model.encode(prompts, use_cache=True)
        
        # Convert all pair embeddings to tensor (N_pairs x 512)
        pair_texts = list(self._pair_embeddings.keys())
        pair_tensor = torch.stack([
            torch.from_numpy(self._pair_embeddings[pt]).float() 
            for pt in pair_texts
        ])
        
        # Compute all similarities: (N_prompts x 512) @ (512 x N_pairs) -> (N_prompts x N_pairs)
        similarities = (prompt_embeddings @ pair_tensor.T).cpu().numpy()
        
        # Build results for each prompt
        results = {}
        for i, prompt in enumerate(prompts):
            # Get similarities for this prompt
            prompt_sims = similarities[i]
            
            # Parse pair texts back to categories and create ranked list
            ranked_pairs = []
            for j, pair_text in enumerate(pair_texts):
                if " and " in pair_text:
                    cat_a, cat_b = pair_text.split(" and ", 1)
                else:
                    cat_a = cat_b = pair_text
                ranked_pairs.append((cat_a, cat_b, float(prompt_sims[j])))
            
            # Convert to probabilities if requested
            if as_probabilities:
                # Auto-detect center from median if not specified
                if prob_center is None:
                    scores = np.array([score for _, _, score in ranked_pairs])
                    prob_center = float(np.median(scores))
                    logger.debug(f"Using median similarity {prob_center:.4f} as sigmoid center")
                
                # Apply sigmoid: 1 / (1 + exp(-gain * (sim - center)))
                scores = np.array([score for _, _, score in ranked_pairs])
                probs = 1.0 / (1.0 + np.exp(-prob_gain * (scores - prob_center)))
                ranked_pairs = [(cat_a, cat_b, float(prob)) 
                               for (cat_a, cat_b, _), prob in zip(ranked_pairs, probs)]
            
            # Sort by score (descending)
            ranked_pairs.sort(key=lambda x: x[2], reverse=True)
            results[prompt] = ranked_pairs
        
        # Cache results for next call
        self._last_prompts = prompts_tuple
        self._last_as_probabilities = as_probabilities
        self._cached_results = results
        
        return results
    
    def get_best_similarity(self, cat_a: str, cat_b: str, prompts: List[str]) -> float:
        """
        Get best similarity score for category pair against prompts.
        
        Args:
            cat_a: First category label
            cat_b: Second category label
            prompts: List of text prompts
            
        Returns:
            Best similarity score in [0, 1]
        """
        results = self.query(cat_a, cat_b, prompts)
        return results[0][1] if results else 0.0
    
    def pre_build_cache(self) -> None:
        """Pre-build and save category pair embeddings cache."""
        logger.info("Pre-building category pair embeddings cache...")
        self._pair_embeddings = self._build_pair_embeddings()
        self._save_cache()
        logger.info("Cache pre-built successfully")
    
    def __repr__(self) -> str:
        return f"ClipSemanticService(cache={self.cache_path.name})"


# Main: Pre-build cache
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pre-build CLIP semantic similarity cache")
    parser.add_argument("--tensorrt", action="store_true", help="Use TensorRT backend")
    parser.add_argument("--engine", type=str, 
                       default="/home/patrick/Dev/model-rt-build/platform/jetson/mobile-clip/text_fp16.engine",
                       help="Path to TensorRT engine (if --tensorrt)")
    parser.add_argument("--cache", type=str, default="category_pair_embeddings.json",
                       help="Path to cache file")
    args = parser.parse_args()
    
    print("=" * 70)
    print("CLIP SEMANTIC SERVICE - Cache Builder")
    print("=" * 70)
    
    # Initialize CLIP text model
    if args.tensorrt:
        print(f"\n✓ Loading TensorRT engine: {args.engine}")
        clip_model = ClipTextTensorRT(
            engine_path=args.engine,
            model_name="MobileCLIP2-S0"
        )
    else:
        print("\n✓ Loading OpenCLIP model: MobileCLIP2-S0")
        clip_model = ClipTextOpenClip(
            model_name="MobileCLIP2-S0",
            device="cpu"
        )
    
    # Create service
    print(f"\n✓ Creating semantic service (cache: {args.cache})")
    service = ClipSemanticService(clip_model, cache_path=args.cache)
    
    # Build cache
    print("\n✓ Building category pair embeddings...")
    service.pre_build_cache()
    
    # Test query
    print("\n--- Testing Query ---")
    test_prompts = ["person riding horse", "cowboy", "horse", "banana"]
    results = service.query("person", "horse", test_prompts)
    
    print(f"Query: person + horse vs {test_prompts}")
    for prompt, sim in results:
        print(f"  '{prompt}': {sim:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ CACHE BUILT SUCCESSFULLY")
    print("=" * 70)
    print(f"Cache saved to: {args.cache}")
