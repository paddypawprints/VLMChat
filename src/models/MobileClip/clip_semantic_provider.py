"""
Connects the CLIPModel to the ObjectClusterer.

This module provides a class that implements the ISemanticCostProvider
interface, using a live CLIPModel as its backend.
"""

import logging
from typing import List, Dict, Optional, Tuple

from object_clusterer import ISemanticCostProvider
from models.clip_model import CLIPModel
from yolo_detector import YoloV8Detector # To get the labels

logger = logging.getLogger(__name__)

class ClipSemanticProvider(ISemanticCostProvider):
    """
    An implementation of ISemanticCostProvider that uses a CLIPModel
    to pre-calculate a semantic cost matrix.
    """
    
    def __init__(self, 
                 clip_model: CLIPModel, 
                 yolo_detector: YoloV8Detector,
                 user_prompts: List[str]):
        """
        Initializes the provider.

        Args:
            clip_model: An initialized CLIPModel instance.
            yolo_detector: An initialized YOLO detector to get the label list from.
            user_prompts: The list of user-defined text prompts to match against.
        """
        self._clip_model = clip_model
        self._yolo_labels = yolo_detector.get_labels()
        self._user_prompts = user_prompts
        self._cost_matrix: Dict[str, Dict[str, float]] = {}
        self._is_ready = False

    def start(self) -> None:
        """
        Builds the semantic cost matrix.
        
        This is a one-time, expensive operation that runs CLIP
        to build the matrix.
        """
        if not self._clip_model.readiness():
            logger.error("ClipSemanticProvider: CLIPModel is not ready.")
            return

        if not self._yolo_labels:
            logger.error("ClipSemanticProvider: YOLO detector did not provide labels.")
            return
            
        if not self._user_prompts:
            logger.warning("ClipSemanticProvider: No user prompts provided. Semantic cost will be 1.0")
            
        logger.info(f"Building semantic cost matrix for {len(self._yolo_labels)} categories...")
        
        try:
            # 1. Pre-calculate all prompt embeddings
            self._clip_model.pre_cache_text_prompts(self._user_prompts)
            
            # Get the cached features
            prompt_features = torch.cat(
                [self._clip_model._text_features_cache[p] for p in self._user_prompts], 
                dim=0
            )

            # 2. Generate all YOLO label pairs
            label_pairs = {} # "person and horse": ("person", "horse")
            for i, cat_a in enumerate(self._yolo_labels):
                for j in range(i, len(self._yolo_labels)):
                    cat_b = self._yolo_labels[j]
                    
                    pair_text = f"{cat_a} and {cat_b}" if cat_a != cat_b else cat_a
                    label_pairs[pair_text] = (cat_a, cat_b)

            # 3. Get text embeddings for all pairs
            pair_texts = list(label_pairs.keys())
            pair_runtime = self._clip_model._runtime_as_clip()
            pair_features = pair_runtime.encode_text(pair_texts)

            # 4. Calculate max similarity for each pair
            # (pair_features @ prompt_features.T) gives a [num_pairs, num_prompts] matrix
            with torch.no_grad():
                similarity_matrix = pair_features @ prompt_features.T
                
                # Find the max similarity (and its index) for each pair
                max_sims, _ = torch.max(similarity_matrix, dim=1) # Shape: [num_pairs]

            # 5. Build the final cost matrix
            self._cost_matrix = {label: {} for label in self._yolo_labels}
            
            for idx, pair_text in enumerate(pair_texts):
                cat_a, cat_b = label_pairs[pair_text]
                max_similarity = max_sims[idx].item()
                
                # Cost = 1.0 - max_similarity
                # A high similarity (1.0) means a low cost (0.0)
                cost = 1.0 - max(0.0, max_similarity) # Clamp at 0
                
                self._cost_matrix[cat_a][cat_b] = cost
                self._cost_matrix[cat_b][cat_a] = cost
                
            self._is_ready = True
            logger.info("Semantic cost matrix built successfully.")
            
        except Exception as e:
            logger.error(f"Failed to build semantic cost matrix: {e}", exc_info=True)
            self._is_ready = False

    def stop(self) -> None:
        """Clears the cost matrix."""
        self._cost_matrix.clear()
        self._is_ready = False
        logger.info("ClipSemanticProvider stopped.")
        
    def get_cost(self, category_a: str, category_b: str) -> float:
        """
Services (WIP)
Gets the pre-calculated cost from the matrix.
        """
        if not self._is_ready:
            # logger.warning("Semantic provider not ready, returning default cost.")
            return 1.0
            
        try:
            return self._cost_matrix[category_a][category_b]
        except KeyError:
            # This can happen if categories are not in the matrix
            # logger.warning(f"No semantic cost for {category_a}/{category_b}, returning default.")
            return 1.0