"""
Detection Labeler Task

Enriches detections with matched prompts and probabilities from CLIP/FashionCLIP comparator.
Filters detections based on probability threshold and adds label information.
"""

import numpy as np
import logging
from typing import List, Dict, Optional

from ..task_base import BaseTask, Context, ContextDataType, register_task
from src.object_detector.detection_base import Detection

logger = logging.getLogger(__name__)


@register_task('detection_labeler')
class DetectionLabelerTask(BaseTask):
    """
    Labels detections with matched prompts from similarity scores.
    
    This task:
    1. Takes SIMILARITY_SCORES and DETECTIONS from context
    2. For each detection, finds prompts above probability threshold
    3. Adds matched_prompts and match_probabilities attributes to detections
    4. Filters out detections with no matches above threshold
    5. Optionally updates object_category to best matching prompt
    
    Input:
        - DETECTIONS: List[Detection] from detector
        - SIMILARITY_SCORES: Dict with 'probabilities', 'texts', 'detection_ids'
    
    Output:
        - DETECTIONS: List[Detection] with added attributes:
            - detection.matched_prompts: List[str] of matching prompts
            - detection.match_probabilities: List[float] of probabilities
            - detection.object_category: Updated to best match (if update_category=True)
    
    Example:
        labeler = DetectionLabelerTask(
            min_probability=0.15,
            update_category=True,
            task_id="detection_labeler"
        )
    """
    
    def __init__(self, 
                 min_probability: float = 0.15,
                 max_labels: int = 3,
                 update_category: bool = True,
                 task_id: str = "detection_labeler"):
        """
        Initialize detection labeler task.
        
        Args:
            min_probability: Minimum probability threshold for a match (0.0-1.0)
            max_labels: Maximum number of labels to attach per detection
            update_category: If True, update detection.object_category to best match
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.min_probability = min_probability
        self.max_labels = max_labels
        self.update_category = update_category
        
        # Define contracts
        self.input_contract = {
            ContextDataType.DETECTIONS: list,
            ContextDataType.SIMILARITY_SCORES: dict
        }
        
        self.output_contract = {
            ContextDataType.DETECTIONS: list
        }
        
        logger.info(f"DetectionLabelerTask '{task_id}' initialized with "
                   f"min_probability={min_probability}, max_labels={max_labels}, "
                   f"update_category={update_category}")
    
    def run(self, context: Context) -> Context:
        """
        Label detections with matched prompts from similarity scores.
        
        Args:
            context: Pipeline context with DETECTIONS and SIMILARITY_SCORES
            
        Returns:
            Context with filtered and labeled DETECTIONS
        """
        # Get input data
        detections_data = context.data.get(ContextDataType.DETECTIONS, [])
        similarity_scores = context.data.get(ContextDataType.SIMILARITY_SCORES)
        
        # Handle nested structure from ordered_merge (happens when DETECTIONS flows through parallel branches)
        # OrderedMerge creates [[branch0_detections], [branch1_detections]] but we just want flat list
        if detections_data and isinstance(detections_data[0], list):
            # Flatten nested structure - all branches should have same detections, so take first
            detections = detections_data[0]
            logger.debug(f"Task '{self.task_id}': flattened nested DETECTIONS structure")
        else:
            detections = detections_data
        
        if not detections:
            logger.warning(f"Task '{self.task_id}': No detections in context")
            return context
        
        if not similarity_scores:
            logger.warning(f"Task '{self.task_id}': No similarity scores in context")
            return context
        
        # Extract similarity data
        probabilities = similarity_scores.get('probabilities')
        texts = similarity_scores.get('texts', [])
        detection_ids = similarity_scores.get('detection_ids', [])
        
        if probabilities is None or len(texts) == 0 or len(detection_ids) == 0:
            logger.warning(f"Task '{self.task_id}': Invalid similarity scores format")
            return context
        
        logger.info(f"Task '{self.task_id}': Labeling {len(detections)} detections "
                   f"with {len(texts)} prompts (min_prob={self.min_probability})")
        
        # Debug: Show what IDs we have
        logger.info(f"Task '{self.task_id}': Detection IDs in detections list: {[d.id for d in detections]}")
        logger.info(f"Task '{self.task_id}': Detection IDs in similarity_scores: {detection_ids}")
        
        # Process each detection and find its similarity scores
        # Note: detection_ids in similarity_scores are raw integer IDs (not strings)
        labeled_detections = []
        
        for det_idx, detection in enumerate(detections):
            # Detection ID should match directly (integer)
            det_id = detection.id
            
            # Try to find column index in similarity matrix
            try:
                col_idx = detection_ids.index(det_id)
            except ValueError:
                # Detection ID not found in similarity scores - might be new from clustering
                logger.debug(f"Task '{self.task_id}': Detection ID {det_id} not found in similarity scores")
                continue
            
            # Get probabilities for this detection (column in matrix)
            det_probs = probabilities[:, col_idx]
            
            # Find matches above threshold
            matches = []
            for prompt_idx, prob in enumerate(det_probs):
                if prob >= self.min_probability:
                    matches.append((texts[prompt_idx], float(prob), prompt_idx))
            
            # Skip detections with no matches
            if not matches:
                logger.debug(f"Task '{self.task_id}': Detection {det_id} has no matches "
                           f"above threshold {self.min_probability}")
                continue
            
            # Sort by probability (descending) and limit to max_labels
            matches.sort(key=lambda x: x[1], reverse=True)
            matches = matches[:self.max_labels]
            
            # Add attributes to detection
            detection.matched_prompts = [m[0] for m in matches]
            detection.match_probabilities = [m[1] for m in matches]
            
            # Update category if requested
            if self.update_category and matches:
                detection.object_category = matches[0][0]
            
            labeled_detections.append(detection)
            
            logger.debug(f"Task '{self.task_id}': Labeled {det_id} with "
                        f"{len(matches)} prompts: {detection.matched_prompts}")
        
        # Update context with filtered and labeled detections
        context.data[ContextDataType.DETECTIONS] = labeled_detections
        
        logger.info(f"Task '{self.task_id}': Kept {len(labeled_detections)} detections "
                   f"with matches above threshold (filtered {len(detections) - len(labeled_detections)})")
        
        return context
    
    def configure(self, **kwargs) -> None:
        """
        Configure task from parameters.
        
        Args:
            **kwargs: Configuration parameters with optional keys:
                - min_probability: float threshold for matches
                - max_labels: int maximum labels per detection
                - update_category: bool whether to update object_category
        """
        if 'min_probability' in kwargs:
            self.min_probability = float(kwargs['min_probability'])
            logger.info(f"Task '{self.task_id}': Updated min_probability={self.min_probability}")
        
        if 'max_labels' in kwargs:
            self.max_labels = int(kwargs['max_labels'])
            logger.info(f"Task '{self.task_id}': Updated max_labels={self.max_labels}")
        
        if 'update_category' in kwargs:
            val = kwargs['update_category']
            self.update_category = val if isinstance(val, bool) else str(val).lower() == 'true'
            logger.info(f"Task '{self.task_id}': Updated update_category={self.update_category}")
    
    def describe(self) -> str:
        """Task description for help system."""
        return ("Labels detections with matched prompts from CLIP/FashionCLIP similarity scores. "
                "Filters detections below probability threshold and adds match information.")
    
    def describe_parameters(self) -> Dict[str, Dict[str, str]]:
        """Parameter descriptions for help system."""
        return {
            'min_probability': {
                'type': 'float',
                'default': '0.15',
                'description': 'Minimum probability threshold for a match (0.0-1.0)'
            },
            'max_labels': {
                'type': 'int',
                'default': '3',
                'description': 'Maximum number of labels to attach per detection'
            },
            'update_category': {
                'type': 'bool',
                'default': 'True',
                'description': 'Whether to update detection.object_category to best match'
            }
        }


if __name__ == "__main__":
    # Example usage
    print("\n--- DetectionLabelerTask Example ---\n")
    
    # Create test detections
    det1 = Detection(box=(100, 100, 200, 200), object_category="person", conf=0.9)
    det2 = Detection(box=(300, 150, 400, 250), object_category="person", conf=0.85)
    det3 = Detection(box=(500, 200, 600, 300), object_category="chair", conf=0.8)
    
    # Create test context
    ctx = Context()
    ctx.data[ContextDataType.DETECTIONS] = [det1, det2, det3]
    
    # Create test similarity scores
    # Probabilities: shape (n_prompts, n_detections)
    # 3 prompts × 3 detections
    probabilities = np.array([
        [0.7, 0.05, 0.02],  # "person" prompt
        [0.2, 0.85, 0.03],  # "person with hat" prompt
        [0.1, 0.1, 0.95]    # "chair" prompt
    ])
    
    ctx.data[ContextDataType.SIMILARITY_SCORES] = {
        'probabilities': probabilities,
        'texts': ['person', 'person with hat', 'chair'],
        'detection_ids': [f'detection_{det1.id}', f'detection_{det2.id}', f'detection_{det3.id}'],
        'metric': 'cosine_similarity'
    }
    
    print(f"Input detections: {len(ctx.data[ContextDataType.DETECTIONS])}")
    for det in ctx.data[ContextDataType.DETECTIONS]:
        print(f"  {det}")
    
    # Create and run labeler with threshold 0.15
    labeler = DetectionLabelerTask(min_probability=0.15, max_labels=2, task_id="test_labeler")
    output_ctx = labeler.run(ctx)
    
    # Check results
    labeled_dets = output_ctx.data.get(ContextDataType.DETECTIONS, [])
    print(f"\nLabeled detections: {len(labeled_dets)}")
    for det in labeled_dets:
        print(f"\n  {det}")
        if hasattr(det, 'matched_prompts'):
            print(f"    Matched prompts: {det.matched_prompts}")
            print(f"    Probabilities: {[f'{p:.3f}' for p in det.match_probabilities]}")
            print(f"    Category: {det.object_category}")
    
    print("\n✓ DetectionLabelerTask labeled detections with probability filtering")
