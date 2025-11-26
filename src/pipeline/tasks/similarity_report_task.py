"""
Similarity Report Task

Generates human-readable report from CLIP similarity analysis.
Takes SIMILARITY_SCORES, TEXT prompts, DETECTIONS, and IMAGE to create
a formatted text report showing which detections match which prompts.
"""

import numpy as np
import logging
from typing import List, Dict, Optional

from ..task_base import BaseTask, Context, ContextDataType, register_task
from ...object_detector.detection_base import Detection

logger = logging.getLogger(__name__)


@register_task('similarity_report')
class SimilarityReportTask(BaseTask):
    """
    Generates human-readable CLIP similarity report.
    
    This task:
    1. Takes SIMILARITY_SCORES (matrix with labels)
    2. Takes DETECTIONS (with YOLO labels and confidences)
    3. Takes TEXT (original prompts)
    4. Formats a readable report showing matches
    5. Outputs to TEXT for display/logging
    
    Report format:
    ```
    CLIP Similarity Analysis
    ========================
    
    Prompts analyzed: person riding horse, horse
    
    Detection Results:
    ------------------
    
    Detection #0 (person @ 0.89 conf):
      - "person riding horse": 0.87 (best match)
      - "horse": 0.43
    
    Detection #1 (horse @ 0.92 conf):
      - "horse": 0.91 (best match)
      - "person riding horse": 0.78
    
    Summary: 3 detections matched across 2 prompts
    ```
    
    Parameters:
        min_score: Minimum similarity score to include (default: 0.0)
        max_matches: Maximum matches to show per detection (default: all)
    """
    
    def __init__(self, 
                 min_score: float = 0.0,
                 max_matches: Optional[int] = None,
                 task_id: str = "similarity_report"):
        """
        Initialize similarity report task.
        
        Args:
            min_score: Minimum similarity score to include in report (default: 0.0)
            max_matches: Maximum matches to show per detection (default: None = all)
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.min_score = min_score
        self.max_matches = max_matches
        
        # Define contracts
        self.input_contract = {
            ContextDataType.SIMILARITY_SCORES: dict,
            ContextDataType.DETECTIONS: list,
            ContextDataType.TEXT: list
        }
        
        self.output_contract = {
            ContextDataType.TEXT: list  # Report text
        }
        
        logger.info(f"SimilarityReportTask '{task_id}' initialized with "
                   f"min_score={min_score}, max_matches={max_matches}")
    
    def run(self, context: Context) -> Context:
        """
        Generate similarity report.
        
        Args:
            context: Pipeline context with SIMILARITY_SCORES, DETECTIONS, TEXT
            
        Returns:
            Context with formatted report added to TEXT
        """
        # Get similarity scores
        similarity_data = context.data.get(ContextDataType.SIMILARITY_SCORES)
        if not similarity_data or not isinstance(similarity_data, dict):
            logger.warning(f"Task '{self.task_id}': No SIMILARITY_SCORES found")
            context.data[ContextDataType.TEXT] = ["No similarity data available"]
            return context
        
        matrix = similarity_data.get('matrix')
        texts = similarity_data.get('texts', [])
        detection_ids = similarity_data.get('detection_ids', [])
        probabilities = similarity_data.get('probabilities')  # May be None
        temperature = similarity_data.get('temperature')
        
        # Decide whether to show probabilities or raw similarities
        use_probabilities = probabilities is not None
        
        logger.info(f"Task '{self.task_id}': Received matrix shape {matrix.shape if hasattr(matrix, 'shape') else 'N/A'}, "
                   f"{len(texts)} prompts, {len(detection_ids)} detections"
                   f"{', with probabilities' if use_probabilities else ''}")
        logger.info(f"Task '{self.task_id}': Similarity matrix values: {matrix.tolist() if hasattr(matrix, 'tolist') else matrix}")
        
        if matrix is None or len(matrix) == 0:
            logger.warning(f"Task '{self.task_id}': Empty similarity matrix")
            context.data[ContextDataType.TEXT] = ["No similarity results"]
            return context
        
        # Get detections for metadata (YOLO labels, confidences)
        detections = context.data.get(ContextDataType.DETECTIONS, [])
        
        # Build detection metadata lookup
        det_metadata = {}
        for i, det in enumerate(detections):
            det_id = detection_ids[i] if i < len(detection_ids) else f"det_{i}"
            det_metadata[det_id] = {
                'label': det.object_category if hasattr(det, 'object_category') else 'unknown',
                'conf': det.conf if hasattr(det, 'conf') else 0.0,
                'box': det.box if hasattr(det, 'box') else None
            }
        
        # Use texts from SIMILARITY_SCORES (authoritative source)
        prompts_str = ', '.join(texts)
        
        # Generate report
        report_lines = []
        report_lines.append("CLIP Similarity Analysis")
        report_lines.append("=" * 50)
        report_lines.append("")
        report_lines.append(f"Prompts analyzed: {prompts_str}")
        if use_probabilities and temperature:
            report_lines.append(f"Using softmax probabilities (temperature={temperature})")
        report_lines.append("")
        report_lines.append("Detection Results:")
        report_lines.append("-" * 50)
        report_lines.append("")
        
        # Iterate through detections
        total_matches = 0
        for det_idx, det_id in enumerate(detection_ids):
            # Get metadata
            metadata = det_metadata.get(det_id, {'label': 'unknown', 'conf': 0.0})
            yolo_label = metadata['label']
            yolo_conf = metadata['conf']
            
            # Get similarity scores for this detection
            det_scores = matrix[:, det_idx] if det_idx < matrix.shape[1] else []
            
            # Get probabilities if available
            if use_probabilities:
                det_probs = probabilities[:, det_idx] if det_idx < probabilities.shape[1] else []
                # Create list of (prompt, score, probability) tuples
                prompt_data = [(texts[i], score, prob) for i, (score, prob) in enumerate(zip(det_scores, det_probs))]
            else:
                # Create list of (prompt, score) tuples  
                prompt_data = [(texts[i], score, None) for i, score in enumerate(det_scores)]
            
            # Filter by minimum score (use raw similarity for threshold)
            prompt_data = [(p, s, pr) for p, s, pr in prompt_data if s >= self.min_score]
            
            if not prompt_data:
                continue  # Skip detections with no matches above threshold
            
            # Sort by score descending (use probability if available, otherwise similarity)
            if use_probabilities:
                prompt_data.sort(key=lambda x: x[2], reverse=True)  # Sort by probability
            else:
                prompt_data.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity
            
            # Limit to max_matches
            if self.max_matches:
                prompt_data = prompt_data[:self.max_matches]
            
            # Format detection header
            report_lines.append(f"Detection #{det_idx} ({yolo_label} @ {yolo_conf:.2f} conf):")
            
            # Format matches
            best_value = prompt_data[0][2] if use_probabilities else prompt_data[0][1]
            for item in prompt_data:
                prompt = item[0]
                similarity = item[1]
                probability = item[2]
                
                current_value = probability if use_probabilities else similarity
                is_best = (current_value == best_value)
                best_marker = " (best match)" if is_best else ""
                
                if use_probabilities:
                    report_lines.append(f"  - \"{prompt}\": {probability:.3f} (sim: {similarity:.2f}){best_marker}")
                else:
                    report_lines.append(f"  - \"{prompt}\": {similarity:.2f}{best_marker}")
                total_matches += 1
            
            report_lines.append("")
        
        # Summary
        report_lines.append("-" * 50)
        report_lines.append(f"Summary: {len(detection_ids)} detections, "
                          f"{len(texts)} prompts, {total_matches} matches above {self.min_score:.2f}")
        
        # Join into single text
        report_text = "\n".join(report_lines)
        
        # Store in TEXT context (as list with single item)
        context.data[ContextDataType.TEXT] = [report_text]
        
        logger.info(f"Task '{self.task_id}': Generated report with {total_matches} matches")
        
        return context
    
    def configure(self, **kwargs) -> None:
        """
        Configure task from parameters.
        
        Args:
            **kwargs: Configuration parameters with optional keys:
                - min_score: float minimum similarity score
                - max_matches: int maximum matches per detection
        """
        if 'min_score' in kwargs:
            self.min_score = float(kwargs['min_score'])
            logger.info(f"Task '{self.task_id}': Updated min_score={self.min_score}")
        
        if 'max_matches' in kwargs:
            self.max_matches = int(kwargs['max_matches'])
            logger.info(f"Task '{self.task_id}': Updated max_matches={self.max_matches}")
    
    def describe(self) -> str:
        """Task description for help system."""
        return ("Generates human-readable report from CLIP similarity analysis. "
                "Shows which detections match which text prompts with scores.")
    
    def describe_parameters(self) -> Dict[str, Dict[str, str]]:
        """Parameter descriptions for help system."""
        return {
            'min_score': {
                'type': 'float',
                'default': '0.0',
                'description': 'Minimum similarity score to include in report'
            },
            'max_matches': {
                'type': 'int',
                'default': 'None',
                'description': 'Maximum number of matches to show per detection'
            }
        }


if __name__ == "__main__":
    # Example usage
    print("\n--- SimilarityReportTask Example ---\n")
    
    # Create test data
    ctx = Context()
    
    # Simulate similarity scores
    matrix = np.array([
        [0.87, 0.91, 0.72],  # "person riding horse" vs 3 detections
        [0.43, 0.88, 0.91]   # "horse" vs 3 detections
    ])
    
    ctx.data[ContextDataType.SIMILARITY_SCORES] = {
        'matrix': matrix,
        'texts': ['person riding horse', 'horse'],
        'detection_ids': ['det_0', 'det_1', 'det_2'],
        'metric': 'cosine_similarity'
    }
    
    # Simulate detections
    from ...object_detector.detection_base import Detection
    detections = [
        Detection(box=(10, 10, 100, 100), object_category="person", conf=0.89),
        Detection(box=(200, 200, 300, 300), object_category="horse", conf=0.92),
        Detection(box=(400, 400, 500, 500), object_category="horse", conf=0.85)
    ]
    ctx.data[ContextDataType.DETECTIONS] = detections
    
    # Simulate original prompts
    ctx.data[ContextDataType.TEXT] = ['person riding horse', 'horse']
    
    # Create and run report task
    report_task = SimilarityReportTask(min_score=0.4, task_id="test_report")
    output_ctx = report_task.run(ctx)
    
    # Display report
    report = output_ctx.data.get(ContextDataType.TEXT, ["No report"])[0]
    print(report)
    
    print("\n✓ SimilarityReportTask generated report")
