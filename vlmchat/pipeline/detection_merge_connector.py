"""
Detection merge connector that intelligently combines cluster and single detections.

This connector prioritizes detections with children (cluster results) and fills
remaining slots with the highest confidence single detections to reach a target count.
"""

from typing import List, Dict
from .task_base import Connector, Context, ContextDataType
from ..object_detector.detection_base import Detection


class DetectionMergeConnector(Connector):
    """
    Merges detection lists by prioritizing clusters over single detections.
    
    When merging multiple detection lists, this connector:
    1. Collects all detections with children (cluster results)
    2. Collects all detections without children (single object detections)
    3. Takes all cluster detections
    4. Fills remaining slots with highest confidence single detections
    5. Returns exactly target_count detections (or fewer if not enough available)
    
    This is useful after clustering to maintain both cluster context and 
    high-confidence individual detections for downstream processing.
    
    Example:
        # After clustering: 2 cluster detections + 10 single detections
        # With target_count=8: Returns 2 clusters + top 6 single detections
        
        merge_conn = DetectionMergeConnector(
            task_id="detection_merge",
            target_count=8
        )
    """
    
    def __init__(self, task_id: str, target_count: int = 8):
        """
        Initialize detection merge connector.
        
        Args:
            task_id: Unique identifier for this connector
            target_count: Target number of detections to return
        """
        super().__init__(task_id)
        self.target_count = target_count
        
        # Define contracts
        self.input_contract = {ContextDataType.DETECTIONS: list}
        self.output_contract = {ContextDataType.DETECTIONS: list}
    
    def merge_strategy(self, contexts: List[Context]) -> Context:
        """
        Merge detection contexts by prioritizing clusters.
        
        This is the proper override method for Connector subclasses.
        
        Args:
            contexts: List of contexts from upstream tasks
            
        Returns:
            Merged context with intelligently selected detections
        """
        return self.merge_contexts(contexts)
    
    def merge_contexts(self, contexts: List[Context]) -> Context:
        """
        Merge detection contexts by prioritizing clusters.
        
        Args:
            contexts: List of contexts from upstream tasks
            
        Returns:
            Merged context with intelligently selected detections
        """
        # Start with first context as base
        if not contexts:
            return Context()
        
        merged_context = Context()
        merged_context.data = contexts[0].data.copy()
        
        # Collect all detections from all contexts
        all_detections: List[Detection] = []
        for ctx in contexts:
            if ContextDataType.DETECTIONS in ctx.data:
                detections = ctx.data[ContextDataType.DETECTIONS]
                if isinstance(detections, list):
                    all_detections.extend(detections)
        
        # Separate cluster detections (with children) from single detections
        cluster_detections: List[Detection] = []
        single_detections: List[Detection] = []
        
        for det in all_detections:
            if hasattr(det, 'children') and len(det.children) > 0:
                cluster_detections.append(det)
            else:
                single_detections.append(det)
        
        # Sort single detections by confidence (highest first)
        single_detections.sort(key=lambda d: d.conf, reverse=True)
        
        # Build final detection list
        selected_detections: List[Detection] = []
        
        # 1. Add all cluster detections first
        selected_detections.extend(cluster_detections)
        
        # 2. Fill remaining slots with highest confidence singles
        remaining_slots = self.target_count - len(cluster_detections)
        if remaining_slots > 0:
            selected_detections.extend(single_detections[:remaining_slots])
        
        # Store merged detections
        merged_context.data[ContextDataType.DETECTIONS] = selected_detections
        
        return merged_context
    
    def split_context(self, context: Context, num_outputs: int) -> List[Context]:
        """
        Split context for downstream tasks (broadcast to all).
        
        Args:
            context: Context to split
            num_outputs: Number of output contexts needed
            
        Returns:
            List of contexts (all identical - broadcast pattern)
        """
        return [context] * num_outputs
    
    def __str__(self) -> str:
        """String representation."""
        return f"DetectionMergeConnector(id={self.task_id}, target={self.target_count})"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return self.__str__()


if __name__ == "__main__":
    # Example usage
    from ..object_detector.detection_base import Detection
    
    print("\n--- Detection Merge Connector Example ---\n")
    
    # Create some test detections
    # Cluster detections (with children)
    cluster1 = Detection(box=(10, 10, 50, 50), object_category="person", conf=0.92)
    child1a = Detection(box=(10, 10, 30, 50), object_category="person", conf=0.85)
    child1b = Detection(box=(30, 10, 50, 50), object_category="person", conf=0.88)
    cluster1.children = [child1a, child1b]
    
    cluster2 = Detection(box=(100, 100, 150, 150), object_category="horse", conf=0.89)
    child2a = Detection(box=(100, 100, 125, 150), object_category="horse", conf=0.82)
    child2b = Detection(box=(125, 100, 150, 150), object_category="horse", conf=0.86)
    cluster2.children = [child2a, child2b]
    
    # Single detections (no children)
    single1 = Detection(box=(200, 200, 250, 250), object_category="dog", conf=0.95)
    single2 = Detection(box=(300, 300, 350, 350), object_category="cat", conf=0.78)
    single3 = Detection(box=(400, 400, 450, 450), object_category="bird", conf=0.88)
    single4 = Detection(box=(500, 500, 550, 550), object_category="car", conf=0.72)
    single5 = Detection(box=(600, 600, 650, 650), object_category="bicycle", conf=0.81)
    
    # Create context with all detections
    ctx = Context()
    ctx.data[ContextDataType.DETECTIONS] = [
        cluster1, cluster2, single1, single2, single3, single4, single5
    ]
    
    # Create merge connector with target of 5
    merge_conn = DetectionMergeConnector(task_id="merge", target_count=5)
    
    print(f"Merge Connector: {merge_conn}")
    print(f"\nInput detections: {len(ctx.data[ContextDataType.DETECTIONS])}")
    print(f"  - Clusters (with children): 2")
    print(f"  - Single detections: 5")
    print(f"\nTarget count: {merge_conn.target_count}")
    
    # Merge (with single context, just passes through with selection)
    merged = merge_conn.merge_contexts([ctx])
    
    result_dets = merged.data[ContextDataType.DETECTIONS]
    print(f"\nOutput detections: {len(result_dets)}")
    
    clusters_out = sum(1 for d in result_dets if hasattr(d, 'children') and len(d.children) > 0)
    singles_out = len(result_dets) - clusters_out
    
    print(f"  - Clusters: {clusters_out}")
    print(f"  - Singles: {singles_out}")
    print(f"\nDetailed output:")
    for i, det in enumerate(result_dets):
        has_children = hasattr(det, 'children') and len(det.children) > 0
        det_type = "CLUSTER" if has_children else "SINGLE"
        print(f"  [{i+1}] {det_type:8} {det.object_category:10} conf={det.conf:.2f}")
        if has_children:
            print(f"       └─ {len(det.children)} children")
    
    print("\n✓ Merge prioritized clusters and selected top singles by confidence")
