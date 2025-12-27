"""
Fork connector that duplicates context to multiple parallel branches.

This connector enables fan-out patterns where the same input needs to be
processed by multiple independent tasks in parallel.
"""

from typing import List
from ..core.task_base import Connector, Context


class ForkConnector(Connector):
    """
    Forks a single input context to multiple parallel output paths.
    
    This connector duplicates the input context so that multiple downstream
    tasks can process the same data independently. Each output gets a complete
    copy of the input context.
    
    This is useful for:
    - Processing detections through different algorithms (e.g., clustering vs filtering)
    - Running parallel analysis pipelines on the same image
    - Creating both clustered and individual detection streams
    
    Example:
        # YOLO -> Fork -> [Clusterer, Passthrough] -> Merge
        fork = ForkConnector(task_id="fork", num_outputs=2)
        # Output 0: Goes to Clusterer
        # Output 1: Bypasses clustering (passthrough to merge)
    """
    
    def __init__(self, task_id: str = "fork", num_outputs: int = 2):
        """
        Initialize fork connector.
        
        Args:
            task_id: Unique identifier for this connector (default: "fork")
            num_outputs: Number of parallel output paths (default: 2)
        """
        super().__init__(task_id)
        self.num_outputs = num_outputs
    
    def merge_contexts(self, contexts: List[Context]) -> Context:
        """
        Pass through the first context (fork operates on single input).
        
        Args:
            contexts: List of input contexts (typically just one)
            
        Returns:
            The first context unchanged
        """
        if not contexts:
            return Context()
        return contexts[0]
    
    def split_context(self, context: Context, num_outputs: int) -> List[Context]:
        """
        Duplicate context to multiple outputs.
        
        Args:
            context: Input context to duplicate
            num_outputs: Number of copies to create
            
        Returns:
            List of independent context copies
        """
        # Create deep copies so each branch can modify independently
        return [Context(data=context.data.copy()) for _ in range(num_outputs)]
    
    def __str__(self) -> str:
        """String representation."""
        return f"ForkConnector(id={self.task_id}, outputs={self.num_outputs})"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return self.__str__()


if __name__ == "__main__":
    # Example usage
    from ..object_detector.detection_base import Detection
    from .task_base import ContextDataType
    
    print("\n--- Fork Connector Example ---\n")
    
    # Create test context with detections
    ctx = Context()
    det1 = Detection(box=(10, 10, 50, 50), object_category="person", conf=0.92)
    det2 = Detection(box=(100, 100, 150, 150), object_category="horse", conf=0.89)
    ctx.data[ContextDataType.DETECTIONS] = [det1, det2]
    
    print(f"Input context: {len(ctx.data[ContextDataType.DETECTIONS])} detections")
    
    # Create fork with 3 outputs
    fork = ForkConnector(task_id="test_fork", num_outputs=3)
    print(f"\nFork Connector: {fork}")
    
    # Split context
    outputs = fork.split_context(ctx, 3)
    
    print(f"\nOutput contexts: {len(outputs)}")
    for i, out_ctx in enumerate(outputs):
        det_count = len(out_ctx.data.get(ContextDataType.DETECTIONS, []))
        print(f"  Output {i}: {det_count} detections")
    
    # Verify independence - modify one output
    outputs[0].data[ContextDataType.DETECTIONS].append(
        Detection(box=(200, 200, 250, 250), object_category="dog", conf=0.88)
    )
    
    print(f"\nAfter modifying output 0:")
    for i, out_ctx in enumerate(outputs):
        det_count = len(out_ctx.data.get(ContextDataType.DETECTIONS, []))
        print(f"  Output {i}: {det_count} detections")
    
    print("\n✓ Fork created independent context copies")
