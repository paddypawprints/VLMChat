"""
Pass-through task that does nothing but copy context from input to output.

Pythonesque "pass" equivalent for pipelines - useful as a placeholder or
to create explicit pass-through branches in complex pipeline topologies.
"""

from .task_base import BaseTask, Context, register_task


@register_task('pass')
class PassTask(BaseTask):
    """
    A no-op task that passes context through unchanged.
    
    This task simply returns its input context without modification.
    It's useful for:
    - Creating explicit pass-through branches in fan-out patterns
    - Placeholder tasks during pipeline development
    - Maintaining pipeline structure while temporarily disabling processing
    
    Example:
        # Fan-out: YOLO -> [Clusterer, Pass] -> Merge
        # Pass branch bypasses clustering, sending original YOLO detections to merge
        pass_task = PassTask(task_id="bypass")
    """
    
    def __init__(self, task_id: str = "pass"):
        """
        Initialize pass-through task.
        
        Args:
            task_id: Unique identifier for this task (default: "pass")
        """
        super().__init__(task_id)
        # No specific input/output contracts - passes everything through
    
    def run(self, context: Context) -> Context:
        """
        Pass through the context unchanged.
        
        Args:
            context: Input context
            
        Returns:
            The same context unchanged
        """
        return context
    
    def __str__(self) -> str:
        """String representation."""
        return f"PassTask(id={self.task_id})"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return self.__str__()


if __name__ == "__main__":
    # Example usage
    from .task_base import ContextDataType
    from ..object_detector.detection_base import Detection
    
    print("\n--- PassTask Example ---\n")
    
    # Create test context
    ctx = Context()
    det1 = Detection(box=(10, 10, 50, 50), object_category="person", conf=0.92)
    det2 = Detection(box=(100, 100, 150, 150), object_category="horse", conf=0.89)
    ctx.data[ContextDataType.DETECTIONS] = [det1, det2]
    ctx.data[ContextDataType.IMAGE] = ["fake_image_data"]
    
    print(f"Input context:")
    print(f"  DETECTIONS: {len(ctx.data[ContextDataType.DETECTIONS])}")
    print(f"  IMAGE: {bool(ctx.data.get(ContextDataType.IMAGE))}")
    
    # Create and run pass task
    pass_task = PassTask(task_id="bypass")
    print(f"\n{pass_task}")
    
    # Run it
    output_ctx = pass_task.run(ctx)
    
    print(f"\nOutput context:")
    print(f"  DETECTIONS: {len(output_ctx.data[ContextDataType.DETECTIONS])}")
    print(f"  IMAGE: {bool(output_ctx.data.get(ContextDataType.IMAGE))}")
    
    # Verify it's the same
    print(f"\nSame object: {ctx is output_ctx}")
    print(f"Same detections: {ctx.data[ContextDataType.DETECTIONS] is output_ctx.data[ContextDataType.DETECTIONS]}")
    
    print("\n✓ PassTask passes context through unchanged")
