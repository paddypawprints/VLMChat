"""Test pipeline diagram generation."""

import pytest
from camera_framework import (
    BaseTask,
    Runner,
    Buffer,
    drop_oldest_policy,
    blocking_policy,
    PipelineTraverser,
    MermaidVisitor,
)


class SourceTask(BaseTask):
    """Source task for testing."""
    def process(self) -> None:
        message = {}
        message.setdefault("data", []).append("test")
        for buf in self.outputs.values():
            buf.put(message)


class ProcessTask(BaseTask):
    """Processing task for testing."""
    def process(self) -> None:
        input_buf = list(self.inputs.values())[0]
        message = input_buf.get()
        if message:
            for buf in self.outputs.values():
                buf.put(message)


class SinkTask(BaseTask):
    """Sink task for testing."""
    def process(self) -> None:
        input_buf = list(self.inputs.values())[0]
        message = input_buf.get()


def test_simple_pipeline_diagram():
    """Test diagram generation for simple pipeline."""
    # Create tasks
    source = SourceTask(name="Camera")
    processor = ProcessTask(name="YOLO")
    sink = SinkTask(name="Display")
    
    # Create buffers
    buf1 = Buffer(size=30, policy=drop_oldest_policy, name="frames")
    buf2 = Buffer(size=10, policy=drop_oldest_policy, name="detections")
    
    # Connect
    source.add_output("out", buf1)
    processor.add_input("in", buf1)
    processor.add_output("out", buf2)
    sink.add_input("in", buf2)
    
    # Create runner
    runner = Runner([source, processor, sink])
    
    # Generate diagram
    visitor = MermaidVisitor()
    traverser = PipelineTraverser(runner)
    traverser.traverse(visitor)
    
    diagram = visitor.get_result()
    
    # Verify diagram structure
    assert "flowchart" in diagram
    assert "([Camera])" in diagram  # Source node
    assert "[YOLO]" in diagram  # Processing node
    assert "[/Display/]" in diagram  # Sink node
    assert "{{frames" in diagram  # Buffer node
    assert "{{detections" in diagram  # Buffer node
    assert "size=30" in diagram
    assert "drop_oldest" in diagram


def test_multi_output_pipeline_diagram():
    """Test diagram generation for pipeline with router."""
    # Create tasks
    source = SourceTask(name="Camera")
    router = ProcessTask(name="CategoryRouter")
    sink1 = SinkTask(name="PersonHandler")
    sink2 = SinkTask(name="VehicleHandler")
    
    # Create buffers
    input_buf = Buffer(size=30, policy=drop_oldest_policy, name="frames")
    person_buf = Buffer(size=10, policy=blocking_policy, name="persons")
    vehicle_buf = Buffer(size=10, policy=blocking_policy, name="vehicles")
    
    # Connect - router has multiple named outputs
    source.add_output("frames", input_buf)
    router.add_input("in", input_buf)
    router.add_output("persons", person_buf)
    router.add_output("vehicles", vehicle_buf)
    sink1.add_input("in", person_buf)
    sink2.add_input("in", vehicle_buf)
    
    # Create runner
    runner = Runner([source, router, sink1, sink2])
    
    # Generate diagram
    visitor = MermaidVisitor()
    traverser = PipelineTraverser(runner)
    traverser.traverse(visitor)
    
    diagram = visitor.get_result()
    
    # Verify multi-output connections
    assert "CategoryRouter" in diagram
    assert "PersonHandler" in diagram
    assert "VehicleHandler" in diagram
    assert "persons" in diagram
    assert "vehicles" in diagram
    assert "blocking" in diagram


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
