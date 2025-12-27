"""
Integration tests for DSL parser with actual task creation.
"""

from vlmchat.pipeline.dsl_parser import DSLParser, create_task_registry
from vlmchat.pipeline.task_base import BaseTask


def test_simple_sequence():
    """Test parsing a simple sequential pipeline."""
    dsl = "pass() -> pass() -> pass()"
    
    registry = create_task_registry()
    parser = DSLParser(registry)
    
    pipeline = parser.parse(dsl)
    
    # Sequence of tasks returns a list
    assert isinstance(pipeline, list) and len(pipeline) == 3
    print("✅ Simple sequence parsed and built successfully")


def test_with_parameters():
    """Test task with parameters."""
    dsl = 'timeout(seconds=60)'
    
    registry = create_task_registry()
    parser = DSLParser(registry)
    
    pipeline = parser.parse(dsl)
    
    assert isinstance(pipeline, BaseTask)
    print("✅ Task with parameters parsed successfully")


def test_parallel():
    """Test parallel execution."""
    dsl = "[pass(), pass(), pass()]"
    
    registry = create_task_registry()
    parser = DSLParser(registry)
    
    pipeline = parser.parse(dsl)
    
    print(f"✅ Parallel tasks parsed: {type(pipeline)}")


def test_loop():
    """Test loop structure."""
    dsl = "{pass() -> pass() ->:timeout(seconds=60)}"
    
    registry = create_task_registry()
    parser = DSLParser(registry)
    
    pipeline = parser.parse(dsl)
    
    print(f"✅ Loop parsed: {type(pipeline)}")


def test_control_task_error():
    """Test that control tasks outside loops raise error."""
    dsl = "pass() ->:timeout(seconds=60)"
    
    registry = create_task_registry()
    parser = DSLParser(registry)
    
    try:
        pipeline = parser.parse(dsl)
        print("❌ Should have raised error for control task outside loop")
    except SyntaxError as e:
        print(f"✅ Correctly rejected control task outside loop: {e}")


def test_unknown_task():
    """Test unknown task name."""
    dsl = "invalid_task_name()"
    
    registry = create_task_registry()
    parser = DSLParser(registry)
    
    try:
        pipeline = parser.parse(dsl)
        print("❌ Should have raised error for unknown task")
    except ValueError as e:
        print(f"✅ Correctly rejected unknown task: {e}")


def test_complex_pipeline():
    """Test complex nested pipeline."""
    dsl = """
    {
        [pass(), pass()] ->
        pass() ->
        :timeout(seconds=300)
    }
    """
    
    registry = create_task_registry()
    parser = DSLParser(registry)
    
    pipeline = parser.parse(dsl)
    
    print(f"✅ Complex pipeline parsed: {type(pipeline)}")


def test_comments():
    """Test that # comments are properly ignored."""
    dsl = """
    # This is a camera capture pipeline
    pass() ->  # First task
    pass()~100 # Second task with timing
    # End of pipeline
    """
    
    registry = create_task_registry()
    parser = DSLParser(registry)
    
    pipeline = parser.parse(dsl)
    
    assert len(pipeline) == 2, f"Expected 2 tasks, got {len(pipeline)}"
    assert pipeline[0].__class__.__name__ == "PassTask"
    assert pipeline[1].__class__.__name__ == "PassTask"
    assert pipeline[1].time_budget_ms == 100, f"Expected timing 100ms, got {pipeline[1].time_budget_ms}"
    
    print(f"✅ Comments correctly ignored, timing preserved")


if __name__ == "__main__":
    print("DSL Parser Integration Tests")
    print("="*60)
    
    tests = [
        test_simple_sequence,
        test_with_parameters,
        test_parallel,
        test_loop,
        test_control_task_error,
        test_unknown_task,
        test_complex_pipeline,
        test_comments,
    ]
    
    for test_func in tests:
        print(f"\n{test_func.__name__}:")
        try:
            test_func()
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Integration tests completed")
