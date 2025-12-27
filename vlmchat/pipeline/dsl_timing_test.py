"""
Test timing syntax in DSL parser.

Tests ~advisory and >=enforced timing constraints.
"""

from src.pipeline.dsl_parser import DSLParser, create_task_registry


def test_advisory_timing():
    """Test ~advisory timing constraint."""
    dsl = "pass()~100"
    
    print("\nTest: Advisory Timing (~)")
    print(f"DSL: {dsl}")
    
    registry = create_task_registry()
    parser = DSLParser(registry)
    pipeline = parser.parse(dsl)
    
    assert hasattr(pipeline, 'time_budget_ms')
    assert pipeline.time_budget_ms == 100
    print(f"✅ Task has advisory time budget: {pipeline.time_budget_ms}ms")


def test_enforced_timing():
    """Test >=enforced minimum timing constraint."""
    dsl = "pass()>=50"
    
    print("\nTest: Enforced Timing (>=)")
    print(f"DSL: {dsl}")
    
    registry = create_task_registry()
    parser = DSLParser(registry)
    pipeline = parser.parse(dsl)
    
    assert hasattr(pipeline, '_enforced_min_ms')
    assert pipeline._enforced_min_ms == 50
    print(f"✅ Task has enforced minimum: {pipeline._enforced_min_ms}ms")


def test_both_timing_constraints():
    """Test that both timing types can coexist."""
    # Note: In practice, you'd use one or the other, but parser should handle both
    dsl = "pass()~100 -> pass()>=33"
    
    print("\nTest: Mixed Timing Constraints")
    print(f"DSL: {dsl}")
    
    registry = create_task_registry()
    parser = DSLParser(registry)
    pipeline = parser.parse(dsl)
    
    assert isinstance(pipeline, list)
    assert len(pipeline) == 2
    assert pipeline[0].time_budget_ms == 100
    assert pipeline[1]._enforced_min_ms == 33
    print(f"✅ Task 1: advisory {pipeline[0].time_budget_ms}ms")
    print(f"✅ Task 2: enforced {pipeline[1]._enforced_min_ms}ms")


def test_loop_timing():
    """Test timing constraints on loops."""
    dsl = "{pass() -> pass()~25}>=33"
    
    print("\nTest: Loop Timing")
    print(f"DSL: {dsl}")
    
    registry = create_task_registry()
    parser = DSLParser(registry)
    pipeline = parser.parse(dsl)
    
    # Loop itself has enforced minimum
    assert hasattr(pipeline, '_enforced_min_ms')
    assert pipeline._enforced_min_ms == 33
    
    # Task inside loop has advisory
    assert len(pipeline.body_tasks) == 2
    assert pipeline.body_tasks[1].time_budget_ms == 25
    
    print(f"✅ Loop enforced minimum: {pipeline._enforced_min_ms}ms per iteration")
    print(f"✅ Inner task advisory: {pipeline.body_tasks[1].time_budget_ms}ms")


def test_realistic_video_pipeline():
    """Test realistic video processing pipeline with timing."""
    dsl = """{
        pass()>=33 ->
        pass()~25 ->
        pass()~5
    }~33"""
    
    print("\nTest: Realistic Video Pipeline (30fps)")
    print(f"DSL: {dsl}")
    
    registry = create_task_registry()
    parser = DSLParser(registry)
    pipeline = parser.parse(dsl)
    
    # Loop has advisory (target 33ms/iteration for 30fps)
    assert pipeline.time_budget_ms == 33
    
    # Camera (first pass) has enforced minimum (33ms frame time)
    camera_task = pipeline.body_tasks[0]
    assert camera_task._enforced_min_ms == 33
    
    # Detector (second pass) has advisory (try to finish in 25ms)
    detect_task = pipeline.body_tasks[1]
    assert detect_task.time_budget_ms == 25
    
    # Processor (third pass) has advisory (5ms remaining)
    process_task = pipeline.body_tasks[2]
    assert process_task.time_budget_ms == 5
    
    print(f"✅ Loop advisory: {pipeline.time_budget_ms}ms (30fps target)")
    print(f"✅ Camera enforced: {camera_task._enforced_min_ms}ms (frame sync)")
    print(f"✅ Detect advisory: {detect_task.time_budget_ms}ms (quality/speed trade)")
    print(f"✅ Process advisory: {process_task.time_budget_ms}ms (remaining time)")


if __name__ == "__main__":
    print("="*70)
    print("DSL TIMING SYNTAX TESTS")
    print("="*70)
    
    tests = [
        test_advisory_timing,
        test_enforced_timing,
        test_both_timing_constraints,
        test_loop_timing,
        test_realistic_video_pipeline,
    ]
    
    passed = 0
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ Test failed: {e}")
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("="*70)
    
    if passed == len(tests):
        print("\n✅ All timing syntax tests passed!")
        print("\nTiming features ready for PipelineRunner integration:")
        print("  • ~ (tilde) = advisory timing (cooperative, task checks should_continue())")
        print("  • >= (greater-equal) = enforced minimum (runner sleeps if task finishes early)")
