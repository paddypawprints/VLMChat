"""
End-to-end DSL integration test with PipelineRunner.

Tests that DSL-parsed pipelines can actually execute, not just parse.
"""

from vlmchat.pipeline.dsl_parser import DSLParser, create_task_registry
from vlmchat.pipeline.pipeline_runner import PipelineRunner
from vlmchat.pipeline.task_base import Context, ContextDataType


def test_dsl_to_execution_simple():
    """Test that a simple DSL pipeline can be parsed and executed."""
    dsl = "pass() -> pass()"
    
    print("\n" + "="*60)
    print("Test: Simple DSL -> Execution")
    print("="*60)
    print(f"DSL: {dsl}")
    
    # Parse DSL
    registry = create_task_registry()
    parser = DSLParser(registry)
    pipeline = parser.parse(dsl)
    
    print(f"✅ Parsed: {pipeline}")
    
    # Create runner
    if isinstance(pipeline, list):
        runner = PipelineRunner(tasks=pipeline, max_workers=1)
    else:
        runner = PipelineRunner(tasks=[pipeline], max_workers=1)
    
    # Execute
    ctx = Context()
    result = runner.run(ctx)
    
    print(f"✅ Executed successfully")
    return True


def test_dsl_to_execution_with_data():
    """Test DSL pipeline that processes actual data."""
    dsl = "pass() -> pass() -> pass()"
    
    print("\n" + "="*60)
    print("Test: DSL Pipeline with Data")
    print("="*60)
    print(f"DSL: {dsl}")
    
    # Parse
    registry = create_task_registry()
    parser = DSLParser(registry)
    pipeline = parser.parse(dsl)
    
    # Setup context with data
    ctx = Context()
    ctx.data[ContextDataType.IMAGE] = ["test_image_data"]
    
    # Execute
    if isinstance(pipeline, list):
        runner = PipelineRunner(tasks=pipeline, max_workers=1)
    else:
        runner = PipelineRunner(tasks=[pipeline], max_workers=1)
    
    result = runner.run(ctx)
    
    # Verify data passed through
    assert ContextDataType.IMAGE in result.data
    assert result.data[ContextDataType.IMAGE] == ["test_image_data"]
    
    print(f"✅ Data passed through pipeline")
    return True


def test_dsl_parallel_execution():
    """Test that parallel DSL structure executes correctly."""
    dsl = "[pass(), pass(), pass()]"
    
    print("\n" + "="*60)
    print("Test: Parallel DSL Execution")
    print("="*60)
    print(f"DSL: {dsl}")
    
    # Parse
    registry = create_task_registry()
    parser = DSLParser(registry)
    pipeline = parser.parse(dsl)
    
    print(f"Parsed structure: {pipeline}")
    print(f"Type: {type(pipeline)}")
    
    # This will likely fail because we need to figure out how to
    # pass fork+tasks to PipelineRunner
    try:
        if isinstance(pipeline, list):
            runner = PipelineRunner(tasks=pipeline, max_workers=3)
        else:
            runner = PipelineRunner(tasks=[pipeline], max_workers=3)
        
        ctx = Context()
        result = runner.run(ctx)
        
        print(f"✅ Parallel execution succeeded")
        return True
    except Exception as e:
        print(f"❌ Parallel execution failed: {e}")
        print(f"   This is expected - need to integrate ForkConnector with PipelineRunner")
        return False


def test_dsl_loop_execution():
    """Test that loop DSL structure executes correctly."""
    dsl = "{pass() ->:timeout(seconds=0.1)}"
    
    print("\n" + "="*60)
    print("Test: Loop DSL Execution")
    print("="*60)
    print(f"DSL: {dsl}")
    
    # Parse
    registry = create_task_registry()
    parser = DSLParser(registry)
    pipeline = parser.parse(dsl)
    
    print(f"Parsed: {type(pipeline).__name__}")
    
    # Execute
    try:
        runner = PipelineRunner(tasks=[pipeline], max_workers=1)
        ctx = Context()
        result = runner.run(ctx)
        
        print(f"✅ Loop execution succeeded")
        return True
    except Exception as e:
        print(f"❌ Loop execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DSL INTEGRATION TESTS - End-to-End Parsing + Execution")
    print("="*70)
    
    tests = [
        ("Simple Pipeline", test_dsl_to_execution_simple),
        ("Pipeline with Data", test_dsl_to_execution_with_data),
        ("Parallel Execution", test_dsl_parallel_execution),
        ("Loop Execution", test_dsl_loop_execution),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n{passed}/{total} tests passed")
    
    if passed < total:
        print("\n⚠️  Some tests failed - these reveal integration gaps between")
        print("   DSL parser and PipelineRunner that need to be addressed.")
