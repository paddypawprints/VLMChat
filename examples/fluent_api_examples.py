"""
Example pipelines demonstrating the fluent API.

These examples show how to use the new Pipeline builder class
to construct task graphs without manual wiring or DSL.
"""

from vlmchat.pipeline import Pipeline
from vlmchat.pipeline.tasks import DiagnosticTask, PassTask, StartTask
from vlmchat.pipeline.connectors import ForkConnector, DetectionMergeConnector


def example_1_simple_chain():
    """Example 1: Simple sequential pipeline."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Sequential Pipeline")
    print("="*70)
    
    # Create tasks
    task1 = DiagnosticTask("task1")
    task1.configure(message="First task", delay_ms=10)
    
    task2 = DiagnosticTask("task2")
    task2.configure(message="Second task", delay_ms=10)
    
    task3 = DiagnosticTask("task3")
    task3.configure(message="Third task", delay_ms=10)
    
    # Build pipeline using chain
    pipeline = Pipeline("simple_chain")
    pipeline.chain([task1, task2, task3])
    
    # Execute
    print(f"\nExecuting pipeline: {pipeline}")
    result = pipeline.run()
    
    print("✅ Example 1 complete")
    return result


def example_2_method_chaining():
    """Example 2: Build pipeline with method chaining."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Method Chaining (.add().then().then())")
    print("="*70)
    
    # Create tasks
    start = StartTask("start")
    process1 = DiagnosticTask("process1")
    process1.configure(message="Processing...", delay_ms=5)
    
    process2 = DiagnosticTask("process2")
    process2.configure(message="More processing...", delay_ms=5)
    
    finish = PassTask("finish")
    
    # Build pipeline with method chaining
    pipeline = Pipeline("chained")
    pipeline.add(start).then(process1).then(process2).then(finish)
    
    # Execute
    print(f"\nExecuting pipeline: {pipeline}")
    result = pipeline.run()
    
    print("✅ Example 2 complete")
    return result


def example_3_parallel_fork():
    """Example 3: Parallel branches with fork/merge."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Parallel Branches (fork/merge)")
    print("="*70)
    
    # Create tasks
    input_task = DiagnosticTask("input")
    input_task.configure(message="Input data", delay_ms=5)
    
    # Parallel processing branches
    branch1 = DiagnosticTask("branch1")
    branch1.configure(message="Branch 1 processing", delay_ms=10)
    
    branch2 = DiagnosticTask("branch2")
    branch2.configure(message="Branch 2 processing", delay_ms=15)
    
    branch3 = DiagnosticTask("branch3")
    branch3.configure(message="Branch 3 processing", delay_ms=12)
    
    output_task = DiagnosticTask("output")
    output_task.configure(message="Merged output", delay_ms=5)
    
    # Build pipeline with fork/merge
    pipeline = Pipeline("parallel")
    pipeline.add(input_task).fork([branch1, branch2, branch3]).merge(merge_task=output_task)
    
    # Execute
    print(f"\nExecuting pipeline: {pipeline}")
    result = pipeline.run(enable_trace=True)
    
    print("✅ Example 3 complete")
    return result


def example_4_nested_branches():
    """Example 4: Nested parallel branches using builder functions."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Nested Parallel Branches")
    print("="*70)
    
    # Create tasks
    start = DiagnosticTask("start")
    start.configure(message="Starting", delay_ms=5)
    
    # Build pipeline with nested branches
    pipeline = Pipeline("nested")
    
    pipeline.add(start).parallel(
        # Branch 1: simple path
        lambda p: p.add(DiagnosticTask("branch1_step1")).then(DiagnosticTask("branch1_step2")),
        
        # Branch 2: another simple path
        lambda p: p.add(DiagnosticTask("branch2_step1")).then(DiagnosticTask("branch2_step2")),
    ).merge()
    
    # Execute
    print(f"\nExecuting pipeline: {pipeline}")
    result = pipeline.run(enable_trace=True)
    
    print("✅ Example 4 complete")
    return result


def example_5_complex_graph():
    """Example 5: Complex multi-stage pipeline."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Complex Multi-Stage Pipeline")
    print("="*70)
    
    # Stage 1: Input
    input_task = DiagnosticTask("input")
    input_task.configure(message="Loading input", delay_ms=10)
    
    # Stage 2: Parallel preprocessing
    preprocess1 = DiagnosticTask("preprocess1")
    preprocess1.configure(message="Preprocessing path 1", delay_ms=15)
    
    preprocess2 = DiagnosticTask("preprocess2")
    preprocess2.configure(message="Preprocessing path 2", delay_ms=12)
    
    # Stage 3: Merge and process
    merge_process = DiagnosticTask("merge_process")
    merge_process.configure(message="Processing merged data", delay_ms=20)
    
    # Stage 4: Parallel analysis
    analyze1 = DiagnosticTask("analyze1")
    analyze1.configure(message="Analysis 1", delay_ms=10)
    
    analyze2 = DiagnosticTask("analyze2")
    analyze2.configure(message="Analysis 2", delay_ms=10)
    
    # Stage 5: Final output
    output_task = DiagnosticTask("output")
    output_task.configure(message="Final output", delay_ms=5)
    
    # Build complex pipeline
    pipeline = Pipeline("complex")
    
    # Stage 1 -> Stage 2 (fork) -> Stage 3 (merge) -> Stage 4 (fork) -> Stage 5 (merge)
    pipeline.add(input_task) \
        .fork([preprocess1, preprocess2]) \
        .merge(merge_task=merge_process) \
        .fork([analyze1, analyze2]) \
        .merge(merge_task=output_task)
    
    # Execute
    print(f"\nExecuting pipeline: {pipeline}")
    result = pipeline.run(enable_trace=True, max_workers=4)
    
    print("✅ Example 5 complete")
    return result


def run_all_examples():
    """Run all examples."""
    print("\n" + "="*70)
    print("FLUENT API EXAMPLES")
    print("="*70)
    
    examples = [
        ("Simple Chain", example_1_simple_chain),
        ("Method Chaining", example_2_method_chaining),
        ("Parallel Fork/Merge", example_3_parallel_fork),
        ("Nested Branches", example_4_nested_branches),
        ("Complex Multi-Stage", example_5_complex_graph),
    ]
    
    for name, example_fn in examples:
        try:
            example_fn()
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("All examples complete!")
    print("="*70)


if __name__ == "__main__":
    run_all_examples()
