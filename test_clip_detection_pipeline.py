#!/usr/bin/env python3
"""
Test CLIP detection pipeline parsing and structure.

Validates that clip_detection.dsl parses correctly and produces expected AST.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from pipeline import dsl_parser

# Get classes from dsl_parser
Lexer = dsl_parser.Lexer
Parser = dsl_parser.Parser
LoopNode = dsl_parser.LoopNode
ParallelNode = dsl_parser.ParallelNode
SequenceNode = dsl_parser.SequenceNode
TaskNode = dsl_parser.TaskNode


def test_clip_detection_dsl_parsing():
    """Test that clip_detection.dsl parses correctly."""
    print("\n" + "="*70)
    print("TEST: CLIP Detection DSL Parsing")
    print("="*70)
    
    # Read the DSL file
    dsl_path = Path(__file__).parent / "pipelines" / "clip_detection.dsl"
    with open(dsl_path, 'r') as f:
        dsl_text = f.read()
    
    print(f"\nParsing: {dsl_path}")
    
    # Parse with low-level parser
    lexer = Lexer(dsl_text)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    
    print(f"\n✓ Parsed successfully")
    print(f"  Root type: {type(ast).__name__}")
    
    # Validate structure
    assert isinstance(ast, LoopNode), f"Expected LoopNode, got {type(ast).__name__}"
    print(f"  ✓ Root is LoopNode (main loop)")
    
    # Check loop body
    body = ast.body
    assert isinstance(body, SequenceNode), f"Expected SequenceNode in loop body, got {type(body).__name__}"
    print(f"  ✓ Loop body is SequenceNode")
    
    # The sequence should have 5 main steps
    tasks = body.tasks
    print(f"\n  Pipeline steps ({len(tasks)}):")
    
    for i, task in enumerate(tasks):
        print(f"    [{i}] {type(task).__name__}", end="")
        if isinstance(task, TaskNode):
            print(f" - {task.name}()", end="")
            if task.is_control:
                print(" [CONTROL]", end="")
        elif isinstance(task, ParallelNode):
            print(f" - {len(task.tasks)} branches", end="")
            if task.merge_strategy:
                print(f" merge={task.merge_strategy}", end="")
        print()
    
    # Step 0: Parallel block with input loop and camera
    assert isinstance(tasks[0], ParallelNode), "Step 0 should be parallel"
    step0 = tasks[0]
    assert len(step0.tasks) == 2, f"Expected 2 parallel branches, got {len(step0.tasks)}"
    
    # First branch: loop with input and break_on
    branch0 = step0.tasks[0]
    assert isinstance(branch0, LoopNode), "First branch should be loop"
    loop_body = branch0.body
    assert isinstance(loop_body, SequenceNode), "Loop body should be sequence"
    assert len(loop_body.tasks) == 2, "Loop should have 2 tasks"
    assert loop_body.tasks[0].name == "input", "First task should be input"
    assert loop_body.tasks[1].name == "break_on", "Second task should be break_on"
    assert loop_body.tasks[1].is_control == True, "break_on should be control task"
    print(f"\n  ✓ Step 0: Parallel input collection validated")
    
    # Step 1: detector
    assert isinstance(tasks[1], TaskNode), "Step 1 should be task"
    assert tasks[1].name == "detector", "Step 1 should be detector"
    print(f"  ✓ Step 1: detector() validated")
    
    # Step 2: filter
    assert isinstance(tasks[2], TaskNode), "Step 2 should be task"
    assert tasks[2].name == "filter", "Step 2 should be filter"
    assert "categories" in tasks[2].params, "filter should have categories param"
    print(f"  ✓ Step 2: filter() validated")
    
    # Step 3: Parallel CLIP encoding with ordered merge
    assert isinstance(tasks[3], ParallelNode), "Step 3 should be parallel"
    step3 = tasks[3]
    assert len(step3.tasks) == 2, f"Expected 2 parallel branches, got {len(step3.tasks)}"
    assert step3.merge_strategy == "ordered_merge", "Should have ordered_merge strategy"
    assert step3.merge_params is not None, "Should have merge params"
    assert step3.merge_params.get("order") == "0,1", "Merge order should be 0,1"
    
    # Check branch names
    branch0 = step3.tasks[0]
    branch1 = step3.tasks[1]
    assert isinstance(branch0, TaskNode), "Branch 0 should be task"
    assert isinstance(branch1, TaskNode), "Branch 1 should be task"
    assert branch0.name == "clip_text_encoder", "Branch 0 should be clip_text_encoder"
    assert branch1.name == "clip_vision", "Branch 1 should be clip_vision"
    print(f"  ✓ Step 3: Parallel CLIP encoding with ordered_merge validated")
    
    # Step 4: clip_comparator
    assert isinstance(tasks[4], TaskNode), "Step 4 should be task"
    assert tasks[4].name == "clip_comparator", "Step 4 should be clip_comparator"
    print(f"  ✓ Step 4: clip_comparator() validated")
    
    # Step 5: output
    assert isinstance(tasks[5], TaskNode), "Step 5 should be task"
    assert tasks[5].name == "output", "Step 5 should be output"
    assert "types" in tasks[5].params, "output should have types param"
    print(f"  ✓ Step 5: output() validated")
    
    print("\n✅ ALL TESTS PASSED")
    print("   - DSL parses correctly")
    print("   - Loop structure validated")
    print("   - Parallel blocks validated")
    print("   - Control operators validated")
    print("   - Ordered merge validated")
    print("   - All 6 pipeline steps present and correct")
    
    return True


if __name__ == "__main__":
    try:
        test_clip_detection_dsl_parsing()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
