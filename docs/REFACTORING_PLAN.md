# Pipeline Runner Refactoring Plan

## Summary
The `__main__` block in `pipeline_runner.py` contains 2,937 lines with massive code duplication across 17 tests. This document outlines the refactoring strategy.

## Completed Changes

### ✅ 1. Created TestConfig Dataclass (Lines ~380-540)
Centralized configuration for all tests with defaults:
- YOLO settings (model, confidence)
- CLIP settings (min_similarity)
- Detection processing (expansion_factor, merge_target)
- Clusterer settings
- Display settings
- Prompts (context, attribute, test)

### ✅ 2. Created Test Infrastructure Helpers (Lines ~540-640)
- `create_test_pipeline()`: Build pipeline from config
- `run_test_pipeline()`: Execute and return results
- `print_pipeline_results()`: Standardized output
- `safe_import_test()`: Check dependencies without scary traces

### ✅ 3. Updated Tests 16-17
- Modified to use `TestConfig` parameter
- Use config values instead of hardcoded constants
- Updated function signatures
- Updated calls to pass `TEST_CONFIGS[16/17]`

## Remaining Work

### Pattern for Test Functions

Each inline test should become:

```python
def run_test_N_description(factory, collector, config: TestConfig, 
                          interactive_mode: bool = False):
    """
    Test N: Brief description
    
    Detailed description of what makes this test unique.
    """
    print_test_header(N, "Title", goal="...", method="...")
    
    try:
        # Create tasks
        # Build pipeline
        # Execute
        result = run_test_pipeline(pipeline, collector)
        
        # Print results
        print_pipeline_results(result)
        
        print_test_footer(success=True, test_num=N)
        
    except ImportError as e:
        print(f"⚠️  Skipping Test {N} (missing dependency: {e.name})")
    except Exception as e:
        print(f"❌ Test {N} failed: {e}")
        if args.log_level == 'DEBUG':
            traceback.print_exc()
```

### Tests 1-5: Simple Connector Tests (~50 lines each → ~30 lines)

**Test 1: Linear Pipeline**
```python
def run_test_1_linear_pipeline(factory, collector, config: TestConfig, interactive_mode: bool = False):
    """Simple linear pipeline: Source → Detector → Embedder"""
    print_test_header(1, "Linear Pipeline: Source → Detector → Embedder")
    
    # Create tasks
    pipeline = Connector("linear_pipeline")
    source = ImageSourceTask()
    detector = DetectorTask("detector")
    embedder = EmbeddingTask()
    
    pipeline.add_task(source)
    pipeline.add_task(detector)
    pipeline.add_task(embedder)
    pipeline.add_edge(source, detector)
    pipeline.add_edge(detector, embedder)
    
    # Execute
    result = run_test_pipeline(pipeline, collector)
    print_pipeline_results(result, ["IMAGE", "CROPS", "EMBEDDINGS"])
    
    # Show metrics
    print(f"\nMetrics Summary:")
    for ts_name, inst in session._instruments:
        exported = inst.export()
        print(f"  {exported['name']}: {json.dumps({k: v for k, v in exported.items() if k not in ['type', 'name', 'binding_keys']}, indent=4)}")
    
    print_test_footer(success=True, test_num=1)
```

**Test 2-5**: Similar pattern, ~30-40 lines each

### Tests 6-9: Factory and Interactive (~80 lines each → ~50 lines)

Use helper functions, reduce duplication.

### Tests 10-15: Full Pipelines (~400-600 lines each → ~200-300 lines)

These are the biggest savings. Extract common patterns:
- Camera setup
- Detector setup
- Clusterer setup  
- CLIP encoding
- Result display

Create sub-helpers like:
```python
def setup_camera_detector(config, camera_type="none"):
    """Common camera + detector setup"""
    camera_task = factory.create_task("camera", f"camera_{camera_type}", {
        "type": camera_type
    })
    detector_task = factory.create_task("yolo_detector", "yolo", {
        "type": "yolo_cpu",
        "model": config.yolo_model,
        "confidence": str(config.yolo_confidence)
    })
    return camera_task, detector_task

def display_similarity_results(matches, prompts, threshold=0.25):
    """Common similarity display logic"""
    # ... extract from multiple tests
```

### Test Registry

At the end of test definitions:

```python
# Test registry: (name, function, requires_interactive, requires_clip)
TESTS = {
    1: ("Linear pipeline", run_test_1_linear_pipeline, False, False),
    2: ("Branching pipeline", run_test_2_branching_pipeline, False, False),
    3: ("FirstComplete connector", run_test_3_first_complete, False, False),
    4: ("OrderedMerge connector", run_test_4_ordered_merge, False, False),
    5: ("PipelineFactory", run_test_5_pipeline_factory, False, False),
    6: ("Factory configuration", run_test_6_factory_config, False, False),
    7: ("Configure test", run_test_7_configure, False, False),
    8: ("Interactive YOLO", run_test_8_interactive_yolo, True, False),
    9: ("Interactive clusterer", run_test_9_interactive_clusterer, True, False),
    10: ("Full pipeline", run_test_10_full_pipeline, False, True),
    11: ("Semantic matching", run_test_11_semantic_matching, False, True),
    12: ("Direct to CLIP", run_test_12_direct_clip, False, True),
    13: ("Detection expander", run_test_13_detection_expander, False, True),
    14: ("Trail riders test", run_test_14_trail_riders, False, True),
    15: ("Dual-path", run_test_15_dual_path, False, True),
    16: ("Color supersaturation", run_test_16_supersaturated_colors, False, True),
    17: ("R/B channel swap", run_test_17_color_channel_swap, False, True),
}
```

### Main Execution Loop

Replace all inline `if should_run_test(N):` blocks with:

```python
# Main test execution loop
for test_num, (name, test_fn, requires_interactive, requires_clip) in TESTS.items():
    if not should_run_test(test_num):
        print(f"\n[TEST {test_num}] Skipped")
        continue
    
    # Check prerequisites
    if requires_interactive and not interactive_mode:
        print(f"\n[TEST {test_num}] Skipped (requires --interactive)")
        continue
    
    if requires_clip and not clip_model:
        print(f"\n[TEST {test_num}] Skipped (requires CLIP model)")
        continue
    
    # Run test
    try:
        config = TEST_CONFIGS.get(test_num, TestConfig())
        if requires_clip:
            test_fn(factory, clip_model, semantic_provider, config, interactive_mode)
        else:
            test_fn(factory, collector, config, interactive_mode)
    except KeyboardInterrupt:
        print(f"\n⚠️  Test {test_num} interrupted by user")
        break
    except Exception as e:
        print(f"\n❌ Test {test_num} ({name}) failed: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
```

## Expected Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Total lines in `__main__` | 2,937 | ~800 | -73% |
| Code duplication | ~70% | ~15% | -55% |
| Average test length | 150 lines | 50 lines | -67% |
| Maintainability | 3/10 | 8/10 | +167% |

## Implementation Steps

1. ✅ Create TestConfig and helpers
2. ✅ Update Tests 16-17
3. **TODO**: Extract Tests 1-5 (simple tests)
4. **TODO**: Extract Tests 6-9 (factory tests)  
5. **TODO**: Extract Tests 10-12 (CLIP tests)
6. **TODO**: Extract Tests 13-15 (complex tests)
7. **TODO**: Create test registry
8. **TODO**: Replace inline calls with execution loop
9. **TODO**: Consolidate factory registration
10. **TODO**: Test all scenarios

## Benefits

1. **Maintainability**: Fix once, applies everywhere
2. **Readability**: See test purpose immediately
3. **Testability**: Each function can be unit tested
4. **Extensibility**: Add new tests by adding to registry
5. **Debugging**: Easier to isolate issues
6. **Documentation**: Function docstrings explain each test
7. **Consistency**: All tests follow same pattern

## Next Steps

To complete this refactoring:

1. Continue extracting test functions following the pattern above
2. Each test should be self-contained with clear inputs/outputs
3. Use helper functions to eliminate duplication
4. Update test registry as you go
5. Replace inline blocks with execution loop
6. Test thoroughly with `--test all` and individual test numbers
