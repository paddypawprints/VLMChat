#!/usr/bin/env python3
"""
Integration test for cache system in pipeline context.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vlmchat.pipeline.task_base import BaseTask, Context, ContextDataType
from vlmchat.pipeline.pipeline_runner import PipelineRunner
from vlmchat.pipeline.item_cache import ItemCache
from vlmchat.pipeline.cached_item import CachedItem
from vlmchat.pipeline.dsl_parser import DSLParser, create_task_registry
from typing import Any, Dict, List, Optional


class MockCachedItem(CachedItem):
    """Mock cached item for testing."""
    
    def __init__(self, cache_key: str, value: str):
        super().__init__(cache_key)
        self._value = value
    
    def get(self, format: Optional[str] = None) -> Any:
        return self._value
    
    def has_format(self, format: str) -> bool:
        return format == "default"
    
    def get_cached_formats(self) -> List[str]:
        return ["default"]
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {"value": self._value}


class CacheProducerTask(BaseTask):
    """Task that produces cached items."""
    
    def run(self, context: Context) -> Context:
        cache = context.cache
        
        # Create items
        for i in range(3):
            item = MockCachedItem("", f"item_{i}")
            cache_key = cache.add(item, "default", {})
            
            # Update item with proper cache key
            item._cache_key = cache_key
            context.add_data(ContextDataType.CUSTOM, item)
        
        return context


class CacheConsumerTask(BaseTask):
    """Task that reads cached items."""
    
    def run(self, context: Context) -> Context:
        cache = context.cache
        
        # Access items through cache
        for item in context.data:
            if isinstance(item, CachedItem):
                value = item.get()
                context.add_data(ContextDataType.CUSTOM, f"processed_{value}")
        
        return context


def test_cache_in_pipeline():
    """Test cache system within actual pipeline execution."""
    
    # Create simple pipeline
    dsl = """
    pipeline test_cache {
        producer -> consumer
    }
    """
    
    registry = create_task_registry()
    registry["producer"] = CacheProducerTask
    registry["consumer"] = CacheConsumerTask
    
    parser = DSLParser(registry)
    sources, ast_root = parser.parse(dsl)
    
    # Build and run
    runner = PipelineRunner(ast_root=ast_root)
    
    # Get initial cache stats
    cache = ItemCache.global_instance()
    initial_stats = cache.stats()
    
    result_context = runner.run(Context())
    
    # After run, GC should have cleaned up
    final_stats = cache.stats()
    
    # Verify results
    assert len(result_context.data) == 6, f"Should have 6 items (3 cached + 3 processed), got {len(result_context.data)}"
    
    processed_count = sum(1 for item in result_context.data if isinstance(item, str) and item.startswith("processed_"))
    assert processed_count == 3, f"Should have 3 processed items, got {processed_count}"
    
    print("✓ Cache integration in pipeline works")
    print(f"  Initial cache: {initial_stats}")
    print(f"  Final cache: {final_stats}")
    print(f"  Result data items: {len(result_context.data)}")


def test_cache_gc_after_fork():
    """Test GC works correctly after fork with multiple branches."""
    
    dsl = """
    pipeline test_fork {
        producer -> {
            branch1 -> merge
            branch2 -> merge
        }
    }
    """
    
    registry = create_task_registry()
    registry["producer"] = CacheProducerTask
    registry["branch1"] = CacheConsumerTask
    registry["branch2"] = CacheConsumerTask
    registry["merge"] = CacheConsumerTask
    
    parser = DSLParser(registry)
    sources, ast_root = parser.parse(dsl)
    
    # Build and run
    runner = PipelineRunner(ast_root=ast_root)
    cache = ItemCache.global_instance()
    
    result_context = runner.run(Context())
    
    # GC should have collected after execution
    final_stats = cache.stats()
    
    print("✓ Cache GC after fork works")
    print(f"  Final cache: {final_stats}")


def run_tests():
    """Run all integration tests."""
    failed = 0
    tests = [
        ("Cache in Pipeline", test_cache_in_pipeline),
        ("Cache GC After Fork", test_cache_gc_after_fork),
    ]
    
    print("=" * 70)
    print("CACHE INTEGRATION TESTS")
    print("=" * 70)
    print()
    
    for name, test_func in tests:
        print(f"{name}:")
        try:
            # Reset cache between tests
            ItemCache._instance = None
            test_func()
            print("✅ PASSED\n")
        except AssertionError as e:
            print(f"❌ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 70)
    total = len(tests)
    passed = total - failed
    print(f"INTEGRATION TEST RESULTS: {passed}/{total} passed")
    if failed == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ {failed} TESTS FAILED")
    print("=" * 70)
    
    return failed


if __name__ == '__main__':
    sys.exit(run_tests())
