#!/usr/bin/env python3
"""
Test ItemCache reference counting and garbage collection.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vlmchat.pipeline.item_cache import ItemCache
from vlmchat.pipeline.cached_item import CachedItem
from typing import Any, Dict, List, Optional


class MockItem(CachedItem):
    """Simple test item."""
    
    def __init__(self, cache_key: str, value: Any):
        super().__init__(cache_key)
        self._value = value
        self._formats = {'default': value}
    
    def get(self, format: Optional[str] = None) -> Any:
        fmt = format or 'default'
        return self._formats.get(fmt)
    
    def has_format(self, format: str) -> bool:
        return format in self._formats
    
    def get_cached_formats(self) -> List[str]:
        return list(self._formats.keys())
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {'value': self._value}


def test_add_and_retrieve():
    """Test basic add and retrieve."""
    cache = ItemCache()
    
    item = MockItem("temp_key", "value1")
    cache_key = cache.add(item, "default", {})
    
    retrieved_data = cache.retrieve(cache_key)
    retrieved_item, fmt, metadata = retrieved_data
    assert retrieved_item is item, "Retrieved item should be same instance"
    assert fmt == "default", "Format should match"
    print("✓ Add and retrieve works")


def test_reference_counting():
    """Test reference counting."""
    cache = ItemCache()
    
    item = MockItem("temp_key", "value2")
    cache_key = cache.add(item, "default", {})  # refcount = 1
    
    cache.retain(cache_key)  # refcount = 2
    cache.retain(cache_key)  # refcount = 3
    
    stats = cache.stats()
    assert stats['total_items'] == 1, f"Should have 1 item, got {stats['total_items']}"
    
    cache.release(cache_key)  # refcount = 2
    cache.release(cache_key)  # refcount = 1
    
    retrieved_item, _, _ = cache.retrieve(cache_key)
    assert retrieved_item is item, "Item should still be in cache"
    
    cache.release(cache_key)  # refcount = 0 - should be removed
    
    try:
        cache.retrieve(cache_key)
        assert False, "Should raise KeyError for removed item"
    except KeyError:
        pass  # Expected
    
    print("✓ Reference counting works")


def test_garbage_collection():
    """Test garbage collection with active keys."""
    cache = ItemCache()
    
    # Add 3 items
    item1 = MockItem("temp1", "value1")
    item2 = MockItem("temp2", "value2")
    item3 = MockItem("temp3", "value3")
    
    key1 = cache.add(item1, "default", {})
    key2 = cache.add(item2, "default", {})
    key3 = cache.add(item3, "default", {})
    
    stats = cache.stats()
    assert stats['total_items'] == 3, f"Should have 3 items, got {stats['total_items']}"
    
    # Mark key1 and key3 as active
    active_keys = {key1, key3}
    released = cache.collect_unreferenced(active_keys)
    
    assert released == 1, f"Should release 1 item (key2), released {released}"
    
    retrieved_item1, _, _ = cache.retrieve(key1)
    assert retrieved_item1 is item1, "key1 should still exist"
    
    try:
        cache.retrieve(key2)
        assert False, "key2 should be collected"
    except KeyError:
        pass  # Expected
    
    retrieved_item3, _, _ = cache.retrieve(key3)
    assert retrieved_item3 is item3, "key3 should still exist"
    
    print("✓ Garbage collection works")


def test_collect_empty_active():
    """Test GC with empty active set (should collect all)."""
    cache = ItemCache()
    
    item1 = MockItem("temp1", "value1")
    item2 = MockItem("temp2", "value2")
    
    key1 = cache.add(item1, "default", {})
    key2 = cache.add(item2, "default", {})
    
    released = cache.collect_unreferenced(set())
    
    assert released == 2, f"Should release 2 items, released {released}"
    
    try:
        cache.retrieve(key1)
        assert False, "key1 should be collected"
    except KeyError:
        pass  # Expected
    
    try:
        cache.retrieve(key2)
        assert False, "key2 should be collected"
    except KeyError:
        pass  # Expected
    
    print("✓ Collect all unreferenced works")


def test_multiple_formats():
    """Test item with multiple formats."""
    cache = ItemCache()
    
    item = MockItem("temp", "base_value")
    item._formats['numpy'] = "numpy_data"
    item._formats['torch'] = "torch_data"
    
    cache_key = cache.add(item, "default", {})
    
    retrieved_item, fmt, _ = cache.retrieve(cache_key)
    assert retrieved_item.get('default') == "base_value"
    assert retrieved_item.get('numpy') == "numpy_data"
    assert retrieved_item.get('torch') == "torch_data"
    
    formats = retrieved_item.get_cached_formats()
    assert 'default' in formats
    assert 'numpy' in formats
    assert 'torch' in formats
    
    print("✓ Multiple formats work")


def test_global_instance():
    """Test global singleton pattern."""
    cache1 = ItemCache.global_instance()
    cache2 = ItemCache.global_instance()
    
    assert cache1 is cache2, "Global instances should be same object"
    
    item = MockItem("temp", "singleton_value")
    cache_key = cache1.add(item, "default", {})
    
    retrieved_item, _, _ = cache2.retrieve(cache_key)
    assert retrieved_item is item, "Both instances should access same cache"
    
    print("✓ Global singleton works")


def run_tests():
    """Run all cache tests."""
    failed = 0
    tests = [
        ("Add and Retrieve", test_add_and_retrieve),
        ("Reference Counting", test_reference_counting),
        ("Garbage Collection", test_garbage_collection),
        ("Collect Empty Active", test_collect_empty_active),
        ("Multiple Formats", test_multiple_formats),
        ("Global Instance", test_global_instance),
    ]
    
    print("=" * 70)
    print("ITEM CACHE TESTS")
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
            failed += 1
    
    print("=" * 70)
    total = len(tests)
    passed = total - failed
    print(f"CACHE TEST RESULTS: {passed}/{total} passed")
    if failed == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ {failed} TESTS FAILED")
    print("=" * 70)
    
    return failed


if __name__ == '__main__':
    sys.exit(run_tests())
