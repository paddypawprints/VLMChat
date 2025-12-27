"""
Smoke tests for contract validation system.

Tests that contracts correctly validate CachedItemType and format compatibility
when connecting tasks in a pipeline.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vlmchat.pipeline.core.task_base import BaseTask, Context, ContextDataType
from vlmchat.pipeline.cache.types import CachedItemType


class DummySourceTask(BaseTask):
    """Source task that produces IMAGE data."""
    def __init__(self, item_type=CachedItemType.IMAGE, format_str="pil", label="default"):
        super().__init__("source")
        self.output_contract = {
            ContextDataType.IMAGE: {label: (item_type, format_str)}
        }
    
    def run(self, context: Context) -> Context:
        return context


class DummyConsumerTask(BaseTask):
    """Consumer task that requires IMAGE data."""
    def __init__(self, item_type=CachedItemType.IMAGE, format_str="pil", label="default"):
        super().__init__("consumer")
        self.input_contract = {
            ContextDataType.IMAGE: {label: (item_type, format_str)}
        }
    
    def run(self, context: Context) -> Context:
        return context


def test_matching_contracts():
    """Test that matching contracts validate successfully."""
    source = DummySourceTask(CachedItemType.IMAGE, "pil")
    consumer = DummyConsumerTask(CachedItemType.IMAGE, "pil")
    
    try:
        consumer.validate_input_contract(source)
        print("✓ Matching contracts validated")
        return True
    except ValueError as e:
        print(f"✗ Matching contracts failed: {e}")
        return False


def test_mismatched_item_type():
    """Test that mismatched CachedItemType fails validation."""
    source = DummySourceTask(CachedItemType.IMAGE, "pil")
    consumer = DummyConsumerTask(CachedItemType.EMBEDDING, "numpy")
    
    try:
        consumer.validate_input_contract(source)
        print("✗ Mismatched item type should have failed")
        return False
    except ValueError as e:
        if "Type mismatch" in str(e) and "image" in str(e) and "embedding" in str(e):
            print("✓ Mismatched item type rejected correctly")
            return True
        else:
            print(f"✗ Wrong error message: {e}")
            return False


def test_missing_data_type():
    """Test that missing ContextDataType fails validation."""
    source = DummySourceTask()
    source.output_contract = {}  # Produces nothing
    consumer = DummyConsumerTask()
    
    try:
        consumer.validate_input_contract(source)
        print("✗ Missing data type should have failed")
        return False
    except ValueError as e:
        if "doesn't produce required" in str(e):
            print("✓ Missing data type rejected correctly")
            return True
        else:
            print(f"✗ Wrong error message: {e}")
            return False


def test_different_formats_same_type():
    """Test that same item type with different formats still validates."""
    source = DummySourceTask(CachedItemType.IMAGE, "numpy")
    consumer = DummyConsumerTask(CachedItemType.IMAGE, "pil")
    
    # Should validate - format transformation is supported by CachedItem
    try:
        consumer.validate_input_contract(source)
        print("✓ Different formats with same type validated")
        return True
    except ValueError as e:
        print(f"✗ Different formats failed: {e}")
        return False


def test_none_format():
    """Test that None format works (no format requirement)."""
    source = DummySourceTask(CachedItemType.DETECTION, None)
    consumer = DummyConsumerTask(CachedItemType.DETECTION, None)
    
    try:
        consumer.validate_input_contract(source)
        print("✓ None format validated")
        return True
    except ValueError as e:
        print(f"✗ None format failed: {e}")
        return False


def test_empty_contracts():
    """Test that empty contracts (source/sink tasks) validate."""
    source = DummySourceTask()
    source.input_contract = {}  # No inputs (source)
    source.output_contract = {ContextDataType.IMAGE: {"default": (CachedItemType.IMAGE, "pil")}}
    
    consumer = DummyConsumerTask()
    consumer.input_contract = {ContextDataType.IMAGE: {"default": (CachedItemType.IMAGE, "pil")}}
    consumer.output_contract = {}  # No outputs (sink)
    
    try:
        consumer.validate_input_contract(source)
        print("✓ Empty contracts validated")
        return True
    except ValueError as e:
        print(f"✗ Empty contracts failed: {e}")
        return False


def test_multiple_data_types():
    """Test validation with multiple data types."""
    source = DummySourceTask()
    source.output_contract = {
        ContextDataType.IMAGE: {"default": (CachedItemType.IMAGE, "pil")},
        ContextDataType.EMBEDDINGS: {"default": (CachedItemType.EMBEDDING, "numpy")}
    }
    
    consumer = DummyConsumerTask()
    consumer.input_contract = {
        ContextDataType.IMAGE: {"default": (CachedItemType.IMAGE, "pil")},
        ContextDataType.EMBEDDINGS: {"default": (CachedItemType.EMBEDDING, "numpy")}
    }
    
    try:
        consumer.validate_input_contract(source)
        print("✓ Multiple data types validated")
        return True
    except ValueError as e:
        print(f"✗ Multiple data types failed: {e}")
        return False


def test_partial_match():
    """Test that consumer requiring subset of producer outputs validates."""
    source = DummySourceTask()
    source.output_contract = {
        ContextDataType.IMAGE: {"default": (CachedItemType.IMAGE, "pil")},
        ContextDataType.EMBEDDINGS: {"default": (CachedItemType.EMBEDDING, "numpy")},
        ContextDataType.TEXT: {"default": (CachedItemType.TEXT, None)}
    }
    
    # Consumer only needs IMAGE
    consumer = DummyConsumerTask(CachedItemType.IMAGE, "pil")
    
    try:
        consumer.validate_input_contract(source)
        print("✓ Partial match validated")
        return True
    except ValueError as e:
        print(f"✗ Partial match failed: {e}")
        return False


def test_label_matching():
    """Test that matching labels validate successfully."""
    source = DummySourceTask(label="camera1")
    consumer = DummyConsumerTask(label="camera1")
    
    try:
        consumer.validate_input_contract(source)
        print("✓ Matching labels validated")
        return True
    except ValueError as e:
        print(f"✗ Matching labels failed: {e}")
        return False


def test_label_missing():
    """Test that missing label fails validation."""
    source = DummySourceTask(label="camera1")
    consumer = DummyConsumerTask(label="camera2")
    
    try:
        consumer.validate_input_contract(source)
        print("✗ Missing label should have failed")
        return False
    except ValueError as e:
        if "doesn't produce required label" in str(e) and "camera2" in str(e):
            print("✓ Missing label rejected correctly")
            return True
        else:
            print(f"✗ Wrong error message: {e}")
            return False


def test_multiple_labels():
    """Test validation with multiple labels."""
    source = DummySourceTask()
    source.output_contract = {
        ContextDataType.IMAGE: {
            "roadview": (CachedItemType.IMAGE, "pil"),
            "cabview": (CachedItemType.IMAGE, "pil")
        }
    }
    
    consumer = DummyConsumerTask()
    consumer.input_contract = {
        ContextDataType.IMAGE: {
            "roadview": (CachedItemType.IMAGE, "pil"),
            "cabview": (CachedItemType.IMAGE, "pil")
        }
    }
    
    try:
        consumer.validate_input_contract(source)
        print("✓ Multiple labels validated")
        return True
    except ValueError as e:
        print(f"✗ Multiple labels failed: {e}")
        return False


def test_partial_label_match():
    """Test that consumer requiring subset of producer labels validates."""
    source = DummySourceTask()
    source.output_contract = {
        ContextDataType.IMAGE: {
            "roadview": (CachedItemType.IMAGE, "pil"),
            "cabview": (CachedItemType.IMAGE, "pil"),
            "rearview": (CachedItemType.IMAGE, "pil")
        }
    }
    
    # Consumer only needs roadview
    consumer = DummyConsumerTask(label="roadview")
    
    try:
        consumer.validate_input_contract(source)
        print("✓ Partial label match validated")
        return True
    except ValueError as e:
        print(f"✗ Partial label match failed: {e}")
        return False


def test_label_type_mismatch():
    """Test that mismatched types within labels fail validation."""
    source = DummySourceTask()
    source.output_contract = {
        ContextDataType.IMAGE: {
            "camera1": (CachedItemType.IMAGE, "pil")
        }
    }
    
    consumer = DummyConsumerTask()
    consumer.input_contract = {
        ContextDataType.IMAGE: {
            "camera1": (CachedItemType.EMBEDDING, "numpy")
        }
    }
    
    try:
        consumer.validate_input_contract(source)
        print("✗ Label type mismatch should have failed")
        return False
    except ValueError as e:
        if "Type mismatch" in str(e) and "camera1" in str(e):
            print("✓ Label type mismatch rejected correctly")
            return True
        else:
            print(f"✗ Wrong error message: {e}")
            return False


def main():
    print("="*70)
    print("CONTRACT VALIDATION TESTS")
    print("="*70)
    print()
    
    tests = [
        ("Matching Contracts", test_matching_contracts),
        ("Mismatched Item Type", test_mismatched_item_type),
        ("Missing Data Type", test_missing_data_type),
        ("Different Formats", test_different_formats_same_type),
        ("None Format", test_none_format),
        ("Empty Contracts", test_empty_contracts),
        ("Multiple Data Types", test_multiple_data_types),
        ("Partial Match", test_partial_match),
        ("Label Matching", test_label_matching),
        ("Label Missing", test_label_missing),
        ("Multiple Labels", test_multiple_labels),
        ("Partial Label Match", test_partial_label_match),
        ("Label Type Mismatch", test_label_type_mismatch),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"{name}:")
        try:
            if test_func():
                passed += 1
                print("✅ PASSED")
            else:
                failed += 1
                print("❌ FAILED")
        except Exception as e:
            print(f"❌ FAILED with exception: {e}")
            failed += 1
        print()
    
    print("="*70)
    print(f"CONTRACT TEST RESULTS: {passed}/{passed+failed} passed")
    if failed == 0:
        print("✅ ALL TESTS PASSED")
        print("="*70)
        return 0
    else:
        print(f"❌ {failed} TESTS FAILED")
        print("="*70)
        return failed


if __name__ == "__main__":
    sys.exit(main())
