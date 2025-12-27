"""
Smoke tests for label-based data flow.

Tests that Context.split() and Connector.merge_strategy() correctly handle
label-based data structure for multi-source provenance.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vlmchat.pipeline.core.task_base import Context, ContextDataType, Connector
from vlmchat.pipeline.cache.text import TextContainer
from vlmchat.pipeline.cache.types import CachedItemType


def test_context_split_with_labels():
    """Test that context.split() correctly copies label structure."""
    context = Context()
    
    # Add text items with multiple labels
    context.data[ContextDataType.TEXT] = {
        "log": [TextContainer("Log 1", "log1"), TextContainer("Log 2", "log2")],
        "output": [TextContainer("Output 1", "out1")]
    }
    
    # Split into 2 branches
    branches = context.split(2, {})
    
    assert len(branches) == 2, "Should create 2 branches"
    
    # Verify both branches have the label structure
    for i, branch in enumerate(branches):
        assert ContextDataType.TEXT in branch.data, f"Branch {i} should have TEXT"
        assert "log" in branch.data[ContextDataType.TEXT], f"Branch {i} should have log label"
        assert "output" in branch.data[ContextDataType.TEXT], f"Branch {i} should have output label"
        assert len(branch.data[ContextDataType.TEXT]["log"]) == 2, f"Branch {i} should have 2 log items"
        assert len(branch.data[ContextDataType.TEXT]["output"]) == 1, f"Branch {i} should have 1 output item"
    
    # Verify items are shallow copied (same references)
    assert branches[0].data[ContextDataType.TEXT]["log"][0] is branches[1].data[ContextDataType.TEXT]["log"][0]
    
    print("✓ Context split preserves label structure")
    return True


def test_merge_with_labels():
    """Test that Connector.merge_strategy() correctly merges by label."""
    # Create two contexts with labeled data
    ctx1 = Context()
    ctx1.data[ContextDataType.TEXT] = {
        "log": [TextContainer("Branch 1 Log", "b1_log")],
        "output": [TextContainer("Branch 1 Output", "b1_out")]
    }
    
    ctx2 = Context()
    ctx2.data[ContextDataType.TEXT] = {
        "log": [TextContainer("Branch 2 Log", "b2_log")],
        "output": [TextContainer("Branch 2 Output", "b2_out")]
    }
    
    # Merge using Connector
    connector = Connector("test_merge")
    merged = connector.merge_strategy([ctx1, ctx2])
    
    # Verify merge preserved label structure
    assert ContextDataType.TEXT in merged.data, "Merged should have TEXT"
    assert "log" in merged.data[ContextDataType.TEXT], "Merged should have log label"
    assert "output" in merged.data[ContextDataType.TEXT], "Merged should have output label"
    
    # Verify items from both branches are in each label
    assert len(merged.data[ContextDataType.TEXT]["log"]) == 2, "Should have 2 log items (one from each branch)"
    assert len(merged.data[ContextDataType.TEXT]["output"]) == 2, "Should have 2 output items (one from each branch)"
    
    # Verify content
    log_texts = [item.get("str") for item in merged.data[ContextDataType.TEXT]["log"]]
    assert "Branch 1 Log" in log_texts, "Should have branch 1 log"
    assert "Branch 2 Log" in log_texts, "Should have branch 2 log"
    
    output_texts = [item.get("str") for item in merged.data[ContextDataType.TEXT]["output"]]
    assert "Branch 1 Output" in output_texts, "Should have branch 1 output"
    assert "Branch 2 Output" in output_texts, "Should have branch 2 output"
    
    print("✓ Merge correctly combines items by label")
    return True


def test_merge_deduplication_per_label():
    """Test that merge deduplicates within each label separately."""
    # Create shared items
    shared_log = TextContainer("Shared Log", "shared_log")
    shared_output = TextContainer("Shared Output", "shared_out")
    
    # Create two contexts sharing some items
    ctx1 = Context()
    ctx1.data[ContextDataType.TEXT] = {
        "log": [shared_log, TextContainer("Unique 1", "uniq1")],
        "output": [shared_output]
    }
    
    ctx2 = Context()
    ctx2.data[ContextDataType.TEXT] = {
        "log": [shared_log, TextContainer("Unique 2", "uniq2")],
        "output": [shared_output]
    }
    
    # Merge
    connector = Connector("test_dedup")
    merged = connector.merge_strategy([ctx1, ctx2])
    
    # Verify deduplication within each label
    # log should have 3 items: shared_log (deduplicated), unique1, unique2
    assert len(merged.data[ContextDataType.TEXT]["log"]) == 3, "Should deduplicate shared log"
    
    # output should have 1 item: shared_output (deduplicated)
    assert len(merged.data[ContextDataType.TEXT]["output"]) == 1, "Should deduplicate shared output"
    
    print("✓ Merge deduplicates correctly within each label")
    return True


def test_mixed_labels_merge():
    """Test merging contexts with different label sets."""
    # Context 1 has log and output labels
    ctx1 = Context()
    ctx1.data[ContextDataType.TEXT] = {
        "log": [TextContainer("Log 1", "log1")],
        "output": [TextContainer("Output 1", "out1")]
    }
    
    # Context 2 has log and error labels
    ctx2 = Context()
    ctx2.data[ContextDataType.TEXT] = {
        "log": [TextContainer("Log 2", "log2")],
        "error": [TextContainer("Error 1", "err1")]
    }
    
    # Merge
    connector = Connector("test_mixed")
    merged = connector.merge_strategy([ctx1, ctx2])
    
    # Verify all labels present
    assert "log" in merged.data[ContextDataType.TEXT], "Should have log label"
    assert "output" in merged.data[ContextDataType.TEXT], "Should have output label"
    assert "error" in merged.data[ContextDataType.TEXT], "Should have error label"
    
    # Verify counts
    assert len(merged.data[ContextDataType.TEXT]["log"]) == 2, "Should have 2 log items"
    assert len(merged.data[ContextDataType.TEXT]["output"]) == 1, "Should have 1 output item"
    assert len(merged.data[ContextDataType.TEXT]["error"]) == 1, "Should have 1 error item"
    
    print("✓ Merge correctly handles different label sets")
    return True


def main():
    print("="*70)
    print("LABEL-BASED DATA FLOW TESTS")
    print("="*70)
    print()
    
    tests = [
        ("Context Split with Labels", test_context_split_with_labels),
        ("Merge with Labels", test_merge_with_labels),
        ("Merge Deduplication per Label", test_merge_deduplication_per_label),
        ("Mixed Labels Merge", test_mixed_labels_merge),
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
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("="*70)
    print(f"LABEL FLOW TEST RESULTS: {passed}/{passed+failed} passed")
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
