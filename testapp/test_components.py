"""
Test script for the annotation tool components.

This tests the core logic without requiring a display or tkinter.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from testapp.mock_data import get_mock_detections, get_mock_prompts, get_mock_scenario


def test_mock_detections():
    """Test mock detection generation."""
    print("Testing mock detections...")
    detections = get_mock_detections()
    
    assert len(detections) == 3, "Should have 3 top-level detections"
    
    # Check first detection
    main_cluster = detections[0]
    assert main_cluster.object_category == "cluster"
    assert len(main_cluster.children) == 2, "Main cluster should have 2 children"
    
    # Check children
    person = main_cluster.children[0]
    bicycle = main_cluster.children[1]
    assert person.object_category == "person"
    assert bicycle.object_category == "bicycle"
    
    print("✓ Mock detections test passed")


def test_mock_prompts():
    """Test mock prompt generation."""
    print("Testing mock prompts...")
    prompts = get_mock_prompts()
    
    assert len(prompts) > 0, "Should have prompts"
    assert "main_query" in prompts
    assert "person riding bicycle" in prompts["main_query"]
    
    print(f"✓ Mock prompts test passed ({len(prompts)} prompts)")


def test_mock_scenario():
    """Test mock scenario generation."""
    print("Testing mock scenario...")
    scenario = get_mock_scenario()
    
    assert "id" in scenario
    assert scenario["id"] == "std_ride_bike_crowded"
    assert "entities" in scenario
    assert "expected_outcome" in scenario
    
    # Check entities
    entities = scenario["entities"]
    assert len(entities) > 0, "Should have entities"
    
    # Check expected outcome
    expected = scenario["expected_outcome"]
    assert "clusters" in expected
    assert "matches" in expected
    
    print(f"✓ Mock scenario test passed ({len(entities)} entities)")


def test_scenario_file():
    """Test loading scenario from YAML file."""
    print("Testing scenario file loading...")
    
    import yaml
    
    scenario_path = os.path.join(
        os.path.dirname(__file__),
        "example_scenarios.yaml"
    )
    
    with open(scenario_path, 'r') as f:
        data = yaml.safe_load(f)
    
    assert "image_settings" in data
    assert "scenarios" in data
    assert len(data["scenarios"]) >= 2, "Should have at least 2 scenarios"
    
    # Check first scenario
    scenario = data["scenarios"][0]
    assert "id" in scenario
    assert "entities" in scenario
    assert "expected_outcome" in scenario
    
    print(f"✓ Scenario file test passed ({len(data['scenarios'])} scenarios)")


def test_detection_hierarchy():
    """Test detection hierarchy structure."""
    print("Testing detection hierarchy...")
    detections = get_mock_detections()
    
    # Count total detections (including children)
    def count_detections(det_list):
        count = len(det_list)
        for det in det_list:
            count += count_detections(det.children)
        return count
    
    total = count_detections(detections)
    print(f"  Total detections (including children): {total}")
    
    # Test depth calculation
    def max_depth(det_list, current=1):
        if not det_list:
            return current - 1
        depths = []
        for det in det_list:
            if det.children:
                depths.append(max_depth(det.children, current + 1))
            else:
                depths.append(current)
        return max(depths)
    
    depth = max_depth(detections)
    print(f"  Maximum depth: {depth}")
    
    assert depth == 2, "Mock data should have depth of 2"
    print("✓ Detection hierarchy test passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Annotation Tool Components")
    print("=" * 60)
    print()
    
    try:
        test_mock_detections()
        test_mock_prompts()
        test_mock_scenario()
        test_scenario_file()
        test_detection_hierarchy()
        
        print()
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print()
        print("Note: GUI requires tkinter and a display to run.")
        print("To run the GUI on a system with display:")
        print("  python -m testapp.annotation_tool")
        
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"Test failed: {e}")
        print("=" * 60)
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"Error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
