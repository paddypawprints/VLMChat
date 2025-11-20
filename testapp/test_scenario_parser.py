"""
Test script for scenario parser.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from testapp.scenario_parser import ScenarioParser, parse_scenario_file


def test_scenario_parser():
    """Test the scenario parser functionality."""
    print("=" * 60)
    print("Testing Scenario Parser")
    print("=" * 60)
    print()
    
    # Load the example scenario file
    scenario_path = os.path.join(
        os.path.dirname(__file__),
        "example_scenarios.yaml"
    )
    
    print(f"Loading scenario from: {scenario_path}")
    parser = parse_scenario_file(scenario_path)
    
    # Test image settings
    print("\n1. Testing image settings...")
    width, height = parser.get_image_size()
    print(f"   Image size: {width}x{height}")
    assert width == 1920
    assert height == 1080
    print("   ✓ Image settings OK")
    
    # Test scenario IDs
    print("\n2. Testing scenario IDs...")
    scenario_ids = parser.get_all_scenario_ids()
    print(f"   Found {len(scenario_ids)} scenarios:")
    for sid in scenario_ids:
        print(f"     - {sid}")
    assert len(scenario_ids) >= 2
    print("   ✓ Scenario IDs OK")
    
    # Test scenario retrieval
    print("\n3. Testing scenario retrieval...")
    first_scenario = parser.get_scenario_by_id(scenario_ids[0])
    assert first_scenario is not None
    print(f"   Retrieved scenario: {first_scenario['id']}")
    print(f"   Type: {first_scenario.get('type')}")
    print("   ✓ Scenario retrieval OK")
    
    # Test prompt extraction
    print("\n4. Testing prompt extraction...")
    prompts = parser.extract_prompts(first_scenario)
    print(f"   Extracted {len(prompts)} prompts:")
    for key, value in list(prompts.items())[:3]:
        print(f"     - {key}: {value[:50]}...")
    assert len(prompts) > 0
    print("   ✓ Prompt extraction OK")
    
    # Test cluster extraction
    print("\n5. Testing cluster extraction...")
    clusters = parser.extract_clusters(first_scenario)
    print(f"   Extracted {len(clusters)} clusters:")
    for cluster in clusters:
        print(f"     - {cluster.get('id')}: {cluster.get('location')} "
              f"({len(cluster.get('members', []))} members)")
    assert len(clusters) > 0
    print("   ✓ Cluster extraction OK")
    
    # Test entity extraction
    print("\n6. Testing entity extraction...")
    entities = parser.extract_entities(first_scenario)
    print(f"   Extracted {len(entities)} entities:")
    for entity in entities[:3]:
        print(f"     - {entity.get('id')}: {entity.get('label')}")
    assert len(entities) > 0
    print("   ✓ Entity extraction OK")
    
    # Test environment info
    print("\n7. Testing environment info...")
    env = parser.get_environment_info(first_scenario)
    print(f"   Environment:")
    for key, value in env.items():
        print(f"     - {key}: {value}")
    assert 'time' in env or 'weather' in env
    print("   ✓ Environment info OK")
    
    # Test prompt update
    print("\n8. Testing prompt update...")
    original_prompt = prompts.get('scene_prompt', '')
    print(f"   Original prompt: {original_prompt[:50]}...")
    
    new_prompts = prompts.copy()
    new_prompts['scene_prompt'] = "Updated test prompt"
    
    success = parser.update_scenario_prompts(scenario_ids[0], new_prompts)
    assert success
    
    # Verify update
    updated_scenario = parser.get_scenario_by_id(scenario_ids[0])
    updated_prompt = updated_scenario.get('prompt', '')
    print(f"   Updated prompt: {updated_prompt[:50]}...")
    assert updated_prompt == "Updated test prompt"
    print("   ✓ Prompt update OK")
    
    # Test to_dict
    print("\n9. Testing to_dict...")
    data_dict = parser.to_dict()
    assert 'scenarios' in data_dict
    assert 'image_settings' in data_dict
    print(f"   Dictionary has {len(data_dict)} top-level keys")
    print("   ✓ to_dict OK")
    
    print("\n" + "=" * 60)
    print("All scenario parser tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_scenario_parser()
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
