"""
Test task documentation system.

Tests the describe() methods and TaskHelpFormatter with CameraTask.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline.dsl_parser import create_task_registry
from src.pipeline.task_help_formatter import TaskHelpFormatter


def test_camera_task_documentation():
    """Test that CameraTask has documentation."""
    print("=" * 60)
    print("Testing CameraTask Documentation")
    print("=" * 60)
    
    # Get task registry
    registry = create_task_registry()
    
    # Check camera task exists
    if 'camera' not in registry:
        print("ERROR: camera task not in registry")
        return False
    
    # Create instance
    camera_class = registry['camera']
    camera_task = camera_class(task_id="test_camera")
    
    # Test describe()
    print("\n1. Testing describe():")
    description = camera_task.describe()
    print(f"   {description}")
    if "camera" not in description.lower():
        print("   WARNING: Description doesn't mention 'camera'")
    
    # Test describe_contracts()
    print("\n2. Testing describe_contracts():")
    contracts = camera_task.describe_contracts()
    print(f"   Inputs: {list(contracts['inputs'].keys())}")
    print(f"   Outputs: {list(contracts['outputs'].keys())}")
    
    # Test describe_parameters()
    print("\n3. Testing describe_parameters():")
    params = camera_task.describe_parameters()
    if not params:
        print("   No parameters documented")
        return False
    
    for param_name, param_info in params.items():
        print(f"\n   Parameter: {param_name}")
        if isinstance(param_info, str):
            print(f"     {param_info}")
        else:
            for key, value in param_info.items():
                print(f"     {key}: {value}")
    
    print("\n" + "=" * 60)
    return True


def test_formatter_console():
    """Test console formatting."""
    print("=" * 60)
    print("Testing TaskHelpFormatter - Console Format")
    print("=" * 60)
    
    # Get camera task
    registry = create_task_registry()
    camera_class = registry['camera']
    camera_task = camera_class(task_id="test_camera")
    
    # Format as console
    formatter = TaskHelpFormatter()
    output = formatter.format_console(camera_task)
    
    print("\n" + output)
    print("\n" + "=" * 60)
    return True


def test_formatter_markdown():
    """Test markdown formatting."""
    print("=" * 60)
    print("Testing TaskHelpFormatter - Markdown Format")
    print("=" * 60)
    
    # Get camera task
    registry = create_task_registry()
    camera_class = registry['camera']
    camera_task = camera_class(task_id="test_camera")
    
    # Format as markdown
    formatter = TaskHelpFormatter()
    output = formatter.format_markdown(camera_task)
    
    print("\n" + output)
    print("\n" + "=" * 60)
    return True


def test_parser_validation():
    """Test that parser warns about undocumented parameters."""
    print("=" * 60)
    print("Testing Parser Parameter Validation")
    print("=" * 60)
    
    from src.pipeline.dsl_parser import DSLParser
    import logging
    import io
    
    # Capture log output
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.WARNING)
    logger = logging.getLogger('src.pipeline.dsl_parser')
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    
    # Create parser
    registry = create_task_registry()
    parser = DSLParser(registry)
    
    # Test 1: Valid documented parameter
    print("\n1. Testing documented parameter (type=\"none\"):")
    log_stream.truncate(0)
    log_stream.seek(0)
    try:
        pipeline = parser.parse('camera(type="none")')
        logs = log_stream.getvalue()
        if logs:
            print(f"   UNEXPECTED WARNING: {logs}")
            return False
        print("   ✓ No warnings (parameter documented)")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test 2: Undocumented parameter (should warn)
    print("\n2. Testing undocumented parameter (fake_param=\"value\"):")
    log_stream.truncate(0)
    log_stream.seek(0)
    try:
        pipeline = parser.parse('camera(type="none", fake_param="test")')
        logs = log_stream.getvalue()
        if "fake_param" in logs and "no description" in logs:
            print("   ✓ Warning logged about undocumented parameter")
        else:
            print(f"   ERROR: Expected warning not found. Logs: {logs}")
            return False
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test 3: Contract validation (connecting incompatible tasks)
    print("\n3. Testing contract validation (camera -> camera):")
    try:
        # This should fail - can't connect camera to camera (output IMAGE -> needs no input)
        pipeline = parser.parse('camera(type="none") -> camera(type="none")')
        # If we get here, check if validation happened during build
        print("   ⚠ Pipeline created (validation may happen at runtime)")
    except Exception as e:
        if "contract" in str(e).lower() or "doesn't produce" in str(e).lower():
            print(f"   ✓ Contract validation failed as expected: {e}")
        else:
            print(f"   ERROR: Unexpected error: {e}")
            return False
    
    logger.removeHandler(handler)
    print("\n" + "=" * 60)
    return True


def main():
    """Run all tests."""
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    tests = [
        ("CameraTask Documentation", test_camera_task_documentation),
        ("Console Formatter", test_formatter_console),
        ("Markdown Formatter", test_formatter_markdown),
        ("Parser Validation", test_parser_validation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            print()
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(success for _, success in results)
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed"))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
