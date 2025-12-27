#!/usr/bin/env python3
"""
Simple test of pipeline commands - just verifying the methods exist and have the right signatures.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_service_methods():
    """Test that service methods are defined correctly."""
    from src.main.chat_services import VLMChatServices
    import inspect
    
    # Get the class methods
    methods = inspect.getmembers(VLMChatServices, predicate=inspect.isfunction)
    method_dict = dict(methods)
    
    # Check that our new methods exist
    required_methods = [
        '_service_pipeline',
        '_service_run',
        '_service_stop',
        '_service_status'
    ]
    
    print("Checking for required service methods...")
    for method_name in required_methods:
        if method_name in method_dict:
            sig = inspect.signature(method_dict[method_name])
            print(f"  ✓ {method_name}{sig}")
        else:
            print(f"  ✗ {method_name} NOT FOUND")
            return False
    
    print("\n✓ All service methods present!")
    return True

def test_command_handlers():
    """Test that command handlers are present."""
    from src.main import chat_console
    import inspect
    
    # Read the process_command function source
    source = inspect.getsource(chat_console.process_command)
    
    # Check for our new command handlers
    required_commands = [
        '/pipeline',
        '/run',
        '/stop',
        '/status'
    ]
    
    print("\nChecking for command handlers...")
    for cmd in required_commands:
        if cmd in source:
            print(f"  ✓ {cmd} handler present")
        else:
            print(f"  ✗ {cmd} handler NOT FOUND")
            return False
    
    print("\n✓ All command handlers present!")
    return True

def test_help_message():
    """Test that help message includes pipeline commands."""
    from src.main import chat_console
    import inspect
    
    # Read the help function source
    source = inspect.getsource(chat_console.print_help_message)
    
    # Check for our new commands in help
    required_in_help = [
        '/pipeline',
        '/run',
        '/stop',
        '/status'
    ]
    
    print("\nChecking help message...")
    for cmd in required_in_help:
        if cmd in source:
            print(f"  ✓ {cmd} in help")
        else:
            print(f"  ✗ {cmd} NOT in help")
            return False
    
    print("\n✓ All commands in help message!")
    return True

def test_config():
    """Test that config has pipeline_dirs."""
    from src.utils.config import PathsConfig
    import inspect
    
    # Check the PathsConfig class
    source = inspect.getsource(PathsConfig)
    
    print("\nChecking config schema...")
    if 'pipeline_dirs' in source:
        print("  ✓ pipeline_dirs in PathsConfig")
    else:
        print("  ✗ pipeline_dirs NOT in PathsConfig")
        return False
    
    print("\n✓ Config schema updated!")
    return True

if __name__ == "__main__":
    print("=== Testing Pipeline Integration ===\n")
    
    tests = [
        test_service_methods,
        test_command_handlers,
        test_help_message,
        test_config
    ]
    
    all_passed = True
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    if all_passed:
        print("\n" + "="*50)
        print("✓ ALL TESTS PASSED!")
        print("="*50)
        sys.exit(0)
    else:
        print("\n" + "="*50)
        print("✗ SOME TESTS FAILED")
        print("="*50)
        sys.exit(1)
