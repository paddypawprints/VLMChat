#!/usr/bin/env python3
"""
Test script for the VLMChat configuration system.

This script tests the configuration loading and validation functionality
to ensure all components work together correctly.
"""

import sys
from pathlib import Path

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent / "src"
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import VLMChatConfig, load_config, create_default_config_file, get_config


def test_default_config():
    """Test loading default configuration."""
    print("Testing default configuration...")
    try:
        config = VLMChatConfig()
        print("✅ Default configuration created successfully")

        # Test all sections
        print(f"  Model path: {config.model.model_path}")
        print(f"  Max tokens: {config.model.max_new_tokens}")
        print(f"  ONNX enabled: {config.model.use_onnx}")
        print(f"  Max pairs: {config.conversation.max_pairs}")
        print(f"  History format: {config.conversation.history_format}")
        print(f"  Word limit: {config.conversation.word_limit}")
        print(f"  Log level: {config.logging.level}")
        print(f"  Project root: {config.paths.project_root}")
        print(f"  Captured images dir: {config.paths.captured_images_dir}")

        return True
    except Exception as e:
        print(f"❌ Failed to create default config: {e}")
        return False


def test_config_file_creation():
    """Test creating configuration file."""
    print("\nTesting configuration file creation...")
    try:
        config_path = "test_config.json"
        create_default_config_file(config_path)

        # Verify file was created
        if Path(config_path).exists():
            print("✅ Configuration file created successfully")

            # Clean up
            Path(config_path).unlink()
            return True
        else:
            print("❌ Configuration file was not created")
            return False
    except Exception as e:
        print(f"❌ Failed to create config file: {e}")
        return False


def test_config_file_loading():
    """Test loading configuration from file."""
    print("\nTesting configuration file loading...")
    try:
        # Create a temporary config file
        config_path = "test_load_config.json"
        create_default_config_file(config_path)

        # Load it back
        config = VLMChatConfig.load_from_file(config_path)
        print("✅ Configuration loaded from file successfully")

        # Test values
        assert config.model.model_path == "HuggingFaceTB/SmolVLM2-256M-Instruct"
        assert config.conversation.max_pairs == 10
        assert config.logging.level.value == "INFO"
        print("✅ Configuration values validated successfully")

        # Clean up
        Path(config_path).unlink()
        return True
    except Exception as e:
        print(f"❌ Failed to load config from file: {e}")
        # Clean up on error
        if Path("test_load_config.json").exists():
            Path("test_load_config.json").unlink()
        return False


def test_environment_loading():
    """Test loading configuration from environment variables."""
    print("\nTesting environment variable loading...")
    try:
        import os

        # Set some test environment variables
        os.environ["VLMCHAT_MODEL_PATH"] = "test/model/path"
        os.environ["VLMCHAT_MAX_PAIRS"] = "5"
        os.environ["VLMCHAT_LOG_LEVEL"] = "DEBUG"

        config = VLMChatConfig.load_from_env()
        print("✅ Configuration loaded from environment successfully")

        # Test values
        assert config.model.model_path == "test/model/path"
        assert config.conversation.max_pairs == 5
        assert config.logging.level.value == "DEBUG"
        print("✅ Environment configuration values validated successfully")

        # Clean up
        del os.environ["VLMCHAT_MODEL_PATH"]
        del os.environ["VLMCHAT_MAX_PAIRS"]
        del os.environ["VLMCHAT_LOG_LEVEL"]

        return True
    except Exception as e:
        print(f"❌ Failed to load config from environment: {e}")
        return False


def test_validation():
    """Test configuration validation."""
    print("\nTesting configuration validation...")
    try:
        # Test invalid values
        try:
            config = VLMChatConfig()
            config.model.max_new_tokens = -1  # Should be positive
            config.model_validate()  # This should raise an error
            print("❌ Validation should have failed for negative max_new_tokens")
            return False
        except:
            print("✅ Validation correctly rejected negative max_new_tokens")

        try:
            config = VLMChatConfig()
            config.conversation.max_pairs = 0  # Should be positive
            config.model_validate()  # This should raise an error
            print("❌ Validation should have failed for zero max_pairs")
            return False
        except:
            print("✅ Validation correctly rejected zero max_pairs")

        return True
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False


def test_global_config():
    """Test global configuration management."""
    print("\nTesting global configuration management...")
    try:
        # Test getting default global config
        config1 = get_config()
        config2 = get_config()

        # Should be the same instance
        assert config1 is config2
        print("✅ Global config singleton works correctly")

        # Test loading global config
        load_config()
        config3 = get_config()
        print("✅ Global config loading works correctly")

        return True
    except Exception as e:
        print(f"❌ Global config test failed: {e}")
        return False


def main():
    """Run all configuration tests."""
    print("VLMChat Configuration System Test")
    print("=" * 40)

    tests = [
        test_default_config,
        test_config_file_creation,
        test_config_file_loading,
        test_environment_loading,
        test_validation,
        test_global_config
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1

    print("\n" + "=" * 40)
    print(f"Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)