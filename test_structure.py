#!/usr/bin/env python3
"""
Test script for the enhanced sentiment analysis pipeline.

This script tests the new modular structure and functionality.
"""

import sys
from pathlib import Path

# Add the divers directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))


def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")

    try:
        from divers.config import SentimentConfig, ConfigManager

        print("✓ Config module imported successfully")
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False

    try:
        from divers.data_module import DataManager

        print("✓ Data module imported successfully")
    except Exception as e:
        print(f"✗ Data module import failed: {e}")
        return False

    try:
        from divers.model_module import ModelManager

        print("✓ Model module imported successfully")
    except Exception as e:
        print(f"✗ Model module import failed: {e}")
        return False

    try:
        from divers.training_module import SentimentTrainer

        print("✓ Training module imported successfully")
    except Exception as e:
        print(f"✗ Training module import failed: {e}")
        return False

    try:
        from divers.evaluation_module import SentimentEvaluator

        print("✓ Evaluation module imported successfully")
    except Exception as e:
        print(f"✗ Evaluation module import failed: {e}")
        return False

    return True


def test_configuration():
    """Test configuration management."""
    print("\nTesting configuration...")

    try:
        from divers.config import SentimentConfig, ConfigManager

        # Test default configuration
        config = SentimentConfig()
        print(f"✓ Default config created with model: {config.model.model_name}")

        # Test configuration validation
        ConfigManager.validate_config(config)
        print("✓ Configuration validation passed")

        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_cli():
    """Test CLI argument parsing."""
    print("\nTesting CLI...")

    try:
        from divers.__main__ import SentimentCLI

        cli = SentimentCLI()
        parser = cli.create_parser()
        print("✓ CLI parser created successfully")

        # Test help for train command
        args = parser.parse_args(["train", "--help"])
        print("✓ CLI help working")

        return True
    except SystemExit:
        # Help command causes SystemExit, which is expected
        print("✓ CLI help working (SystemExit expected)")
        return True
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("ENHANCED SENTIMENT ANALYSIS - STRUCTURE TEST")
    print("=" * 60)

    tests = [
        test_imports,
        test_configuration,
        test_cli,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        else:
            print("Test failed, continuing...")

    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The enhanced structure is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
