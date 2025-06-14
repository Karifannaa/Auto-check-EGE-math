#!/usr/bin/env python3
"""
Simple test script to verify the evaluation system works correctly.
This script tests the basic functionality without requiring an API key.
"""

import os
import sys
import json
from datasets import load_from_disk

# Set a dummy API key for testing before importing anything
os.environ["OPENROUTER_API_KEY"] = "test_key_for_testing"

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

def test_dataset_loading():
    """Test that we can load the dataset correctly."""
    print("Testing dataset loading...")
    
    # Test both datasets
    datasets_to_test = [
        ("dataset_benchmark_hf", "Original dataset"),
        ("dataset_benchmark_hf_updated", "Updated dataset with true solutions")
    ]
    
    for dataset_dir, description in datasets_to_test:
        if os.path.exists(dataset_dir):
            try:
                dataset = load_from_disk(dataset_dir)
                print(f"✓ {description}: {len(dataset)} examples loaded")
                
                # Check for true solution images
                true_solution_count = sum(1 for ex in dataset if ex.get("images_with_true_solution"))
                print(f"  - Examples with true solution images: {true_solution_count}")
                
                # Check task distribution
                task_counts = {}
                for ex in dataset:
                    task_id = ex["task_id"]
                    task_counts[task_id] = task_counts.get(task_id, 0) + 1
                
                print(f"  - Task distribution: {dict(sorted(task_counts.items()))}")
                
            except Exception as e:
                print(f"✗ {description}: Error loading - {e}")
        else:
            print(f"✗ {description}: Directory not found")
    
    return True

def test_imports():
    """Test that all required modules can be imported."""
    print("\nTesting imports...")
    
    try:
        from app.utils.prompt_utils import PromptGenerator
        print("✓ PromptGenerator imported successfully")
    except Exception as e:
        print(f"✗ PromptGenerator import failed: {e}")
        return False
    
    try:
        from app.utils.image_utils import prepare_image_for_api
        print("✓ Image utils imported successfully")
    except Exception as e:
        print(f"✗ Image utils import failed: {e}")
        return False
    
    try:
        from app.api.openrouter_client import OpenRouterClient
        print("✓ OpenRouter client imported successfully")
    except Exception as e:
        print(f"✗ OpenRouter client import failed: {e}")
        return False
    
    try:
        from app.core.config import settings
        print("✓ Settings imported successfully")
        print(f"  - Available models: {len(settings.AVAILABLE_MODELS)}")
        print(f"  - Default model: {settings.DEFAULT_MODEL}")
    except Exception as e:
        print(f"✗ Settings import failed: {e}")
        return False
    
    return True

def test_benchmark_class():
    """Test that the ModelBenchmark class can be instantiated."""
    print("\nTesting ModelBenchmark class...")

    try:
        # Import the benchmark class
        sys.path.append("dataset_benchmark")
        from benchmark_models import ModelBenchmark

        # Try to create an instance with the updated dataset
        benchmark = ModelBenchmark(dataset_dir="dataset_benchmark_hf_updated")
        print("✓ ModelBenchmark instantiated successfully")
        print(f"  - Dataset loaded: {len(benchmark.dataset)} examples")

        # Test filtering
        task13_examples = benchmark.filter_dataset("13")
        print(f"  - Task 13 examples: {len(task13_examples)}")

        # Test that we can access the first example
        if task13_examples:
            first_example = task13_examples[0]
            print(f"  - First example ID: {first_example['solution_id']}")
            print(f"  - Has true solution images: {bool(first_example.get('images_with_true_solution'))}")

        return True

    except Exception as e:
        print(f"✗ ModelBenchmark test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Simple Evaluation System Test ===\n")
    
    tests = [
        test_dataset_loading,
        test_imports,
        test_benchmark_class
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! The system is ready for evaluation.")
        return True
    else:
        print("✗ Some tests failed. Please fix the issues before running evaluation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
