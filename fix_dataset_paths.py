#!/usr/bin/env python3
"""
<<<<<<< HEAD
Script to fix the image paths in the dataset to use the correct directory and path separators.
=======
Script to fix the image paths in the dataset to use the correct directory.
>>>>>>> bddc85f (Add Qwen 2.5 VL 32B model evaluation results)
"""

import os
from datasets import load_from_disk, Dataset

<<<<<<< HEAD
def fix_paths(dataset_dir="dataset_benchmark_hf_updated"):
    """Fix the image paths in the dataset."""
    print(f"Loading dataset from {dataset_dir}...")
    dataset = load_from_disk(dataset_dir)
    
    print(f"Loaded {len(dataset)} examples")
=======
def fix_dataset_paths():
    """Fix the image paths in the dataset."""
    
    print("Loading dataset...")
    dataset = load_from_disk("dataset_benchmark_hf_updated")
    
    print(f"Dataset loaded with {len(dataset)} examples")
>>>>>>> bddc85f (Add Qwen 2.5 VL 32B model evaluation results)
    
    # Fix paths for each example
    fixed_examples = []
    
    for i, example in enumerate(dataset):
        fixed_example = dict(example)
        
        # Fix with_answer paths
<<<<<<< HEAD
        if 'images_with_answer' in fixed_example:
            fixed_paths = []
            for path in fixed_example['images_with_answer']:
                # Replace old directory name and fix path separators
                fixed_path = path.replace('dataset_benchmark_hf\\', 'dataset_benchmark_hf_updated/')
                fixed_path = fixed_path.replace('\\', '/')
=======
        if example['images_with_answer']:
            fixed_paths = []
            for path in example['images_with_answer']:
                # Replace old directory with new directory
                fixed_path = path.replace('dataset_benchmark_hf\\', 'dataset_benchmark_hf_updated/')
                fixed_path = fixed_path.replace('dataset_benchmark_hf/', 'dataset_benchmark_hf_updated/')
                fixed_path = fixed_path.replace('\\', '/')  # Convert Windows paths to Unix
>>>>>>> bddc85f (Add Qwen 2.5 VL 32B model evaluation results)
                fixed_paths.append(fixed_path)
            fixed_example['images_with_answer'] = fixed_paths
        
        # Fix without_answer paths
<<<<<<< HEAD
        if 'images_without_answer' in fixed_example:
            fixed_paths = []
            for path in fixed_example['images_without_answer']:
                # Replace old directory name and fix path separators
                fixed_path = path.replace('dataset_benchmark_hf\\', 'dataset_benchmark_hf_updated/')
                fixed_path = fixed_path.replace('\\', '/')
=======
        if example['images_without_answer']:
            fixed_paths = []
            for path in example['images_without_answer']:
                # Replace old directory with new directory
                fixed_path = path.replace('dataset_benchmark_hf\\', 'dataset_benchmark_hf_updated/')
                fixed_path = fixed_path.replace('dataset_benchmark_hf/', 'dataset_benchmark_hf_updated/')
                fixed_path = fixed_path.replace('\\', '/')  # Convert Windows paths to Unix
>>>>>>> bddc85f (Add Qwen 2.5 VL 32B model evaluation results)
                fixed_paths.append(fixed_path)
            fixed_example['images_without_answer'] = fixed_paths
        
        # Fix with_true_solution paths (these might already be correct)
<<<<<<< HEAD
        if 'images_with_true_solution' in fixed_example:
            fixed_paths = []
            for path in fixed_example['images_with_true_solution']:
                # Fix path separators
                fixed_path = path.replace('\\', '/')
=======
        if example['images_with_true_solution']:
            fixed_paths = []
            for path in example['images_with_true_solution']:
                # Replace old directory with new directory
                fixed_path = path.replace('dataset_benchmark_hf\\', 'dataset_benchmark_hf_updated/')
                fixed_path = fixed_path.replace('dataset_benchmark_hf/', 'dataset_benchmark_hf_updated/')
                fixed_path = fixed_path.replace('\\', '/')  # Convert Windows paths to Unix
>>>>>>> bddc85f (Add Qwen 2.5 VL 32B model evaluation results)
                fixed_paths.append(fixed_path)
            fixed_example['images_with_true_solution'] = fixed_paths
        
        fixed_examples.append(fixed_example)
        
<<<<<<< HEAD
        if (i + 1) % 10 == 0:
            print(f"Fixed {i + 1}/{len(dataset)} examples...")
    
    # Create new dataset with fixed paths
    print("Creating new dataset with fixed paths...")
    fixed_dataset = Dataset.from_list(fixed_examples)
    
    # Save the fixed dataset
    print(f"Saving fixed dataset to {dataset_dir}...")
    fixed_dataset.save_to_disk(dataset_dir)
    
    print("Dataset paths fixed successfully!")
    
    # Verify the fix
    print("\nVerifying fix...")
    test_dataset = load_from_disk(dataset_dir)
    print("First example paths after fix:")
    print("With answer:", test_dataset[0]['images_with_answer'])
    print("Without answer:", test_dataset[0]['images_without_answer'])
    print("With true solution:", test_dataset[0]['images_with_true_solution'])
    
    # Check if files exist
    print("\nChecking if files exist:")
    for path in test_dataset[0]['images_with_answer']:
        exists = os.path.exists(path)
        print(f"  {path}: {'✓' if exists else '✗'}")
    
    for path in test_dataset[0]['images_without_answer']:
        exists = os.path.exists(path)
        print(f"  {path}: {'✓' if exists else '✗'}")
    
    for path in test_dataset[0]['images_with_true_solution']:
        exists = os.path.exists(path)
        print(f"  {path}: {'✓' if exists else '✗'}")

if __name__ == "__main__":
    fix_paths()
=======
        if (i + 1) % 20 == 0:
            print(f"Fixed {i + 1}/{len(dataset)} examples...")
    
    print("Creating new dataset with fixed paths...")
    
    # Create new dataset with fixed paths
    fixed_dataset = Dataset.from_list(fixed_examples)
    
    # Save the fixed dataset
    print("Saving fixed dataset...")
    fixed_dataset.save_to_disk("dataset_benchmark_hf_updated_fixed")
    
    print("Dataset saved to dataset_benchmark_hf_updated_fixed")
    
    # Verify a few examples
    print("\nVerifying fixed paths...")
    for i in range(min(3, len(fixed_dataset))):
        example = fixed_dataset[i]
        print(f"\nExample {i+1}:")
        print(f"  Solution ID: {example['solution_id']}")
        
        # Check if files exist
        for path_type, paths in [
            ("with_answer", example['images_with_answer']),
            ("without_answer", example['images_without_answer']),
            ("with_true_solution", example['images_with_true_solution'])
        ]:
            if paths:
                print(f"  {path_type}: {len(paths)} files")
                for path in paths[:2]:  # Check first 2 files
                    exists = os.path.exists(path)
                    print(f"    {path} - {'✓' if exists else '✗'}")
            else:
                print(f"  {path_type}: No files")
    
    return True

if __name__ == "__main__":
    print("=== Dataset Path Fixer ===")
    print("This script will fix the image paths in the dataset to use the correct directory.")
    
    try:
        success = fix_dataset_paths()
        if success:
            print("\n✓ Dataset paths fixed successfully!")
            print("You can now use 'dataset_benchmark_hf_updated_fixed' for evaluations.")
        else:
            print("\n✗ Failed to fix dataset paths.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
>>>>>>> bddc85f (Add Qwen 2.5 VL 32B model evaluation results)
