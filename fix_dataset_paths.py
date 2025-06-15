#!/usr/bin/env python3
"""
Script to fix the image paths in the dataset to use the correct directory and path separators.
"""

import os
from datasets import load_from_disk, Dataset

def fix_paths(dataset_dir="dataset_benchmark_hf_updated"):
    """Fix the image paths in the dataset."""
    print(f"Loading dataset from {dataset_dir}...")
    dataset = load_from_disk(dataset_dir)
    
    print(f"Loaded {len(dataset)} examples")
    
    # Fix paths for each example
    fixed_examples = []
    
    for i, example in enumerate(dataset):
        fixed_example = dict(example)
        
        # Fix with_answer paths
        if 'images_with_answer' in fixed_example:
            fixed_paths = []
            for path in fixed_example['images_with_answer']:
                # Replace old directory name and fix path separators
                fixed_path = path.replace('dataset_benchmark_hf\\', 'dataset_benchmark_hf_updated/')
                fixed_path = fixed_path.replace('\\', '/')
                fixed_paths.append(fixed_path)
            fixed_example['images_with_answer'] = fixed_paths
        
        # Fix without_answer paths
        if 'images_without_answer' in fixed_example:
            fixed_paths = []
            for path in fixed_example['images_without_answer']:
                # Replace old directory name and fix path separators
                fixed_path = path.replace('dataset_benchmark_hf\\', 'dataset_benchmark_hf_updated/')
                fixed_path = fixed_path.replace('\\', '/')
                fixed_paths.append(fixed_path)
            fixed_example['images_without_answer'] = fixed_paths
        
        # Fix with_true_solution paths (these might already be correct)
        if 'images_with_true_solution' in fixed_example:
            fixed_paths = []
            for path in fixed_example['images_with_true_solution']:
                # Fix path separators
                fixed_path = path.replace('\\', '/')
                fixed_paths.append(fixed_path)
            fixed_example['images_with_true_solution'] = fixed_paths
        
        fixed_examples.append(fixed_example)
        
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
