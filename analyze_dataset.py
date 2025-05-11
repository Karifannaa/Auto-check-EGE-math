"""
Script to analyze the dataset structure and count examples.
"""

import os
import sys
from datasets import load_from_disk

def analyze_dataset(dataset_dir="dataset_benchmark_hf"):
    """Analyze the dataset structure and count examples."""
    print(f"Loading dataset from {dataset_dir}...")
    
    try:
        # Load the dataset
        dataset = load_from_disk(dataset_dir)
        
        # Count total examples
        total_examples = len(dataset)
        print(f"Total examples in dataset: {total_examples}")
        
        # Count examples by task
        task_counts = {}
        for example in dataset:
            task_id = example["task_id"]
            task_counts[task_id] = task_counts.get(task_id, 0) + 1
        
        # Print task counts
        print("\nExamples by task:")
        for task_id, count in sorted(task_counts.items()):
            print(f"Task {task_id}: {count} examples")
        
        # Check if examples have true solution images
        true_solution_count = 0
        for example in dataset:
            if "images_with_true_solution" in example and example["images_with_true_solution"]:
                true_solution_count += 1
        
        print(f"\nExamples with true solution images: {true_solution_count}")
        
        # Check task 13 examples with true solution
        task13_true_solution = 0
        for example in dataset:
            if example["task_id"] == "13" and "images_with_true_solution" in example and example["images_with_true_solution"]:
                task13_true_solution += 1
        
        print(f"Task 13 examples with true solution images: {task13_true_solution}")
        
        # Print score distribution
        score_counts = {}
        for example in dataset:
            score = example["score"]
            score_counts[score] = score_counts.get(score, 0) + 1
        
        print("\nScore distribution:")
        for score, count in sorted(score_counts.items()):
            print(f"Score {score}: {count} examples ({count/total_examples*100:.2f}%)")
        
    except Exception as e:
        print(f"Error analyzing dataset: {e}")

if __name__ == "__main__":
    # Use custom dataset directory if provided
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset_benchmark_hf"
    analyze_dataset(dataset_dir)
