import os
import json
from datasets import load_from_disk

def verify_dataset(dataset_dir="dataset_benchmark_hf"):
    """Verify the Hugging Face dataset structure and content."""
    # Check if the directory exists
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory {dataset_dir} does not exist.")
        return False
    
    # Check for with_answer and without_answer subdirectories
    with_answer_dir = os.path.join(dataset_dir, "with_answer")
    without_answer_dir = os.path.join(dataset_dir, "without_answer")
    
    if not os.path.exists(with_answer_dir):
        print(f"Error: with_answer directory {with_answer_dir} does not exist.")
    else:
        with_answer_files = [f for f in os.listdir(with_answer_dir) if f.endswith('.png')]
        print(f"Found {len(with_answer_files)} files in with_answer directory.")
    
    if not os.path.exists(without_answer_dir):
        print(f"Error: without_answer directory {without_answer_dir} does not exist.")
    else:
        without_answer_files = [f for f in os.listdir(without_answer_dir) if f.endswith('.png')]
        print(f"Found {len(without_answer_files)} files in without_answer directory.")
    
    # Try to load the dataset
    try:
        dataset = load_from_disk(dataset_dir)
        print(f"Successfully loaded dataset with {len(dataset)} examples.")
        
        # Check image paths in the dataset
        with_answer_count = 0
        without_answer_count = 0
        missing_with_answer = 0
        missing_without_answer = 0
        
        for example in dataset:
            # Check with_answer images
            for img_path in example['images_with_answer']:
                with_answer_count += 1
                if not os.path.exists(img_path):
                    missing_with_answer += 1
                    if missing_with_answer <= 5:  # Show only first 5 missing files
                        print(f"Missing with_answer image: {img_path}")
            
            # Check without_answer images
            for img_path in example['images_without_answer']:
                without_answer_count += 1
                if not os.path.exists(img_path):
                    missing_without_answer += 1
                    if missing_without_answer <= 5:  # Show only first 5 missing files
                        print(f"Missing without_answer image: {img_path}")
        
        print(f"Dataset references {with_answer_count} with_answer images (Missing: {missing_with_answer})")
        print(f"Dataset references {without_answer_count} without_answer images (Missing: {missing_without_answer})")
        
        return True
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

if __name__ == "__main__":
    verify_dataset()