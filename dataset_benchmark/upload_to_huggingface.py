import os
from huggingface_hub import HfApi
from datasets import load_from_disk

def upload_to_huggingface(dataset_dir="dataset_benchmark_hf", repo_name="russian-math-exam-benchmark"):
    """Upload the dataset to Hugging Face Hub."""
    # Load the prepared dataset
    dataset = load_from_disk(dataset_dir)
    
    # Initialize Hugging Face API
    api = HfApi()
    
    # Create a new dataset repository
    api.create_repo(
        repo_id=f"{os.environ.get('HF_USERNAME', 'your-username')}/{repo_name}",
        repo_type="dataset",
        private=False
    )
    
    # Push the dataset to Hugging Face Hub
    dataset.push_to_hub(
        f"{os.environ.get('HF_USERNAME', 'your-username')}/{repo_name}",
        token=os.environ.get("HF_TOKEN"),
        embed_external_files=True  # Ensure all image files are uploaded
    )
    
    print(f"Dataset uploaded to Hugging Face Hub: {os.environ.get('HF_USERNAME', 'your-username')}/{repo_name}")

if __name__ == "__main__":
    # Make sure to set environment variables HF_USERNAME and HF_TOKEN before running
    if not os.environ.get("HF_TOKEN"):
        print("Please set the HF_TOKEN environment variable with your Hugging Face API token")
    elif not os.environ.get("HF_USERNAME"):
        print("Please set the HF_USERNAME environment variable with your Hugging Face username")
    else:
        upload_to_huggingface()
