import os
import json
import shutil
from PIL import Image
import pandas as pd
from datasets import Dataset, Features, Value, ClassLabel, Image as HFImage

def prepare_huggingface_dataset(metadata_file="dataset_benchmark/metadata.json", output_dir="dataset_benchmark_hf"):
    """Prepare the dataset for Hugging Face."""
    # Create output directory and subdirectories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "with_answer"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "without_answer"), exist_ok=True)

    # Load metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Prepare data for Hugging Face Dataset
    dataset_dict = {
        "solution_id": [],
        "task_id": [],
        "example_id": [],
        "task_type": [],
        "score": [],
        "parts_count": [],
        "images_with_answer": [],
        "images_without_answer": [],
        "images_with_true_solution": []
    }

    # Track copied files for debugging
    copied_files = {"with_answer": 0, "without_answer": 0, "with_true_solution": 0}
    missing_files = {"with_answer": 0, "without_answer": 0, "with_true_solution": 0}

    # Create output directory for true solution images
    os.makedirs(os.path.join(output_dir, "with_true_solution"), exist_ok=True)

    for item in metadata:
        # Copy and process images
        with_answer_images = []
        without_answer_images = []
        with_true_solution_images = []

        # Process with_answer images
        for img_path in item["file_paths"]["with_answer"]:
            # Ensure the path uses forward slashes for consistency
            img_path = img_path.replace('\\', '/')

            # Create a destination path in the with_answer subdirectory
            filename = os.path.basename(img_path)
            new_path = os.path.join(output_dir, "with_answer", filename)

            # Ensure the source file exists before copying
            if os.path.exists(img_path):
                shutil.copy(img_path, new_path)
                with_answer_images.append(new_path)
                copied_files["with_answer"] += 1
            else:
                print(f"Warning: Source file not found: {img_path}")
                missing_files["with_answer"] += 1

        # Process without_answer images
        for img_path in item["file_paths"]["without_answer"]:
            # Ensure the path uses forward slashes for consistency
            img_path = img_path.replace('\\', '/')

            # Create a destination path in the without_answer subdirectory
            filename = os.path.basename(img_path)
            new_path = os.path.join(output_dir, "without_answer", filename)

            # Ensure the source file exists before copying
            if os.path.exists(img_path):
                shutil.copy(img_path, new_path)
                without_answer_images.append(new_path)
                copied_files["without_answer"] += 1
            else:
                print(f"Warning: Source file not found: {img_path}")
                missing_files["without_answer"] += 1

        # Process with_true_solution images if available
        if "with_true_solution" in item["file_paths"]:
            for img_path in item["file_paths"]["with_true_solution"]:
                # Ensure the path uses forward slashes for consistency
                img_path = img_path.replace('\\', '/')

                # Create a destination path in the with_true_solution subdirectory
                filename = os.path.basename(img_path)
                new_path = os.path.join(output_dir, "with_true_solution", filename)

                # Ensure the source file exists before copying
                if os.path.exists(img_path):
                    shutil.copy(img_path, new_path)
                    with_true_solution_images.append(new_path)
                    copied_files["with_true_solution"] += 1
                else:
                    print(f"Warning: Source file not found: {img_path}")
                    missing_files["with_true_solution"] += 1

        # Only add to dataset if we have at least one image
        if with_answer_images or without_answer_images or with_true_solution_images:
            # Add to dataset dictionary
            dataset_dict["solution_id"].append(item["solution_id"])
            dataset_dict["task_id"].append(item["task_id"])
            dataset_dict["example_id"].append(item["example_id"])
            dataset_dict["task_type"].append(item["task_type"])
            dataset_dict["score"].append(item["score"])
            dataset_dict["parts_count"].append(item["parts_count"])
            dataset_dict["images_with_answer"].append(with_answer_images)
            dataset_dict["images_without_answer"].append(without_answer_images)
            dataset_dict["images_with_true_solution"].append(with_true_solution_images)
        else:
            print(f"Warning: No images found for solution {item['solution_id']}")

    # Create Hugging Face Dataset
    if dataset_dict["solution_id"]:  # Check if we have any data
        df = pd.DataFrame(dataset_dict)
        dataset = Dataset.from_pandas(df)

        # Save dataset
        dataset.save_to_disk(output_dir)

        # Create dataset card
        create_dataset_card(metadata, output_dir)

        print(f"Dataset prepared for Hugging Face in {output_dir}")
        print(f"Total solutions processed: {len(dataset_dict['solution_id'])}")
        print(f"Total with_answer images copied: {copied_files['with_answer']} (Missing: {missing_files['with_answer']})")
        print(f"Total without_answer images copied: {copied_files['without_answer']} (Missing: {missing_files['without_answer']})")
        print(f"Total with_true_solution images copied: {copied_files['with_true_solution']} (Missing: {missing_files['with_true_solution']})")
        print(f"Total with_answer images in dataset: {sum(len(imgs) for imgs in dataset_dict['images_with_answer'])}")
        print(f"Total without_answer images in dataset: {sum(len(imgs) for imgs in dataset_dict['images_without_answer'])}")
        print(f"Total with_true_solution images in dataset: {sum(len(imgs) for imgs in dataset_dict['images_with_true_solution'])}")
        return dataset
    else:
        print("Error: No data was processed. Check file paths in metadata.")
        return None

def create_dataset_card(metadata, output_dir):
    """Create a dataset card (README.md) for Hugging Face."""
    task_counts = {}
    score_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for item in metadata:
        task_id = item["task_id"]
        score = item["score"]

        task_counts[task_id] = task_counts.get(task_id, 0) + 1
        score_distribution[score] += 1

    with open(os.path.join(output_dir, "README.md"), 'w', encoding='utf-8') as f:
        f.write("# Russian Math Exam Solutions Benchmark Dataset\n\n")
        f.write("## Dataset Description\n\n")
        f.write("This dataset contains student solutions to Russian Unified State Exam (EGE) mathematics problems, with reference scores for benchmarking automated evaluation systems.\n\n")

        f.write("### Task Types\n\n")
        f.write("| Task ID | Description | Count |\n")
        f.write("|---------|-------------|-------|\n")
        for task_id, count in sorted(task_counts.items()):
            task_type = next((item["task_type"] for item in metadata if item["task_id"] == task_id), "Unknown")
            f.write(f"| {task_id} | {task_type} | {count} |\n")

        f.write("\n### Score Distribution\n\n")
        f.write("| Score | Count | Percentage |\n")
        f.write("|-------|-------|------------|\n")
        total = sum(score_distribution.values())
        for score, count in sorted(score_distribution.items()):
            percentage = (count / total) * 100 if total > 0 else 0
            f.write(f"| {score} | {count} | {percentage:.2f}% |\n")

        f.write("\n## Dataset Structure\n\n")
        f.write("Each example contains:\n\n")
        f.write("- `solution_id`: Unique identifier for the solution\n")
        f.write("- `task_id`: Task type ID (13-19)\n")
        f.write("- `example_id`: Specific example identifier\n")
        f.write("- `task_type`: Description of the task type\n")
        f.write("- `score`: Reference score (0-4)\n")
        f.write("- `parts_count`: Number of parts in the solution\n")
        f.write("- `images_with_answer`: List of image paths containing student solution with correct answer\n")
        f.write("- `images_without_answer`: List of image paths containing only student solution\n")
        f.write("- `images_with_true_solution`: List of image paths containing task with true solution\n\n")

        f.write("## Usage\n\n")
        f.write("```python\n")
        f.write("from datasets import load_dataset\n\n")
        f.write("# Load the dataset\n")
        f.write("dataset = load_dataset('username/russian-math-exam-benchmark')\n\n")
        f.write("# Access an example\n")
        f.write("example = dataset['train'][0]\n")
        f.write("print(example['solution_id'], example['score'])\n")
        f.write("```\n\n")

        f.write("## License\n\n")
        f.write("This dataset is provided for research purposes only.\n")

if __name__ == "__main__":
    prepare_huggingface_dataset()

