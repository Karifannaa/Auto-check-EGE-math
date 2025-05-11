"""
Script to update the dataset with true solution images.
"""

import os
import json
import shutil
from datasets import load_from_disk, Dataset
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("update_dataset.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("update_dataset")

def update_dataset_with_true_solutions(
    metadata_file="dataset_benchmark/metadata.json",
    dataset_dir="dataset_benchmark_hf",
    output_dir="dataset_benchmark_hf_updated"
):
    """Update the dataset with true solution images."""
    # Load metadata
    logger.info(f"Loading metadata from {metadata_file}")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Load dataset
    logger.info(f"Loading dataset from {dataset_dir}")
    dataset = load_from_disk(dataset_dir)
    logger.info(f"Loaded dataset with {len(dataset)} examples")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)

    # Copy existing image directories
    for subdir in ["with_answer", "without_answer"]:
        src_dir = os.path.join(dataset_dir, subdir)
        dst_dir = os.path.join(output_dir, subdir)
        if os.path.exists(src_dir):
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir, exist_ok=True)
            for file in os.listdir(src_dir):
                src_file = os.path.join(src_dir, file)
                dst_file = os.path.join(dst_dir, file)
                if not os.path.exists(dst_file):
                    shutil.copy(src_file, dst_file)

    # Create output directory for true solution images
    true_solution_dir = os.path.join(output_dir, "with_true_solution")
    os.makedirs(true_solution_dir, exist_ok=True)

    # Convert dataset to pandas DataFrame for easier manipulation
    df = pd.DataFrame(dataset)

    # Add empty column for true solution images if it doesn't exist
    if 'images_with_true_solution' not in df.columns:
        df['images_with_true_solution'] = [[] for _ in range(len(df))]

    # Track copied files for debugging
    copied_files = 0
    missing_files = 0
    updated_examples = 0

    # Create a mapping from solution_id to row index
    solution_id_to_index = {row['solution_id']: i for i, row in df.iterrows()}

    # Update dataset with true solution images
    for item in metadata:
        solution_id = item["solution_id"]

        # Skip if solution_id not in dataset
        if solution_id not in solution_id_to_index:
            logger.warning(f"Solution {solution_id} not found in dataset")
            continue

        # Get row index
        idx = solution_id_to_index[solution_id]

        # Skip if no true solution images
        if "with_true_solution" not in item["file_paths"] or not item["file_paths"]["with_true_solution"]:
            logger.warning(f"No true solution images for solution {solution_id}")
            continue

        # Process true solution images
        true_solution_images = []
        for img_path in item["file_paths"]["with_true_solution"]:
            # Ensure the path uses forward slashes for consistency
            img_path = img_path.replace('\\', '/')

            # Create a destination path in the with_true_solution subdirectory
            filename = os.path.basename(img_path)
            new_path = os.path.join(true_solution_dir, filename)

            # Ensure the source file exists before copying
            if os.path.exists(img_path):
                # Copy file if it doesn't exist in destination
                if not os.path.exists(new_path):
                    shutil.copy(img_path, new_path)
                # Use relative path in dataset
                rel_path = os.path.join(output_dir, "with_true_solution", filename)
                true_solution_images.append(rel_path)
                copied_files += 1
            else:
                logger.warning(f"Source file not found: {img_path}")
                missing_files += 1

        # Update dataset with true solution images
        if true_solution_images:
            df.at[idx, 'images_with_true_solution'] = true_solution_images
            updated_examples += 1

    # Convert back to Hugging Face Dataset
    updated_dataset = Dataset.from_pandas(df)

    # Save updated dataset
    logger.info(f"Saving updated dataset to {output_dir}")
    updated_dataset.save_to_disk(output_dir)

    # Update README.md to include images_with_true_solution
    readme_path = os.path.join(dataset_dir, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()

        # Update README to include images_with_true_solution
        if "- `images_with_true_solution`:" not in readme_content:
            readme_content = readme_content.replace(
                "- `images_without_answer`: List of image paths containing only student solution\n",
                "- `images_without_answer`: List of image paths containing only student solution\n- `images_with_true_solution`: List of image paths containing task with true solution\n"
            )

        # Save updated README
        with open(os.path.join(output_dir, "README.md"), 'w', encoding='utf-8') as f:
            f.write(readme_content)

    # Print summary
    logger.info(f"Dataset update complete")
    logger.info(f"Total examples in dataset: {len(updated_dataset)}")
    logger.info(f"Examples updated with true solution images: {updated_examples}")
    logger.info(f"Total true solution images copied: {copied_files}")
    logger.info(f"Missing true solution images: {missing_files}")

    return updated_dataset

if __name__ == "__main__":
    update_dataset_with_true_solutions()
