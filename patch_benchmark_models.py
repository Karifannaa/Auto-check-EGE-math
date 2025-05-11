"""
Script to patch the benchmark_models.py file to handle true solution images from metadata.json.
"""

import os
import json
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("patch_benchmark.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("patch_benchmark")

def patch_benchmark_models(
    benchmark_file="dataset_benchmark/benchmark_models.py",
    metadata_file="dataset_benchmark/metadata.json"
):
    """Patch the benchmark_models.py file to handle true solution images from metadata.json."""
    # Load metadata
    logger.info(f"Loading metadata from {metadata_file}")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Load benchmark_models.py
    logger.info(f"Loading benchmark file from {benchmark_file}")
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        benchmark_code = f.read()

    # Find the evaluate_solution method
    evaluate_solution_pattern = r'async def evaluate_solution\([^)]*\):[^#]*?# Determine which image paths to use based on the options(.*?)else:[^#]*?# Use regular with_answer or without_answer images'
    evaluate_solution_match = re.search(evaluate_solution_pattern, benchmark_code, re.DOTALL)

    if not evaluate_solution_match:
        logger.error("Could not find the evaluate_solution method in benchmark_models.py")
        return False

    # Extract the current code for handling true solution images
    current_code = evaluate_solution_match.group(1)

    # Create a mapping from solution_id to true solution images
    solution_to_true_solution = {}
    for item in metadata:
        solution_id = item["solution_id"]
        if "with_true_solution" in item["file_paths"] and item["file_paths"]["with_true_solution"]:
            solution_to_true_solution[solution_id] = item["file_paths"]["with_true_solution"]

    # Create new code for handling true solution images
    new_code = """
        if use_true_solution:
            # Get solution_id
            solution_id = example["solution_id"]
            
            # Load true solution images from metadata.json
            true_solution_images = []
            
            # Mapping of solution_id to true solution images
            solution_to_true_solution = {
"""

    # Add the mapping to the code
    for solution_id, image_paths in solution_to_true_solution.items():
        new_code += f'                "{solution_id}": {json.dumps(image_paths)},\n'

    new_code += """            }
            
            # Get true solution images for this solution
            if solution_id in solution_to_true_solution:
                image_paths = solution_to_true_solution[solution_id]
                # Set prompt variant to with_solution for tasks 13-19
                if task_type in ["task_13", "task_14", "task_15", "task_16", "task_17", "task_18", "task_19"]:
                    prompt_variant = "with_solution"
            else:
                logger.warning(f"No true solution images for solution {solution_id}, falling back to without_answer")
                image_paths = example["images_without_answer"]
                use_true_solution = False  # Reset flag since we're not using true solution images
"""

    # Replace the current code with the new code
    patched_code = benchmark_code.replace(current_code, new_code)

    # Save the patched file
    backup_file = benchmark_file + ".bak"
    logger.info(f"Creating backup of benchmark file at {backup_file}")
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(benchmark_code)

    logger.info(f"Saving patched benchmark file to {benchmark_file}")
    with open(benchmark_file, 'w', encoding='utf-8') as f:
        f.write(patched_code)

    logger.info("Benchmark file patched successfully")
    return True

if __name__ == "__main__":
    patch_benchmark_models()
