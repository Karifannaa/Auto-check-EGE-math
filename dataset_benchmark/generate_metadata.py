import os
import json
import csv
import re
from collections import defaultdict

def normalize_path(path):
    """Normalize file path for consistent handling across platforms."""
    # Convert backslashes to forward slashes for consistency
    normalized = path.replace('\\', '/')
    # Ensure the path is absolute
    if not os.path.isabs(normalized):
        normalized = os.path.abspath(normalized)
    return normalized

# Define task type descriptions
TASK_TYPES = {
    "13": "Тригонометрические уравнения",
    "14": "Стереометрическая задача",
    "15": "Логарифмические неравенства",
    "16": "Планиметрическая задача",
    "17": "Финансовая математика",
    "18": "Задача с параметром",
    "19": "Задача по теории чисел"
}

def extract_info_from_filename(filename):
    """Extract metadata from filename."""
    # Example: 19.1.1_solve_part_1_estimate_4.png
    pattern = r'(\d+)\.(\d+)\.(\d+)_solve(?:_part_(\d+))?_estimate_(\d+)\.png'
    match = re.match(pattern, filename)
    
    if match:
        task_id, example_num, solution_num, part_num, score = match.groups()
        return {
            "task_id": task_id,
            "example_id": f"{task_id}.{example_num}",
            "solution_id": f"{task_id}.{example_num}.{solution_num}",
            "part": int(part_num) if part_num else 1,
            "score": int(score)
        }
    return None

def generate_metadata(dataset_dir="dataset_benchmark"):
    """Generate metadata from the dataset directory."""
    metadata = []
    solution_files = defaultdict(lambda: {"with_answer": [], "without_answer": []})
    
    # Walk through the directory structure
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if not file.endswith('.png'):
                continue
                
            file_info = extract_info_from_filename(file)
            if not file_info:
                continue
                
            full_path = normalize_path(os.path.join(root, file))
            solution_key = file_info["solution_id"]
            
            # Determine if this is with_answer or without_answer
            if "with_answer" in root:
                solution_files[solution_key]["with_answer"].append(full_path)
            else:
                solution_files[solution_key]["without_answer"].append(full_path)
    
    # Process collected files into metadata entries
    for solution_id, paths in solution_files.items():
        task_id = solution_id.split('.')[0]
        
        # Sort paths by part number
        for key in ["with_answer", "without_answer"]:
            paths[key].sort(key=lambda p: extract_info_from_filename(os.path.basename(p))["part"])
        
        # Get score from any file (should be the same for all parts)
        sample_file = paths["with_answer"][0] if paths["with_answer"] else paths["without_answer"][0]
        score = extract_info_from_filename(os.path.basename(sample_file))["score"]
        
        # Count parts
        parts_count = max(len(paths["with_answer"]), len(paths["without_answer"]))
        
        metadata.append({
            "task_id": task_id,
            "example_id": ".".join(solution_id.split(".")[:2]),
            "solution_id": solution_id,
            "file_paths": paths,
            "score": score,
            "parts_count": parts_count,
            "task_type": TASK_TYPES.get(task_id, "Unknown task type")
        })
    
    return metadata

def save_metadata_json(metadata, output_file="dataset_benchmark/metadata.json"):
    """Save metadata as JSON."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"JSON metadata saved to {output_file}")

def save_metadata_csv(metadata, output_file="dataset_benchmark/metadata.csv"):
    """Save metadata as CSV."""
    # Flatten the nested structure for CSV
    flattened = []
    for item in metadata:
        with_answer_files = "|".join(item["file_paths"]["with_answer"])
        without_answer_files = "|".join(item["file_paths"]["without_answer"])
        
        flattened.append({
            "task_id": item["task_id"],
            "example_id": item["example_id"],
            "solution_id": item["solution_id"],
            "with_answer_files": with_answer_files,
            "without_answer_files": without_answer_files,
            "score": item["score"],
            "parts_count": item["parts_count"],
            "task_type": item["task_type"]
        })
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = flattened[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)
    print(f"CSV metadata saved to {output_file}")

if __name__ == "__main__":
    metadata = generate_metadata()
    save_metadata_json(metadata)
    save_metadata_csv(metadata)
