import json
import os
import re

def update_metadata():
    """Update metadata.json to include true solution images for all examples."""
    # Load metadata
    with open('dataset_benchmark/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Update each item
    for item in metadata:
        task_id = item['task_id']
        example_id = item['example_id']
        
        # Skip if already has with_true_solution
        if 'with_true_solution' in item['file_paths']:
            continue
        
        # Create path for true solution image
        true_solution_path = f"C:\\Users\\xpylb\\Documents\\augment-projects\\Auto check EGE math\\dataset_benchmark\\{task_id}\\{example_id}\\{example_id}_task+true_solve.png"
        
        # Check if task 16 uses a different naming convention
        if task_id == "16":
            alt_path = f"C:\\Users\\xpylb\\Documents\\augment-projects\\Auto check EGE math\\dataset_benchmark\\{task_id}\\{example_id}\\{example_id}_solve_true.png"
            if os.path.exists(alt_path):
                true_solution_path = alt_path
        
        # Add the path to the item
        item['file_paths']['with_true_solution'] = [true_solution_path]
    
    # Save updated metadata
    with open('dataset_benchmark/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print("Metadata updated successfully!")

if __name__ == "__main__":
    update_metadata()
