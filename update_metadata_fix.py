import json
import os
import glob

def update_metadata():
    """Update metadata.json to use the correct file naming format for true solution images."""
    # Load metadata
    with open('dataset_benchmark/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Track statistics
    stats = {
        "single_image": 0,
        "multi_part": 0,
        "old_style": 0,
        "not_found": 0
    }

    # Update each item
    for item in metadata:
        task_id = item['task_id']
        example_id = item['example_id']

        # Base directory for this example
        base_dir = f"dataset_benchmark/{task_id}/{example_id}"

        # Check for single true solution image
        single_image_path = f"{base_dir}/{example_id}_solve_true.png"
        single_image_full_path = f"C:\\Users\\xpylb\\Documents\\augment-projects\\Auto check EGE math\\{single_image_path}"

        # Check for multi-part true solution images
        multi_image_pattern = f"{base_dir}/{example_id}_solve_true_part_*.png"
        multi_image_paths = sorted(glob.glob(multi_image_pattern))
        multi_image_full_paths = [f"C:\\Users\\xpylb\\Documents\\augment-projects\\Auto check EGE math\\{path}" for path in multi_image_paths]

        # Check for old-style image
        old_path = f"{base_dir}/{example_id}_task+true_solve.png"
        old_full_path = f"C:\\Users\\xpylb\\Documents\\augment-projects\\Auto check EGE math\\{old_path}"

        # Determine which path(s) to use
        if os.path.exists(single_image_full_path):
            # Use single image path
            item['file_paths']['with_true_solution'] = [single_image_full_path]
            stats["single_image"] += 1
            print(f"Updated {example_id} with single true solution image")
        elif multi_image_paths:
            # Use multi-part image paths
            item['file_paths']['with_true_solution'] = multi_image_full_paths
            stats["multi_part"] += 1
            print(f"Updated {example_id} with {len(multi_image_paths)} multi-part true solution images")
        elif os.path.exists(old_full_path):
            # Fallback to old naming convention if neither exists
            item['file_paths']['with_true_solution'] = [old_full_path]
            stats["old_style"] += 1
            print(f"Updated {example_id} with old-style true solution image")
        else:
            # No true solution image found
            stats["not_found"] += 1
            print(f"WARNING: No true solution image found for {example_id}")

    # Save updated metadata
    with open('dataset_benchmark/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print("\nMetadata update statistics:")
    print(f"Single image format: {stats['single_image']}")
    print(f"Multi-part image format: {stats['multi_part']}")
    print(f"Old-style image format: {stats['old_style']}")
    print(f"No true solution found: {stats['not_found']}")
    print("\nMetadata updated successfully!")

if __name__ == "__main__":
    update_metadata()
