import glob
import os

# Test for task 18.4
base_dir = "dataset_benchmark/18/18.4"
example_id = "18.4"

# Check for single true solution image
single_image_path = f"{base_dir}/{example_id}_solve_true.png"
print(f"Single image path: {single_image_path}")
print(f"Exists: {os.path.exists(single_image_path)}")

# Check for multi-part true solution images
multi_image_pattern = f"{base_dir}/{example_id}_solve_true_part_*.png"
print(f"Multi-image pattern: {multi_image_pattern}")
multi_image_paths = sorted(glob.glob(multi_image_pattern))
print(f"Found multi-part images: {multi_image_paths}")

# Check for old-style image
old_path = f"{base_dir}/{example_id}_task+true_solve.png"
print(f"Old-style path: {old_path}")
print(f"Exists: {os.path.exists(old_path)}")

# List all files in the directory
print("\nAll files in directory:")
all_files = os.listdir(base_dir)
for file in all_files:
    print(f"  {file}")
