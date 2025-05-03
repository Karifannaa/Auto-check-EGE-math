import os
import json

def debug_metadata_paths(metadata_file="dataset_benchmark/metadata.json"):
    """Debug file paths in metadata to ensure they exist."""
    # Load metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    missing_files = {
        "with_answer": [],
        "without_answer": []
    }
    
    for item in metadata:
        # Check with_answer paths
        for img_path in item["file_paths"]["with_answer"]:
            # Try different path formats
            paths_to_try = [
                img_path,
                img_path.replace('\\', '/'),
                os.path.normpath(img_path)
            ]
            
            exists = False
            for path in paths_to_try:
                if os.path.exists(path):
                    exists = True
                    break
            
            if not exists:
                missing_files["with_answer"].append(img_path)
        
        # Check without_answer paths
        for img_path in item["file_paths"]["without_answer"]:
            # Try different path formats
            paths_to_try = [
                img_path,
                img_path.replace('\\', '/'),
                os.path.normpath(img_path)
            ]
            
            exists = False
            for path in paths_to_try:
                if os.path.exists(path):
                    exists = True
                    break
            
            if not exists:
                missing_files["without_answer"].append(img_path)
    
    # Print results
    print(f"Total solutions in metadata: {len(metadata)}")
    print(f"Missing with_answer files: {len(missing_files['with_answer'])}")
    print(f"Missing without_answer files: {len(missing_files['without_answer'])}")
    
    if missing_files["with_answer"] or missing_files["without_answer"]:
        print("\nSample of missing files:")
        for category, files in missing_files.items():
            if files:
                print(f"\n{category.upper()}:")
                for file in files[:5]:  # Show first 5 examples
                    print(f"  - {file}")
    
    return missing_files

if __name__ == "__main__":
    debug_metadata_paths()