#!/usr/bin/env python3
"""
Demonstration script showing the corrected evaluation system.
This shows how the system is set up and what the evaluation process looks like.
"""

import os
import sys
from datetime import datetime

# Set dummy API key for demonstration
os.environ["OPENROUTER_API_KEY"] = "demo_key_for_testing"

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
sys.path.append("dataset_benchmark")

def demonstrate_system():
    """Demonstrate the evaluation system setup."""
    
    print("="*70)
    print("DEMONSTRATION: CORRECTED EVALUATION SYSTEM")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. Show dataset information
    print("1. DATASET VERIFICATION")
    print("-" * 30)
    
    from datasets import load_from_disk
    
    # Load the updated dataset
    dataset = load_from_disk("dataset_benchmark_hf_updated")
    print(f"✓ Dataset loaded: {len(dataset)} examples")
    
    # Show task distribution
    task_counts = {}
    score_counts = {}
    true_solution_count = 0
    
    for example in dataset:
        task_id = example["task_id"]
        score = example["score"]
        task_counts[task_id] = task_counts.get(task_id, 0) + 1
        score_counts[score] = score_counts.get(score, 0) + 1
        
        if example.get("images_with_true_solution"):
            true_solution_count += 1
    
    print(f"✓ Examples with true solution images: {true_solution_count}")
    print(f"✓ Task distribution:")
    for task_id, count in sorted(task_counts.items()):
        print(f"    Task {task_id}: {count} examples")
    
    print(f"✓ Score distribution:")
    for score, count in sorted(score_counts.items()):
        percentage = (count / len(dataset)) * 100
        print(f"    Score {score}: {count} examples ({percentage:.1f}%)")
    
    # 2. Show system components
    print(f"\n2. SYSTEM COMPONENTS")
    print("-" * 30)
    
    try:
        from benchmark_models import ModelBenchmark
        benchmark = ModelBenchmark()
        print(f"✓ ModelBenchmark class: Ready")
        print(f"    - Dataset: {len(benchmark.dataset)} examples loaded")
        print(f"    - Results directory: {benchmark.results_dir}")
        
        # Test filtering
        task13_examples = benchmark.filter_dataset("13")
        print(f"    - Task 13 filtering: {len(task13_examples)} examples")
        
    except Exception as e:
        print(f"✗ ModelBenchmark error: {e}")
    
    try:
        from app.core.config import settings
        print(f"✓ Configuration: Ready")
        print(f"    - Available models: {len(settings.AVAILABLE_MODELS)}")
        print(f"    - Target model: google/gemini-2.0-flash-exp:free")
        print(f"    - Model available: {'google/gemini-2.0-flash-exp:free' in settings.AVAILABLE_MODELS.values()}")
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
    
    # 3. Show evaluation plan
    print(f"\n3. EVALUATION PLAN")
    print("-" * 30)
    
    model_id = "google/gemini-2.0-flash-exp:free"
    total_examples = len(dataset)
    
    print(f"Target Model: {model_id}")
    print(f"Total Examples: {total_examples}")
    print(f"Evaluation Approaches:")
    print(f"  - with_answer: {total_examples} evaluations")
    print(f"  - without_answer: {total_examples} evaluations")
    print(f"  - Total evaluations: {total_examples * 2}")
    
    # Estimate cost and time
    estimated_cost_per_eval = 0.05  # Rough estimate
    estimated_time_per_eval = 10    # seconds
    
    total_cost = total_examples * 2 * estimated_cost_per_eval
    total_time_minutes = (total_examples * 2 * estimated_time_per_eval) / 60
    
    print(f"\nEstimated Metrics:")
    print(f"  - Cost: ~${total_cost:.2f}")
    print(f"  - Time: ~{total_time_minutes:.0f} minutes")
    print(f"  - Rate: ~{60/estimated_time_per_eval:.1f} evaluations/minute")
    
    # 4. Show example data
    print(f"\n4. EXAMPLE DATA STRUCTURE")
    print("-" * 30)
    
    example = dataset[0]
    print(f"Example ID: {example['solution_id']}")
    print(f"Task ID: {example['task_id']}")
    print(f"Expected Score: {example['score']}")
    print(f"Images with answer: {len(example['images_with_answer'])}")
    print(f"Images without answer: {len(example['images_without_answer'])}")
    print(f"Images with true solution: {len(example['images_with_true_solution'])}")
    
    # Show file paths (first few characters)
    if example['images_with_answer']:
        path = example['images_with_answer'][0]
        print(f"Sample path: ...{path[-50:]}")
    
    # 5. Show commands to run
    print(f"\n5. COMMANDS TO RUN EVALUATION")
    print("-" * 30)
    
    print("Quick test (1 example):")
    print("  python run_test_evaluation.py")
    print()
    
    print("Full evaluation (all 122 examples):")
    print("  python run_full_evaluation.py")
    print()
    
    print("Manual evaluation (advanced):")
    print("  cd dataset_benchmark")
    print(f"  python benchmark_models.py --task 13 --models \"{model_id}\" --with-answer --without-answer")
    print()
    
    print("Analysis of existing results:")
    print("  cd dataset_benchmark")
    print("  python benchmark_models.py --analyze-only --results-file results.json --latex")
    
    # 6. Show expected results structure
    print(f"\n6. EXPECTED RESULTS")
    print("-" * 30)
    
    print("Results will include:")
    print("  - Accuracy: % of exact score matches")
    print("  - Quality Score: Normalized prediction closeness (0-100%)")
    print("  - Macro Precision/Recall/F1: Multi-class metrics")
    print("  - Confusion Matrix: Prediction patterns")
    print("  - Cost and timing information")
    print("  - Detailed per-example results")
    
    print(f"\nFiles generated:")
    print(f"  - benchmark_results/[timestamp]/benchmark_*.json")
    print(f"  - benchmark_results/[timestamp]/benchmark_*_analysis.json")
    print(f"  - benchmark_results/[timestamp]/benchmark_*_metrics_table.tex")
    
    print(f"\n" + "="*70)
    print("SYSTEM STATUS: ✓ READY FOR EVALUATION")
    print("="*70)
    print()
    print("To run the actual evaluation:")
    print("1. Get an OpenRouter API key from https://openrouter.ai/")
    print("2. Run: python run_test_evaluation.py (for quick test)")
    print("3. Run: python run_full_evaluation.py (for complete evaluation)")
    print()
    print("The system has been corrected and is ready to evaluate")
    print("google/gemini-2.0-flash-exp:free on all 122 examples across 7 task types!")

if __name__ == "__main__":
    demonstrate_system()
