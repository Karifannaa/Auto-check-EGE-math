#!/usr/bin/env python3
"""
Script to run a full evaluation for all tasks (13-19) with the specified model.
This will evaluate google/gemini-2.0-flash-exp:free on all 122 examples.
"""

import os
import sys
import asyncio
import getpass
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
sys.path.append("dataset_benchmark")

async def run_full_evaluation():
    """Run a full evaluation for all tasks."""
    
    # Get API key from user
    api_key = getpass.getpass("Enter your OpenRouter API key: ")
    if not api_key or api_key.strip() == "":
        print("No API key provided. Exiting.")
        return False
    
    # Set the API key
    os.environ["OPENROUTER_API_KEY"] = api_key.strip()
    
    try:
        # Import after setting the API key
        from benchmark_models import ModelBenchmark
        
        print("=== Full Evaluation for All Tasks ===")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Create benchmark instance with updated dataset
        print("Initializing benchmark...")
        benchmark = ModelBenchmark(
            dataset_dir="dataset_benchmark_hf_updated",
            max_retries=10,
            initial_delay=15,
            max_delay=120
        )
        
        print(f"Dataset loaded: {len(benchmark.dataset)} examples")
        
        # Model to evaluate
        model_id = "google/gemini-2.0-flash-exp:free"
        print(f"Evaluating model: {model_id}")
        print("Running evaluation on ALL examples from ALL tasks (13-19)...")
        print("This will take a significant amount of time and may cost money.")
        
        # Confirm with user
        confirm = input("\nDo you want to proceed? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Evaluation cancelled.")
            return False
        
        print("\nStarting evaluation...")
        start_time = datetime.now()
        
        # Run benchmark for all tasks
        results, results_file = await benchmark.run_benchmark(
            task_id=None,  # All tasks
            model_ids=[model_id],
            with_answer=True,
            without_answer=True,  # Evaluate both approaches
            with_true_solution=False,  # Skip true_solution for now
            max_examples=None,  # All examples
            prompt_variant="detailed",
            include_examples=False
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\nEvaluation completed!")
        print(f"Duration: {duration}")
        print(f"Results saved to: {results_file}")
        
        # Analyze results
        print("\nAnalyzing results...")
        analysis = benchmark.analyze_results(results)
        analysis_file = benchmark.save_analysis(analysis, results_file)
        
        # Print comprehensive summary
        print("\n" + "="*60)
        print("FULL EVALUATION RESULTS SUMMARY")
        print("="*60)
        print(f"Model: {model_id}")
        print(f"Total examples: {analysis['total_examples']}")
        print(f"Total evaluations: {analysis['total_evaluations']}")
        print(f"Total cost: ${analysis['summary']['total_cost']:.4f}")
        print(f"Average evaluation time: {analysis['summary']['avg_evaluation_time']:.2f}s")
        print(f"Total duration: {duration}")
        
        if analysis['summary'].get('avg_quality_score') is not None:
            print(f"Average quality score: {analysis['summary']['avg_quality_score'] * 100:.2f}%")
        
        # Print detailed results by approach
        for model_id_key, model_data in analysis["models"].items():
            print(f"\nModel: {model_id_key}")
            for answer_type, metrics in model_data.items():
                print(f"\n  === {answer_type.upper()} ===")
                print(f"    Accuracy: {metrics['accuracy']:.2f}%")
                
                if metrics.get('quality_score') is not None:
                    print(f"    Quality score: {metrics['quality_score']:.2f}%")
                if metrics.get('avg_score_distance') is not None:
                    print(f"    Avg. score distance: {metrics['avg_score_distance']:.2f}")
                
                # Print precision, recall, F1
                if 'macro_precision' in metrics:
                    print(f"    Macro precision: {metrics['macro_precision']:.2f}%")
                    print(f"    Macro recall: {metrics['macro_recall']:.2f}%")
                    print(f"    Macro F1: {metrics['macro_f1']:.2f}%")
                
                print(f"    Evaluations: {metrics['evaluations']}")
                print(f"    Avg. evaluation time: {metrics['avg_evaluation_time']:.2f}s")
                print(f"    Total cost: ${metrics['total_cost']:.4f}")
                
                # Print confusion matrix if available and not too large
                if 'confusion_matrix' in metrics and metrics['confusion_matrix'] and len(metrics['confusion_matrix']) <= 5:
                    print(f"    Confusion matrix:")
                    scores = sorted(metrics['confusion_matrix'].keys())
                    header = "      True\\Pred |"
                    for score in scores:
                        header += f" {score} |"
                    print(header)
                    for true_score in scores:
                        row = f"      {true_score}        |"
                        for pred_score in scores:
                            count = metrics['confusion_matrix'][true_score].get(pred_score, 0)
                            row += f" {count} |"
                        print(row)
        
        print(f"\nDetailed results saved to: {results_file}")
        print(f"Analysis saved to: {analysis_file}")
        
        # Count examples by task
        task_counts = {}
        for result in results:
            task_id = result['task_id']
            task_counts[task_id] = task_counts.get(task_id, 0) + 1
        
        print(f"\nEvaluations by task:")
        for task_id, count in sorted(task_counts.items()):
            print(f"  Task {task_id}: {count} evaluations")
        
        return True
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'benchmark' in locals():
            await benchmark.close()

def main():
    """Main function."""
    print("=== Full Evaluation Script ===")
    print("This script will run a complete evaluation for ALL tasks (13-19) with the model:")
    print("google/gemini-2.0-flash-exp:free")
    print()
    print("This will:")
    print("- Evaluate 122 examples across 7 task types")
    print("- Test both 'with_answer' and 'without_answer' approaches")
    print("- Take significant time (potentially hours)")
    print("- Cost money (estimated $5-20 depending on the model)")
    print()
    
    # Check if we have the required files
    required_files = [
        "dataset_benchmark_hf_updated",
        "dataset_benchmark/benchmark_models.py",
        "backend/app"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Error: Missing required files/directories:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    # Run the evaluation
    try:
        success = asyncio.run(run_full_evaluation())
        if success:
            print("\n" + "="*60)
            print("✓ FULL EVALUATION COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("Check the results files for detailed analysis.")
        else:
            print("\n✗ Full evaluation failed.")
        return success
    except KeyboardInterrupt:
        print("\nEvaluation cancelled by user.")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
