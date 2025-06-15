#!/usr/bin/env python3
"""
Script to run evaluation for Qwen 2.5 VL 32B model with "With True Solution" mode.
This completes the full 3-mode evaluation.
"""

import os
import sys
import asyncio
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
sys.path.append("dataset_benchmark")

async def run_qwen_with_true_solution():
    """Run evaluation for Qwen 32B model with true solution mode."""
    
    # Set the API key directly
    api_key = "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"
    os.environ["OPENROUTER_API_KEY"] = api_key
    
    try:
        # Import after setting the API key
        from benchmark_models import ModelBenchmark
        
        print("=== Qwen 2.5 VL 32B - With True Solution Mode ===")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Create benchmark instance with fixed dataset
        print("Initializing benchmark...")
        benchmark = ModelBenchmark(
            dataset_dir="dataset_benchmark_hf_updated_fixed",
            api_key=api_key,  # Pass API key directly
            max_retries=10,
            initial_delay=15,
            max_delay=120
        )
        
        print(f"Dataset loaded: {len(benchmark.dataset)} examples")
        
        # Model to evaluate
        model_id = "qwen-2.5-vl-32b"  # This maps to qwen/qwen2.5-vl-32b-instruct
        print(f"Evaluating model: {model_id}")
        print("Running evaluation with 'With True Solution' mode...")
        print("This will evaluate 122 examples with images containing the true solution")
        print()
        
        print("Starting evaluation automatically...")
        start_time = datetime.now()
        
        # Run benchmark with true solution mode only
        results, results_file = await benchmark.run_benchmark(
            task_id=None,  # All tasks
            model_ids=[model_id],
            with_answer=False,
            without_answer=False,
            with_true_solution=True,  # Only this mode
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
        print("QWEN 2.5 VL 32B - WITH TRUE SOLUTION RESULTS")
        print("="*60)
        print(f"Model: {model_id}")
        print(f"Total examples: {analysis['total_examples']}")
        print(f"Total evaluations: {analysis['total_evaluations']}")
        print(f"Total cost: ${analysis['summary']['total_cost']:.4f}")
        print(f"Average evaluation time: {analysis['summary']['avg_evaluation_time']:.2f}s")
        print(f"Total duration: {duration}")
        
        if analysis['summary'].get('avg_quality_score') is not None:
            print(f"Average quality score: {analysis['summary']['avg_quality_score'] * 100:.2f}%")
        
        # Print detailed results
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
        
        # Copy results to model directory
        await copy_results_to_model_dir(model_id, results_file, analysis_file)
        
        return True
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'benchmark' in locals():
            await benchmark.close()

async def copy_results_to_model_dir(model_id, results_file, analysis_file):
    """Copy the new results to the existing model directory."""
    import shutil
    import os
    
    # Model directory already exists
    model_name = model_id.replace("/", "_").replace(":", "_")
    model_dir = os.path.join("model_results", model_name)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    # Copy new results files to model directory
    if os.path.exists(results_file):
        new_results_file = os.path.join(model_dir, os.path.basename(results_file))
        shutil.copy2(results_file, new_results_file)
        print(f"True solution results copied to: {new_results_file}")
    
    if os.path.exists(analysis_file):
        new_analysis_file = os.path.join(model_dir, os.path.basename(analysis_file))
        shutil.copy2(analysis_file, new_analysis_file)
        print(f"True solution analysis copied to: {new_analysis_file}")
    
    # Copy LaTeX tables if they exist
    latex_file = results_file.replace('.json', '_metrics_table.tex')
    if os.path.exists(latex_file):
        new_latex_file = os.path.join(model_dir, os.path.basename(latex_file))
        shutil.copy2(latex_file, new_latex_file)
        print(f"True solution LaTeX table copied to: {new_latex_file}")
    
    print(f"\nTrue solution results added to: {model_dir}")

def main():
    """Main function."""
    print("=== Qwen 2.5 VL 32B - With True Solution Evaluation ===")
    print("This script will run evaluation with 'With True Solution' mode:")
    print("qwen/qwen2.5-vl-32b-instruct")
    print()
    print("This will:")
    print("- Evaluate 122 examples across 7 task types")
    print("- Test 'with_true_solution' approach only")
    print("- Take about 45-60 minutes")
    print("- Cost approximately $0.45")
    print()
    print("RUNNING AUTOMATICALLY - NO USER CONFIRMATION REQUIRED")
    print()
    
    # Check if we have the required files
    required_files = [
        "dataset_benchmark_hf_updated_fixed",
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
        success = asyncio.run(run_qwen_with_true_solution())
        if success:
            print("\n" + "="*60)
            print("✓ QWEN 2.5 VL 32B WITH TRUE SOLUTION COMPLETED!")
            print("="*60)
            print("Now you have all 3 evaluation modes completed.")
        else:
            print("\n✗ Qwen with true solution evaluation failed.")
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
