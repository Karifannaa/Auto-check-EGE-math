#!/usr/bin/env python3
"""
Script to run or resume OpenAI o4-mini evaluation.
This script can detect where the evaluation stopped and continue from there.
"""

import os
import sys
import asyncio
import json
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
sys.path.append("dataset_benchmark")

def check_existing_results():
    """Check what o4-mini results already exist."""
    results_dir = "dataset_benchmark/benchmark_results"
    model_results_dir = "model_results"
    
    existing_results = {
        "without_answer": None,
        "with_answer": None, 
        "with_true_solution": None
    }
    
    completed_evaluations = 0
    
    # Check benchmark results directory
    if os.path.exists(results_dir):
        for item in os.listdir(results_dir):
            if "o4" in item.lower() and "mini" in item.lower():
                print(f"Found existing result: {item}")
                # Try to determine which mode this is
                if "without_answer" in item or "both_approaches" in item:
                    existing_results["without_answer"] = os.path.join(results_dir, item)
                elif "with_answer" in item:
                    existing_results["with_answer"] = os.path.join(results_dir, item)
                elif "true_solution" in item:
                    existing_results["with_true_solution"] = os.path.join(results_dir, item)
    
    # Check model results directory
    if os.path.exists(model_results_dir):
        for item in os.listdir(model_results_dir):
            if "o4" in item.lower() and "mini" in item.lower():
                print(f"Found existing model results: {item}")
                model_dir = os.path.join(model_results_dir, item)
                # Count existing result files
                for file in os.listdir(model_dir):
                    if file.endswith('.json') and 'analysis' not in file:
                        # Try to load and count evaluations
                        try:
                            with open(os.path.join(model_dir, file), 'r') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    completed_evaluations += len(data)
                        except:
                            pass
    
    print(f"Existing results summary:")
    for mode, path in existing_results.items():
        status = "✓ COMPLETED" if path else "✗ MISSING"
        print(f"  {mode}: {status}")
    
    print(f"Total completed evaluations found: {completed_evaluations}")
    
    return existing_results, completed_evaluations

async def run_o4_mini_evaluation_resume():
    """Run or resume o4-mini evaluation."""
    
    # Set the API key directly
    api_key = "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"
    os.environ["OPENROUTER_API_KEY"] = api_key
    
    print("=== OpenAI o4-mini Evaluation (Resume Capable) ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check existing results
    existing_results, completed_count = check_existing_results()
    
    if completed_count > 0:
        print(f"\n⚠️  Found {completed_count} existing evaluations!")
        print("This script will continue from where it left off.")
    else:
        print("\n🆕 Starting fresh evaluation.")
    
    try:
        # Import after setting the API key
        from benchmark_models import ModelBenchmark
        
        # Create benchmark instance
        print("\nInitializing benchmark...")
        benchmark = ModelBenchmark(
            dataset_dir="dataset_benchmark_hf_updated_fixed",
            api_key=api_key,
            max_retries=10,
            initial_delay=15,
            max_delay=120
        )
        
        print(f"Dataset loaded: {len(benchmark.dataset)} examples")
        
        model_id = "o4-mini"
        print(f"Evaluating model: {model_id} (openai/o4-mini)")
        
        all_results = []
        
        # Mode 1: Without Answer
        if not existing_results["without_answer"]:
            print("\n" + "="*60)
            print("MODE 1: WITHOUT ANSWER (Starting)")
            print("="*60)
            
            results1, results_file1 = await benchmark.run_benchmark(
                task_id=None,
                model_ids=[model_id],
                with_answer=False,
                without_answer=True,
                with_true_solution=False,
                max_examples=None,
                prompt_variant="detailed",
                include_examples=False
            )
            all_results.extend(results1)
            print(f"Mode 1 completed: {len(results1)} evaluations")
        else:
            print("\n" + "="*60)
            print("MODE 1: WITHOUT ANSWER (Already completed)")
            print("="*60)
            # Load existing results
            try:
                result_file = None
                for file in os.listdir(existing_results["without_answer"]):
                    if file.endswith('.json') and 'analysis' not in file:
                        result_file = os.path.join(existing_results["without_answer"], file)
                        break
                
                if result_file:
                    with open(result_file, 'r') as f:
                        results1 = json.load(f)
                        all_results.extend(results1)
                        print(f"Loaded existing Mode 1 results: {len(results1)} evaluations")
            except Exception as e:
                print(f"Error loading existing results: {e}")
        
        # Mode 2: With Answer
        if not existing_results["with_answer"]:
            print("\n" + "="*60)
            print("MODE 2: WITH ANSWER (Starting)")
            print("="*60)
            
            results2, results_file2 = await benchmark.run_benchmark(
                task_id=None,
                model_ids=[model_id],
                with_answer=True,
                without_answer=False,
                with_true_solution=False,
                max_examples=None,
                prompt_variant="detailed",
                include_examples=False
            )
            all_results.extend(results2)
            print(f"Mode 2 completed: {len(results2)} evaluations")
        else:
            print("\n" + "="*60)
            print("MODE 2: WITH ANSWER (Already completed)")
            print("="*60)
            # Load existing results
            try:
                result_file = None
                for file in os.listdir(existing_results["with_answer"]):
                    if file.endswith('.json') and 'analysis' not in file:
                        result_file = os.path.join(existing_results["with_answer"], file)
                        break
                
                if result_file:
                    with open(result_file, 'r') as f:
                        results2 = json.load(f)
                        all_results.extend(results2)
                        print(f"Loaded existing Mode 2 results: {len(results2)} evaluations")
            except Exception as e:
                print(f"Error loading existing results: {e}")
        
        # Mode 3: With True Solution
        if not existing_results["with_true_solution"]:
            print("\n" + "="*60)
            print("MODE 3: WITH TRUE SOLUTION (Starting)")
            print("="*60)
            
            results3, results_file3 = await benchmark.run_benchmark(
                task_id=None,
                model_ids=[model_id],
                with_answer=False,
                without_answer=False,
                with_true_solution=True,
                max_examples=None,
                prompt_variant="detailed",
                include_examples=False
            )
            all_results.extend(results3)
            print(f"Mode 3 completed: {len(results3)} evaluations")
        else:
            print("\n" + "="*60)
            print("MODE 3: WITH TRUE SOLUTION (Already completed)")
            print("="*60)
            # Load existing results
            try:
                result_file = None
                for file in os.listdir(existing_results["with_true_solution"]):
                    if file.endswith('.json') and 'analysis' not in file:
                        result_file = os.path.join(existing_results["with_true_solution"], file)
                        break
                
                if result_file:
                    with open(result_file, 'r') as f:
                        results3 = json.load(f)
                        all_results.extend(results3)
                        print(f"Loaded existing Mode 3 results: {len(results3)} evaluations")
            except Exception as e:
                print(f"Error loading existing results: {e}")
        
        print(f"\n🎉 All modes completed!")
        print(f"Total evaluations: {len(all_results)}")
        
        if len(all_results) > 0:
            # Save combined results
            print("\nSaving combined results...")
            combined_results_file = await save_combined_results(benchmark, all_results, model_id)
            
            # Analyze results
            print("Analyzing results...")
            analysis = benchmark.analyze_results(all_results)
            analysis_file = benchmark.save_analysis(analysis, combined_results_file)
            
            # Print summary
            print("\n" + "="*70)
            print("OPENAI O4-MINI EVALUATION SUMMARY")
            print("="*70)
            print(f"Total evaluations: {len(all_results)}")
            print(f"Total cost: ${analysis['summary']['total_cost']:.4f}")
            print(f"Average evaluation time: {analysis['summary']['avg_evaluation_time']:.2f}s")
            
            # Organize results
            await organize_results(model_id, combined_results_file, analysis_file, analysis)
            
            return True
        else:
            print("❌ No results to process!")
            return False
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'benchmark' in locals():
            await benchmark.close()

async def save_combined_results(benchmark, all_results, model_id):
    """Save combined results from all modes."""
    import json
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"benchmark_all_tasks_o4-mini_all_modes_{timestamp}.json"
    filepath = os.path.join(benchmark.results_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"Combined results saved to: {filepath}")
    return filepath

async def organize_results(model_id, results_file, analysis_file, analysis):
    """Organize results into model directory."""
    import shutil
    
    model_dir = os.path.join("model_results", "openai_o4-mini")
    os.makedirs(model_dir, exist_ok=True)
    
    # Copy files
    if os.path.exists(results_file):
        new_results_file = os.path.join(model_dir, os.path.basename(results_file))
        shutil.copy2(results_file, new_results_file)
        print(f"Results copied to: {new_results_file}")
    
    if os.path.exists(analysis_file):
        new_analysis_file = os.path.join(model_dir, os.path.basename(analysis_file))
        shutil.copy2(analysis_file, new_analysis_file)
        print(f"Analysis copied to: {new_analysis_file}")
    
    # Generate README
    readme_content = f"""# OpenAI o4-mini Evaluation Results

## Summary
- Total Evaluations: {analysis['total_evaluations']}
- Total Cost: ${analysis['summary']['total_cost']:.4f}
- Average Time: {analysis['summary']['avg_evaluation_time']:.2f}s

## Performance by Mode
"""
    
    for model_id_key, model_data in analysis["models"].items():
        for mode, metrics in model_data.items():
            readme_content += f"\n### {mode.replace('_', ' ').title()}\n"
            readme_content += f"- Accuracy: {metrics['accuracy']:.2f}%\n"
            readme_content += f"- Evaluations: {metrics['evaluations']}\n"
            readme_content += f"- Cost: ${metrics['total_cost']:.4f}\n"
    
    readme_path = os.path.join(model_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"README generated: {readme_path}")

def main():
    """Main function."""
    print("=== OpenAI o4-mini Resume Evaluation ===")
    print("This script can detect and resume from partial evaluations.")
    print()
    
    try:
        success = asyncio.run(run_o4_mini_evaluation_resume())
        if success:
            print("\n✅ OpenAI o4-mini evaluation completed successfully!")
        else:
            print("\n❌ OpenAI o4-mini evaluation failed.")
        return success
    except Exception as e:
        print(f"\nError: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
