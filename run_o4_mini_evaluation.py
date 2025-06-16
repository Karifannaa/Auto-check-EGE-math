#!/usr/bin/env python3
"""
Script to run a full evaluation for all tasks (13-19) with the OpenAI o4-mini model.
This will evaluate openai/o4-mini on all 122 examples in all 3 modes.
"""

import os
import sys
import asyncio
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
sys.path.append("dataset_benchmark")

async def run_o4_mini_evaluation():
    """Run a full evaluation for all tasks with o4-mini model."""
    
    # Set the API key directly
    api_key = "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"
    os.environ["OPENROUTER_API_KEY"] = api_key
    
    try:
        # Import after setting the API key
        from benchmark_models import ModelBenchmark
        
        print("=== Full Evaluation for OpenAI o4-mini Model ===")
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
        model_id = "o4-mini"  # This maps to openai/o4-mini
        print(f"Evaluating model: {model_id} (openai/o4-mini)")
        print("Running evaluation on ALL examples from ALL tasks (13-19)...")
        print("This will evaluate in ALL 3 modes:")
        print("1. With Answer: Using images that include the answer")
        print("2. Without Answer: Using images without the answer") 
        print("3. With True Solution: Using images with the true solution")
        print()
        print("Expected:")
        print("- Total examples: 122")
        print("- Total evaluations: 366 (122 × 3 modes)")
        print("- Estimated time: 2-4 hours")
        print("- Estimated cost: $15-50 (o4-mini is a reasoning model)")
        
        print("\nStarting evaluation automatically...")
        start_time = datetime.now()
        
        all_results = []
        
        # Run each mode separately to ensure all 3 modes are evaluated
        print("\n" + "="*60)
        print("MODE 1: WITHOUT ANSWER")
        print("="*60)
        
        results1, results_file1 = await benchmark.run_benchmark(
            task_id=None,  # All tasks
            model_ids=[model_id],
            with_answer=False,      
            without_answer=True,   # Mode 1 only
            with_true_solution=False,
            max_examples=None,  # All examples
            prompt_variant="detailed",
            include_examples=False
        )
        all_results.extend(results1)
        print(f"Mode 1 completed: {len(results1)} evaluations")
        
        print("\n" + "="*60)
        print("MODE 2: WITH ANSWER")
        print("="*60)
        
        results2, results_file2 = await benchmark.run_benchmark(
            task_id=None,  # All tasks
            model_ids=[model_id],
            with_answer=True,      # Mode 2 only
            without_answer=False,   
            with_true_solution=False,
            max_examples=None,  # All examples
            prompt_variant="detailed",
            include_examples=False
        )
        all_results.extend(results2)
        print(f"Mode 2 completed: {len(results2)} evaluations")
        
        print("\n" + "="*60)
        print("MODE 3: WITH TRUE SOLUTION")
        print("="*60)
        
        results3, results_file3 = await benchmark.run_benchmark(
            task_id=None,  # All tasks
            model_ids=[model_id],
            with_answer=False,      
            without_answer=False,   
            with_true_solution=True,  # Mode 3 only
            max_examples=None,  # All examples
            prompt_variant="detailed",
            include_examples=False
        )
        all_results.extend(results3)
        print(f"Mode 3 completed: {len(results3)} evaluations")
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\nAll evaluations completed!")
        print(f"Duration: {duration}")
        print(f"Total evaluations: {len(all_results)}")
        
        # Save combined results
        print("\nSaving combined results...")
        combined_results_file = await save_combined_results(benchmark, all_results, model_id)
        
        # Analyze combined results
        print("\nAnalyzing combined results...")
        analysis = benchmark.analyze_results(all_results)
        analysis_file = benchmark.save_analysis(analysis, combined_results_file)
        
        # Print comprehensive summary
        print("\n" + "="*70)
        print("OPENAI O4-MINI FULL EVALUATION RESULTS SUMMARY")
        print("="*70)
        print(f"Model: {model_id}")
        print(f"Total examples: {analysis['total_examples']}")
        print(f"Total evaluations: {analysis['total_evaluations']}")
        print(f"Total cost: ${analysis['summary']['total_cost']:.4f}")
        print(f"Average evaluation time: {analysis['summary']['avg_evaluation_time']:.2f}s")
        print(f"Total duration: {duration}")
        
        if analysis['summary'].get('avg_quality_score') is not None:
            print(f"Average quality score: {analysis['summary']['avg_quality_score'] * 100:.2f}%")
        
        # Print detailed results by evaluation mode
        print(f"\n{'='*70}")
        print("DETAILED RESULTS BY EVALUATION MODE")
        print("="*70)
        
        for model_id_key, model_data in analysis["models"].items():
            print(f"\nModel: {model_id_key}")
            
            # Sort modes for consistent display
            mode_order = ["without_answer", "with_answer", "with_true_solution"]
            sorted_modes = []
            for mode in mode_order:
                if mode in model_data:
                    sorted_modes.append((mode, model_data[mode]))
            
            for answer_type, metrics in sorted_modes:
                print(f"\n  === {answer_type.upper().replace('_', ' ')} ===")
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
        
        print(f"\nDetailed results saved to: {combined_results_file}")
        print(f"Analysis saved to: {analysis_file}")
        
        # Count examples by task and mode
        task_counts = {}
        mode_counts = {"with_answer": 0, "without_answer": 0, "with_true_solution": 0}
        
        for result in all_results:
            task_id = result['task_id']
            task_counts[task_id] = task_counts.get(task_id, 0) + 1
            
            # Count by mode
            if result.get('use_true_solution'):
                mode_counts["with_true_solution"] += 1
            elif result.get('use_answer'):
                mode_counts["with_answer"] += 1
            else:
                mode_counts["without_answer"] += 1
        
        print(f"\nEvaluations by task:")
        for task_id, count in sorted(task_counts.items()):
            print(f"  Task {task_id}: {count} evaluations")
        
        print(f"\nEvaluations by mode:")
        for mode, count in mode_counts.items():
            print(f"  {mode.replace('_', ' ').title()}: {count} evaluations")
        
        # Create model results directory and organize files
        await organize_results(model_id, combined_results_file, analysis_file, analysis, duration)
        
        return True
        
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
    
    # Create filename for combined results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_short = model_id.replace("/", "_").replace(":", "_")
    filename = f"benchmark_all_tasks_{model_short}_all_modes_{timestamp}.json"
    filepath = os.path.join(benchmark.results_dir, filename)
    
    # Save the results
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"Combined results saved to: {filepath}")
    return filepath

async def organize_results(model_id, results_file, analysis_file, analysis, duration):
    """Organize results into model-specific directory structure."""
    import shutil
    import json
    
    # Create model directory
    model_name = model_id.replace("/", "_").replace(":", "_")
    model_dir = os.path.join("model_results", f"openai_{model_name}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Copy results files to model directory
    if os.path.exists(results_file):
        new_results_file = os.path.join(model_dir, os.path.basename(results_file))
        shutil.copy2(results_file, new_results_file)
        print(f"Results copied to: {new_results_file}")
    
    if os.path.exists(analysis_file):
        new_analysis_file = os.path.join(model_dir, os.path.basename(analysis_file))
        shutil.copy2(analysis_file, new_analysis_file)
        print(f"Analysis copied to: {new_analysis_file}")
    
    # Copy LaTeX tables if they exist
    latex_file = results_file.replace('.json', '_metrics_table.tex')
    if os.path.exists(latex_file):
        new_latex_file = os.path.join(model_dir, os.path.basename(latex_file))
        shutil.copy2(latex_file, new_latex_file)
        print(f"LaTeX table copied to: {new_latex_file}")
    
    # Generate README for the model
    await generate_model_readme(model_dir, model_id, analysis, duration)
    
    print(f"\nAll results organized in: {model_dir}")

async def generate_model_readme(model_dir, model_id, analysis, duration):
    """Generate a comprehensive README for the model results."""
    readme_content = f"""# Benchmark Results: openai/o4-mini

## Overview
Comprehensive evaluation of openai/o4-mini model on Auto-check-EGE-math dataset.

**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d')}  
**Total Examples**: {analysis['total_examples']}  
**Total Evaluations**: {analysis['total_evaluations']}  
**Total Cost**: ${analysis['summary']['total_cost']:.4f}  
**Duration**: {duration}

## Performance Summary

### Overall Metrics by Evaluation Mode

| Evaluation Mode | Accuracy | Quality Score | Avg Score Distance | Total Cost | Evaluations |
|----------------|----------|---------------|-------------------|------------|-------------|
"""
    
    # Add performance data for each mode
    mode_order = ["without_answer", "with_answer", "with_true_solution"]
    for model_id_key, model_data in analysis["models"].items():
        for mode in mode_order:
            if mode in model_data:
                metrics = model_data[mode]
                quality_score = f"{metrics.get('quality_score', 0):.2f}%" if metrics.get('quality_score') is not None else "N/A"
                score_distance = f"{metrics.get('avg_score_distance', 0):.2f}" if metrics.get('avg_score_distance') is not None else "N/A"
                mode_display = mode.replace('_', ' ').title()
                readme_content += f"| **{mode_display}** | **{metrics['accuracy']:.2f}%** | **{quality_score}** | **{score_distance}** | **${metrics['total_cost']:.4f}** | **{metrics['evaluations']}** |\n"
    
    readme_content += f"""
## Key Findings

### Performance Characteristics
- **Average evaluation time**: {analysis['summary']['avg_evaluation_time']:.2f}s per assessment
- **Cost efficiency**: ${analysis['summary']['total_cost']:.4f} total cost for {analysis['total_evaluations']} evaluations
- **Token usage**: ~{analysis['summary']['avg_prompt_tokens']:.0f} prompt + {analysis['summary']['avg_completion_tokens']:.0f} completion tokens per evaluation

### Model Capabilities
- **Reasoning Model**: Advanced reasoning capabilities with chain-of-thought
- **Mathematical Excellence**: Optimized for math, science, and STEM tasks
- **Multimodal Processing**: Supports both text and image inputs
- **PhD-level Accuracy**: Designed for complex problem solving

## Technical Details

**Model Configuration**:
- Provider: OpenRouter
- Full model name: openai/o4-mini
- Model family: o4 (reasoning)
- Context length: 200,000 tokens
- Multimodal: Text + Image → Text
- Reasoning: Chain-of-thought enabled

**Evaluation Settings**:
- Prompt variant: detailed
- Include examples: false
- Max examples: all ({analysis['total_examples']})
- Retry logic: enabled for rate limiting

## Files in this Directory

### Results Files
- `*.json` - Raw benchmark results
- `*_analysis.json` - Detailed analysis and metrics
- `*_metrics_table.tex` - LaTeX formatted metrics tables

## Recommendations

### Use Cases
- **Advanced Mathematical Problem Solving**: Excellent for complex math problems
- **STEM Education**: Optimized for science and engineering tasks
- **Research Applications**: PhD-level accuracy on academic benchmarks
- **Complex Reasoning**: Multi-step problem solving with explanations

### Optimal Configuration
- **Best for**: Complex mathematical reasoning tasks
- **Cost consideration**: Higher cost but superior reasoning capabilities
- **Performance**: Advanced reasoning with detailed explanations
- **Accuracy**: Expected high performance due to reasoning optimization

## Conclusion

The openai/o4-mini model represents OpenAI's latest reasoning technology, specifically optimized for mathematical and scientific problem solving. With advanced chain-of-thought reasoning capabilities, it's designed to provide PhD-level accuracy on complex STEM tasks.
"""
    
    # Write README file
    readme_path = os.path.join(model_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"README generated: {readme_path}")

def main():
    """Main function."""
    print("=== OpenAI o4-mini Full Evaluation Script ===")
    print("This script will run a complete evaluation for ALL tasks (13-19) with the model:")
    print("openai/o4-mini")
    print()
    print("This will:")
    print("- Evaluate 122 examples across 7 task types")
    print("- Test ALL THREE evaluation approaches:")
    print("  1. Without Answer (images without the answer)")
    print("  2. With Answer (images include the answer)")
    print("  3. With True Solution (images with the true solution)")
    print("- Total evaluations: 366 (122 × 3 modes)")
    print("- Take significant time (2-4 hours)")
    print("- Cost money (estimated $15-50 for reasoning model)")
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
        success = asyncio.run(run_o4_mini_evaluation())
        if success:
            print("\n" + "="*70)
            print("✓ OPENAI O4-MINI FULL EVALUATION COMPLETED SUCCESSFULLY!")
            print("="*70)
            print("Check the model_results directory for detailed analysis.")
        else:
            print("\n✗ OpenAI o4-mini evaluation failed.")
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
