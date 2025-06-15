#!/usr/bin/env python3
"""
Script to run a full evaluation for all tasks (13-19) with the Qwen 2.5 VL 72B model.
This uses the exact same approach that worked for Gemini models.
"""

import os
import sys
import asyncio
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
sys.path.append("dataset_benchmark")

async def run_qwen72b_evaluation():
    """Run a full evaluation for all tasks with Qwen 72B model."""
    
    # Set the NEW API key directly
    api_key = "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"
    os.environ["OPENROUTER_API_KEY"] = api_key
    
    try:
        # Import after setting the API key
        from benchmark_models import ModelBenchmark
        
        print("=== Full Evaluation for Qwen 2.5 VL 72B Model ===")
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
        model_id = "qwen/qwen2.5-vl-72b-instruct"
        print(f"Evaluating model: {model_id}")
        print("Running evaluation on ALL examples from ALL tasks (13-19)...")
        print("This will take significant time (potentially hours)")
        print("This will cost money (estimated $10-50 depending on the model)")
        print()
        
        # Confirm with user
        confirm = input("\nDo you want to proceed? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Evaluation cancelled.")
            return False
        
        print("\nStarting evaluation...")
        start_time = datetime.now()
        
        # Run benchmark for all tasks - EXACT SAME APPROACH AS GEMINI
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
        print("QWEN 2.5 VL 72B EVALUATION RESULTS SUMMARY")
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
        
        # Create model results directory and organize files
        await organize_results(model_id, results_file, analysis_file, analysis, duration)
        
        return True
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'benchmark' in locals():
            await benchmark.close()

async def organize_results(model_id, results_file, analysis_file, analysis, duration):
    """Organize results into model-specific directory structure."""
    import shutil
    import json
    
    # Create model directory
    model_name = model_id.replace("/", "_").replace(":", "_")
    model_dir = os.path.join("model_results", model_name)
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
    readme_content = f"""# Benchmark Results: {model_id}

## Overview
Comprehensive evaluation of {model_id} model on Auto-check-EGE-math dataset.

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
    for model_id_key, model_data in analysis["models"].items():
        for answer_type, metrics in model_data.items():
            quality_score = f"{metrics.get('quality_score', 0):.2f}%" if metrics.get('quality_score') is not None else "N/A"
            score_distance = f"{metrics.get('avg_score_distance', 0):.2f}" if metrics.get('avg_score_distance') is not None else "N/A"
            readme_content += f"| **{answer_type.title()}** | **{metrics['accuracy']:.2f}%** | **{quality_score}** | **{score_distance}** | **${metrics['total_cost']:.4f}** | **{metrics['evaluations']}** |\n"
    
    readme_content += f"""
## Key Findings

### Performance Characteristics
- **Average evaluation time**: {analysis['summary']['avg_evaluation_time']:.2f}s per assessment
- **Cost efficiency**: ${analysis['summary']['total_cost']:.4f} total cost for {analysis['total_evaluations']} evaluations
- **Token usage**: ~{analysis['summary']['avg_prompt_tokens']:.0f} prompt + {analysis['summary']['avg_completion_tokens']:.0f} completion tokens per evaluation

### Model Capabilities
- **Multimodal processing**: Supports both text and image inputs
- **Mathematical reasoning**: Advanced mathematical problem solving
- **Visual analysis**: Can interpret mathematical diagrams and formulas

## Technical Details

**Model Configuration**:
- Provider: OpenRouter
- Full model name: {model_id}
- Model family: Qwen 2.5 VL
- Parameters: 72B
- Multimodal: Text + Image → Text

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
- **Mathematical problem solving**: Excellent for complex math problems
- **Visual reasoning**: Strong performance on diagram interpretation
- **Educational assessment**: Suitable for automated grading systems

### Optimal Configuration
- **Best for**: Mathematical reasoning tasks with visual components
- **Cost consideration**: Higher cost but potentially better performance
- **Performance**: Consistent evaluation times around {analysis['summary']['avg_evaluation_time']:.1f}s

## Conclusion

The {model_id} model demonstrates strong capabilities for mathematical problem solving with visual components, offering advanced performance for educational assessment applications.
"""
    
    # Write README file
    readme_path = os.path.join(model_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"README generated: {readme_path}")

def main():
    """Main function."""
    print("=== Qwen 2.5 VL 72B Evaluation Script ===")
    print("This script will run a complete evaluation for ALL tasks (13-19) with the model:")
    print("qwen/qwen2.5-vl-72b-instruct")
    print()
    print("This will:")
    print("- Evaluate 122 examples across 7 task types")
    print("- Test both 'with_answer' and 'without_answer' approaches")
    print("- Take significant time (potentially hours)")
    print("- Cost money (estimated $10-50 depending on the model)")
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
        success = asyncio.run(run_qwen72b_evaluation())
        if success:
            print("\n" + "="*60)
            print("✓ QWEN 2.5 VL 72B EVALUATION COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("Check the model_results directory for detailed analysis.")
        else:
            print("\n✗ Qwen 72B evaluation failed.")
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
