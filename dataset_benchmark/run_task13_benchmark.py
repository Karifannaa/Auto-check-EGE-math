"""
Script to run a benchmark for task 13 using a free model.
"""

import os
import sys
import asyncio
import subprocess
from datetime import datetime
from benchmark_models import ModelBenchmark

# OpenRouter API key
OPENROUTER_API_KEY = "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"

async def run_task13_benchmark():
    """Run a benchmark for task 13 using a free model."""
    # Use a free model for testing
    model_id = "google/gemini-2.0-flash-exp:free"  # Try a different model

    # Create model-specific results directory
    model_short_name = model_id.split("/")[-1].split(":")[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "benchmark_results",
        f"task13_{model_short_name}_{timestamp}"
    )
    os.makedirs(model_results_dir, exist_ok=True)
    print(f"Created model results directory: {model_results_dir}")

    # Create benchmark instance with API key
    benchmark = ModelBenchmark(api_key=OPENROUTER_API_KEY)

    try:
        # Run benchmark for task 13 with a limited number of examples
        print(f"Running benchmark for task 13 with model: {model_id}")
        results, results_file = await benchmark.run_benchmark(
            task_id="13",
            model_ids=[model_id],
            with_answer=True,
            without_answer=True,
            max_examples=3,  # Limit to 3 examples for quick testing
            prompt_variant="detailed",
            include_examples=False,
            output_dir=model_results_dir  # Use the model-specific directory
        )

        # Analyze results
        try:
            analysis = benchmark.analyze_results(results)
            analysis_file = benchmark.save_analysis(analysis, results_file)

            # Print brief summary
            print("\nBenchmark Summary:")
            print(f"Total examples: {analysis['total_examples']}")
            print(f"Total evaluations: {analysis['total_evaluations']}")
            print(f"Total cost: ${analysis['summary']['total_cost']:.4f}")
        except Exception as e:
            print(f"\nError analyzing results: {e}")
            print("Continuing with metrics table generation...")

        # Run the analysis script to generate the metrics table
        print("\nGenerating metrics table...")
        try:
            # Use the Python executable from the current environment
            python_executable = sys.executable

            # Run the analysis script
            subprocess.run([
                python_executable,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_existing_results.py"),
                results_file
            ], check=True)
            print(f"Metrics table generated successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error generating metrics table: {e}")

        print(f"\nDetailed results saved to: {results_file}")
        if 'analysis_file' in locals():
            print(f"Analysis saved to: {analysis_file}")
        print(f"All results and analysis files are in: {model_results_dir}")

    finally:
        await benchmark.close()

if __name__ == "__main__":
    asyncio.run(run_task13_benchmark())
