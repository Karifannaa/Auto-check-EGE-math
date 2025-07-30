"""
Script to run a full benchmark for all tasks using specified models.
"""

import asyncio
import argparse
from benchmark_models import ModelBenchmark

async def run_full_benchmark(models=None, max_examples=None):
    """Run a full benchmark for all tasks using specified models."""
    # Default to using free models if none specified
    if not models:
        models = [
            "moonshotai/kimi-vl-a3b-thinking:free",
            "google/gemini-2.5-flash-preview:thinking",
            "qwen/qwen2.5-vl-32b-instruct:free"
        ]

    # Create benchmark instance with updated dataset
    benchmark = ModelBenchmark(dataset_dir="dataset_benchmark_hf_updated")

    try:
        # Run benchmark for all tasks
        print(f"Running full benchmark with models: {models}")
        results, results_file = await benchmark.run_benchmark(
            task_id=None,  # All tasks
            model_ids=models,
            with_answer=True,
            without_answer=True,
            max_examples=max_examples,
            prompt_variant="detailed",
            include_examples=False
        )

        # Analyze results
        analysis = benchmark.analyze_results(results)
        analysis_file = benchmark.save_analysis(analysis, results_file)

        # Print summary
        print("\nBenchmark Summary:")
        print(f"Total examples: {analysis['total_examples']}")
        print(f"Total evaluations: {analysis['total_evaluations']}")
        print(f"Total cost: ${analysis['summary']['total_cost']:.4f}")
        if analysis['summary'].get('avg_quality_score') is not None:
            print(f"Average quality score: {analysis['summary']['avg_quality_score'] * 100:.2f}%")

        print("\nModel Performance:")
        for model_id, model_data in analysis["models"].items():
            print(f"\n{model_id}:")
            for answer_type, metrics in model_data.items():
                print(f"  {answer_type}:")
                print(f"    Accuracy: {metrics['accuracy']:.2f}%")

                # Print new metrics
                if metrics.get('quality_score') is not None:
                    print(f"    Quality score: {metrics['quality_score']:.2f}%")
                if metrics.get('avg_score_distance') is not None:
                    print(f"    Avg. score distance: {metrics['avg_score_distance']:.2f}")

                # Print precision, recall, F1
                if 'macro_precision' in metrics:
                    print(f"    Macro precision: {metrics['macro_precision']:.2f}%")
                    print(f"    Macro recall: {metrics['macro_recall']:.2f}%")
                    print(f"    Macro F1: {metrics['macro_f1']:.2f}%")

                # Print confusion matrix if available and not too large
                if 'confusion_matrix' in metrics and metrics['confusion_matrix'] and len(metrics['confusion_matrix']) <= 5:
                    print(f"    Confusion matrix:")
                    # Print header
                    scores = sorted(metrics['confusion_matrix'].keys())
                    header = "      True\\Pred |"
                    for score in scores:
                        header += f" {score} |"
                    print(header)
                    # Print rows
                    for true_score in scores:
                        row = f"      {true_score}        |"
                        for pred_score in scores:
                            count = metrics['confusion_matrix'][true_score].get(pred_score, 0)
                            row += f" {count} |"
                        print(row)

                print(f"    Evaluations: {metrics['evaluations']}")
                print(f"    Avg. evaluation time: {metrics['avg_evaluation_time']:.2f}s")
                print(f"    Total cost: ${metrics['total_cost']:.4f}")

        print(f"\nDetailed results saved to: {results_file}")
        print(f"Analysis saved to: {analysis_file}")

    finally:
        await benchmark.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a full benchmark for all tasks")
    parser.add_argument("--models", type=str, nargs="+", help="Model IDs to benchmark")
    parser.add_argument("--max-examples", type=int, help="Maximum number of examples per task")

    args = parser.parse_args()

    asyncio.run(run_full_benchmark(models=args.models, max_examples=args.max_examples))
