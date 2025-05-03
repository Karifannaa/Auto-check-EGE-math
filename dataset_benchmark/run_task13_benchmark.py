"""
Script to run a benchmark for task 13 using a free model.
"""

import asyncio
from benchmark_models import ModelBenchmark

async def run_task13_benchmark():
    """Run a benchmark for task 13 using a free model."""
    # Use a free model for testing
    model_id = "moonshotai/kimi-vl-a3b-thinking:free"

    # Create benchmark instance
    benchmark = ModelBenchmark()

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
    asyncio.run(run_task13_benchmark())
