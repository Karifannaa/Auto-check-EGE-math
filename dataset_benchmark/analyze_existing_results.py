"""
Script to analyze existing benchmark results without running the benchmark again.
"""

import os
import sys
import json
import argparse
import pandas as pd
from typing import Dict, List

# Add the backend directory to the path so we can import from app
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

class ResultsAnalyzer:
    """Class to analyze existing benchmark results."""

    def __init__(self, results_dir: str = "benchmark_results"):
        """Initialize the analyzer."""
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), results_dir)

    def load_results(self, results_file: str) -> List[Dict]:
        """Load results from a file."""
        # Handle different path formats
        if os.path.isabs(results_file):
            # Absolute path
            full_path = results_file
        elif "dataset_benchmark/benchmark_results/" in results_file:
            # Extract just the filename
            filename = os.path.basename(results_file)
            full_path = os.path.join(self.results_dir, filename)
        else:
            # Relative path
            full_path = os.path.join(self.results_dir, results_file)

        # Try alternative paths if file not found
        if not os.path.exists(full_path):
            # Try the exact path as provided
            if os.path.exists(results_file):
                full_path = results_file
            else:
                raise FileNotFoundError(f"Results file not found: {full_path} or {results_file}")

        with open(full_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

        print(f"Loaded {len(results)} results from {full_path}")
        return results

    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze benchmark results with enhanced metrics."""
        if not results:
            return {"error": "No results to analyze"}

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)

        # Filter out rows with None scores
        df_valid = df.dropna(subset=['score', 'expected_score'])

        # Calculate accuracy (score matches expected_score)
        df['correct'] = df.apply(lambda row: row['score'] == row['expected_score'] if row['score'] is not None else False, axis=1)

        # Calculate close accuracy (score is within 1 point of expected_score)
        df['close_correct'] = df.apply(
            lambda row: abs(row['score'] - row['expected_score']) <= 1
            if row['score'] is not None and not pd.isna(row['score']) and not pd.isna(row['expected_score'])
            else False,
            axis=1
        )

        # Calculate distance between predicted and expected scores
        if len(df_valid) > 0:
            df['score_distance'] = float('nan')  # Initialize with NaN
            df_valid['score_distance'] = df_valid.apply(lambda row: abs(row['score'] - row['expected_score']), axis=1)

            # Copy values to df where indices match
            for idx in df_valid.index:
                if idx in df.index:
                    df.at[idx, 'score_distance'] = df_valid.at[idx, 'score_distance']
        else:
            df['score_distance'] = float('nan')

        # Calculate normalized distance (as a percentage of the maximum possible distance)
        # First, determine max possible score for each task type
        max_scores = {
            'task_13': 2,
            'task_14': 3,
            'task_15': 2,
            'task_16': 2,  # Fixed: Economic task has maximum 2 points
            'task_17': 3,
            'task_18': 4,
            'task_19': 4
        }

        # Calculate normalized distance
        # Add max_score column to both dataframes
        df['max_score'] = df['task_type'].map(lambda x: max_scores.get(x, 4))  # Default to 4 if task_type not found
        df_valid['max_score'] = df_valid['task_type'].map(lambda x: max_scores.get(x, 4))

        # Now calculate normalized distance with safe division
        if len(df_valid) > 0:
            df['normalized_distance'] = float('nan')  # Initialize with NaN
            df_valid['normalized_distance'] = df_valid.apply(
                lambda row: abs(row['score'] - row['expected_score']) / row['max_score']
                if row['max_score'] > 0 else 0.0, axis=1
            )

            # Copy values to df where indices match
            for idx in df_valid.index:
                if idx in df.index:
                    df.at[idx, 'normalized_distance'] = df_valid.at[idx, 'normalized_distance']
        else:
            df['normalized_distance'] = float('nan')

        # Calculate quality score (1 - normalized_distance)
        # This gives a score from 0 to 1, where 1 is perfect prediction and 0 is worst possible prediction
        df['quality_score'] = 1 - df['normalized_distance']

        # Group by model and answer type
        grouped = df.groupby(['model_id', 'use_answer'])

        analysis = {
            "total_examples": len(df['solution_id'].unique()),
            "total_evaluations": len(df),
            "models": {},
            "summary": {
                "avg_evaluation_time": df['evaluation_time'].mean(),
                "avg_prompt_tokens": df['prompt_tokens'].mean(),
                "avg_completion_tokens": df['completion_tokens'].mean(),
                "avg_total_tokens": df['total_tokens'].mean(),
                "total_cost": df['cost'].sum(),
                "avg_quality_score": df['quality_score'].mean() if 'quality_score' in df else None,
                "accuracy": df['correct'].mean() * 100,
                "close_accuracy": df['close_correct'].mean() * 100
            }
        }

        # Analyze each model
        for (model_id, use_answer), group in grouped:
            if model_id not in analysis["models"]:
                analysis["models"][model_id] = {}

            answer_type = "with_answer" if use_answer else "without_answer"

            # Basic metrics
            accuracy = group['correct'].mean() * 100
            close_accuracy = group['close_correct'].mean() * 100  # New metric: accuracy within 1 point
            avg_quality_score = group['quality_score'].mean() if 'quality_score' in group else None
            avg_distance = group['score_distance'].mean() if 'score_distance' in group else None

            # Calculate precision, recall, and F1 score for each possible score value
            # Treat each score as a class for multi-class classification metrics
            precision_recall_f1 = {}

            # Get unique expected scores in this group
            unique_scores = sorted(group['expected_score'].dropna().unique())

            for score_value in unique_scores:
                # For each score value, calculate precision, recall, and F1
                true_positives = sum((group['score'] == score_value) & (group['expected_score'] == score_value))
                false_positives = sum((group['score'] == score_value) & (group['expected_score'] != score_value))
                false_negatives = sum((group['score'] != score_value) & (group['expected_score'] == score_value))

                # Calculate precision, recall, and F1 with safe division
                try:
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
                    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                except (ZeroDivisionError, TypeError, ValueError):
                    precision = recall = f1 = 0.0

                precision_recall_f1[str(score_value)] = {
                    "precision": precision * 100,  # Convert to percentage
                    "recall": recall * 100,        # Convert to percentage
                    "f1": f1 * 100                 # Convert to percentage
                }

            # Calculate macro-average precision, recall, and F1 with safe division
            if precision_recall_f1:
                try:
                    macro_precision = sum(item["precision"] for item in precision_recall_f1.values()) / len(precision_recall_f1)
                    macro_recall = sum(item["recall"] for item in precision_recall_f1.values()) / len(precision_recall_f1)
                    macro_f1 = sum(item["f1"] for item in precision_recall_f1.values()) / len(precision_recall_f1)
                except (ZeroDivisionError, TypeError, ValueError):
                    macro_precision = macro_recall = macro_f1 = 0.0
            else:
                macro_precision = macro_recall = macro_f1 = 0.0

            # Create confusion matrix
            if len(unique_scores) > 0:
                confusion_matrix = {}
                for true_score in unique_scores:
                    confusion_matrix[str(true_score)] = {}
                    for pred_score in unique_scores:
                        count = sum((group['expected_score'] == true_score) & (group['score'] == pred_score))
                        confusion_matrix[str(true_score)][str(pred_score)] = int(count)
            else:
                confusion_matrix = {}

            # Store all metrics in the analysis
            analysis["models"][model_id][answer_type] = {
                "accuracy": accuracy,
                "close_accuracy": close_accuracy,  # New metric: accuracy within 1 point
                "quality_score": avg_quality_score * 100 if avg_quality_score is not None else None,  # Convert to percentage
                "avg_score_distance": avg_distance,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "per_score_metrics": precision_recall_f1,
                "confusion_matrix": confusion_matrix,
                "evaluations": len(group),
                "avg_evaluation_time": group['evaluation_time'].mean(),
                "avg_prompt_tokens": group['prompt_tokens'].mean(),
                "avg_completion_tokens": group['completion_tokens'].mean(),
                "avg_total_tokens": group['total_tokens'].mean(),
                "total_cost": group['cost'].sum()
            }

        return analysis

    def save_analysis(self, analysis: Dict, results_file: str) -> str:
        """Save analysis results to a file."""
        # Create filename based on results file
        filename = os.path.basename(results_file)
        analysis_filename = filename.replace('.json', '_analysis.json')

        # Save in the same directory as the results file
        if os.path.isabs(results_file) and os.path.exists(os.path.dirname(results_file)):
            analysis_path = os.path.join(os.path.dirname(results_file), analysis_filename)
        else:
            analysis_path = os.path.join(self.results_dir, analysis_filename)

        # Save analysis to file
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

        print(f"Saved analysis to {analysis_path}")

        return analysis_path

    def print_analysis(self, analysis: Dict):
        """Print analysis results."""
        print("\nBenchmark Summary:")
        print(f"Total examples: {analysis['total_examples']}")
        print(f"Total evaluations: {analysis['total_evaluations']}")
        print(f"Total cost: ${analysis['summary']['total_cost']:.4f}")
        print(f"Overall accuracy: {analysis['summary']['accuracy']:.2f}%")
        print(f"Overall close accuracy (±1 point): {analysis['summary']['close_accuracy']:.2f}%")
        if analysis['summary'].get('avg_quality_score') is not None:
            print(f"Average quality score: {analysis['summary']['avg_quality_score'] * 100:.2f}%")

        print("\nModel Performance:")
        for model_id, model_data in analysis["models"].items():
            print(f"\n{model_id}:")
            for answer_type, metrics in model_data.items():
                print(f"  {answer_type}:")
                print(f"    Accuracy: {metrics['accuracy']:.2f}%")
                print(f"    Close accuracy (±1 point): {metrics['close_accuracy']:.2f}%")

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

    def generate_metrics_latex_table(self, analysis: Dict) -> str:
        """Generate a LaTeX table with all metrics from the benchmark results."""
        latex_table = []

        # Start the table
        latex_table.append("\\begin{table}[htbp]")
        latex_table.append("\\centering")
        latex_table.append("\\caption{Benchmark Results for Task 13 Evaluation}")
        latex_table.append("\\begin{tabular}{lcc}")
        latex_table.append("\\toprule")
        latex_table.append("\\textbf{Metric} & \\textbf{With Answer} & \\textbf{Without Answer} \\\\")
        latex_table.append("\\midrule")

        # Get the model data (assuming only one model for simplicity)
        model_id = list(analysis["models"].keys())[0]
        model_data = analysis["models"][model_id]

        # Add metrics rows
        metrics_to_include = [
            ("Accuracy (\\%)", "accuracy"),
            ("Close Accuracy ($\\pm$1 point) (\\%)", "close_accuracy"),
            ("Quality Score (\\%)", "quality_score"),
            ("Average Score Distance", "avg_score_distance"),
            ("Macro Precision (\\%)", "macro_precision"),
            ("Macro Recall (\\%)", "macro_recall"),
            ("Macro F1 (\\%)", "macro_f1"),
            ("Evaluations (count)", "evaluations"),
            ("Average Evaluation Time (s)", "avg_evaluation_time"),
            ("Total Cost (\\$)", "total_cost")
        ]

        for metric_name, metric_key in metrics_to_include:
            with_value = model_data.get("with_answer", {}).get(metric_key, "N/A")
            without_value = model_data.get("without_answer", {}).get(metric_key, "N/A")

            # Format values appropriately
            if isinstance(with_value, (int, float)) and metric_key != "evaluations":
                if metric_key == "total_cost":
                    with_value = f"{with_value:.2f}"
                    without_value = f"{without_value:.2f}"
                elif metric_key == "avg_evaluation_time":
                    with_value = f"{with_value:.2f}"
                    without_value = f"{without_value:.2f}"
                else:
                    with_value = f"{with_value:.2f}"
                    without_value = f"{without_value:.2f}"

            latex_table.append(f"{metric_name} & {with_value} & {without_value} \\\\")

        # End the table
        latex_table.append("\\bottomrule")
        latex_table.append("\\end{tabular}")
        latex_table.append("\\end{table}")

        return "\n".join(latex_table)

    def generate_explanation_latex_table(self) -> str:
        """Generate a LaTeX table explaining each metric."""
        latex_table = []

        # Start the table
        latex_table.append("\\begin{table}[htbp]")
        latex_table.append("\\centering")
        latex_table.append("\\caption{Explanation of Benchmark Evaluation Metrics}")
        latex_table.append("\\begin{tabular}{p{3cm}p{9cm}}")
        latex_table.append("\\toprule")
        latex_table.append("\\textbf{Metric} & \\textbf{Explanation} \\\\")
        latex_table.append("\\midrule")

        # Add explanations for each metric
        explanations = [
            ("Accuracy", "Percentage of cases where the model's predicted score exactly matches the expected score. Higher is better."),
            ("Close Accuracy", "Percentage of cases where the model's predicted score is within $\\pm$1 point of the expected score. Measures near-correctness. Higher is better."),
            ("Quality Score", "Normalized measure (0-100\\%) indicating how close predictions are to expected scores relative to the maximum possible error. Calculated as 100\\% $\\times$ (1 - normalized\\_distance). Higher is better."),
            ("Average Score Distance", "Average absolute difference between predicted and expected scores. Lower is better."),
            ("Macro Precision", "Average precision across all score classes. Measures how many of the model's predictions for each score value were correct. Higher is better."),
            ("Macro Recall", "Average recall across all score classes. Measures how many of the actual instances of each score value were correctly identified. Higher is better."),
            ("Macro F1", "Harmonic mean of precision and recall across all score classes. Balanced measure of model performance. Higher is better."),
            ("Evaluations", "Number of evaluations performed."),
            ("Average Evaluation Time", "Average time in seconds to complete one evaluation."),
            ("Total Cost", "Total cost in USD for all evaluations.")
        ]

        for metric, explanation in explanations:
            latex_table.append(f"{metric} & {explanation} \\\\")
            latex_table.append("\\addlinespace")

        # End the table
        latex_table.append("\\bottomrule")
        latex_table.append("\\end{tabular}")
        latex_table.append("\\end{table}")

        return "\n".join(latex_table)

    def generate_quality_score_explanation_latex_table(self) -> str:
        """Generate a LaTeX table with detailed explanation of the quality score metric."""
        latex_table = []

        # Start the table
        latex_table.append("\\begin{table}[htbp]")
        latex_table.append("\\centering")
        latex_table.append("\\caption{Detailed Explanation of Quality Score Metric}")
        latex_table.append("\\begin{tabular}{p{12cm}}")
        latex_table.append("\\toprule")
        latex_table.append("\\textbf{Quality Score: In-Depth Explanation} \\\\")
        latex_table.append("\\midrule")

        # Add detailed explanation
        explanation = [
            "The Quality Score is a normalized measure that indicates how close the model's predictions are to the expected scores, taking into account the maximum possible error for each task type.",
            "",
            "\\textbf{Calculation:}",
            "1. For each prediction, calculate the absolute difference between predicted and expected score",
            "2. Normalize this difference by dividing by the maximum possible score for the task type (for Task 13, max score = 2)",
            "3. Subtract this normalized distance from 1 to get a quality score between 0 and 1",
            "4. Convert to percentage by multiplying by 100",
            "",
            "\\textbf{Formula:} Quality Score = 100\\% $\\times$ (1 - $|$predicted\\_score - expected\\_score$|$ / max\\_possible\\_score)",
            "",
            "\\textbf{Example:}",
            "- If predicted = 1, expected = 2, max\\_score = 2:",
            "- Quality Score = 100\\% $\\times$ (1 - $|$1-2$|$/2) = 100\\% $\\times$ (1 - 0.5) = 50\\%",
            "",
            "\\textbf{Interpretation:}",
            "- 100\\%: Perfect prediction (predicted = expected)",
            "- 50\\%: Prediction off by half the maximum possible error",
            "- 0\\%: Prediction off by the maximum possible error",
            "",
            "For Task 13 with \"With Answer\" (66.67\\%), this means predictions are on average about 1/3 of the maximum possible error away from the expected scores."
        ]

        for line in explanation:
            if line == "":
                latex_table.append("\\addlinespace")
            else:
                latex_table.append(f"{line} \\\\")

        # End the table
        latex_table.append("\\bottomrule")
        latex_table.append("\\end{tabular}")
        latex_table.append("\\end{table}")

        return "\n".join(latex_table)

def main():
    """Main function to analyze existing benchmark results."""
    parser = argparse.ArgumentParser(description="Analyze existing benchmark results")
    parser.add_argument("results_file", type=str, help="Results file to analyze (e.g., benchmark_task13_kimi-vl-a3b-thinking_20250505_002804.json)")
    parser.add_argument("--latex", action="store_true", help="Generate all LaTeX tables")
    parser.add_argument("--full-output", action="store_true", help="Show full analysis output")
    parser.add_argument("--explanation", action="store_true", help="Generate explanation table")
    parser.add_argument("--quality", action="store_true", help="Generate quality score explanation table")

    args = parser.parse_args()

    analyzer = ResultsAnalyzer()
    results = analyzer.load_results(args.results_file)
    analysis = analyzer.analyze_results(results)
    analyzer.save_analysis(analysis, args.results_file)

    # Only print full analysis if requested
    if args.full_output:
        analyzer.print_analysis(analysis)

    # Always generate metrics table
    print("\nGenerating metrics table...")
    metrics_table = analyzer.generate_metrics_latex_table(analysis)
    metrics_table_file = args.results_file.replace('.json', '_metrics_table.tex')
    if os.path.isabs(args.results_file):
        metrics_table_path = os.path.join(os.path.dirname(args.results_file), os.path.basename(metrics_table_file))
    else:
        metrics_table_path = os.path.join(analyzer.results_dir, os.path.basename(metrics_table_file))

    with open(metrics_table_path, 'w', encoding='utf-8') as f:
        f.write(metrics_table)
    print(f"Saved metrics table to {metrics_table_path}")

    # Print the metrics table to console
    print("\n=== METRICS TABLE ===")
    print(metrics_table)

    # Generate explanation table if requested
    if args.latex or args.explanation:
        print("\nGenerating explanation table...")
        explanation_table = analyzer.generate_explanation_latex_table()
        explanation_table_file = args.results_file.replace('.json', '_explanation_table.tex')
        if os.path.isabs(args.results_file):
            explanation_table_path = os.path.join(os.path.dirname(args.results_file), os.path.basename(explanation_table_file))
        else:
            explanation_table_path = os.path.join(analyzer.results_dir, os.path.basename(explanation_table_file))

        with open(explanation_table_path, 'w', encoding='utf-8') as f:
            f.write(explanation_table)
        print(f"Saved explanation table to {explanation_table_path}")

        if args.latex:
            print("\n=== EXPLANATION TABLE ===")
            print(explanation_table)

    # Generate quality score explanation table if requested
    if args.latex or args.quality:
        print("\nGenerating quality score explanation table...")
        quality_table = analyzer.generate_quality_score_explanation_latex_table()
        quality_table_file = args.results_file.replace('.json', '_quality_explanation_table.tex')
        if os.path.isabs(args.results_file):
            quality_table_path = os.path.join(os.path.dirname(args.results_file), os.path.basename(quality_table_file))
        else:
            quality_table_path = os.path.join(analyzer.results_dir, os.path.basename(quality_table_file))

        with open(quality_table_path, 'w', encoding='utf-8') as f:
            f.write(quality_table)
        print(f"Saved quality score explanation table to {quality_table_path}")

        if args.latex:
            print("\n=== QUALITY SCORE EXPLANATION TABLE ===")
            print(quality_table)

if __name__ == "__main__":
    main()
