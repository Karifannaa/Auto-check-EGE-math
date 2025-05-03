"""
Script to benchmark models on the Russian Math Exam Solutions dataset.

This script evaluates model performance on the dataset, with options to:
1. Filter by specific task types (e.g., only task_13)
2. Compare performance with and without answer images
3. Test specific models or multiple models
4. Save results for analysis
"""

import os
import sys
import asyncio
import json
import logging
import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image
import io
import pandas as pd
from datasets import load_from_disk

# Add the backend directory to the path so we can import from app
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from app.utils.prompt_utils import PromptGenerator
from app.utils.image_utils import prepare_image_for_api
from app.api.openrouter_client import OpenRouterClient
from app.core.config import settings

# Import score extractor or define it inline if import fails
try:
    from app.utils.score_extractor import extract_score_from_text
except ImportError:
    import re
    import logging

    def extract_score_from_text(result_text: str, task_type: str = None) -> int:
        """
        Extract the score from the model's response text.

        Args:
            result_text: The text response from the model
            task_type: Optional task type for validation (e.g., "task_13")

        Returns:
            Extracted score as an integer
        """
        score = 0

        try:
            # Look for explicit score sections
            score_sections = [
                r'итоговая оценка[\s\S]*?(\d+)\s*балл',
                r'оценка[\s\S]*?(\d+)\s*балл',
                r'\[оценка[\s\S]*?(\d+)\s*балл',
                r'итоговый балл[\s\S]*?(\d+)',
                r'итоговая оценка[\s\S]*?(\d+)',
                r'оценка:\s*(\d+)',
                r'выставляется\s*(\d+)\s*балл'
            ]

            for pattern in score_sections:
                matches = re.findall(pattern, result_text.lower())
                if matches:
                    score = int(matches[-1])  # Use the last match as it's likely the final score
                    logger.info(f"Found score {score} using pattern: {pattern}")
                    break

            # Look for specific formats in the entire text
            if score == 0:
                # Look for patterns like "1 балл" or "2 балла" in the explanation
                score_patterns = [
                    r'(\d+)\s*балл',  # "2 балла"
                    r'оценка\s*[:-]\s*(\d+)',  # "Оценка: 2"
                    r'\[(\d+)\s*балл',  # "[2 балла"
                    r'\[оценка\s*[:-]\s*(\d+)\]'  # "[Оценка: 2]"
                ]

                for pattern in score_patterns:
                    matches = re.findall(pattern, result_text.lower())
                    if matches:
                        score = int(matches[-1])
                        logger.info(f"Found score {score} using pattern: {pattern}")
                        break

            # Validate the score based on task type
            if task_type and task_type.startswith("task_"):
                task_id = task_type.split("_")[1]
                if task_id == "13" and score > 2:
                    logger.warning(f"Score {score} exceeds maximum of 2 for task_13, capping at 2")
                    score = 2
                elif task_id == "14" and score > 3:
                    logger.warning(f"Score {score} exceeds maximum of 3 for task_14, capping at 3")
                    score = 3

        except Exception as e:
            logger.error(f"Error extracting score: {str(e)}")
            score = 0

        return score

# Configure logging
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_logs")
os.makedirs(logs_dir, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(logs_dir, f"benchmark_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("benchmark")

class ModelBenchmark:
    """Class to benchmark models on the dataset."""

    def __init__(
        self,
        dataset_dir: str = "dataset_benchmark_hf",
        results_dir: str = "benchmark_results",
        api_key: Optional[str] = None
    ):
        """Initialize the benchmark."""
        self.dataset_dir = dataset_dir
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), results_dir)
        os.makedirs(self.results_dir, exist_ok=True)

        # Load dataset
        logger.info(f"Loading dataset from {dataset_dir}")
        self.dataset = load_from_disk(dataset_dir)
        logger.info(f"Loaded dataset with {len(self.dataset)} examples")

        # Initialize OpenRouter client
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable.")

        self.client = OpenRouterClient(
            api_key=self.api_key,
            site_url=settings.SITE_URL,
            site_name=settings.SITE_NAME
        )

        # Initialize prompt generator
        self.prompt_generator = PromptGenerator()

    async def close(self):
        """Close the client."""
        await self.client.close()

    def filter_dataset(self, task_id: Optional[str] = None) -> List[Dict]:
        """Filter the dataset by task ID."""
        if task_id:
            # Filter by task ID (e.g., "13")
            filtered = [ex for ex in self.dataset if ex["task_id"] == task_id]
            logger.info(f"Filtered dataset to {len(filtered)} examples for task {task_id}")
            return filtered
        return list(self.dataset)

    async def evaluate_solution(
        self,
        example: Dict,
        model_id: str,
        use_answer: bool = True,
        prompt_variant: str = "detailed",
        include_examples: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 10000
    ) -> Dict:
        """Evaluate a solution using the specified model."""
        task_id = example["task_id"]
        task_type = f"task_{task_id}"
        solution_id = example["solution_id"]

        # Get image paths
        image_paths = example["images_with_answer"] if use_answer else example["images_without_answer"]
        if not image_paths:
            logger.warning(f"No {'with_answer' if use_answer else 'without_answer'} images for solution {solution_id}")
            return {
                "solution_id": solution_id,
                "task_id": task_id,
                "task_type": task_type,
                "model_id": model_id,
                "use_answer": use_answer,
                "error": "No images available",
                "score": None,
                "expected_score": example["score"],
                "evaluation_time": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0,
                "result_text": ""
            }

        try:
            # Process all image parts of the solution
            all_image_data = []

            # Log the number of parts
            logger.info(f"Processing {len(image_paths)} parts for solution {solution_id}")

            # Process each image part in order
            for i, image_path in enumerate(image_paths):
                # Load the image
                with open(image_path, "rb") as f:
                    img_data = f.read()

                img = Image.open(io.BytesIO(img_data))

                # Prepare image for API
                image_data = prepare_image_for_api(
                    img,
                    max_size=2048,  # Balanced size for quality and token count
                    enhance=True,
                    contrast_factor=1.3
                )

                all_image_data.append(image_data)

            # Generate messages with all image parts
            # For multi-part solutions, we need to create a custom prompt that includes all parts

            # First, create a base message structure
            messages = []

            # System message
            system_message = {"role": "system", "content": "You are a helpful assistant that evaluates math solutions."}
            messages.append(system_message)

            # User message with content array for multiple images
            user_content = []

            # Get the appropriate prompt text based on task type and variant
            prompt_text = self.prompt_generator.get_prompt_text(
                task_type=task_type,
                prompt_variant=prompt_variant
            )

            # Add prompt text
            user_content.append({"type": "text", "text": prompt_text})

            # Add instruction for multi-part solutions if needed
            if len(all_image_data) > 1:
                part_instruction = f"\n\nЭто решение состоит из {len(all_image_data)} частей. Рассмотрите все части вместе как одно полное решение.\n\n"
                user_content.append({"type": "text", "text": part_instruction})

            # Add all image parts in order
            for i, img_data in enumerate(all_image_data):
                if len(all_image_data) > 1:
                    user_content.append({"type": "text", "text": f"\nЧасть {i+1}:\n"})
                user_content.append(img_data)

            # Add final instruction
            user_content.append({"type": "text", "text": "\n\nОцените это решение и укажите итоговый балл в разделе 'Итоговый балл'."})

            # Create the user message
            user_message = {"role": "user", "content": user_content}
            messages.append(user_message)

            # Start timing
            start_time = time.time()

            # Call the API
            logger.info(f"Calling API with model: {model_id} for solution {solution_id}")

            # Add thinking mode for models that support it
            extra_body = {}
            if "thinking" in model_id.lower():
                extra_body["thinking"] = True

            response = await self.client.chat_completion(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body
            )

            # Calculate evaluation time
            evaluation_time = time.time() - start_time

            # Extract result text
            result_text = response["choices"][0]["message"]["content"]

            # Extract token usage
            prompt_tokens = response.get("usage", {}).get("prompt_tokens", 0)
            completion_tokens = response.get("usage", {}).get("completion_tokens", 0)
            total_tokens = response.get("usage", {}).get("total_tokens", 0)

            # Extract score
            try:
                # Pass the task_type to help with score validation
                score = extract_score_from_text(result_text, task_type=task_type)
            except Exception as e:
                logger.error(f"Error extracting score: {str(e)}")
                score = 0

            # Calculate cost
            try:
                # Import the cost calculator
                from app.utils.cost_calculator import calculate_actual_cost

                # Calculate cost based on the number of images used
                cost = calculate_actual_cost(
                    model_id=model_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    num_images=len(all_image_data)  # Use the actual number of images
                )
            except Exception as e:
                logger.error(f"Error calculating cost: {str(e)}")
                # Fallback to a simple estimation if the cost calculator fails
                cost = 0.0

            return {
                "solution_id": solution_id,
                "task_id": task_id,
                "task_type": task_type,
                "model_id": model_id,
                "use_answer": use_answer,
                "score": score,
                "expected_score": example["score"],
                "evaluation_time": evaluation_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": cost,
                "result_text": result_text
            }

        except Exception as e:
            logger.error(f"Error evaluating solution {solution_id} with model {model_id}: {str(e)}")
            return {
                "solution_id": solution_id,
                "task_id": task_id,
                "task_type": task_type,
                "model_id": model_id,
                "use_answer": use_answer,
                "error": str(e),
                "score": None,
                "expected_score": example["score"],
                "evaluation_time": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0,
                "result_text": ""
            }

    async def run_benchmark(
        self,
        task_id: Optional[str] = None,
        model_ids: Optional[List[str]] = None,
        with_answer: bool = True,
        without_answer: bool = True,
        max_examples: Optional[int] = None,
        prompt_variant: str = "detailed",
        include_examples: bool = False
    ) -> Tuple[List[Dict], str]:
        """Run the benchmark on the dataset."""
        # Filter dataset
        filtered_dataset = self.filter_dataset(task_id)

        # Limit number of examples if specified
        if max_examples and max_examples > 0:
            filtered_dataset = filtered_dataset[:max_examples]
            logger.info(f"Limited to {len(filtered_dataset)} examples")

        # Use default model if none specified
        if not model_ids:
            model_ids = [settings.DEFAULT_MODEL]

        # Validate models
        valid_model_ids = []
        for model_id in model_ids:
            if model_id in settings.AVAILABLE_MODELS.values():
                valid_model_ids.append(model_id)
            elif model_id in settings.AVAILABLE_MODELS:
                valid_model_ids.append(settings.AVAILABLE_MODELS[model_id])
            else:
                logger.warning(f"Unknown model: {model_id}")

        if not valid_model_ids:
            raise ValueError("No valid models specified")

        logger.info(f"Running benchmark with models: {valid_model_ids}")

        # Run evaluations
        results = []

        for example in filtered_dataset:
            for model_id in valid_model_ids:
                # Evaluate with answer if requested
                if with_answer:
                    result = await self.evaluate_solution(
                        example=example,
                        model_id=model_id,
                        use_answer=True,
                        prompt_variant=prompt_variant,
                        include_examples=include_examples
                    )
                    results.append(result)

                # Evaluate without answer if requested
                if without_answer:
                    result = await self.evaluate_solution(
                        example=example,
                        model_id=model_id,
                        use_answer=False,
                        prompt_variant=prompt_variant,
                        include_examples=include_examples
                    )
                    results.append(result)

        # Save results
        task_str = f"task{task_id}" if task_id else "all_tasks"
        models_str = "_".join([m.split("/")[-1].split(":")[0] for m in valid_model_ids])
        if len(models_str) > 50:  # Truncate if too long
            models_str = models_str[:47] + "..."

        filename = f"benchmark_{task_str}_{models_str}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved results to {filepath}")

        return results, filepath

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

        # Calculate distance between predicted and expected scores
        df['score_distance'] = df_valid.apply(lambda row: abs(row['score'] - row['expected_score']), axis=1)

        # Calculate normalized distance (as a percentage of the maximum possible distance)
        # First, determine max possible score for each task type
        max_scores = {
            'task_13': 2,
            'task_14': 3,
            'task_15': 2,
            'task_16': 3,
            'task_17': 3,
            'task_18': 4,
            'task_19': 4
        }

        # Calculate normalized distance
        df['max_score'] = df['task_type'].map(lambda x: max_scores.get(x, 4))  # Default to 4 if task_type not found
        df['normalized_distance'] = df_valid.apply(
            lambda row: abs(row['score'] - row['expected_score']) / row['max_score'], axis=1
        )

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
                "avg_quality_score": df['quality_score'].mean() if 'quality_score' in df else None
            }
        }

        # Analyze each model
        for (model_id, use_answer), group in grouped:
            if model_id not in analysis["models"]:
                analysis["models"][model_id] = {}

            answer_type = "with_answer" if use_answer else "without_answer"

            # Basic metrics
            accuracy = group['correct'].mean() * 100
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

                # Calculate precision, recall, and F1
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                precision_recall_f1[str(score_value)] = {
                    "precision": precision * 100,  # Convert to percentage
                    "recall": recall * 100,        # Convert to percentage
                    "f1": f1 * 100                 # Convert to percentage
                }

            # Calculate macro-average precision, recall, and F1
            if precision_recall_f1:
                macro_precision = sum(item["precision"] for item in precision_recall_f1.values()) / len(precision_recall_f1)
                macro_recall = sum(item["recall"] for item in precision_recall_f1.values()) / len(precision_recall_f1)
                macro_f1 = sum(item["f1"] for item in precision_recall_f1.values()) / len(precision_recall_f1)
            else:
                macro_precision = macro_recall = macro_f1 = 0

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
        """Save analysis results."""
        analysis_file = results_file.replace(".json", "_analysis.json")

        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved analysis to {analysis_file}")

        return analysis_file

async def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark models on the Russian Math Exam Solutions dataset")
    parser.add_argument("--task", type=str, help="Task ID to benchmark (e.g., '13')")
    parser.add_argument("--models", type=str, nargs="+", help="Model IDs to benchmark")
    parser.add_argument("--with-answer", action="store_true", help="Run with answer images")
    parser.add_argument("--without-answer", action="store_true", help="Run without answer images")
    parser.add_argument("--max-examples", type=int, help="Maximum number of examples to process")
    parser.add_argument("--prompt-variant", type=str, default="detailed", help="Prompt variant to use")
    parser.add_argument("--include-examples", action="store_true", help="Include examples in prompts")
    parser.add_argument("--dataset-dir", type=str, default="dataset_benchmark_hf", help="Dataset directory")

    args = parser.parse_args()

    # Default to both with and without answer if neither is specified
    if not args.with_answer and not args.without_answer:
        args.with_answer = True
        args.without_answer = True

    try:
        benchmark = ModelBenchmark(dataset_dir=args.dataset_dir)

        results, results_file = await benchmark.run_benchmark(
            task_id=args.task,
            model_ids=args.models,
            with_answer=args.with_answer,
            without_answer=args.without_answer,
            max_examples=args.max_examples,
            prompt_variant=args.prompt_variant,
            include_examples=args.include_examples
        )

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

                # Print per-score metrics if available
                if 'per_score_metrics' in metrics and metrics['per_score_metrics']:
                    print(f"    Per-score metrics:")
                    for score, score_metrics in metrics['per_score_metrics'].items():
                        print(f"      Score {score}:")
                        print(f"        Precision: {score_metrics['precision']:.2f}%")
                        print(f"        Recall: {score_metrics['recall']:.2f}%")
                        print(f"        F1: {score_metrics['f1']:.2f}%")

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
        if 'benchmark' in locals():
            await benchmark.close()

if __name__ == "__main__":
    asyncio.run(main())
