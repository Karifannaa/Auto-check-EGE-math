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
import random
import numpy as np
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

# Import score extractor - use centralized version for consistency
try:
    from app.utils.score_extractor import extract_score_from_text
except ImportError:
    # Fallback implementation if import fails
    def extract_score_from_text(result_text: str, task_type: str = None) -> int:
        """
        Fallback score extraction function.

        Args:
            result_text: The text response from the model
            task_type: Optional task type for validation (e.g., "task_13")

        Returns:
            Extracted score as an integer
        """
        import re

        if not isinstance(result_text, str) or not result_text.strip():
            return 0

        score = 0
        try:
            # Simple pattern matching for score extraction
            patterns = [
                r'итоговая оценка[\s\S]*?(\d+)\s*балл',
                r'оценка:\s*(\d+)',
                r'(\d+)\s*балл'
            ]

            for pattern in patterns:
                matches = re.findall(pattern, result_text.lower())
                if matches:
                    score = int(matches[-1])
                    break

            # Basic validation
            if score < 0:
                score = 0
            elif task_type == "task_13" and score > 2:
                score = 2
            elif task_type == "task_14" and score > 3:
                score = 3
            elif task_type == "task_15" and score > 2:
                score = 2
            elif task_type == "task_16" and score > 2:  # Fixed: Economic task max 2 points
                score = 2
            elif task_type == "task_17" and score > 3:
                score = 3
            elif task_type == "task_18" and score > 4:
                score = 4
            elif task_type == "task_19" and score > 4:
                score = 4
            elif score > 4:  # Maximum possible score
                score = 4

        except (ValueError, TypeError):
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
        api_key: Optional[str] = None,
        max_retries: int = 10,
        initial_delay: int = 15,
        max_delay: int = 120
    ):
        """Initialize the benchmark."""
        self.dataset_dir = dataset_dir
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), results_dir)
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize current run directory (will be set during run_benchmark)
        self.current_run_dir = None

        # Rate limit retry parameters
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay

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
        use_true_solution: bool = False,
        prompt_variant: str = "detailed",
        include_examples: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 10000
    ) -> Dict:
        """Evaluate a solution using the specified model.

        Args:
            example: Example from the dataset
            model_id: Model ID to use for evaluation
            use_answer: Whether to use images with answer
            use_true_solution: Whether to use images with true solution
            prompt_variant: Prompt variant to use
            include_examples: Whether to include examples in the prompt
            temperature: Temperature for the model
            max_tokens: Maximum tokens for the model response

        Returns:
            Dictionary with evaluation results
        """
        task_id = example["task_id"]
        task_type = f"task_{task_id}"
        solution_id = example["solution_id"]

        # Determine which image paths to use based on the options
        if use_true_solution:
            # Use true solution images if available
            if "images_with_true_solution" in example and example["images_with_true_solution"]:
                image_paths = example["images_with_true_solution"]
                # Set prompt variant to with_solution for tasks 13-19
                if task_type in ["task_13", "task_14", "task_15", "task_16", "task_17", "task_18", "task_19"]:
                    prompt_variant = "with_solution"
            else:
                logger.warning(f"No images_with_true_solution for solution {solution_id}, falling back to without_answer")
                image_paths = example["images_without_answer"]
                use_true_solution = False  # Reset flag since we're not using true solution images
        else:
            # Use regular with_answer or without_answer images
            image_paths = example["images_with_answer"] if use_answer else example["images_without_answer"]

        if not image_paths:
            image_type = "with_true_solution" if use_true_solution else ("with_answer" if use_answer else "without_answer")
            logger.warning(f"No {image_type} images for solution {solution_id}")
            return {
                "solution_id": solution_id,
                "task_id": task_id,
                "task_type": task_type,
                "model_id": model_id,
                "use_answer": use_answer,
                "use_true_solution": use_true_solution,
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
            if use_true_solution:
                logger.info(f"Processing {len(image_paths)} true solution parts and {len(example['images_without_answer'])} student solution parts for solution {solution_id} in a single API call")
            else:
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

            # Add true solution images with clear separation
            if use_true_solution:
                # Add header for true solution
                user_content.append({"type": "text", "text": "\n\n## ВЕРНОЕ РЕШЕНИЕ (ЭТАЛОН ДЛЯ СРАВНЕНИЯ):\n\nНиже представлено верное решение задачи. Используй его как эталон для понимания задачи и правильного ответа. Обрати внимание на ключевые шаги и итоговый ответ.\n\nПомни: ученик может использовать другой подход к решению, и это нормально, если подход математически корректен и приводит к верному ответу.\n\n"})

                # Add true solution images
                for i, img_data in enumerate(all_image_data):
                    if len(all_image_data) > 1:
                        user_content.append({"type": "text", "text": f"\nЧасть {i+1} верного решения:\n"})
                    user_content.append(img_data)

                # Add separator between true solution and student solution
                user_content.append({"type": "text", "text": "\n\n## РЕШЕНИЕ УЧЕНИКА (ДЛЯ ОЦЕНКИ):\n\nВыше представлено верное решение задачи (эталон). Ниже представлено решение ученика, которое нужно оценить.\n\nВАЖНО: Решение ученика может отличаться от эталонного по подходу и оформлению. Это нормально! Главное, чтобы решение было математически верным и приводило к правильному ответу. Оценивай математическую корректность, а не точное совпадение с эталоном.\n\nКРИТИЧЕСКИ ВАЖНО: Проверь, что ответ ученика математически верный! Если ответ ученика неверный, это ОБЯЗАТЕЛЬНО должно быть учтено в оценке!\n\n"})

                # Get student solution images from without_answer
                student_image_paths = example["images_without_answer"]
                student_image_data = []

                # Process student solution images
                for i, image_path in enumerate(student_image_paths):
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

                    student_image_data.append(image_data)

                # Add student solution images
                for i, img_data in enumerate(student_image_data):
                    if len(student_image_data) > 1:
                        user_content.append({"type": "text", "text": f"\nЧасть {i+1} решения ученика:\n"})
                    user_content.append(img_data)
            else:
                # Add all image parts in order (original behavior for non-true-solution cases)
                for i, img_data in enumerate(all_image_data):
                    if len(all_image_data) > 1:
                        user_content.append({"type": "text", "text": f"\nЧасть {i+1}:\n"})
                    user_content.append(img_data)

            # Add final instruction
            user_content.append({"type": "text", "text": "\n\nОцените это решение и укажите итоговый балл в разделе 'Итоговая оценка'. Оценивайте математическую корректность решения, а не точное совпадение с эталоном. Разные подходы к решению могут быть верными, если они математически обоснованы и приводят к правильному ответу.\n\nВАЖНО: Раздел 'Итоговая оценка' ОБЯЗАТЕЛЬНО должен быть оформлен в формате: '### Итоговая оценка' (заголовок) и '[Оценка: X баллов]' (где X - это число баллов)."})

            # Create the user message
            user_message = {"role": "user", "content": user_content}
            messages.append(user_message)

            # Start timing
            start_time = time.time()

            # Call the API with retry logic for rate limits
            if use_true_solution:
                logger.info(f"Calling API with model: {model_id} for solution {solution_id} with true solution and student solution")
            else:
                logger.info(f"Calling API with model: {model_id} for solution {solution_id}")

            # Add thinking mode for models that support it
            extra_body = {}
            if "thinking" in model_id.lower():
                extra_body["thinking"] = True

            # Use instance retry parameters
            max_retries = self.max_retries
            initial_delay = self.initial_delay  # seconds
            max_delay = self.max_delay  # seconds
            retry_count = 0

            while True:
                try:
                    response = await self.client.chat_completion(
                        model=model_id,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        extra_body=extra_body
                    )

                    # Check if response contains an error field indicating rate limit
                    if 'error' in response and 'code' in response['error'] and response['error']['code'] == 429:
                        error_msg = response['error'].get('message', 'Rate limit exceeded')

                        # Check if we've reached max retries
                        if retry_count >= max_retries:
                            logger.error(f"Max retries ({max_retries}) reached for rate limit. Giving up.")
                            raise ValueError(f"Rate limit exceeded after {max_retries} retries: {error_msg}")

                        # Calculate delay with exponential backoff and jitter
                        delay = min(initial_delay * (2 ** retry_count), max_delay)
                        # Add jitter (±20%)
                        jitter = random.uniform(0.8, 1.2)
                        delay = delay * jitter

                        retry_count += 1
                        # Get rate limit info for logging
                        remaining = response['error'].get('metadata', {}).get('headers', {}).get('X-RateLimit-Remaining', '0')
                        limit = response['error'].get('metadata', {}).get('headers', {}).get('X-RateLimit-Limit', 'unknown')

                        logger.warning(f"Rate limit hit ({remaining}/{limit}). Retrying in {delay:.1f} seconds (attempt {retry_count}/{max_retries}). Error: {error_msg}")

                        # Wait before retrying
                        await asyncio.sleep(delay)
                        continue

                    # If we get here, the request was successful
                    break

                except Exception as e:
                    # Check if the exception message indicates a rate limit
                    error_str = str(e).lower()
                    if "rate limit" in error_str or "429" in error_str:
                        # Check if we've reached max retries
                        if retry_count >= max_retries:
                            logger.error(f"Max retries ({max_retries}) reached for rate limit. Giving up.")
                            raise ValueError(f"Rate limit exceeded after {max_retries} retries: {str(e)}")

                        # Calculate delay with exponential backoff and jitter
                        delay = min(initial_delay * (2 ** retry_count), max_delay)
                        # Add jitter (±20%)
                        jitter = random.uniform(0.8, 1.2)
                        delay = delay * jitter

                        retry_count += 1
                        logger.warning(f"Rate limit exception. Retrying in {delay:.1f} seconds (attempt {retry_count}/{max_retries}). Error: {str(e)}")

                        # Wait before retrying
                        await asyncio.sleep(delay)
                        continue
                    else:
                        # Not a rate limit error, re-raise
                        raise

            # Calculate evaluation time
            evaluation_time = time.time() - start_time

            # Extract result text
            if "choices" in response and len(response["choices"]) > 0 and "message" in response["choices"][0]:
                result_text = response["choices"][0]["message"].get("content", "")
            elif "error" in response:
                # Handle error response
                error_code = response.get("error", {}).get("code")
                error_message = response.get("error", {}).get("message", "Unknown error")

                if error_code == 429:
                    # This should have been caught by the retry logic, but just in case
                    logger.error(f"Rate limit error not caught by retry logic: {error_message}")
                    raise ValueError(f"Rate limit error: {error_message}")
                else:
                    # Other API error
                    logger.error(f"API error: {error_message} (code: {error_code})")
                    raise ValueError(f"API error: {error_message} (code: {error_code})")
            else:
                # Log the response for debugging
                logger.error(f"Unexpected API response format: {response}")
                raise ValueError(f"Invalid API response format: missing 'choices' or 'message' in response")

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
                from app.core.config import settings

                # Get the full model name from settings if available
                full_model_name = model_id
                if model_id in settings.AVAILABLE_MODELS.values():
                    # Model ID is already a full name
                    pass
                elif model_id in settings.AVAILABLE_MODELS:
                    # Convert short name to full name
                    full_model_name = settings.AVAILABLE_MODELS[model_id]

                # Calculate cost based on the number of images used
                total_images = len(all_image_data)
                if use_true_solution:
                    # Add the number of student solution images
                    total_images += len(student_image_data)

                cost = calculate_actual_cost(
                    model_id=full_model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    num_images=total_images  # Use the total number of images
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
                "use_true_solution": use_true_solution,
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
                "use_true_solution": use_true_solution,
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
        with_true_solution: bool = False,
        max_examples: Optional[int] = None,
        prompt_variant: str = "detailed",
        include_examples: bool = False,
        output_dir: Optional[str] = None
    ) -> Tuple[List[Dict], str]:
        """Run the benchmark on the dataset.

        Args:
            task_id: Optional task ID to filter the dataset
            model_ids: List of model IDs to benchmark
            with_answer: Whether to run with answer images
            without_answer: Whether to run without answer images
            with_true_solution: Whether to run with true solution images
            max_examples: Maximum number of examples to process
            prompt_variant: Prompt variant to use
            include_examples: Whether to include examples in prompts
            output_dir: Optional directory to save results (defaults to self.results_dir)

        Returns:
            Tuple of (results, filepath)
        """
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
                if with_answer and not with_true_solution:  # Skip if with_true_solution is True to avoid duplicate API calls
                    result = await self.evaluate_solution(
                        example=example,
                        model_id=model_id,
                        use_answer=True,
                        use_true_solution=False,
                        prompt_variant=prompt_variant,
                        include_examples=include_examples
                    )
                    results.append(result)

                # Evaluate without answer if requested
                if without_answer and not with_true_solution:  # Skip if with_true_solution is True to avoid duplicate API calls
                    result = await self.evaluate_solution(
                        example=example,
                        model_id=model_id,
                        use_answer=False,
                        use_true_solution=False,
                        prompt_variant=prompt_variant,
                        include_examples=include_examples
                    )
                    results.append(result)

                # Evaluate with true solution if requested
                if with_true_solution:
                    result = await self.evaluate_solution(
                        example=example,
                        model_id=model_id,
                        use_answer=False,
                        use_true_solution=True,
                        prompt_variant=prompt_variant,
                        include_examples=include_examples
                    )
                    results.append(result)

        # Create a unique folder for this run
        task_str = f"task{task_id}" if task_id else "all_tasks"
        models_str = "_".join([m.split("/")[-1].split(":")[0] for m in valid_model_ids])
        if len(models_str) > 30:  # Truncate if too long
            models_str = models_str[:27] + "..."

        # Create a timestamp for the run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Determine the approach name
        approach_name = ""
        if with_true_solution:
            approach_name = "true_solution"
        elif with_answer and without_answer:
            approach_name = "both_approaches"
        elif with_answer:
            approach_name = "with_answer"
        elif without_answer:
            approach_name = "without_answer"

        # Create folder name with approach
        folder_name = f"{task_str}_{models_str}_{approach_name}_{timestamp}"

        # Use the specified output directory if provided, otherwise use the default
        base_results_dir = output_dir if output_dir else self.results_dir
        run_results_dir = os.path.join(base_results_dir, folder_name)

        # Create the directory
        os.makedirs(run_results_dir, exist_ok=True)

        # Create the results filename
        filename = f"benchmark_{task_str}_{models_str}_{timestamp}.json"
        filepath = os.path.join(run_results_dir, filename)

        # Save the results
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved results to {filepath}")

        # Store the run directory in the object for later use
        self.current_run_dir = run_results_dir

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
            'task_16': 2,  # Fixed: Economic task has maximum 2 points
            'task_17': 3,
            'task_18': 4,
            'task_19': 4
        }

        # Calculate normalized distance
        df['max_score'] = df['task_type'].map(lambda x: max_scores.get(x, 4))  # Default to 4 if task_type not found

        # Add max_score to df_valid as well
        df_valid['max_score'] = df_valid['task_type'].map(lambda x: max_scores.get(x, 4))  # Default to 4 if task_type not found

        # Handle empty df_valid case with safe division
        if len(df_valid) > 0:
            # Create normalized_distance column in df_valid first with safe division
            df_valid['normalized_distance'] = df_valid.apply(
                lambda row: abs(row['score'] - row['expected_score']) / row['max_score']
                if row['max_score'] > 0 else 0.0, axis=1
            )

            # Then copy values to df where indices match
            df['normalized_distance'] = float('nan')  # Initialize with NaN
            for idx in df_valid.index:
                if idx in df.index:
                    df.at[idx, 'normalized_distance'] = df_valid.at[idx, 'normalized_distance']
        else:
            # If no valid scores, set all to NaN
            df['normalized_distance'] = float('nan')

        # Calculate quality score (1 - normalized_distance)
        # This gives a score from 0 to 1, where 1 is perfect prediction and 0 is worst possible prediction
        df['quality_score'] = 1 - df['normalized_distance']

        # Group by model, answer type, and true solution type
        grouped = df.groupby(['model_id', 'use_answer', 'use_true_solution'])

        # Helper function to convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        analysis = {
            "total_examples": convert_to_native(len(df['solution_id'].unique())),
            "total_evaluations": convert_to_native(len(df)),
            "models": {},
            "summary": {
                "avg_evaluation_time": convert_to_native(df['evaluation_time'].mean()),
                "avg_prompt_tokens": convert_to_native(df['prompt_tokens'].mean()),
                "avg_completion_tokens": convert_to_native(df['completion_tokens'].mean()),
                "avg_total_tokens": convert_to_native(df['total_tokens'].mean()),
                "total_cost": convert_to_native(df['cost'].sum()),
                "avg_quality_score": convert_to_native(df['quality_score'].mean() if 'quality_score' in df else None)
            },
            "task_type": df['task_type'].iloc[0] if 'task_type' in df.columns and len(df) > 0 else None
        }

        # Analyze each model
        for (model_id, use_answer, use_true_solution), group in grouped:
            if model_id not in analysis["models"]:
                analysis["models"][model_id] = {}

            # Determine the evaluation type
            if use_true_solution:
                answer_type = "with_true_solution"
            else:
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
                "accuracy": convert_to_native(accuracy),
                "quality_score": convert_to_native(avg_quality_score * 100 if avg_quality_score is not None else None),  # Convert to percentage
                "avg_score_distance": convert_to_native(avg_distance),
                "macro_precision": convert_to_native(macro_precision),
                "macro_recall": convert_to_native(macro_recall),
                "macro_f1": convert_to_native(macro_f1),
                "per_score_metrics": precision_recall_f1,  # Already contains native types
                "confusion_matrix": confusion_matrix,  # Already contains native types
                "evaluations": convert_to_native(len(group)),
                "avg_evaluation_time": convert_to_native(group['evaluation_time'].mean()),
                "avg_prompt_tokens": convert_to_native(group['prompt_tokens'].mean()),
                "avg_completion_tokens": convert_to_native(group['completion_tokens'].mean()),
                "avg_total_tokens": convert_to_native(group['total_tokens'].mean()),
                "total_cost": convert_to_native(group['cost'].sum())
            }

        return analysis

    def save_analysis(self, analysis: Dict, results_file: str) -> str:
        """Save analysis results to a file."""
        # Create filename based on results file
        analysis_file = results_file.replace('.json', '_analysis.json')

        # Ensure the directory exists
        os.makedirs(os.path.dirname(analysis_file), exist_ok=True)

        # Save analysis to file
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved analysis to {analysis_file}")

        return analysis_file

    def load_results(self, results_file: str) -> Tuple[List[Dict], str]:
        """Load benchmark results from a file and return the results and full path."""
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

        logger.info(f"Loaded {len(results)} results from {full_path}")
        return results, full_path

    def generate_metrics_latex_table(self, analysis: Dict) -> str:
        """Generate a LaTeX table with all metrics from the benchmark results."""
        latex_table = []

        # Get the model data (assuming only one model for simplicity)
        model_id = list(analysis["models"].keys())[0]
        model_data = analysis["models"][model_id]

        # Determine which approaches are used
        approaches = []
        if "with_answer" in model_data and any(v != "N/A" for v in model_data["with_answer"].values()):
            approaches.append("With Answer")
        if "without_answer" in model_data and any(v != "N/A" for v in model_data["without_answer"].values()):
            approaches.append("Without Answer")
        if "with_true_solution" in model_data and any(v != "N/A" for v in model_data["with_true_solution"].values()):
            approaches.append("With True Solution")

        # Try to get task ID from the analysis
        task_id = None

        # Check if task_type is directly in the analysis
        if "task_type" in analysis and analysis["task_type"]:
            task_type = analysis["task_type"]
            if task_type and isinstance(task_type, str) and task_type.startswith("task_"):
                task_id = task_type.split("_")[1]

        # If not found, check in the model data
        if not task_id:
            for approach, approach_data in model_data.items():
                if approach_data and isinstance(approach_data, dict):
                    # Check if there's a task_type field in the approach data
                    if "task_type" in approach_data:
                        task_type = approach_data.get("task_type")
                        if task_type and isinstance(task_type, str) and task_type.startswith("task_"):
                            task_id = task_type.split("_")[1]
                            break

        # Create caption with model name and approaches
        model_display_name = model_id.split("/")[-1] if "/" in model_id else model_id
        approaches_str = ", ".join(approaches)
        caption = f"Benchmark Results for Model: {model_display_name}"
        if task_id:
            caption += f", Task {task_id}"
        if approaches:
            caption += f" ({approaches_str})"

        # Start the table
        latex_table.append("\\begin{table}[htbp]")
        latex_table.append("\\centering")
        latex_table.append(f"\\caption{{{caption}}}")
        latex_table.append("\\begin{tabular}{lcc}")
        latex_table.append("\\toprule")
        latex_table.append("\\textbf{Metric} & \\textbf{With Answer} & \\textbf{Without Answer} \\\\")
        latex_table.append("\\midrule")

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
            with_true_solution_value = model_data.get("with_true_solution", {}).get(metric_key, "N/A")

            # Format values appropriately
            if isinstance(with_value, (int, float)) and metric_key != "evaluations":
                if metric_key == "total_cost":
                    with_value = f"{with_value:.2f}"
                    without_value = f"{without_value:.2f}"
                    with_true_solution_value = f"{with_true_solution_value:.2f}" if isinstance(with_true_solution_value, (int, float)) else with_true_solution_value
                elif metric_key == "avg_evaluation_time":
                    with_value = f"{with_value:.2f}"
                    without_value = f"{without_value:.2f}"
                    with_true_solution_value = f"{with_true_solution_value:.2f}" if isinstance(with_true_solution_value, (int, float)) else with_true_solution_value
                else:
                    with_value = f"{with_value:.2f}"
                    without_value = f"{without_value:.2f}"
                    with_true_solution_value = f"{with_true_solution_value:.2f}" if isinstance(with_true_solution_value, (int, float)) else with_true_solution_value

            # Check if we have with_true_solution data
            if with_true_solution_value != "N/A":
                # Update table header to include with_true_solution
                latex_table[3] = "\\begin{tabular}{lccc}"
                latex_table[5] = "\\textbf{Metric} & \\textbf{With Answer} & \\textbf{Without Answer} & \\textbf{With True Solution} \\\\"
                latex_table.append(f"{metric_name} & {with_value} & {without_value} & {with_true_solution_value} \\\\")
            else:
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
            "- 0\\%: Prediction off by the maximum possible error"
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

async def main():
    """Main function to run the benchmark or analyze existing results."""
    parser = argparse.ArgumentParser(description="Benchmark models on the Russian Math Exam Solutions dataset or analyze existing results")

    # Analysis-only mode arguments
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing results without running benchmark")
    parser.add_argument("--results-file", type=str, help="Path to existing results file to analyze (required with --analyze-only)")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX tables for metrics, explanations, and quality score")
    parser.add_argument("--metrics-latex", action="store_true", help="Generate LaTeX table for metrics only")
    parser.add_argument("--explanation-latex", action="store_true", help="Generate LaTeX table explaining metrics")
    parser.add_argument("--quality-latex", action="store_true", help="Generate LaTeX table explaining quality score")

    # Benchmark mode arguments
    parser.add_argument("--task", type=str, help="Task ID to benchmark (e.g., '13')")
    parser.add_argument("--models", type=str, nargs="+", help="Model IDs to benchmark")
    parser.add_argument("--with-answer", action="store_true", help="Run with answer images")
    parser.add_argument("--without-answer", action="store_true", help="Run without answer images")
    parser.add_argument("--with-true-solution", action="store_true", help="Run with true solution images")
    parser.add_argument("--max-examples", type=int, help="Maximum number of examples to process")
    parser.add_argument("--prompt-variant", type=str, default="detailed", help="Prompt variant to use")
    parser.add_argument("--include-examples", action="store_true", help="Include examples in prompts")
    parser.add_argument("--dataset-dir", type=str, default="dataset_benchmark_hf", help="Dataset directory")

    # Rate limit handling parameters
    parser.add_argument("--max-retries", type=int, default=10, help="Maximum number of retries for rate-limited requests")
    parser.add_argument("--initial-delay", type=int, default=15, help="Initial delay in seconds before retrying rate-limited requests")
    parser.add_argument("--max-delay", type=int, default=120, help="Maximum delay in seconds before retrying rate-limited requests")

    args = parser.parse_args()

    # Create benchmark instance
    benchmark = ModelBenchmark(
        dataset_dir=args.dataset_dir,
        max_retries=args.max_retries,
        initial_delay=args.initial_delay,
        max_delay=args.max_delay
    )

    try:
        # Analysis-only mode
        if args.analyze_only:
            if not args.results_file:
                print("Error: --results-file is required with --analyze-only")
                return

            # Load and analyze existing results
            results, full_path = benchmark.load_results(args.results_file)
            analysis = benchmark.analyze_results(results)
            # Add filename to analysis for reference
            analysis["filename"] = os.path.basename(full_path)
            analysis_file = benchmark.save_analysis(analysis, full_path)

            # Generate LaTeX tables if requested
            if args.latex or args.metrics_latex:
                print("\nGenerating metrics LaTeX table...")
                metrics_table = benchmark.generate_metrics_latex_table(analysis)
                metrics_table_file = full_path.replace('.json', '_metrics_table.tex')
                with open(metrics_table_file, 'w', encoding='utf-8') as f:
                    f.write(metrics_table)
                print(f"Saved metrics table to {metrics_table_file}")

                if args.latex:
                    print("\n=== METRICS TABLE ===")
                    print(metrics_table)

            if args.latex or args.explanation_latex:
                print("\nGenerating explanation LaTeX table...")
                explanation_table = benchmark.generate_explanation_latex_table()
                explanation_table_file = full_path.replace('.json', '_explanation_table.tex')
                with open(explanation_table_file, 'w', encoding='utf-8') as f:
                    f.write(explanation_table)
                print(f"Saved explanation table to {explanation_table_file}")

                if args.latex:
                    print("\n=== EXPLANATION TABLE ===")
                    print(explanation_table)

            if args.latex or args.quality_latex:
                print("\nGenerating quality score explanation LaTeX table...")
                quality_table = benchmark.generate_quality_score_explanation_latex_table()
                quality_table_file = full_path.replace('.json', '_quality_explanation_table.tex')
                with open(quality_table_file, 'w', encoding='utf-8') as f:
                    f.write(quality_table)
                print(f"Saved quality score explanation table to {quality_table_file}")

                if args.latex:
                    print("\n=== QUALITY SCORE EXPLANATION TABLE ===")
                    print(quality_table)

        # Benchmark mode
        else:
            # Validate required arguments for benchmark mode
            if not args.task or not args.models:
                print("Error: --task and --models are required for benchmark mode")
                return

            # Default to both with and without answer if neither is specified and not using true solution
            if not args.with_answer and not args.without_answer and not args.with_true_solution:
                args.with_answer = True
                args.without_answer = True

            # If using true solution, disable with_answer and without_answer to avoid duplicate API calls
            if args.with_true_solution:
                if args.with_answer or args.without_answer:
                    logger.info("Using --with-true-solution flag, disabling --with-answer and --without-answer to avoid duplicate API calls")
                args.with_answer = False
                args.without_answer = False

            # Run benchmark
            results, results_file = await benchmark.run_benchmark(
                task_id=args.task,
                model_ids=args.models,
                with_answer=args.with_answer,
                without_answer=args.without_answer,
                with_true_solution=args.with_true_solution,
                max_examples=args.max_examples,
                prompt_variant=args.prompt_variant,
                include_examples=args.include_examples
            )

            # Analyze results
            analysis = benchmark.analyze_results(results)
            # Add filename to analysis for reference
            analysis["filename"] = os.path.basename(results_file)
            analysis_file = benchmark.save_analysis(analysis, results_file)

            # Generate LaTeX tables if requested
            if args.latex or args.metrics_latex:
                print("\nGenerating metrics LaTeX table...")
                metrics_table = benchmark.generate_metrics_latex_table(analysis)
                metrics_table_file = results_file.replace('.json', '_metrics_table.tex')
                with open(metrics_table_file, 'w', encoding='utf-8') as f:
                    f.write(metrics_table)
                print(f"Saved metrics table to {metrics_table_file}")

            if args.latex or args.explanation_latex:
                print("\nGenerating explanation LaTeX table...")
                explanation_table = benchmark.generate_explanation_latex_table()
                explanation_table_file = results_file.replace('.json', '_explanation_table.tex')
                with open(explanation_table_file, 'w', encoding='utf-8') as f:
                    f.write(explanation_table)
                print(f"Saved explanation table to {explanation_table_file}")

            if args.latex or args.quality_latex:
                print("\nGenerating quality score explanation LaTeX table...")
                quality_table = benchmark.generate_quality_score_explanation_latex_table()
                quality_table_file = results_file.replace('.json', '_quality_explanation_table.tex')
                with open(quality_table_file, 'w', encoding='utf-8') as f:
                    f.write(quality_table)
                print(f"Saved quality score explanation table to {quality_table_file}")

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

        if args.analyze_only:
            print(f"\nAnalysis saved to: {analysis_file}")
        else:
            print(f"\nDetailed results saved to: {results_file}")
            print(f"Analysis saved to: {analysis_file}")

    finally:
        if 'benchmark' in locals():
            await benchmark.close()

if __name__ == "__main__":
    asyncio.run(main())
