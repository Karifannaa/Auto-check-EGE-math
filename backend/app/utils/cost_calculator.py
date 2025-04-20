"""
Cost Calculator Module

This module provides utilities for calculating the cost of using different models
for evaluating math solutions.
"""

from typing import Dict, List, Any, Tuple
from app.core.config import settings


# Actual pricing per 1M tokens (in USD) for different models based on OpenRouter documentation
# Prices are current as of April 2025 and may change over time
MODEL_PRICING = {
    # Reasoning models
    "openai/o4-mini-high": {"input": 1.10, "output": 4.40, "image": 0.842},
    "openai/o4-mini": {"input": 1.10, "output": 4.40, "image": 0.842},
    "openai/o3": {"input": 10.0, "output": 40.0, "image": 7.65},
    "google/gemini-2.5-pro-preview": {"input": 3.0, "output": 10.0, "image": 3.0},  # Estimated based on similar models
    "google/gemini-2.5-flash-preview-thinking": {"input": 0.15, "output": 1.20, "image": 0.619},  # Higher output for thinking
    "anthropic/claude-3.7-sonnet-thinking": {"input": 3.0, "output": 30.0, "image": 4.80},  # Higher output for thinking

    # Non-reasoning models
    "openai/gpt-4o-2024-11-20": {"input": 2.50, "output": 10.0, "image": 3.613},
    "openai/gpt-4o-mini": {"input": 0.50, "output": 1.50, "image": 0.50},  # Estimated
    "google/gemini-2.5-flash-preview": {"input": 0.15, "output": 0.60, "image": 0.619},
    "google/gemini-2.0-flash": {"input": 0.125, "output": 0.375, "image": 0.50},  # Estimated
    "google/gemini-1.5-flash": {"input": 0.10, "output": 0.30, "image": 0.40},  # Estimated
    "meta/llama-4-maverick": {"input": 0.50, "output": 1.50, "image": 0.50},  # Estimated
    "meta/llama-4-scout": {"input": 0.15, "output": 0.45, "image": 0.40},  # Estimated
    "qwen/qwen2.5-vl-32b-instruct": {"input": 0.50, "output": 1.50, "image": 0.50},  # Estimated
    "mistral/mistral-small-3.1-24b": {"input": 0.20, "output": 0.60, "image": 0.00},  # No image support
    "anthropic/claude-3.7-sonnet": {"input": 3.0, "output": 15.0, "image": 4.80},
    "google/gemma-3-27b": {"input": 0.25, "output": 0.75, "image": 0.00},  # No image support
    "microsoft/phi-4-multimodal-instruct": {"input": 0.20, "output": 0.60, "image": 0.40}  # Estimated
}

# Average token counts for different components
AVG_TOKENS = {
    "system_message": 150,
    "task_description": 200,
    "solution_analysis": 500  # Average output tokens for solution analysis
}

# Average image processing costs
AVG_IMAGE = {
    "images_per_solution": 1,  # Typically one image per solution
    "image_size": "medium"  # Medium complexity/resolution image
}


def estimate_tokens_per_solution(task_type: str) -> Tuple[int, int]:
    """
    Estimate the number of tokens needed for processing a single solution.

    Args:
        task_type: Type of task (e.g., "task_13", "task_17")

    Returns:
        Tuple of (input_tokens, output_tokens)
    """
    # Base token count for system message
    input_tokens = AVG_TOKENS["system_message"]

    # Add task description tokens
    input_tokens += AVG_TOKENS["task_description"]

    # Adjust based on task type (some tasks may require more tokens)
    if task_type in ["task_17", "task_18", "task_19"]:
        input_tokens += 100  # These tasks typically have more complex descriptions

    # Estimate output tokens
    output_tokens = AVG_TOKENS["solution_analysis"]

    # Adjust based on task type (some tasks may require more detailed analysis)
    if task_type in ["task_17", "task_18", "task_19"]:
        output_tokens += 200  # These tasks typically require more detailed analysis

    return input_tokens, output_tokens


def calculate_cost_per_solution(model_id: str, task_type: str) -> float:
    """
    Calculate the estimated cost for evaluating a single solution.

    Args:
        model_id: ID of the model to use
        task_type: Type of task

    Returns:
        Estimated cost in USD
    """
    # Get the full model name
    if model_id in settings.REASONING_MODELS:
        model_name = settings.REASONING_MODELS[model_id]
    elif model_id in settings.NON_REASONING_MODELS:
        model_name = settings.NON_REASONING_MODELS[model_id]
    else:
        raise ValueError(f"Unknown model ID: {model_id}")

    # Get pricing for the model
    if model_name not in MODEL_PRICING:
        raise ValueError(f"Pricing information not available for model: {model_name}")

    pricing = MODEL_PRICING[model_name]

    # Estimate token counts
    input_tokens, output_tokens = estimate_tokens_per_solution(task_type)

    # Calculate token costs
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    # Calculate image cost if the model supports images
    image_cost = 0.0
    if "image" in pricing and pricing["image"] > 0:
        # Cost per image in USD
        image_cost = AVG_IMAGE["images_per_solution"] * (pricing["image"] / 1000)

    # Total cost
    total_cost = input_cost + output_cost + image_cost

    return total_cost


def calculate_dataset_cost(model_id: str) -> Dict[str, Any]:
    """
    Calculate the estimated cost for evaluating the entire dataset.

    Args:
        model_id: ID of the model to use

    Returns:
        Dictionary with cost information
    """
    # Dataset structure: 7 task types, 5 examples per task, ~3 solutions per example
    task_counts = {
        task_type: {"examples": 5, "solutions_per_example": 3}
        for task_type in settings.TASK_TYPES.keys()
    }

    total_cost = 0.0
    total_token_cost = 0.0
    total_image_cost = 0.0
    cost_breakdown = {}
    total_solutions = 0

    # Get the full model name
    if model_id in settings.REASONING_MODELS:
        model_name = settings.REASONING_MODELS[model_id]
    elif model_id in settings.NON_REASONING_MODELS:
        model_name = settings.NON_REASONING_MODELS[model_id]
    else:
        raise ValueError(f"Unknown model ID: {model_id}")

    # Get pricing for the model
    if model_name not in MODEL_PRICING:
        raise ValueError(f"Pricing information not available for model: {model_name}")

    pricing = MODEL_PRICING[model_name]
    supports_images = "image" in pricing and pricing["image"] > 0

    # Calculate cost for each task type
    for task_type, counts in task_counts.items():
        num_examples = counts["examples"]
        solutions_per_example = counts["solutions_per_example"]
        total_solutions_for_task = num_examples * solutions_per_example

        # Calculate cost per solution
        cost_per_solution = calculate_cost_per_solution(model_id, task_type)
        task_cost = cost_per_solution * total_solutions_for_task

        # Calculate token and image costs separately
        input_tokens, output_tokens = estimate_tokens_per_solution(task_type)
        token_cost = ((input_tokens / 1_000_000) * pricing["input"] +
                     (output_tokens / 1_000_000) * pricing["output"]) * total_solutions_for_task

        image_cost = 0.0
        if supports_images:
            image_cost = (AVG_IMAGE["images_per_solution"] * (pricing["image"] / 1000)) * total_solutions_for_task

        cost_breakdown[task_type] = {
            "solutions": total_solutions_for_task,
            "cost_per_solution": cost_per_solution,
            "token_cost": token_cost,
            "image_cost": image_cost,
            "total_cost": task_cost
        }

        total_cost += task_cost
        total_token_cost += token_cost
        total_image_cost += image_cost
        total_solutions += total_solutions_for_task

    return {
        "model_id": model_id,
        "model_name": model_name,
        "total_solutions": total_solutions,
        "total_cost": total_cost,
        "total_token_cost": total_token_cost,
        "total_image_cost": total_image_cost,
        "average_cost_per_solution": total_cost / total_solutions if total_solutions > 0 else 0,
        "supports_images": supports_images,
        "breakdown": cost_breakdown
    }


def compare_model_costs() -> List[Dict[str, Any]]:
    """
    Compare the costs of using different models for the entire dataset.

    Returns:
        List of dictionaries with cost information for each model
    """
    results = []

    # Process reasoning models
    for model_id in settings.REASONING_MODELS:
        try:
            cost_info = calculate_dataset_cost(model_id)
            cost_info["category"] = "reasoning"
            results.append(cost_info)
        except ValueError:
            # Skip models with missing pricing information
            continue

    # Process non-reasoning models
    for model_id in settings.NON_REASONING_MODELS:
        try:
            cost_info = calculate_dataset_cost(model_id)
            cost_info["category"] = "non_reasoning"
            results.append(cost_info)
        except ValueError:
            # Skip models with missing pricing information
            continue

    # Sort by total cost
    results.sort(key=lambda x: x["total_cost"])

    return results
