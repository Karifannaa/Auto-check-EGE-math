"""
Solutions Router Module

This module provides API endpoints for evaluating math solutions.
"""

import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import httpx
from PIL import Image, UnidentifiedImageError
import io

from app.api.openrouter_client import OpenRouterClient
from app.core.config import settings
from app.models.solution import SolutionRequest, SolutionEvaluation, EvaluationResult
from app.utils.image_utils import load_and_prepare_image, prepare_image_for_api
from app.utils.prompt_utils import PromptGenerator
from app.utils.cost_calculator import calculate_cost_per_solution

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Create prompt generator
prompt_generator = PromptGenerator()


async def get_openrouter_client() -> OpenRouterClient:
    """
    Get an instance of the OpenRouter client.

    Returns:
        OpenRouterClient instance
    """
    if not settings.OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")

    client = OpenRouterClient(
        api_key=settings.OPENROUTER_API_KEY,
        site_url=settings.SITE_URL,
        site_name=settings.SITE_NAME
    )

    try:
        yield client
    finally:
        await client.close()


@router.post("/evaluate", response_model=SolutionEvaluation)
async def evaluate_solution(
    task_type: str = Form(...),
    task_description: str = Form(...),
    model_id: str = Form(...),
    solution_image: UploadFile = File(...),
    include_examples: bool = Form(False),
    prompt_variant: Optional[str] = Form(None),
    temperature: float = Form(0.7),
    max_tokens: Optional[int] = Form(None),
    client: OpenRouterClient = Depends(get_openrouter_client)
):
    """
    Evaluate a math solution using a reasoning model.

    Args:
        task_type: Type of task (e.g., "task_13", "task_17")
        task_description: Description of the task
        model_id: ID of the model to use for evaluation
        solution_image: Image file containing the solution
        include_examples: Whether to include examples in the prompt
        prompt_variant: Specific prompt variant to use (e.g., "basic", "detailed")
        temperature: Sampling temperature for the model
        max_tokens: Maximum number of tokens to generate
        client: OpenRouter client instance

    Returns:
        Evaluation result
    """
    # Validate task type
    if task_type not in settings.TASK_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid task type: {task_type}")

    # Validate model ID and check if it supports image processing
    if model_id in settings.REASONING_MODELS:
        model_name = settings.REASONING_MODELS[model_id]
    elif model_id in settings.NON_REASONING_MODELS:
        # Check if the non-reasoning model supports image processing
        # This is a simplified check - in a real system, you'd want a more robust way to determine this
        image_capable_models = [
            "gpt-4o", "gpt-4o-mini", "gemini-2.5-flash-preview", "gemini-2.0-flash",
            "gemini-1.5-flash", "qwen-2.5-vl-32b", "phi-4-multimodal"
        ]
        if model_id not in image_capable_models:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_id} does not support image processing. Please choose a reasoning model or an image-capable non-reasoning model."
            )
        model_name = settings.NON_REASONING_MODELS[model_id]
    else:
        raise HTTPException(status_code=400, detail=f"Invalid model ID: {model_id}")

    try:
        # Read and process the image
        image_content = await solution_image.read()
        try:
            image = Image.open(io.BytesIO(image_content))
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Prepare the image for the API
        image_data = prepare_image_for_api(
            image,
            resize=True,
            enhance=settings.ENHANCE_IMAGES,
            max_size=settings.MAX_IMAGE_SIZE
        )

        # Create messages for the API
        messages = prompt_generator.create_messages_with_image(
            task_type=task_type,
            task_description=task_description,
            image_data=image_data,
            include_examples=include_examples,
            examples=None,  # We're not including examples for now
            prompt_variant=prompt_variant
        )

        # Start timing
        start_time = time.time()

        # Call the API
        response = await client.chat_completion(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Calculate evaluation time
        evaluation_time = time.time() - start_time

        # Extract the result
        result_text = response["choices"][0]["message"]["content"]

        # Extract usage information if available
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        # Parse the score from the result
        # This is a simple implementation - in a real system, you'd want more robust parsing
        score = 0
        for line in result_text.split("\n"):
            if "итоговая оценка" in line.lower() or "итоговый балл" in line.lower():
                # Try to extract the score
                try:
                    # Look for digits in the line
                    digits = [int(s) for s in line.split() if s.isdigit()]
                    if digits:
                        score = digits[0]
                        break
                except Exception as e:
                    logger.warning(f"Failed to extract score from line: {line}, error: {str(e)}")

        # Create the evaluation result
        evaluation_result = EvaluationResult(
            score=score,
            explanation=result_text,
            model_id=model_id,
            task_type=task_type,
            evaluation_time=evaluation_time,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )

        # Calculate the cost of this evaluation
        try:
            cost = calculate_cost_per_solution(model_id, task_type)
        except ValueError:
            cost = 0.0

        # Create the solution evaluation
        solution_evaluation = SolutionEvaluation(
            id=str(uuid.uuid4()),
            task_type=task_type,
            task_description=task_description,
            model_id=model_id,
            result=evaluation_result,
            estimated_cost=cost
        )

        return solution_evaluation

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        logger.error(f"Request error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
