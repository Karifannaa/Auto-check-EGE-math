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
from app.utils.cost_calculator import calculate_cost_per_solution, calculate_actual_cost

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
    task_description: str = Form(""),
    model_id: str = Form(...),
    student_solution_image: UploadFile = File(...),
    correct_solution_image: Optional[UploadFile] = File(None),
    include_examples: bool = Form(False),
    prompt_variant: Optional[str] = Form(None),
    temperature: float = Form(0.7),
    max_tokens: Optional[int] = Form(10000),
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
    # Log request details for debugging
    logger.info(f"Received evaluation request - Task type: {task_type}, Model ID: {model_id}")
    logger.info(f"Student image filename: {student_solution_image.filename if student_solution_image else 'None'}")
    logger.info(f"Correct solution image: {correct_solution_image.filename if correct_solution_image else 'None'}")

    # Validate task type
    if task_type not in settings.TASK_TYPES:
        logger.error(f"Invalid task type: {task_type}")
        raise HTTPException(status_code=400, detail=f"Invalid task type: {task_type}")

    # Validate model ID and check if it supports image processing
    if model_id in settings.REASONING_MODELS:
        model_name = settings.REASONING_MODELS[model_id]
        logger.info(f"Using reasoning model: {model_name}")
    elif model_id in settings.NON_REASONING_MODELS:
        # Check if the non-reasoning model supports image processing
        # This is a simplified check - in a real system, you'd want a more robust way to determine this
        image_capable_models = [
            "gpt-4o", "gpt-4o-mini", "gemini-2.5-flash-preview", "gemini-2.0-flash",
            "gemini-1.5-flash", "qwen-2.5-vl-32b", "phi-4-multimodal"
        ]
        if model_id not in image_capable_models:
            logger.error(f"Model {model_id} does not support image processing")
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_id} does not support image processing. Please choose a reasoning model or an image-capable non-reasoning model."
            )
        model_name = settings.NON_REASONING_MODELS[model_id]
        logger.info(f"Using non-reasoning model: {model_name}")
    else:
        logger.error(f"Invalid model ID: {model_id}")
        raise HTTPException(status_code=400, detail=f"Invalid model ID: {model_id}")

    try:
        # Read and process the student solution image
        student_image_content = await student_solution_image.read()
        try:
            student_image = Image.open(io.BytesIO(student_image_content))
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid student solution image format")

        # Prepare the student solution image for the API
        student_image_data = prepare_image_for_api(
            student_image,
            resize=True,
            enhance=settings.ENHANCE_IMAGES,
            max_size=settings.MAX_IMAGE_SIZE
        )

        # Process correct solution image if provided
        correct_image_data = None
        if correct_solution_image:
            correct_image_content = await correct_solution_image.read()
            try:
                correct_image = Image.open(io.BytesIO(correct_image_content))
                correct_image_data = prepare_image_for_api(
                    correct_image,
                    resize=True,
                    enhance=settings.ENHANCE_IMAGES,
                    max_size=settings.MAX_IMAGE_SIZE
                )
            except UnidentifiedImageError:
                raise HTTPException(status_code=400, detail="Invalid correct solution image format")

        # Create messages for the API
        messages = prompt_generator.create_messages_with_image(
            task_type=task_type,
            task_description=task_description,
            student_solution_image=student_image_data,
            correct_solution_image=correct_image_data,
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

        # Parse the score from the result with improved robustness and semantic section detection
        score = 0

        try:
            import re

            # First approach: Semantic section detection - look for the "Итоговая оценка" section
            logger.info("Attempting to extract score using semantic section detection")

            # Split the text into sections by markdown headers
            sections = re.split(r'###\s+', result_text)

            # Look for the "Итоговая оценка" section
            score_section = None
            for section in sections:
                if section.strip().lower().startswith('итоговая оценка'):
                    score_section = section.strip()
                    logger.info(f"Found 'Итоговая оценка' section: {score_section[:100]}...")
                    break

            # If we found the section, extract the score from it
            if score_section:
                # Look for the pattern [Оценка: X баллов] or similar
                score_patterns = [
                    r'\[оценка:\s*(\d+)\s*балл',  # [Оценка: 2 балла]
                    r'оценка:\s*(\d+)\s*балл',    # Оценка: 2 балла
                    r'(\d+)\s*балл'               # 2 балла
                ]

                for pattern in score_patterns:
                    matches = re.findall(pattern, score_section.lower())
                    if matches:
                        score = int(matches[0])  # Use the first match in the section
                        logger.info(f"Found score {score} in 'Итоговая оценка' section using pattern: {pattern}")
                        break

            # Second approach: If semantic section detection failed, fall back to traditional methods
            if score == 0:
                logger.info("Semantic section detection failed, falling back to traditional methods")

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

                # Look for lines with "оценка" and extract digits
                if score == 0:
                    for line in result_text.split("\n"):
                        if "оценка" in line.lower() or "балл" in line.lower():
                            # Extract all digits from the line
                            digits = [int(s) for s in re.findall(r'\d+', line)]
                            if digits:
                                score = digits[-1]  # Use the last digit as the score
                                logger.info(f"Found score {score} from line: {line}")
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

                # Most aggressive - find any number after "оценка"
                if score == 0:
                    # Find any occurrence of "оценка" followed by a number within 30 characters
                    score_sections = re.findall(r'оценка.{1,30}?(\d+)', result_text.lower())
                    if score_sections:
                        score = int(score_sections[-1])
                        logger.info(f"Found score {score} using aggressive approach")

                # Check for specific score mentions in the text
                if score == 0:
                    if "2 балла" in result_text.lower() or "два балла" in result_text.lower():
                        score = 2
                        logger.info("Found score 2 from direct text mention")
                    elif "1 балл" in result_text.lower() or "один балл" in result_text.lower():
                        score = 1
                        logger.info("Found score 1 from direct text mention")
                    elif "0 баллов" in result_text.lower() or "ноль баллов" in result_text.lower():
                        score = 0
                        logger.info("Found score 0 from direct text mention")

            # Validate the score is within expected range for task 13
            if task_type == "task_13" and score > 2:
                logger.warning(f"Score {score} exceeds maximum of 2 for task_13, capping at 2")
                score = 2

        except Exception as e:
            logger.error(f"Error extracting score: {str(e)}")
            # Default to 0 if we couldn't extract a score
            score = 0

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

        # Calculate the cost of this evaluation based on actual token usage
        try:
            # If we have actual token usage data, use it for more accurate cost calculation
            if prompt_tokens is not None and completion_tokens is not None:
                # Calculate number of images (1 or 2 depending on whether correct solution was provided)
                num_images = 1 if correct_solution_image is None else 2

                # Calculate actual cost based on token usage from API response
                cost = calculate_actual_cost(
                    model_id=model_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    num_images=num_images
                )
                logger.info(f"Calculated actual cost: ${cost:.6f} based on {prompt_tokens} prompt tokens and {completion_tokens} completion tokens")
            else:
                # Fall back to estimation if token usage data is not available
                cost = calculate_cost_per_solution(model_id, task_type)
                logger.info(f"Using estimated cost: ${cost:.6f} (token usage data not available)")
        except ValueError as e:
            logger.warning(f"Error calculating cost: {str(e)}")
            cost = 0.0

        # Create the solution evaluation
        solution_evaluation = SolutionEvaluation(
            id=str(uuid.uuid4()),
            task_type=task_type,
            task_description=task_description,
            model_id=model_id,
            result=evaluation_result,
            estimated_cost=calculate_cost_per_solution(model_id, task_type),
            actual_cost=cost  # Use the calculated actual cost based on token usage
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
        logger.error(f"Error details: {type(e).__name__}, {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
