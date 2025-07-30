"""
Models Router Module

This module provides API endpoints for retrieving information about available models.
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.openrouter_client import OpenRouterClient
from app.core.config import settings
from app.routers.solutions import get_openrouter_client
from app.utils.cost_calculator import calculate_dataset_cost, compare_model_costs

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.get("/available")
async def get_available_models(category: Optional[str] = Query(None, description="Filter models by category: 'reasoning' or 'non_reasoning'")):
    """
    Get a list of available models for solution evaluation.

    Args:
        category: Optional filter for model category

    Returns:
        Dictionary with available models
    """
    if category == "reasoning":
        models_dict = settings.REASONING_MODELS
        category_name = "Reasoning Models"
    elif category == "non_reasoning":
        models_dict = settings.NON_REASONING_MODELS
        category_name = "Non-Reasoning Models"
    else:
        models_dict = settings.AVAILABLE_MODELS
        category_name = "All Models"

    return {
        "category": category_name,
        "models": [
            {
                "id": model_id,
                "name": model_id.replace("-", " ").title(),
                "full_name": model_name
            }
            for model_id, model_name in models_dict.items()
        ]
    }


@router.get("/task-types")
async def get_task_types():
    """
    Get a list of available task types.

    Returns:
        Dictionary with available task types
    """
    return {
        "task_types": [
            {
                "id": task_id,
                "name": task_name
            }
            for task_id, task_name in settings.TASK_TYPES.items()
        ]
    }


@router.get("/openrouter-models")
async def get_openrouter_models(client: OpenRouterClient = Depends(get_openrouter_client)):
    """
    Get a list of all available models from OpenRouter.

    Args:
        client: OpenRouter client instance

    Returns:
        List of models from OpenRouter
    """
    try:
        models = await client.list_models()
        return models
    except Exception as e:
        logger.error(f"Error fetching models from OpenRouter: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")


@router.get("/credits")
async def get_credits(client: OpenRouterClient = Depends(get_openrouter_client)):
    """
    Get account credits information from OpenRouter.

    Args:
        client: OpenRouter client instance

    Returns:
        Credit information
    """
    try:
        credits = await client.get_credits()
        return credits
    except Exception as e:
        logger.error(f"Error fetching credits from OpenRouter: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching credits: {str(e)}")


@router.get("/cost-estimate/{model_id}")
async def get_cost_estimate(model_id: str):
    """
    Get an estimate of the cost for evaluating the entire dataset with a specific model.

    Args:
        model_id: ID of the model to use

    Returns:
        Cost estimate information
    """
    try:
        # Check if the model exists
        if model_id not in settings.AVAILABLE_MODELS:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        # Calculate the cost
        cost_info = calculate_dataset_cost(model_id)
        return cost_info
    except ValueError as e:
        logger.error(f"Error calculating cost: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get("/compare-costs")
async def compare_costs():
    """
    Compare the costs of using different models for the entire dataset.

    Returns:
        List of cost information for each model
    """
    try:
        cost_comparison = compare_model_costs()
        return {"models": cost_comparison}
    except Exception as e:
        logger.error(f"Error comparing costs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing costs: {str(e)}")
