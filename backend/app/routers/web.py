"""
Web Router Module

This module provides routes for the web interface.
"""

import os
import logging
from fastapi import APIRouter, Request, Form, File, UploadFile, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Optional, List

from app.core.config import settings
from app.api.openrouter_client import OpenRouterClient
from app.routers.solutions import get_openrouter_client

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Set up templates
templates_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")
templates = Jinja2Templates(directory=templates_dir)

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Render the main page.
    
    Args:
        request: Request object
        
    Returns:
        HTML response with the main page
    """
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "task_types": settings.TASK_TYPES,
            "reasoning_models": settings.REASONING_MODELS,
            "non_reasoning_models": settings.NON_REASONING_MODELS
        }
    )

@router.get("/batch", response_class=HTMLResponse)
async def batch_page(request: Request):
    """
    Render the batch processing page.
    
    Args:
        request: Request object
        
    Returns:
        HTML response with the batch processing page
    """
    return templates.TemplateResponse(
        "batch.html", 
        {
            "request": request,
            "task_types": settings.TASK_TYPES,
            "reasoning_models": settings.REASONING_MODELS,
            "non_reasoning_models": settings.NON_REASONING_MODELS
        }
    )
