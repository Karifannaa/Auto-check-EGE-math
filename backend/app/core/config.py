"""
Configuration Module

This module provides configuration settings for the application.
"""

import os
from typing import Dict, List, Optional, Any
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "EGE Math Solution Checker"

    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    # OpenRouter API settings
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    SITE_URL: Optional[str] = os.getenv("SITE_URL", "http://localhost:3000")
    SITE_NAME: Optional[str] = os.getenv("SITE_NAME", "EGE Math Solution Checker")

    # Model settings
    DEFAULT_MODEL: str = "openai/gpt-4o"

    # Models categorized by type
    REASONING_MODELS: Dict[str, str] = {
        "o4-mini-high": "openai/o4-mini-high",
        "o4-mini": "openai/o4-mini",
        "o3": "openai/o3",
        "gemini-2.5-pro-preview": "google/gemini-2.5-pro-preview",
        "gemini-2.5-pro-exp": "google/gemini-2.5-pro-exp-03-25",
        "gemini-2.5-flash-preview-thinking": "google/gemini-2.5-flash-preview-thinking",
        "gemini-2.5-flash-preview-thinking-free": "google/gemini-2.5-flash-preview:thinking",
        "claude-3.7-sonnet-thinking": "anthropic/claude-3.7-sonnet-thinking",
        "kimi-vl-a3b-thinking": "moonshotai/kimi-vl-a3b-thinking:free"
    }

    NON_REASONING_MODELS: Dict[str, str] = {
        "gpt-4o": "openai/gpt-4o-2024-11-20",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "gpt-4.1": "openai/gpt-4.1",
        "gpt-4.1-mini": "openai/gpt-4.1-mini",
        "gpt-4.1-nano": "openai/gpt-4.1-nano",
        "gemini-2.5-flash-preview": "google/gemini-2.5-flash-preview",
        "gemini-2.5-flash-preview-thinking": "google/gemini-2.5-flash-preview:thinking",
        "gemini-2.0-flash": "google/gemini-2.0-flash",
        "gemini-2.0-flash-001": "google/gemini-2.0-flash-001",
        "gemini-2.0-flash-lite": "google/gemini-2.0-flash-lite-001",
        "gemini-2.0-flash-exp-free": "google/gemini-2.0-flash-exp:free",
        "gemini-1.5-flash": "google/gemini-1.5-flash",
        "gemma-3-27b": "google/gemma-3-27b-it",
        "gemma-3-27b-free": "google/gemma-3-27b-it:free",
        "gemma-3-12b": "google/gemma-3-12b-it",
        "gemma-3-12b-free": "google/gemma-3-12b-it:free",
        "gemma-3-4b": "google/gemma-3-4b-it",
        "gemma-3-4b-free": "google/gemma-3-4b-it:free",
        "gemma-3-1b": "google/gemma-3-1b-it:free",
        "llama-4-maverick": "meta-llama/llama-4-maverick",
        "llama-4-maverick-free": "meta-llama/llama-4-maverick:free",
        "llama-4-scout": "meta-llama/llama-4-scout",
        "llama-4-scout-free": "meta-llama/llama-4-scout:free",
        "qwen-2.5-vl-32b": "qwen/qwen2.5-vl-32b-instruct",
        "qwen-2.5-vl-32b-free": "qwen/qwen2.5-vl-32b-instruct:free",
        "mistral-small-3.1-24b": "mistralai/mistral-small-3.1-24b-instruct",
        "mistral-small-3.1-24b-free": "mistralai/mistral-small-3.1-24b-instruct:free",
        "claude-3.7-sonnet": "anthropic/claude-3.7-sonnet",
        "phi-4-multimodal": "microsoft/phi-4-multimodal-instruct",
        "arcee-spotlight": "arcee-ai/spotlight"
    }

    # Combined models for API use
    AVAILABLE_MODELS: Dict[str, str] = {**REASONING_MODELS, **NON_REASONING_MODELS}

    # Task types
    TASK_TYPES: Dict[str, str] = {
        "task_13": "Trigonometric, logarithmic or exponential equation",
        "task_14": "Stereometric problem",
        "task_15": "Inequality",
        "task_16": "Planimetric problem",
        "task_17": "Planimetric problem with proof",
        "task_18": "Problem with parameters",
        "task_19": "Number theory problem"
    }

    # Image settings
    MAX_IMAGE_SIZE: int = 4096
    ENHANCE_IMAGES: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()
