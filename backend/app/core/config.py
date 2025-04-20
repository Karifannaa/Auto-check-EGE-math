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
        "gemini-2.5-flash-preview-thinking": "google/gemini-2.5-flash-preview-thinking",
        "claude-3.7-sonnet-thinking": "anthropic/claude-3.7-sonnet-thinking"
    }

    NON_REASONING_MODELS: Dict[str, str] = {
        "gpt-4o": "openai/gpt-4o-2024-11-20",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "gemini-2.5-flash-preview": "google/gemini-2.5-flash-preview",
        "gemini-2.0-flash": "google/gemini-2.0-flash",
        "gemini-1.5-flash": "google/gemini-1.5-flash",
        "llama-4-maverick": "meta/llama-4-maverick",
        "llama-4-scout": "meta/llama-4-scout",
        "qwen-2.5-vl-32b": "qwen/qwen2.5-vl-32b-instruct",
        "mistral-small-3.1-24b": "mistral/mistral-small-3.1-24b",
        "claude-3.7-sonnet": "anthropic/claude-3.7-sonnet",
        "gemma-3-27b": "google/gemma-3-27b",
        "phi-4-multimodal": "microsoft/phi-4-multimodal-instruct"
    }

    # Combined models for API use
    AVAILABLE_MODELS: Dict[str, str] = {**REASONING_MODELS, **NON_REASONING_MODELS}

    # Task types
    TASK_TYPES: Dict[str, str] = {
        "task_13": "Тригонометрическое, логарифмическое или показательное уравнение",
        "task_14": "Стереометрическая задача",
        "task_15": "Неравенство",
        "task_16": "Планиметрическая задача",
        "task_17": "Планиметрическая задача с доказательством",
        "task_18": "Задача с параметром",
        "task_19": "Задача по теории чисел"
    }

    # Image settings
    MAX_IMAGE_SIZE: int = 4096
    ENHANCE_IMAGES: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()
