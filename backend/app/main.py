"""
Main Application Module

This is the main entry point for the FastAPI application.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.routers import solutions, models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

# Set up CORS
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include routers
app.include_router(solutions.router, prefix=f"{settings.API_V1_STR}/solutions", tags=["solutions"])
app.include_router(models.router, prefix=f"{settings.API_V1_STR}/models", tags=["models"])

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the EGE Math Solution Checker API"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
