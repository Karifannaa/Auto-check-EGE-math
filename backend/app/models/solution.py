"""
Solution Models Module

This module defines the data models for math solutions and evaluations.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class SolutionRequest(BaseModel):
    """Request model for solution evaluation."""

    task_type: str = Field(..., description="Type of task (e.g., 'task_13', 'task_17')")
    task_description: str = Field(..., description="Description of the task")
    model_id: str = Field(..., description="ID of the model to use for evaluation")
    include_examples: bool = Field(False, description="Whether to include examples in the prompt")
    prompt_variant: Optional[str] = Field(None, description="Specific prompt variant to use (e.g., 'basic', 'detailed', 'with_examples')")
    temperature: float = Field(0.7, description="Sampling temperature for the model")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate")


class EvaluationResult(BaseModel):
    """Model for evaluation results."""

    score: int = Field(..., description="Score assigned to the solution (in points)")
    explanation: str = Field(..., description="Explanation for the score")
    model_id: str = Field(..., description="ID of the model used for evaluation")
    task_type: str = Field(..., description="Type of task that was evaluated")
    evaluation_time: float = Field(..., description="Time taken for evaluation (in seconds)")
    prompt_tokens: Optional[int] = Field(None, description="Number of prompt tokens used")
    completion_tokens: Optional[int] = Field(None, description="Number of completion tokens used")
    total_tokens: Optional[int] = Field(None, description="Total number of tokens used")


class SolutionEvaluation(BaseModel):
    """Model for a complete solution evaluation."""

    id: str = Field(..., description="Unique ID for the evaluation")
    task_type: str = Field(..., description="Type of task")
    task_description: str = Field(..., description="Description of the task")
    model_id: str = Field(..., description="ID of the model used for evaluation")
    result: EvaluationResult = Field(..., description="Evaluation result")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    estimated_cost: float = Field(0.0, description="Estimated cost of the evaluation in USD")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "task_type": "task_13",
                "task_description": "а) Решите уравнение 2sin(x+π/3)+cos2x=√3cosx+1. б) Укажите корни этого уравнения, принадлежащие отрезку [-3π;-3π/2].",
                "model_id": "gpt-4o",
                "result": {
                    "score": 2,
                    "explanation": "Решение полностью верное. В пункте а) корректно выполнены все преобразования и найдены все корни уравнения. В пункте б) правильно отобраны корни, принадлежащие указанному отрезку.",
                    "model_id": "gpt-4o",
                    "task_type": "task_13",
                    "evaluation_time": 3.45,
                    "prompt_tokens": 1250,
                    "completion_tokens": 750,
                    "total_tokens": 2000
                },
                "created_at": "2023-04-01T12:00:00",
                "estimated_cost": 0.0025
            }
        }
