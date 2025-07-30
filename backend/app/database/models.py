"""
Database Models Module

This module defines the SQLAlchemy models for the centralized metrics system.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.sql import func
from datetime import datetime
from .config import Base


class MetricsEntry(Base):
    """
    Centralized metrics table for storing all evaluation metrics.
    
    This table aggregates metrics from all model evaluations with proper
    categorization and metadata for comprehensive analysis.
    """
    __tablename__ = "metrics"

    # Primary key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Timestamps
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Model identification
    model_id = Column(String(255), nullable=False, index=True)
    model_name = Column(String(255), nullable=True, index=True)  # Human-readable model name
    model_type = Column(String(100), nullable=True, index=True)  # e.g., "reasoning", "non_reasoning"
    
    # Task identification
    task_id = Column(String(50), nullable=False, index=True)  # e.g., "13", "14"
    task_type = Column(String(100), nullable=False, index=True)  # e.g., "task_13", "task_14"
    task_description = Column(Text, nullable=True)
    
    # Solution identification
    solution_id = Column(String(100), nullable=False, index=True)  # e.g., "13.1.1"
    example_id = Column(String(100), nullable=True, index=True)  # e.g., "13.1"
    
    # Evaluation configuration
    use_answer = Column(Boolean, nullable=False, default=False, index=True)
    use_true_solution = Column(Boolean, nullable=False, default=False, index=True)
    prompt_variant = Column(String(100), nullable=True, index=True)
    include_examples = Column(Boolean, nullable=False, default=False)
    temperature = Column(Float, nullable=True)
    max_tokens = Column(Integer, nullable=True)
    
    # Performance metrics
    score = Column(Integer, nullable=True, index=True)  # Predicted score
    expected_score = Column(Integer, nullable=True, index=True)  # Ground truth score
    accuracy = Column(Boolean, nullable=True, index=True)  # score == expected_score
    score_distance = Column(Float, nullable=True, index=True)  # abs(score - expected_score)
    normalized_distance = Column(Float, nullable=True, index=True)  # score_distance / max_possible_score
    quality_score = Column(Float, nullable=True, index=True)  # 1 - normalized_distance
    
    # Timing metrics
    evaluation_time = Column(Float, nullable=True, index=True)  # Time in seconds
    
    # Token usage metrics
    prompt_tokens = Column(Integer, nullable=True, index=True)
    completion_tokens = Column(Integer, nullable=True, index=True)
    total_tokens = Column(Integer, nullable=True, index=True)
    
    # Cost metrics
    estimated_cost = Column(Float, nullable=True, index=True)  # Estimated cost in USD
    actual_cost = Column(Float, nullable=True, index=True)  # Actual cost in USD
    
    # Result data
    result_text = Column(Text, nullable=True)  # Full model response
    explanation = Column(Text, nullable=True)  # Extracted explanation
    error_message = Column(Text, nullable=True)  # Error message if evaluation failed
    
    # Metadata
    benchmark_run_id = Column(String(255), nullable=True, index=True)  # ID of the benchmark run
    benchmark_config = Column(JSON, nullable=True)  # Configuration used for the benchmark
    additional_metadata = Column(JSON, nullable=True)  # Any additional metadata
    
    # Status
    status = Column(String(50), nullable=False, default="completed", index=True)  # completed, failed, pending
    
    def __repr__(self):
        return f"<MetricsEntry(id={self.id}, model_id='{self.model_id}', task_type='{self.task_type}', solution_id='{self.solution_id}', score={self.score})>"
    
    def to_dict(self):
        """Convert the model instance to a dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "task_description": self.task_description,
            "solution_id": self.solution_id,
            "example_id": self.example_id,
            "use_answer": self.use_answer,
            "use_true_solution": self.use_true_solution,
            "prompt_variant": self.prompt_variant,
            "include_examples": self.include_examples,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "score": self.score,
            "expected_score": self.expected_score,
            "accuracy": self.accuracy,
            "score_distance": self.score_distance,
            "normalized_distance": self.normalized_distance,
            "quality_score": self.quality_score,
            "evaluation_time": self.evaluation_time,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.estimated_cost,
            "actual_cost": self.actual_cost,
            "result_text": self.result_text,
            "explanation": self.explanation,
            "error_message": self.error_message,
            "benchmark_run_id": self.benchmark_run_id,
            "benchmark_config": self.benchmark_config,
            "additional_metadata": self.additional_metadata,
            "status": self.status
        }
