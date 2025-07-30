"""
Metrics Service Module

This module provides services for collecting, storing, and retrieving metrics
from the centralized metrics database.
"""

import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func

from ..database.models import MetricsEntry
from ..database.config import get_db
from ..core.config import settings


class MetricsService:
    """Service for handling metrics collection and retrieval."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def store_evaluation_result(
        self,
        evaluation_result: Dict[str, Any],
        benchmark_run_id: Optional[str] = None,
        benchmark_config: Optional[Dict[str, Any]] = None
    ) -> MetricsEntry:
        """
        Store an evaluation result in the centralized metrics table.
        
        Args:
            evaluation_result: Dictionary containing evaluation results
            benchmark_run_id: Optional ID of the benchmark run
            benchmark_config: Optional configuration used for the benchmark
            
        Returns:
            MetricsEntry: The created metrics entry
        """
        # Extract model information
        model_id = evaluation_result.get("model_id", "")
        model_name = self._extract_model_name(model_id)
        model_type = self._determine_model_type(model_id)
        
        # Extract task information
        task_id = evaluation_result.get("task_id", "")
        task_type = evaluation_result.get("task_type", "")
        solution_id = evaluation_result.get("solution_id", "")
        example_id = self._extract_example_id(solution_id)
        
        # Calculate derived metrics
        score = evaluation_result.get("score")
        expected_score = evaluation_result.get("expected_score")
        accuracy = None
        score_distance = None
        normalized_distance = None
        quality_score = None
        
        if score is not None and expected_score is not None:
            accuracy = score == expected_score
            score_distance = abs(score - expected_score)
            
            # Get max score for the task type
            max_score = self._get_max_score_for_task(task_type)
            if max_score > 0:
                normalized_distance = score_distance / max_score
                quality_score = 1.0 - normalized_distance
        
        # Create metrics entry
        metrics_entry = MetricsEntry(
            # Model identification
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            
            # Task identification
            task_id=task_id,
            task_type=task_type,
            solution_id=solution_id,
            example_id=example_id,
            
            # Evaluation configuration
            use_answer=evaluation_result.get("use_answer", False),
            use_true_solution=evaluation_result.get("use_true_solution", False),
            prompt_variant=evaluation_result.get("prompt_variant"),
            include_examples=evaluation_result.get("include_examples", False),
            temperature=evaluation_result.get("temperature"),
            max_tokens=evaluation_result.get("max_tokens"),
            
            # Performance metrics
            score=score,
            expected_score=expected_score,
            accuracy=accuracy,
            score_distance=score_distance,
            normalized_distance=normalized_distance,
            quality_score=quality_score,
            
            # Timing metrics
            evaluation_time=evaluation_result.get("evaluation_time"),
            
            # Token usage metrics
            prompt_tokens=evaluation_result.get("prompt_tokens"),
            completion_tokens=evaluation_result.get("completion_tokens"),
            total_tokens=evaluation_result.get("total_tokens"),
            
            # Cost metrics
            actual_cost=evaluation_result.get("cost"),
            
            # Result data
            result_text=evaluation_result.get("result_text"),
            error_message=evaluation_result.get("error"),
            
            # Metadata
            benchmark_run_id=benchmark_run_id,
            benchmark_config=benchmark_config,
            
            # Status
            status="completed" if not evaluation_result.get("error") else "failed"
        )
        
        # Add to database
        self.db.add(metrics_entry)
        self.db.commit()
        self.db.refresh(metrics_entry)
        
        return metrics_entry
    
    def store_batch_results(
        self,
        evaluation_results: List[Dict[str, Any]],
        benchmark_run_id: Optional[str] = None,
        benchmark_config: Optional[Dict[str, Any]] = None
    ) -> List[MetricsEntry]:
        """
        Store multiple evaluation results in batch.
        
        Args:
            evaluation_results: List of evaluation result dictionaries
            benchmark_run_id: Optional ID of the benchmark run
            benchmark_config: Optional configuration used for the benchmark
            
        Returns:
            List[MetricsEntry]: List of created metrics entries
        """
        entries = []
        for result in evaluation_results:
            entry = self.store_evaluation_result(
                result, benchmark_run_id, benchmark_config
            )
            entries.append(entry)
        
        return entries

    def get_metrics(
        self,
        model_ids: Optional[List[str]] = None,
        task_types: Optional[List[str]] = None,
        task_ids: Optional[List[str]] = None,
        use_answer: Optional[bool] = None,
        use_true_solution: Optional[bool] = None,
        benchmark_run_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: str = "timestamp",
        order_direction: str = "desc"
    ) -> List[MetricsEntry]:
        """
        Retrieve metrics with optional filtering.

        Args:
            model_ids: Filter by model IDs
            task_types: Filter by task types
            task_ids: Filter by task IDs
            use_answer: Filter by use_answer flag
            use_true_solution: Filter by use_true_solution flag
            benchmark_run_id: Filter by benchmark run ID
            limit: Maximum number of results
            offset: Number of results to skip
            order_by: Field to order by
            order_direction: "asc" or "desc"

        Returns:
            List[MetricsEntry]: List of metrics entries
        """
        query = self.db.query(MetricsEntry)

        # Apply filters
        if model_ids:
            query = query.filter(MetricsEntry.model_id.in_(model_ids))

        if task_types:
            query = query.filter(MetricsEntry.task_type.in_(task_types))

        if task_ids:
            query = query.filter(MetricsEntry.task_id.in_(task_ids))

        if use_answer is not None:
            query = query.filter(MetricsEntry.use_answer == use_answer)

        if use_true_solution is not None:
            query = query.filter(MetricsEntry.use_true_solution == use_true_solution)

        if benchmark_run_id:
            query = query.filter(MetricsEntry.benchmark_run_id == benchmark_run_id)

        # Apply ordering
        order_field = getattr(MetricsEntry, order_by, MetricsEntry.timestamp)
        if order_direction.lower() == "desc":
            query = query.order_by(desc(order_field))
        else:
            query = query.order_by(asc(order_field))

        # Apply pagination
        if offset:
            query = query.offset(offset)

        if limit:
            query = query.limit(limit)

        return query.all()

    def get_aggregated_metrics(
        self,
        group_by: List[str] = ["model_id", "task_type"],
        model_ids: Optional[List[str]] = None,
        task_types: Optional[List[str]] = None,
        task_ids: Optional[List[str]] = None,
        use_answer: Optional[bool] = None,
        use_true_solution: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Get aggregated metrics grouped by specified fields.

        Args:
            group_by: Fields to group by
            model_ids: Filter by model IDs
            task_types: Filter by task types
            task_ids: Filter by task IDs
            use_answer: Filter by use_answer flag
            use_true_solution: Filter by use_true_solution flag

        Returns:
            List[Dict[str, Any]]: Aggregated metrics
        """
        # Build group by fields
        group_fields = []
        for field in group_by:
            if hasattr(MetricsEntry, field):
                group_fields.append(getattr(MetricsEntry, field))

        if not group_fields:
            group_fields = [MetricsEntry.model_id, MetricsEntry.task_type]

        # Build query
        query = self.db.query(
            *group_fields,
            func.count(MetricsEntry.id).label("total_evaluations"),
            func.avg(MetricsEntry.accuracy.cast(float)).label("accuracy"),
            func.avg(MetricsEntry.quality_score).label("avg_quality_score"),
            func.avg(MetricsEntry.score_distance).label("avg_score_distance"),
            func.avg(MetricsEntry.evaluation_time).label("avg_evaluation_time"),
            func.avg(MetricsEntry.prompt_tokens).label("avg_prompt_tokens"),
            func.avg(MetricsEntry.completion_tokens).label("avg_completion_tokens"),
            func.avg(MetricsEntry.total_tokens).label("avg_total_tokens"),
            func.sum(MetricsEntry.actual_cost).label("total_cost")
        )

        # Apply filters
        if model_ids:
            query = query.filter(MetricsEntry.model_id.in_(model_ids))

        if task_types:
            query = query.filter(MetricsEntry.task_type.in_(task_types))

        if task_ids:
            query = query.filter(MetricsEntry.task_id.in_(task_ids))

        if use_answer is not None:
            query = query.filter(MetricsEntry.use_answer == use_answer)

        if use_true_solution is not None:
            query = query.filter(MetricsEntry.use_true_solution == use_true_solution)

        # Group by specified fields
        query = query.group_by(*group_fields)

        # Execute query and convert to dictionaries
        results = []
        for row in query.all():
            result = {}
            for i, field in enumerate(group_by):
                result[field] = row[i]

            # Add aggregated metrics
            result.update({
                "total_evaluations": row.total_evaluations,
                "accuracy": float(row.accuracy * 100) if row.accuracy is not None else None,
                "avg_quality_score": float(row.avg_quality_score * 100) if row.avg_quality_score is not None else None,
                "avg_score_distance": float(row.avg_score_distance) if row.avg_score_distance is not None else None,
                "avg_evaluation_time": float(row.avg_evaluation_time) if row.avg_evaluation_time is not None else None,
                "avg_prompt_tokens": float(row.avg_prompt_tokens) if row.avg_prompt_tokens is not None else None,
                "avg_completion_tokens": float(row.avg_completion_tokens) if row.avg_completion_tokens is not None else None,
                "avg_total_tokens": float(row.avg_total_tokens) if row.avg_total_tokens is not None else None,
                "total_cost": float(row.total_cost) if row.total_cost is not None else None
            })

            results.append(result)

        return results

    def _extract_model_name(self, model_id: str) -> str:
        """Extract human-readable model name from model ID."""
        if "/" in model_id:
            return model_id.split("/")[-1]
        return model_id

    def _determine_model_type(self, model_id: str) -> str:
        """Determine model type (reasoning/non_reasoning) from model ID."""
        if model_id in settings.REASONING_MODELS.values():
            return "reasoning"
        elif model_id in settings.NON_REASONING_MODELS.values():
            return "non_reasoning"
        else:
            return "unknown"

    def _extract_example_id(self, solution_id: str) -> str:
        """Extract example ID from solution ID (e.g., '13.1' from '13.1.1')."""
        parts = solution_id.split(".")
        if len(parts) >= 2:
            return f"{parts[0]}.{parts[1]}"
        return solution_id

    def _get_max_score_for_task(self, task_type: str) -> int:
        """Get maximum possible score for a task type."""
        max_scores = {
            'task_13': 2,
            'task_14': 3,
            'task_15': 2,
            'task_16': 3,
            'task_17': 3,
            'task_18': 4,
            'task_19': 4
        }
        return max_scores.get(task_type, 4)  # Default to 4 if unknown
