"""
Metrics Router Module

This module provides API endpoints for retrieving and analyzing metrics
from the centralized metrics database.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ..database.config import get_db
from ..services.metrics_service import MetricsService
from ..database.models import MetricsEntry


router = APIRouter(prefix="/metrics", tags=["metrics"])


class MetricsFilter(BaseModel):
    """Model for metrics filtering parameters."""
    model_ids: Optional[List[str]] = Field(None, description="Filter by model IDs")
    task_types: Optional[List[str]] = Field(None, description="Filter by task types")
    task_ids: Optional[List[str]] = Field(None, description="Filter by task IDs")
    use_answer: Optional[bool] = Field(None, description="Filter by use_answer flag")
    use_true_solution: Optional[bool] = Field(None, description="Filter by use_true_solution flag")
    benchmark_run_id: Optional[str] = Field(None, description="Filter by benchmark run ID")


class MetricsResponse(BaseModel):
    """Model for metrics response."""
    id: int
    timestamp: str
    model_id: str
    model_name: Optional[str]
    model_type: Optional[str]
    task_id: str
    task_type: str
    solution_id: str
    use_answer: bool
    use_true_solution: bool
    score: Optional[int]
    expected_score: Optional[int]
    accuracy: Optional[bool]
    quality_score: Optional[float]
    evaluation_time: Optional[float]
    total_tokens: Optional[int]
    actual_cost: Optional[float]
    status: str


class AggregatedMetricsResponse(BaseModel):
    """Model for aggregated metrics response."""
    group_fields: Dict[str, Any]
    total_evaluations: int
    accuracy: Optional[float]
    avg_quality_score: Optional[float]
    avg_score_distance: Optional[float]
    avg_evaluation_time: Optional[float]
    avg_total_tokens: Optional[float]
    total_cost: Optional[float]


@router.get("/", response_model=List[MetricsResponse])
async def get_metrics(
    model_ids: Optional[List[str]] = Query(None, description="Filter by model IDs"),
    task_types: Optional[List[str]] = Query(None, description="Filter by task types"),
    task_ids: Optional[List[str]] = Query(None, description="Filter by task IDs"),
    use_answer: Optional[bool] = Query(None, description="Filter by use_answer flag"),
    use_true_solution: Optional[bool] = Query(None, description="Filter by use_true_solution flag"),
    benchmark_run_id: Optional[str] = Query(None, description="Filter by benchmark run ID"),
    limit: Optional[int] = Query(100, description="Maximum number of results"),
    offset: Optional[int] = Query(0, description="Number of results to skip"),
    order_by: str = Query("timestamp", description="Field to order by"),
    order_direction: str = Query("desc", description="Order direction (asc/desc)"),
    db: Session = Depends(get_db)
):
    """
    Retrieve metrics with optional filtering and pagination.
    """
    try:
        metrics_service = MetricsService(db)
        
        metrics = metrics_service.get_metrics(
            model_ids=model_ids,
            task_types=task_types,
            task_ids=task_ids,
            use_answer=use_answer,
            use_true_solution=use_true_solution,
            benchmark_run_id=benchmark_run_id,
            limit=limit,
            offset=offset,
            order_by=order_by,
            order_direction=order_direction
        )
        
        # Convert to response models
        response = []
        for metric in metrics:
            response.append(MetricsResponse(
                id=metric.id,
                timestamp=metric.timestamp.isoformat() if metric.timestamp else "",
                model_id=metric.model_id,
                model_name=metric.model_name,
                model_type=metric.model_type,
                task_id=metric.task_id,
                task_type=metric.task_type,
                solution_id=metric.solution_id,
                use_answer=metric.use_answer,
                use_true_solution=metric.use_true_solution,
                score=metric.score,
                expected_score=metric.expected_score,
                accuracy=metric.accuracy,
                quality_score=metric.quality_score,
                evaluation_time=metric.evaluation_time,
                total_tokens=metric.total_tokens,
                actual_cost=metric.actual_cost,
                status=metric.status
            ))
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")


@router.get("/aggregated", response_model=List[AggregatedMetricsResponse])
async def get_aggregated_metrics(
    group_by: List[str] = Query(["model_id", "task_type"], description="Fields to group by"),
    model_ids: Optional[List[str]] = Query(None, description="Filter by model IDs"),
    task_types: Optional[List[str]] = Query(None, description="Filter by task types"),
    task_ids: Optional[List[str]] = Query(None, description="Filter by task IDs"),
    use_answer: Optional[bool] = Query(None, description="Filter by use_answer flag"),
    use_true_solution: Optional[bool] = Query(None, description="Filter by use_true_solution flag"),
    db: Session = Depends(get_db)
):
    """
    Get aggregated metrics grouped by specified fields.
    """
    try:
        metrics_service = MetricsService(db)
        
        aggregated = metrics_service.get_aggregated_metrics(
            group_by=group_by,
            model_ids=model_ids,
            task_types=task_types,
            task_ids=task_ids,
            use_answer=use_answer,
            use_true_solution=use_true_solution
        )
        
        # Convert to response models
        response = []
        for item in aggregated:
            group_fields = {}
            for field in group_by:
                if field in item:
                    group_fields[field] = item[field]
            
            response.append(AggregatedMetricsResponse(
                group_fields=group_fields,
                total_evaluations=item.get("total_evaluations", 0),
                accuracy=item.get("accuracy"),
                avg_quality_score=item.get("avg_quality_score"),
                avg_score_distance=item.get("avg_score_distance"),
                avg_evaluation_time=item.get("avg_evaluation_time"),
                avg_total_tokens=item.get("avg_total_tokens"),
                total_cost=item.get("total_cost")
            ))
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving aggregated metrics: {str(e)}")


@router.get("/summary")
async def get_metrics_summary(
    model_ids: Optional[List[str]] = Query(None, description="Filter by model IDs"),
    task_types: Optional[List[str]] = Query(None, description="Filter by task types"),
    db: Session = Depends(get_db)
):
    """
    Get a summary of metrics including counts and basic statistics.
    """
    try:
        metrics_service = MetricsService(db)
        
        # Get basic counts
        all_metrics = metrics_service.get_metrics(
            model_ids=model_ids,
            task_types=task_types
        )
        
        total_evaluations = len(all_metrics)
        
        if total_evaluations == 0:
            return {
                "total_evaluations": 0,
                "unique_models": 0,
                "unique_tasks": 0,
                "unique_solutions": 0,
                "overall_accuracy": None,
                "overall_quality_score": None,
                "total_cost": None
            }
        
        # Calculate summary statistics
        unique_models = len(set(m.model_id for m in all_metrics))
        unique_tasks = len(set(m.task_type for m in all_metrics))
        unique_solutions = len(set(m.solution_id for m in all_metrics))
        
        # Calculate overall metrics
        valid_accuracy = [m.accuracy for m in all_metrics if m.accuracy is not None]
        overall_accuracy = sum(valid_accuracy) / len(valid_accuracy) * 100 if valid_accuracy else None
        
        valid_quality = [m.quality_score for m in all_metrics if m.quality_score is not None]
        overall_quality_score = sum(valid_quality) / len(valid_quality) * 100 if valid_quality else None
        
        valid_costs = [m.actual_cost for m in all_metrics if m.actual_cost is not None]
        total_cost = sum(valid_costs) if valid_costs else None
        
        return {
            "total_evaluations": total_evaluations,
            "unique_models": unique_models,
            "unique_tasks": unique_tasks,
            "unique_solutions": unique_solutions,
            "overall_accuracy": overall_accuracy,
            "overall_quality_score": overall_quality_score,
            "total_cost": total_cost
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics summary: {str(e)}")


@router.post("/import")
async def import_benchmark_results(
    file_path: str,
    benchmark_run_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Import benchmark results from a JSON file into the centralized metrics table.
    """
    try:
        import json
        import os
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if not isinstance(results, list):
            raise HTTPException(status_code=400, detail="File must contain a list of evaluation results")
        
        metrics_service = MetricsService(db)
        
        # Generate benchmark run ID if not provided
        if not benchmark_run_id:
            import uuid
            benchmark_run_id = str(uuid.uuid4())
        
        # Store results
        entries = metrics_service.store_batch_results(
            results,
            benchmark_run_id=benchmark_run_id,
            benchmark_config={"imported_from": file_path}
        )
        
        return {
            "message": f"Successfully imported {len(entries)} evaluation results",
            "benchmark_run_id": benchmark_run_id,
            "imported_count": len(entries)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing results: {str(e)}")
