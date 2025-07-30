"""
Metrics Integration Utility

This module provides utilities to integrate the centralized metrics collection
with the existing benchmark system.
"""

import uuid
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session

from ..database.config import SessionLocal
from ..services.metrics_service import MetricsService


class MetricsCollector:
    """
    Utility class to collect and store metrics from benchmark runs.
    
    This class can be used by the benchmark system to automatically
    store evaluation results in the centralized metrics database.
    """
    
    def __init__(self, benchmark_run_id: Optional[str] = None):
        """
        Initialize the metrics collector.
        
        Args:
            benchmark_run_id: Optional ID for the benchmark run.
                             If not provided, a new UUID will be generated.
        """
        self.benchmark_run_id = benchmark_run_id or str(uuid.uuid4())
        self.benchmark_config = {}
        self.collected_results = []
    
    def set_benchmark_config(self, config: Dict[str, Any]):
        """
        Set the configuration for the current benchmark run.
        
        Args:
            config: Dictionary containing benchmark configuration
        """
        self.benchmark_config = config
    
    def collect_result(self, evaluation_result: Dict[str, Any]):
        """
        Collect a single evaluation result.
        
        Args:
            evaluation_result: Dictionary containing evaluation results
        """
        self.collected_results.append(evaluation_result)
    
    def collect_batch_results(self, evaluation_results: List[Dict[str, Any]]):
        """
        Collect multiple evaluation results.
        
        Args:
            evaluation_results: List of evaluation result dictionaries
        """
        self.collected_results.extend(evaluation_results)
    
    def store_to_database(self) -> List[int]:
        """
        Store all collected results to the centralized metrics database.
        
        Returns:
            List[int]: List of database IDs for the stored metrics entries
        """
        if not self.collected_results:
            return []
        
        db = SessionLocal()
        try:
            metrics_service = MetricsService(db)
            
            entries = metrics_service.store_batch_results(
                self.collected_results,
                benchmark_run_id=self.benchmark_run_id,
                benchmark_config=self.benchmark_config
            )
            
            return [entry.id for entry in entries]
            
        finally:
            db.close()
    
    def get_benchmark_run_id(self) -> str:
        """Get the benchmark run ID."""
        return self.benchmark_run_id
    
    def get_collected_count(self) -> int:
        """Get the number of collected results."""
        return len(self.collected_results)
    
    def clear_collected_results(self):
        """Clear all collected results."""
        self.collected_results.clear()


def store_evaluation_result(
    evaluation_result: Dict[str, Any],
    benchmark_run_id: Optional[str] = None,
    benchmark_config: Optional[Dict[str, Any]] = None
) -> int:
    """
    Convenience function to store a single evaluation result.
    
    Args:
        evaluation_result: Dictionary containing evaluation results
        benchmark_run_id: Optional ID of the benchmark run
        benchmark_config: Optional configuration used for the benchmark
        
    Returns:
        int: Database ID of the stored metrics entry
    """
    db = SessionLocal()
    try:
        metrics_service = MetricsService(db)
        
        entry = metrics_service.store_evaluation_result(
            evaluation_result,
            benchmark_run_id=benchmark_run_id,
            benchmark_config=benchmark_config
        )
        
        return entry.id
        
    finally:
        db.close()


def store_batch_results(
    evaluation_results: List[Dict[str, Any]],
    benchmark_run_id: Optional[str] = None,
    benchmark_config: Optional[Dict[str, Any]] = None
) -> List[int]:
    """
    Convenience function to store multiple evaluation results.
    
    Args:
        evaluation_results: List of evaluation result dictionaries
        benchmark_run_id: Optional ID of the benchmark run
        benchmark_config: Optional configuration used for the benchmark
        
    Returns:
        List[int]: List of database IDs for the stored metrics entries
    """
    db = SessionLocal()
    try:
        metrics_service = MetricsService(db)
        
        entries = metrics_service.store_batch_results(
            evaluation_results,
            benchmark_run_id=benchmark_run_id,
            benchmark_config=benchmark_config
        )
        
        return [entry.id for entry in entries]
        
    finally:
        db.close()


def import_json_results(
    file_path: str,
    benchmark_run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Import evaluation results from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing evaluation results
        benchmark_run_id: Optional ID for the benchmark run
        
    Returns:
        Dict[str, Any]: Import summary with statistics
    """
    import json
    import os
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    if not isinstance(results, list):
        raise ValueError("File must contain a list of evaluation results")
    
    # Generate benchmark run ID if not provided
    if not benchmark_run_id:
        benchmark_run_id = str(uuid.uuid4())
    
    # Store results
    entry_ids = store_batch_results(
        results,
        benchmark_run_id=benchmark_run_id,
        benchmark_config={"imported_from": file_path}
    )
    
    return {
        "benchmark_run_id": benchmark_run_id,
        "imported_count": len(entry_ids),
        "file_path": file_path,
        "entry_ids": entry_ids
    }


def get_benchmark_summary(benchmark_run_id: str) -> Dict[str, Any]:
    """
    Get a summary of a specific benchmark run.
    
    Args:
        benchmark_run_id: ID of the benchmark run
        
    Returns:
        Dict[str, Any]: Summary statistics for the benchmark run
    """
    db = SessionLocal()
    try:
        metrics_service = MetricsService(db)
        
        # Get all metrics for this benchmark run
        metrics = metrics_service.get_metrics(benchmark_run_id=benchmark_run_id)
        
        if not metrics:
            return {
                "benchmark_run_id": benchmark_run_id,
                "total_evaluations": 0,
                "error": "No metrics found for this benchmark run"
            }
        
        # Calculate summary statistics
        total_evaluations = len(metrics)
        unique_models = len(set(m.model_id for m in metrics))
        unique_tasks = len(set(m.task_type for m in metrics))
        unique_solutions = len(set(m.solution_id for m in metrics))
        
        # Calculate accuracy and quality scores
        valid_accuracy = [m.accuracy for m in metrics if m.accuracy is not None]
        overall_accuracy = sum(valid_accuracy) / len(valid_accuracy) * 100 if valid_accuracy else None
        
        valid_quality = [m.quality_score for m in metrics if m.quality_score is not None]
        overall_quality_score = sum(valid_quality) / len(valid_quality) * 100 if valid_quality else None
        
        # Calculate costs and timing
        valid_costs = [m.actual_cost for m in metrics if m.actual_cost is not None]
        total_cost = sum(valid_costs) if valid_costs else None
        
        valid_times = [m.evaluation_time for m in metrics if m.evaluation_time is not None]
        avg_evaluation_time = sum(valid_times) / len(valid_times) if valid_times else None
        
        # Get configuration
        benchmark_config = metrics[0].benchmark_config if metrics else None
        
        return {
            "benchmark_run_id": benchmark_run_id,
            "total_evaluations": total_evaluations,
            "unique_models": unique_models,
            "unique_tasks": unique_tasks,
            "unique_solutions": unique_solutions,
            "overall_accuracy": overall_accuracy,
            "overall_quality_score": overall_quality_score,
            "total_cost": total_cost,
            "avg_evaluation_time": avg_evaluation_time,
            "benchmark_config": benchmark_config,
            "first_evaluation": metrics[0].timestamp.isoformat() if metrics else None,
            "last_evaluation": metrics[-1].timestamp.isoformat() if metrics else None
        }
        
    finally:
        db.close()
