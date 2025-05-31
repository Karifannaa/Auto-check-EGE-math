#!/usr/bin/env python3
"""
Metrics CLI Tool

Command-line interface for interacting with the centralized metrics system.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from app.utils.metrics_integration import (
    import_json_results, 
    get_benchmark_summary,
    store_batch_results
)
from app.database.config import create_tables, SessionLocal
from app.services.metrics_service import MetricsService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("metrics_cli")


def init_database():
    """Initialize the database tables."""
    try:
        create_tables()
        print("✓ Database tables initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize database: {e}")
        return False


def import_results(file_path: str, benchmark_run_id: Optional[str] = None) -> bool:
    """Import results from a JSON file."""
    try:
        if not os.path.exists(file_path):
            print(f"✗ File not found: {file_path}")
            return False
        
        print(f"Importing results from: {file_path}")
        
        result = import_json_results(file_path, benchmark_run_id)
        
        print(f"✓ Successfully imported {result['imported_count']} results")
        print(f"  Benchmark Run ID: {result['benchmark_run_id']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to import results: {e}")
        return False


def show_summary(benchmark_run_id: Optional[str] = None):
    """Show metrics summary."""
    try:
        db = SessionLocal()
        metrics_service = MetricsService(db)
        
        if benchmark_run_id:
            # Show summary for specific benchmark run
            summary = get_benchmark_summary(benchmark_run_id)
            
            print(f"\n=== Benchmark Run Summary ===")
            print(f"Benchmark Run ID: {summary['benchmark_run_id']}")
            print(f"Total Evaluations: {summary['total_evaluations']}")
            print(f"Unique Models: {summary['unique_models']}")
            print(f"Unique Tasks: {summary['unique_tasks']}")
            print(f"Unique Solutions: {summary['unique_solutions']}")
            
            if summary.get('overall_accuracy') is not None:
                print(f"Overall Accuracy: {summary['overall_accuracy']:.2f}%")
            
            if summary.get('overall_quality_score') is not None:
                print(f"Overall Quality Score: {summary['overall_quality_score']:.2f}%")
            
            if summary.get('total_cost') is not None:
                print(f"Total Cost: ${summary['total_cost']:.4f}")
            
            if summary.get('avg_evaluation_time') is not None:
                print(f"Average Evaluation Time: {summary['avg_evaluation_time']:.2f}s")
        
        else:
            # Show overall summary
            all_metrics = metrics_service.get_metrics()
            
            if not all_metrics:
                print("No metrics found in the database")
                return
            
            total_evaluations = len(all_metrics)
            unique_models = len(set(m.model_id for m in all_metrics))
            unique_tasks = len(set(m.task_type for m in all_metrics))
            unique_benchmark_runs = len(set(m.benchmark_run_id for m in all_metrics if m.benchmark_run_id))
            
            # Calculate overall metrics
            valid_accuracy = [m.accuracy for m in all_metrics if m.accuracy is not None]
            overall_accuracy = sum(valid_accuracy) / len(valid_accuracy) * 100 if valid_accuracy else None
            
            valid_quality = [m.quality_score for m in all_metrics if m.quality_score is not None]
            overall_quality_score = sum(valid_quality) / len(valid_quality) * 100 if valid_quality else None
            
            valid_costs = [m.actual_cost for m in all_metrics if m.actual_cost is not None]
            total_cost = sum(valid_costs) if valid_costs else None
            
            print(f"\n=== Overall Metrics Summary ===")
            print(f"Total Evaluations: {total_evaluations}")
            print(f"Unique Models: {unique_models}")
            print(f"Unique Tasks: {unique_tasks}")
            print(f"Unique Benchmark Runs: {unique_benchmark_runs}")
            
            if overall_accuracy is not None:
                print(f"Overall Accuracy: {overall_accuracy:.2f}%")
            
            if overall_quality_score is not None:
                print(f"Overall Quality Score: {overall_quality_score:.2f}%")
            
            if total_cost is not None:
                print(f"Total Cost: ${total_cost:.4f}")
        
        db.close()
        
    except Exception as e:
        print(f"✗ Failed to show summary: {e}")


def list_benchmark_runs():
    """List all benchmark runs."""
    try:
        db = SessionLocal()
        metrics_service = MetricsService(db)
        
        # Get all unique benchmark run IDs
        all_metrics = metrics_service.get_metrics()
        
        if not all_metrics:
            print("No benchmark runs found")
            return
        
        # Group by benchmark run ID
        runs = {}
        for metric in all_metrics:
            run_id = metric.benchmark_run_id or "unknown"
            if run_id not in runs:
                runs[run_id] = {
                    "count": 0,
                    "first_timestamp": metric.timestamp,
                    "last_timestamp": metric.timestamp,
                    "models": set(),
                    "tasks": set()
                }
            
            runs[run_id]["count"] += 1
            runs[run_id]["models"].add(metric.model_id)
            runs[run_id]["tasks"].add(metric.task_type)
            
            if metric.timestamp < runs[run_id]["first_timestamp"]:
                runs[run_id]["first_timestamp"] = metric.timestamp
            if metric.timestamp > runs[run_id]["last_timestamp"]:
                runs[run_id]["last_timestamp"] = metric.timestamp
        
        print(f"\n=== Benchmark Runs ({len(runs)} total) ===")
        
        for run_id, info in sorted(runs.items(), key=lambda x: x[1]["first_timestamp"], reverse=True):
            print(f"\nRun ID: {run_id}")
            print(f"  Evaluations: {info['count']}")
            print(f"  Models: {len(info['models'])} ({', '.join(sorted(info['models'])[:3])}{'...' if len(info['models']) > 3 else ''})")
            print(f"  Tasks: {len(info['tasks'])} ({', '.join(sorted(info['tasks']))})")
            print(f"  First: {info['first_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Last: {info['last_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        db.close()
        
    except Exception as e:
        print(f"✗ Failed to list benchmark runs: {e}")


def export_metrics(output_file: str, benchmark_run_id: Optional[str] = None):
    """Export metrics to a JSON file."""
    try:
        db = SessionLocal()
        metrics_service = MetricsService(db)
        
        # Get metrics
        if benchmark_run_id:
            metrics = metrics_service.get_metrics(benchmark_run_id=benchmark_run_id)
        else:
            metrics = metrics_service.get_metrics()
        
        if not metrics:
            print("No metrics found to export")
            return
        
        # Convert to dictionaries
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_entries": len(metrics),
            "benchmark_run_id": benchmark_run_id,
            "metrics": [metric.to_dict() for metric in metrics]
        }
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Exported {len(metrics)} metrics to {output_file}")
        
        db.close()
        
    except Exception as e:
        print(f"✗ Failed to export metrics: {e}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Centralized Metrics CLI Tool")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize database tables")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import results from JSON file")
    import_parser.add_argument("file", help="Path to JSON file containing results")
    import_parser.add_argument("--run-id", help="Optional benchmark run ID")
    
    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Show metrics summary")
    summary_parser.add_argument("--run-id", help="Show summary for specific benchmark run")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all benchmark runs")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export metrics to JSON file")
    export_parser.add_argument("output", help="Output file path")
    export_parser.add_argument("--run-id", help="Export specific benchmark run only")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute commands
    if args.command == "init":
        init_database()
    
    elif args.command == "import":
        if not init_database():
            return
        import_results(args.file, args.run_id)
    
    elif args.command == "summary":
        show_summary(args.run_id)
    
    elif args.command == "list":
        list_benchmark_runs()
    
    elif args.command == "export":
        export_metrics(args.output, args.run_id)


if __name__ == "__main__":
    main()
