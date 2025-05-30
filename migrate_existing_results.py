#!/usr/bin/env python3
"""
Migration Script for Existing Benchmark Results

This script migrates existing benchmark results from JSON files
to the new centralized metrics database.
"""

import os
import sys
import json
import glob
import logging
from datetime import datetime
from typing import List, Dict, Any

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from app.utils.metrics_integration import import_json_results, get_benchmark_summary
from app.database.config import create_tables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("migration.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("migration")


def find_benchmark_result_files(results_dir: str = "dataset_benchmark/benchmark_results") -> List[str]:
    """
    Find all benchmark result JSON files in the results directory.
    
    Args:
        results_dir: Directory containing benchmark results
        
    Returns:
        List[str]: List of file paths to benchmark result files
    """
    if not os.path.exists(results_dir):
        logger.warning(f"Results directory not found: {results_dir}")
        return []
    
    # Find all JSON files that look like benchmark results
    patterns = [
        os.path.join(results_dir, "benchmark_*.json"),
        os.path.join(results_dir, "*", "benchmark_*.json"),
        os.path.join(results_dir, "*", "*", "benchmark_*.json")
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    
    # Filter out analysis files
    result_files = [f for f in files if not f.endswith("_analysis.json")]
    
    logger.info(f"Found {len(result_files)} benchmark result files")
    return result_files


def migrate_file(file_path: str) -> Dict[str, Any]:
    """
    Migrate a single benchmark result file.
    
    Args:
        file_path: Path to the benchmark result file
        
    Returns:
        Dict[str, Any]: Migration result summary
    """
    try:
        logger.info(f"Migrating file: {file_path}")
        
        # Extract benchmark run ID from filename if possible
        filename = os.path.basename(file_path)
        benchmark_run_id = filename.replace("benchmark_", "").replace(".json", "")
        
        # Import the results
        result = import_json_results(file_path, benchmark_run_id)
        
        logger.info(f"Successfully migrated {result['imported_count']} results from {file_path}")
        
        return {
            "file_path": file_path,
            "success": True,
            "benchmark_run_id": result["benchmark_run_id"],
            "imported_count": result["imported_count"],
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Error migrating file {file_path}: {str(e)}")
        return {
            "file_path": file_path,
            "success": False,
            "benchmark_run_id": None,
            "imported_count": 0,
            "error": str(e)
        }


def migrate_all_results(results_dir: str = "dataset_benchmark/benchmark_results") -> Dict[str, Any]:
    """
    Migrate all benchmark results from the specified directory.
    
    Args:
        results_dir: Directory containing benchmark results
        
    Returns:
        Dict[str, Any]: Migration summary
    """
    logger.info("Starting migration of existing benchmark results")
    
    # Ensure database tables exist
    create_tables()
    logger.info("Database tables initialized")
    
    # Find all result files
    result_files = find_benchmark_result_files(results_dir)
    
    if not result_files:
        logger.warning("No benchmark result files found")
        return {
            "total_files": 0,
            "successful_migrations": 0,
            "failed_migrations": 0,
            "total_imported_results": 0,
            "migration_results": []
        }
    
    # Migrate each file
    migration_results = []
    successful_migrations = 0
    failed_migrations = 0
    total_imported_results = 0
    
    for file_path in result_files:
        result = migrate_file(file_path)
        migration_results.append(result)
        
        if result["success"]:
            successful_migrations += 1
            total_imported_results += result["imported_count"]
        else:
            failed_migrations += 1
    
    summary = {
        "total_files": len(result_files),
        "successful_migrations": successful_migrations,
        "failed_migrations": failed_migrations,
        "total_imported_results": total_imported_results,
        "migration_results": migration_results
    }
    
    logger.info(f"Migration completed: {successful_migrations}/{len(result_files)} files migrated successfully")
    logger.info(f"Total imported results: {total_imported_results}")
    
    return summary


def print_migration_summary(summary: Dict[str, Any]):
    """
    Print a formatted migration summary.
    
    Args:
        summary: Migration summary dictionary
    """
    print("\n" + "="*60)
    print("MIGRATION SUMMARY")
    print("="*60)
    print(f"Total files found: {summary['total_files']}")
    print(f"Successful migrations: {summary['successful_migrations']}")
    print(f"Failed migrations: {summary['failed_migrations']}")
    print(f"Total imported results: {summary['total_imported_results']}")
    
    if summary['failed_migrations'] > 0:
        print("\nFAILED MIGRATIONS:")
        print("-" * 40)
        for result in summary['migration_results']:
            if not result['success']:
                print(f"File: {result['file_path']}")
                print(f"Error: {result['error']}")
                print()
    
    if summary['successful_migrations'] > 0:
        print("\nSUCCESSFUL MIGRATIONS:")
        print("-" * 40)
        for result in summary['migration_results']:
            if result['success']:
                print(f"File: {os.path.basename(result['file_path'])}")
                print(f"Benchmark Run ID: {result['benchmark_run_id']}")
                print(f"Imported Results: {result['imported_count']}")
                print()


def main():
    """Main migration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate existing benchmark results to centralized metrics database")
    parser.add_argument(
        "--results-dir",
        default="dataset_benchmark/benchmark_results",
        help="Directory containing benchmark results (default: dataset_benchmark/benchmark_results)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually migrating"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual migration will be performed")
        result_files = find_benchmark_result_files(args.results_dir)
        print(f"\nFound {len(result_files)} files that would be migrated:")
        for file_path in result_files:
            print(f"  - {file_path}")
        return
    
    # Perform the migration
    summary = migrate_all_results(args.results_dir)
    
    # Print summary
    print_migration_summary(summary)
    
    # Save summary to file
    summary_file = f"migration_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Migration summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
