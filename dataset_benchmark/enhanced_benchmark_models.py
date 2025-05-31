"""
Enhanced Benchmark Models with Centralized Metrics Collection

This module extends the existing benchmark_models.py with integrated
centralized metrics collection capabilities.
"""

import os
import sys
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

# Import the original benchmark class
from benchmark_models import ModelBenchmark as OriginalModelBenchmark

# Import the metrics integration utilities
from app.utils.metrics_integration import MetricsCollector, store_batch_results
from app.database.config import create_tables

logger = logging.getLogger("enhanced_benchmark")


class EnhancedModelBenchmark(OriginalModelBenchmark):
    """
    Enhanced ModelBenchmark class with centralized metrics collection.
    
    This class extends the original ModelBenchmark to automatically
    store evaluation results in the centralized metrics database
    while maintaining backward compatibility.
    """
    
    def __init__(
        self,
        dataset_dir: str = "dataset_benchmark_hf",
        results_dir: str = "benchmark_results",
        api_key: Optional[str] = None,
        max_retries: int = 10,
        initial_delay: int = 15,
        max_delay: int = 120,
        enable_centralized_metrics: bool = True
    ):
        """
        Initialize the enhanced benchmark.
        
        Args:
            dataset_dir: Directory containing the dataset
            results_dir: Directory to save results
            api_key: OpenRouter API key
            max_retries: Maximum retries for rate-limited requests
            initial_delay: Initial delay for retries
            max_delay: Maximum delay for retries
            enable_centralized_metrics: Whether to enable centralized metrics collection
        """
        # Initialize the parent class
        super().__init__(
            dataset_dir=dataset_dir,
            results_dir=results_dir,
            api_key=api_key,
            max_retries=max_retries,
            initial_delay=initial_delay,
            max_delay=max_delay
        )
        
        self.enable_centralized_metrics = enable_centralized_metrics
        self.metrics_collector = None
        
        # Initialize database tables if centralized metrics are enabled
        if self.enable_centralized_metrics:
            try:
                create_tables()
                logger.info("Database tables initialized for centralized metrics")
            except Exception as e:
                logger.warning(f"Failed to initialize database tables: {e}")
                self.enable_centralized_metrics = False
    
    async def run_benchmark(
        self,
        task_id: Optional[str] = None,
        model_ids: Optional[List[str]] = None,
        with_answer: bool = True,
        without_answer: bool = True,
        with_true_solution: bool = False,
        max_examples: Optional[int] = None,
        prompt_variant: str = "detailed",
        include_examples: bool = False,
        output_dir: Optional[str] = None
    ) -> Tuple[List[Dict], str]:
        """
        Run the benchmark with enhanced metrics collection.
        
        This method extends the parent's run_benchmark to automatically
        collect metrics in the centralized database.
        """
        # Initialize metrics collector if enabled
        if self.enable_centralized_metrics:
            self.metrics_collector = MetricsCollector()
            
            # Set benchmark configuration
            benchmark_config = {
                "task_id": task_id,
                "model_ids": model_ids,
                "with_answer": with_answer,
                "without_answer": without_answer,
                "with_true_solution": with_true_solution,
                "max_examples": max_examples,
                "prompt_variant": prompt_variant,
                "include_examples": include_examples,
                "dataset_dir": self.dataset_dir,
                "timestamp": str(datetime.now())
            }
            self.metrics_collector.set_benchmark_config(benchmark_config)
        
        # Run the original benchmark
        results, filepath = await super().run_benchmark(
            task_id=task_id,
            model_ids=model_ids,
            with_answer=with_answer,
            without_answer=without_answer,
            with_true_solution=with_true_solution,
            max_examples=max_examples,
            prompt_variant=prompt_variant,
            include_examples=include_examples,
            output_dir=output_dir
        )
        
        # Store results in centralized metrics database
        if self.enable_centralized_metrics and self.metrics_collector:
            try:
                logger.info("Storing results in centralized metrics database...")
                
                # Collect all results
                self.metrics_collector.collect_batch_results(results)
                
                # Store to database
                entry_ids = self.metrics_collector.store_to_database()
                
                logger.info(f"Successfully stored {len(entry_ids)} metrics entries in centralized database")
                logger.info(f"Benchmark run ID: {self.metrics_collector.get_benchmark_run_id()}")
                
                # Add metrics information to the results file metadata
                self._add_metrics_metadata(filepath, self.metrics_collector.get_benchmark_run_id(), entry_ids)
                
            except Exception as e:
                logger.error(f"Failed to store results in centralized metrics database: {e}")
        
        return results, filepath
    
    def _add_metrics_metadata(self, results_filepath: str, benchmark_run_id: str, entry_ids: List[int]):
        """
        Add metrics metadata to the results file.
        
        Args:
            results_filepath: Path to the results JSON file
            benchmark_run_id: ID of the benchmark run in the metrics database
            entry_ids: List of database entry IDs
        """
        try:
            import json
            
            # Create metadata file
            metadata_filepath = results_filepath.replace('.json', '_metrics_metadata.json')
            
            metadata = {
                "benchmark_run_id": benchmark_run_id,
                "centralized_metrics_enabled": True,
                "database_entry_ids": entry_ids,
                "total_entries": len(entry_ids),
                "results_file": os.path.basename(results_filepath),
                "created_at": str(datetime.now())
            }
            
            with open(metadata_filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved metrics metadata to {metadata_filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics metadata: {e}")
    
    def get_centralized_metrics_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get a summary of the current benchmark run from the centralized metrics database.
        
        Returns:
            Optional[Dict[str, Any]]: Summary of the benchmark run or None if not available
        """
        if not self.enable_centralized_metrics or not self.metrics_collector:
            return None
        
        try:
            from app.utils.metrics_integration import get_benchmark_summary
            
            benchmark_run_id = self.metrics_collector.get_benchmark_run_id()
            return get_benchmark_summary(benchmark_run_id)
            
        except Exception as e:
            logger.error(f"Failed to get centralized metrics summary: {e}")
            return None
    
    def import_existing_results_to_centralized_db(self, results_file: str) -> Dict[str, Any]:
        """
        Import existing benchmark results to the centralized metrics database.
        
        Args:
            results_file: Path to the existing results JSON file
            
        Returns:
            Dict[str, Any]: Import summary
        """
        if not self.enable_centralized_metrics:
            raise ValueError("Centralized metrics are not enabled")
        
        try:
            from app.utils.metrics_integration import import_json_results
            
            logger.info(f"Importing existing results from {results_file}")
            
            # Import the results
            import_summary = import_json_results(results_file)
            
            logger.info(f"Successfully imported {import_summary['imported_count']} results")
            logger.info(f"Benchmark run ID: {import_summary['benchmark_run_id']}")
            
            return import_summary
            
        except Exception as e:
            logger.error(f"Failed to import existing results: {e}")
            raise


# Convenience function to create an enhanced benchmark instance
def create_enhanced_benchmark(**kwargs) -> EnhancedModelBenchmark:
    """
    Create an enhanced benchmark instance with sensible defaults.
    
    Args:
        **kwargs: Arguments to pass to EnhancedModelBenchmark constructor
        
    Returns:
        EnhancedModelBenchmark: Configured benchmark instance
    """
    return EnhancedModelBenchmark(**kwargs)


# Function to migrate all existing results
def migrate_all_existing_results(results_dir: str = "benchmark_results") -> Dict[str, Any]:
    """
    Migrate all existing benchmark results to the centralized metrics database.
    
    Args:
        results_dir: Directory containing benchmark results
        
    Returns:
        Dict[str, Any]: Migration summary
    """
    import glob
    import json
    from datetime import datetime
    
    # Find all result files
    result_files = []
    patterns = [
        os.path.join(results_dir, "benchmark_*.json"),
        os.path.join(results_dir, "*", "benchmark_*.json")
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern)
        # Filter out analysis files
        result_files.extend([f for f in files if not f.endswith("_analysis.json")])
    
    logger.info(f"Found {len(result_files)} result files to migrate")
    
    # Create enhanced benchmark instance
    benchmark = create_enhanced_benchmark()
    
    # Migrate each file
    migration_results = []
    successful_migrations = 0
    failed_migrations = 0
    total_imported_results = 0
    
    for file_path in result_files:
        try:
            import_summary = benchmark.import_existing_results_to_centralized_db(file_path)
            
            migration_results.append({
                "file_path": file_path,
                "success": True,
                "benchmark_run_id": import_summary["benchmark_run_id"],
                "imported_count": import_summary["imported_count"],
                "error": None
            })
            
            successful_migrations += 1
            total_imported_results += import_summary["imported_count"]
            
        except Exception as e:
            logger.error(f"Failed to migrate {file_path}: {e}")
            
            migration_results.append({
                "file_path": file_path,
                "success": False,
                "benchmark_run_id": None,
                "imported_count": 0,
                "error": str(e)
            })
            
            failed_migrations += 1
    
    summary = {
        "total_files": len(result_files),
        "successful_migrations": successful_migrations,
        "failed_migrations": failed_migrations,
        "total_imported_results": total_imported_results,
        "migration_results": migration_results,
        "migration_timestamp": str(datetime.now())
    }
    
    logger.info(f"Migration completed: {successful_migrations}/{len(result_files)} files migrated")
    logger.info(f"Total imported results: {total_imported_results}")
    
    return summary
