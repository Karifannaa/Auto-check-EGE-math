# Centralized Metrics Collection System

This document describes the new centralized metrics collection system that aggregates all evaluation metrics into a unified database structure.

## Overview

The centralized metrics system provides:

- **Unified Storage**: All metrics are stored in a single SQLite database table
- **Categorization**: Metrics are automatically categorized by model type, task type, and evaluation configuration
- **Aggregation**: Automatic calculation of derived metrics like accuracy, quality scores, and normalized distances
- **API Access**: RESTful API endpoints for querying and analyzing metrics
- **Backward Compatibility**: Existing benchmark scripts continue to work unchanged

## Database Schema

The central `metrics` table includes the following key columns:

### Identification
- `id`: Primary key
- `timestamp`: When the evaluation was performed
- `benchmark_run_id`: Groups related evaluations together

### Model Information
- `model_id`: Full model identifier (e.g., "google/gemini-2.0-flash-exp:free")
- `model_name`: Human-readable model name
- `model_type`: "reasoning" or "non_reasoning"

### Task Information
- `task_id`: Task number (e.g., "13", "14")
- `task_type`: Full task type (e.g., "task_13", "task_14")
- `solution_id`: Specific solution identifier (e.g., "13.1.1")

### Evaluation Configuration
- `use_answer`: Whether answer images were used
- `use_true_solution`: Whether true solution images were used
- `prompt_variant`: Prompt variant used
- `temperature`: Model temperature setting

### Performance Metrics
- `score`: Predicted score
- `expected_score`: Ground truth score
- `accuracy`: Whether prediction was exactly correct
- `score_distance`: Absolute difference between predicted and expected
- `normalized_distance`: Distance normalized by maximum possible score
- `quality_score`: 1 - normalized_distance (0-1 scale)

### Resource Metrics
- `evaluation_time`: Time taken for evaluation (seconds)
- `prompt_tokens`, `completion_tokens`, `total_tokens`: Token usage
- `actual_cost`: Cost in USD

## Installation

1. **Install Dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Initialize Database**:
   ```bash
   python metrics_cli.py init
   ```

## Usage

### 1. Using the Enhanced Benchmark System

The new `EnhancedModelBenchmark` class automatically stores results in the centralized database:

```python
from dataset_benchmark.enhanced_benchmark_models import create_enhanced_benchmark

# Create enhanced benchmark instance
benchmark = create_enhanced_benchmark()

# Run benchmark (automatically stores in centralized DB)
results, filepath = await benchmark.run_benchmark(
    task_id="13",
    model_ids=["google/gemini-2.0-flash-exp:free"],
    with_true_solution=True,
    max_examples=10
)

# Get centralized metrics summary
summary = benchmark.get_centralized_metrics_summary()
print(f"Benchmark Run ID: {summary['benchmark_run_id']}")
```

### 2. Migrating Existing Results

#### Option A: Use the Migration Script
```bash
python migrate_existing_results.py --results-dir dataset_benchmark/benchmark_results
```

#### Option B: Use the CLI Tool
```bash
# Import a specific file
python metrics_cli.py import dataset_benchmark/benchmark_results/benchmark_task13_gemini-2.0-flash-exp_20250511_150610.json

# Import with custom run ID
python metrics_cli.py import results.json --run-id my-custom-run-id
```

### 3. Using the CLI Tool

```bash
# Show overall summary
python metrics_cli.py summary

# Show summary for specific benchmark run
python metrics_cli.py summary --run-id benchmark_task13_gemini-2.0-flash-exp_20250511_150610

# List all benchmark runs
python metrics_cli.py list

# Export metrics to JSON
python metrics_cli.py export all_metrics.json
python metrics_cli.py export run_metrics.json --run-id specific-run-id
```

### 4. Using the API Endpoints

Start the FastAPI server:
```bash
cd backend
uvicorn app.main:app --reload
```

#### Get Metrics
```bash
# Get all metrics
curl "http://localhost:8000/api/v1/metrics/"

# Filter by model
curl "http://localhost:8000/api/v1/metrics/?model_ids=google/gemini-2.0-flash-exp:free"

# Filter by task type
curl "http://localhost:8000/api/v1/metrics/?task_types=task_13"

# Get with pagination
curl "http://localhost:8000/api/v1/metrics/?limit=50&offset=0"
```

#### Get Aggregated Metrics
```bash
# Group by model and task
curl "http://localhost:8000/api/v1/metrics/aggregated?group_by=model_id&group_by=task_type"

# Group by model type only
curl "http://localhost:8000/api/v1/metrics/aggregated?group_by=model_type"
```

#### Get Summary
```bash
curl "http://localhost:8000/api/v1/metrics/summary"
```

#### Import Results via API
```bash
curl -X POST "http://localhost:8000/api/v1/metrics/import" \
     -H "Content-Type: application/json" \
     -d '{"file_path": "path/to/results.json", "benchmark_run_id": "optional-run-id"}'
```

### 5. Programmatic Access

```python
from backend.app.database.config import SessionLocal
from backend.app.services.metrics_service import MetricsService

# Create database session
db = SessionLocal()
metrics_service = MetricsService(db)

# Get metrics with filtering
metrics = metrics_service.get_metrics(
    model_ids=["google/gemini-2.0-flash-exp:free"],
    task_types=["task_13"],
    use_true_solution=True
)

# Get aggregated metrics
aggregated = metrics_service.get_aggregated_metrics(
    group_by=["model_id", "task_type"],
    task_types=["task_13"]
)

# Store new evaluation result
result = {
    "solution_id": "13.1.1",
    "task_id": "13",
    "task_type": "task_13",
    "model_id": "google/gemini-2.0-flash-exp:free",
    "score": 2,
    "expected_score": 2,
    "evaluation_time": 5.2,
    "prompt_tokens": 1500,
    "completion_tokens": 300,
    "total_tokens": 1800,
    "cost": 0.002
}

entry = metrics_service.store_evaluation_result(result)
print(f"Stored with ID: {entry.id}")

db.close()
```

## Key Features

### 1. Automatic Metric Calculation

The system automatically calculates derived metrics:

- **Accuracy**: Whether predicted score exactly matches expected score
- **Score Distance**: Absolute difference between predicted and expected scores
- **Normalized Distance**: Score distance divided by maximum possible score for the task
- **Quality Score**: 1 - normalized_distance (higher is better, 0-1 scale)

### 2. Model Type Classification

Models are automatically classified as:
- **Reasoning Models**: Models with built-in reasoning capabilities
- **Non-Reasoning Models**: Standard language models

### 3. Flexible Querying

The API supports filtering by:
- Model IDs or model types
- Task IDs or task types
- Evaluation configuration (with/without answer, with/without true solution)
- Benchmark run IDs
- Date ranges

### 4. Aggregation and Analysis

Get aggregated metrics grouped by any combination of:
- Model ID or model type
- Task ID or task type
- Evaluation configuration
- Custom groupings

## File Structure

```
backend/
├── app/
│   ├── database/
│   │   ├── __init__.py
│   │   ├── config.py          # Database configuration
│   │   └── models.py          # SQLAlchemy models
│   ├── services/
│   │   ├── __init__.py
│   │   └── metrics_service.py # Metrics business logic
│   ├── routers/
│   │   └── metrics.py         # API endpoints
│   └── utils/
│       └── metrics_integration.py # Integration utilities
dataset_benchmark/
├── enhanced_benchmark_models.py   # Enhanced benchmark class
└── benchmark_models.py           # Original benchmark class
migrate_existing_results.py       # Migration script
metrics_cli.py                    # CLI tool
```

## Backward Compatibility

The existing `benchmark_models.py` continues to work unchanged. The new system is additive:

- Existing scripts continue to save JSON files as before
- New enhanced scripts additionally store in the centralized database
- Migration tools can import existing JSON files into the database
- All existing analysis tools continue to work

## Benefits

1. **Centralized Analysis**: All metrics in one place for comprehensive analysis
2. **Better Querying**: SQL-based filtering and aggregation
3. **Historical Tracking**: Track performance trends over time
4. **Resource Monitoring**: Monitor costs and token usage across all evaluations
5. **Standardized Schema**: Consistent data structure across all evaluations
6. **API Access**: Programmatic access to metrics data
7. **Scalability**: Database-backed storage scales better than individual JSON files
