# Benchmark Results: google/gemini-2.0-flash-lite-001

## Overview
Comprehensive evaluation of google/gemini-2.0-flash-lite-001 model on Auto-check-EGE-math dataset.

**Evaluation Date**: 2025-06-14  
**Total Examples**: 122 (across tasks 13-19)  
**Total Evaluations**: 366 (3 modes × 122 examples)  
**Total Cost**: $0.107  

## Performance Summary

### Overall Metrics by Evaluation Mode

| Evaluation Mode | Accuracy | Quality Score | Avg Score Distance | Total Cost | Evaluations |
|----------------|----------|---------------|-------------------|------------|-------------|
| **With True Solution** | **38.52%** | **70.22%** | **0.84** | **$0.0369** | **122** |
| **With Answer** | **35.25%** | **67.83%** | **0.90** | **$0.0350** | **122** |
| **Without Answer** | **31.97%** | **64.96%** | **1.00** | **$0.0351** | **122** |

### Task-by-Task Performance (With True Solution Mode)

| Task | Examples | Accuracy | Avg Score | Expected Score | Cost |
|------|----------|----------|-----------|----------------|------|
| Task 13 | 21 | **57.1%** | 1.38 | 0.95 | $0.0059 |
| Task 14 | 18 | 22.2% | 1.50 | 1.28 | $0.0056 |
| Task 15 | 19 | 31.6% | 1.26 | 1.11 | $0.0052 |
| Task 16 | 17 | **52.9%** | 1.29 | 1.29 | $0.0051 |
| Task 17 | 15 | 40.0% | 0.87 | 1.20 | $0.0048 |
| Task 18 | 16 | 25.0% | 2.62 | 2.38 | $0.0054 |
| Task 19 | 16 | 37.5% | 2.06 | 2.06 | $0.0049 |


### Performance Characteristics
- **Best mode**: "With True Solution" provides highest accuracy
- **Task variation**: Tasks 13 and 16 show strongest performance
- **Scoring tendency**: Slightly more generous than expected scores
- **Speed**: Fast evaluation suitable for large-scale assessment

### Score Distribution Analysis
The model tends to give slightly higher scores than expected:
- Predicted 2-point scores: 39.3% vs Expected: 28.7% (+10.6%)
- Predicted 0-point scores: 14.8% vs Expected: 23.0% (-8.2%)

## Files in this Directory

### Results Files
- `benchmark_all_tasks_gemini-2.0-flash-lite-001_20250614_164756.json` - With True Solution mode results
- `benchmark_all_tasks_gemini-2.0-flash-lite-001_20250614_164756_analysis.json` - Analysis for True Solution mode
- `benchmark_all_tasks_gemini-2.0-flash-lite-001_20250614_170136.json` - With/Without Answer modes results
- `benchmark_all_tasks_gemini-2.0-flash-lite-001_20250614_170136_analysis.json` - Analysis for both modes

### LaTeX Tables
- `benchmark_all_tasks_gemini-2.0-flash-lite-001_20250614_164756_metrics_table.tex` - Metrics table (True Solution)
- `benchmark_all_tasks_gemini-2.0-flash-lite-001_20250614_170136_metrics_table.tex` - Metrics table (All modes)

## Technical Details

**Model Configuration**:
- Provider: OpenRouter
- Full model name: google/gemini-2.0-flash-lite-001
- Pricing: $0.075/1M input tokens, $0.30/1M output tokens
- Average tokens per evaluation: ~3,066 prompt + 266 completion

**Evaluation Settings**:
- Prompt variant: detailed
- Include examples: false
- Max examples: all (122)
- Retry logic: enabled for rate limiting
