# Benchmark Results: google/gemini-2.0-flash-001

## Overview
Comprehensive evaluation of google/gemini-2.0-flash-001 model on Auto-check-EGE-math dataset.

**Evaluation Date**: 2025-06-14  
**Total Examples**: 122 (across tasks 13-19)  
**Total Evaluations**: 366 (3 modes × 122 examples)  
**Total Cost**: $0.492  

## Performance Summary

### Overall Metrics by Evaluation Mode

| Evaluation Mode | Accuracy | Quality Score | Avg Score Distance | Total Cost | Evaluations |
|----------------|----------|---------------|-------------------|------------|-------------|
| **With True Solution** | **46.72%** | **75.82%** | **0.71** | **$0.2057** | **122** |
| **With Answer** | **47.54%** | **74.04%** | **0.75** | **$0.1445** | **122** |
| **Without Answer** | **36.89%** | **71.04%** | **0.84** | **$0.1422** | **122** |

### Task-by-Task Performance (With True Solution Mode)

| Task | Examples | Accuracy | Avg Score | Expected Score | Cost |
|------|----------|----------|-----------|----------------|------|
| Task 13 | 21 | **61.9%** | 1.48 | 0.95 | $0.0295 |
| Task 14 | 18 | 33.3% | 1.61 | 1.28 | $0.0284 |
| Task 15 | 19 | 42.1% | 1.42 | 1.11 | $0.0276 |
| Task 16 | 17 | **58.8%** | 1.47 | 1.29 | $0.0270 |
| Task 17 | 15 | 40.0% | 0.93 | 1.20 | $0.0254 |
| Task 18 | 16 | 37.5% | 2.50 | 2.38 | $0.0286 |
| Task 19 | 16 | 43.8% | 2.31 | 2.06 | $0.0392 |


### Performance Characteristics
- **Best mode**: "With Answer" shows highest accuracy (47.54%)
- **Task excellence**: Tasks 13 (61.9%) and 16 (58.8%) show exceptional performance
- **Balanced evaluation**: Good performance across all three evaluation modes
- **Speed**: Consistent 3-4 second evaluation time per assessment

### Score Distribution Analysis
The model demonstrates more accurate scoring compared to gemini-2.0-flash-lite-001:
- Better alignment with expected score distributions
- More conservative and precise scoring approach
- Reduced tendency for over-generous scoring

## Comparison with gemini-2.0-flash-lite-001

### Performance Improvements
- **Accuracy**: +8.2% improvement (46.72% vs 38.52%)
- **Quality Score**: +5.6% improvement (75.82% vs 70.22%)
- **Score Distance**: -0.13 improvement (0.71 vs 0.84)
- **Task Performance**: Better across all task types

### Cost Analysis
- **Cost per evaluation**: ~$0.0017 (vs $0.0003 for lite version)
- **Cost increase**: ~5.7x higher cost
- **Performance/cost ratio**: Significant quality improvement justifies cost increase
- **Total cost**: $0.492 for 366 evaluations

## Files in this Directory

### Results Files
- `benchmark_all_tasks_gemini-2.0-flash-001_20250614_171902.json` - With True Solution mode results
- `benchmark_all_tasks_gemini-2.0-flash-001_20250614_171902_analysis.json` - Analysis for True Solution mode
- `benchmark_all_tasks_gemini-2.0-flash-001_20250614_173838.json` - With/Without Answer modes results
- `benchmark_all_tasks_gemini-2.0-flash-001_20250614_173838_analysis.json` - Analysis for both modes

### LaTeX Tables
- `benchmark_all_tasks_gemini-2.0-flash-001_20250614_171902_metrics_table.tex` - Metrics table (True Solution)
- `benchmark_all_tasks_gemini-2.0-flash-001_20250614_173838_metrics_table.tex` - Metrics table (All modes)

## Technical Details

**Model Configuration**:
- Provider: OpenRouter
- Full model name: google/gemini-2.0-flash-001
- Pricing: $0.125/1M input tokens, $0.375/1M output tokens
- Average tokens per evaluation: ~3,200 prompt + 280 completion

**Evaluation Settings**:
- Prompt variant: detailed
- Include examples: false
- Max examples: all (122)
- Retry logic: enabled for rate limiting

