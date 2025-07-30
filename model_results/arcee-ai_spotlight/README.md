# Benchmark Results: arcee-ai/spotlight

## Overview
Comprehensive evaluation of arcee-ai/spotlight model on Auto-check-EGE-math dataset.

**Evaluation Date**: 2025-06-15
**Total Examples**: 122 (across tasks 13-19)
**Total Evaluations**: 366 (3 modes × 122 examples)
**Total Duration**: 46 minutes 29 seconds
**Total Cost**: $0.0000 (pricing information not available)

## Performance Summary

### Overall Metrics by Evaluation Mode

| Evaluation Mode | Accuracy | Quality Score | Avg Score Distance | Evaluations | Avg Time |
|----------------|----------|---------------|-------------------|-------------|----------|
| **Without Answer** | **27.87%** | **64.48%** | **1.04** | **122** | **8.80s** |
| **With Answer** | **26.23%** | **63.18%** | **1.09** | **122** | **6.99s** |
| **With True Solution** | **25.41%** | **59.22%** | **1.16** | **122** | **6.98s** |

### Task-by-Task Performance (Without Answer Mode - Best Performance)

| Task | Examples | Accuracy | Quality Score | Avg Score Distance |
|------|----------|----------|---------------|-------------------|
| Task 13 (Trigonometry) | 21 | 28.6% | 65.2% | 1.02 |
| Task 14 (Stereometry) | 18 | 27.8% | 64.1% | 1.05 |
| Task 15 (Logarithmic inequalities) | 19 | 26.3% | 63.8% | 1.08 |
| Task 16 (Planimetry) | 17 | 29.4% | 65.5% | 0.98 |
| Task 17 (Financial math) | 15 | 26.7% | 62.9% | 1.12 |
| Task 18 (Parametric problems) | 16 | 31.3% | 66.1% | 0.94 |
| Task 19 (Number theory) | 16 | 25.0% | 63.2% | 1.15 |


### Performance Characteristics
- **Best mode**: "Without Answer" shows highest accuracy (27.87%) and quality score (64.48%)
- **Most consistent**: All three modes show similar performance (25-28% accuracy)
- **True solution impact**: With true solution mode shows slightly lower quality score (59.22%)
- **Score distance**: Average 1.0-1.2 points from correct score across all modes
- **Quality vs accuracy**: Quality scores (59-64%) indicate good partial understanding


## Comparison with Other Models

### Performance vs google/gemini-2.5-flash-preview
- **Accuracy**: -15.39% lower (27.87% vs 44.26% without answer)
- **Quality Score**: -6.56% lower (64.48% vs 71.04%)
- **Evaluation Time**: -7.2s faster (8.80s vs 16.0s)
- **Cost**: Significantly lower (free vs $0.32 per mode)

### Performance vs google/gemini-2.0-flash-001
- **Accuracy**: -19.67% lower (27.87% vs 47.54%)
- **Quality Score**: -11.34% lower (64.48% vs 75.82%)
- **Evaluation Time**: +4.6s slower (8.80s vs 4.2s)
- **Cost**: Lower (free vs $0.492)

## Files in this Directory

### Results Files
- `benchmark_all_tasks_spotlight_20250615_121720.json` - With/Without Answer modes results
- `benchmark_all_tasks_spotlight_20250615_121720_analysis.json` - Analysis for both modes
- `benchmark_all_tasks_spotlight_20250615_140030.json` - With True Solution mode results
- `benchmark_all_tasks_spotlight_20250615_140030_analysis.json` - Analysis for true solution mode
- `benchmark_all_tasks_spotlight_combined_20250615_152639.json` - Combined all modes results (corrected)
- `benchmark_all_tasks_spotlight_combined_20250615_152639_analysis.json` - Complete analysis (corrected)

## Technical Details

**Model Information**:
- **Provider**: OpenRouter (Arcee AI)
- **Full model name**: arcee-ai/spotlight
- **Base model**: Qwen 2.5-VL derived (7B parameters)
- **Pricing**: Not available in OpenRouter pricing data
- **Capabilities**: Vision-language model with multimodal understanding

**Evaluation Settings**:
- **Prompt variant**: detailed
- **Include examples**: false
- **Max examples**: all (122)
- **Retry logic**: enabled (max 10 retries)
- **Modes tested**: with_answer, without_answer, with_true_solution
- **Complete coverage**: All three evaluation modes successfully completed

## Performance Insights

### Statistical Metrics (Without Answer Mode)
- **Macro Precision**: 27.20%
- **Macro Recall**: 29.16%
- **Macro F1**: 26.83%
- **Average Evaluation Time**: 8.80s

### Statistical Metrics (With Answer Mode)
- **Macro Precision**: 25.16%
- **Macro Recall**: 27.59%
- **Macro F1**: 24.04%
- **Average Evaluation Time**: 6.99s

### Statistical Metrics (With True Solution Mode)
- **Macro Precision**: 24.58%
- **Macro Recall**: 27.23%
- **Macro F1**: 24.09%
- **Average Evaluation Time**: 6.98s
