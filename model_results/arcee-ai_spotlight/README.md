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

## Key Findings

### Strengths
- **Consistent performance**: Similar accuracy across all three evaluation modes (25-28%)
- **Good quality scoring**: Quality scores 59-64% indicate reasonable partial understanding
- **Fast evaluation**: Average 7.6 seconds per evaluation
- **Stable API integration**: All 366 API calls successful
- **Complete task coverage**: Evaluated across all 7 task types (13-19) with all three modes
- **Decent partial credit**: Quality scores around 60% show good partial understanding

### Performance Characteristics
- **Best mode**: "Without Answer" shows highest accuracy (27.87%) and quality score (64.48%)
- **Most consistent**: All three modes show similar performance (25-28% accuracy)
- **True solution impact**: With true solution mode shows slightly lower quality score (59.22%)
- **Score distance**: Average 1.0-1.2 points from correct score across all modes
- **Quality vs accuracy**: Quality scores (59-64%) indicate good partial understanding

### Model Behavior Analysis
The arcee-ai/spotlight model demonstrates:
- Consistent scoring patterns across different problem types
- Better performance on geometric and parametric problems
- Reasonable quality scores indicating understanding of problem structure
- Stable performance without significant mode-dependent variations

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

### Quality Analysis
- **Score Distribution**: Reasonable spread across score ranges
- **Partial Credit**: High quality scores suggest good partial understanding
- **Consistency**: Stable performance across different task types
- **Error Patterns**: Consistent scoring methodology across evaluations

## Recommendations

### Primary Use Cases
- **Budget-conscious applications**: Free model with reasonable performance
- **Research and experimentation**: Good for testing and development
- **Baseline comparisons**: Useful reference point for other models
- **Educational applications**: Suitable for non-critical educational tools

### Optimal Configuration
- **Recommended mode**: "Without Answer" for slightly better accuracy (27.87%)
- **Best tasks**: Focus on Tasks 16 and 18 for optimal performance
- **Evaluation approach**: Use for scenarios where partial credit is valuable
- **Cost consideration**: Excellent choice for budget-constrained projects

### When to Choose This Model
- **Cost sensitivity**: When budget is a primary constraint
- **Moderate accuracy needs**: When 27-28% accuracy is acceptable
- **Development/testing**: For prototyping and initial development
- **Educational research**: Academic studies with limited funding

### When to Choose Other Models Instead
- **High accuracy requirements**: Choose Gemini models for better performance
- **Critical applications**: Use premium models for production systems
- **Speed requirements**: Consider faster models if evaluation time is critical
- **Advanced reasoning**: Choose models with better mathematical reasoning

## Conclusion

The arcee-ai/spotlight model offers:
- **Moderate accuracy** (27.87% best mode across all three evaluation approaches)
- **Good quality scoring** (59-64% quality scores indicating reasonable partial understanding)
- **Fast evaluation** (7.6s average across all modes)
- **Cost-effective solution** (free through OpenRouter)
- **Complete evaluation coverage** (all three modes: with/without answer, with true solution)

This model serves as an excellent budget option for applications where moderate accuracy is sufficient and cost is a primary concern. While it doesn't match the performance of premium models like Gemini, it provides reasonable results for educational, research, and development purposes. The quality scores of 59-64% across all evaluation modes suggest the model has good partial understanding of mathematical problems, often providing solutions that earn partial credit even when not achieving perfect scores. This makes it particularly valuable for scenarios where partial credit recognition and cost-effectiveness are important considerations.
