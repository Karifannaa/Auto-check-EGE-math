# Benchmark Results: qwen/qwen2.5-vl-32b-instruct

## Overview
Comprehensive evaluation of qwen/qwen2.5-vl-32b-instruct model on Auto-check-EGE-math dataset.

**Evaluation Date**: 2025-06-15  
**Total Examples**: 122  
**Total Evaluations**: 366 (ALL 3 MODES)  
**Total Cost**: $1.55  
**Duration**: ~2.5 hours (across all modes)

## Performance Summary

### Overall Metrics by Evaluation Mode

| Evaluation Mode | Accuracy | Quality Score | Avg Score Distance | Total Cost | Evaluations |
|----------------|----------|---------------|-------------------|------------|-------------|
| **Without Answer** | **31.15%** | **62.09%** | **1.09** | **$0.46** | **122** |
| **With Answer** | **30.33%** | **61.95%** | **1.08** | **$0.46** | **122** |
| **With True Solution** | **43.44%** | **70.49%** | **0.81** | **$0.63** | **122** |

## Key Findings

### Performance Comparison
1. **🥇 With True Solution**: Best performance (43.44% accuracy, 70.49% quality)
2. **🥈 Without Answer**: Moderate performance (31.15% accuracy, 62.09% quality)  
3. **🥉 With Answer**: Similar to without answer (30.33% accuracy, 61.95% quality)

### Performance Characteristics
- **Average evaluation time**: ~25s per assessment
- **Cost efficiency**: $1.55 total cost for 366 evaluations (~$0.004 per evaluation)
- **Token usage**: ~2,500-3,500 tokens per evaluation
- **Best mode**: With True Solution shows 40%+ improvement in accuracy

### Model Capabilities
- **Multimodal processing**: Excellent support for text and image inputs
- **Mathematical reasoning**: Strong mathematical problem solving capabilities
- **Visual analysis**: Can interpret mathematical diagrams and formulas effectively
- **Reference-based evaluation**: Significantly better when provided with true solutions

## Technical Details

**Model Configuration**:
- Provider: OpenRouter
- Full model name: qwen/qwen2.5-vl-32b-instruct
- Model family: Qwen 2.5 VL
- Parameters: 32B
- Context length: 128,000 tokens
- Multimodal: Text + Image → Text

**Evaluation Settings**:
- Prompt variant: detailed
- Include examples: false
- Max examples: all (122)
- Retry logic: enabled for rate limiting
- All 3 evaluation modes tested

## Task Performance

### Evaluations by Task Type
- Task 13 (Trigonometric/Logarithmic/Exponential): 63 evaluations
- Task 14 (Stereometric): 54 evaluations
- Task 15 (Inequalities): 57 evaluations
- Task 16 (Planimetric): 51 evaluations
- Task 17 (Planimetric with proof): 45 evaluations
- Task 18 (Parameter problems): 48 evaluations
- Task 19 (Number theory): 48 evaluations

## Files in this Directory

### Results Files
- `benchmark_all_tasks_qwen2.5-vl-32b-instruct_*_both_approaches_*.json` - With/Without answer results
- `benchmark_all_tasks_qwen2.5-vl-32b-instruct_*_true_solution_*.json` - True solution results
- `*_analysis.json` - Detailed analysis and metrics for each mode
- `*_metrics_table.tex` - LaTeX formatted metrics tables
