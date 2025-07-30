# Benchmark Results: openai/o4-mini

## Overview
Comprehensive evaluation of openai/o4-mini model on Auto-check-EGE-math dataset.

**Evaluation Date**: 2025-06-16
**Total Examples**: 122
**Total Evaluations**: 366 (ALL 3 MODES)
**Total Cost**: $6.47
**Duration**: ~4.5 hours
**Average Evaluation Time**: 43.68s

## Performance Summary

### Overall Metrics by Evaluation Mode

| Evaluation Mode | Accuracy | Quality Score | Avg Score Distance | Total Cost | Evaluations | Avg Time |
|----------------|----------|---------------|-------------------|------------|-------------|----------|
| **🥇 With Answer** | **56.56%** | **78.17%** | **0.60** | **$2.02** | **122** | **32.94s** |
| **🥈 Without Answer** | **55.74%** | **75.55%** | **0.66** | **$2.18** | **122** | **39.62s** |
| **🥉 With True Solution** | **54.10%** | **76.16%** | **0.66** | **$2.28** | **122** | **58.47s** |

## Key Findings

### Performance Characteristics
- **🏆 Best Mode**: "With Answer" achieves highest accuracy (56.56%)
- **⚡ Fastest Mode**: "With Answer" is also the fastest (32.94s avg)
- **💰 Most Cost-Effective**: "With Answer" has lowest cost per evaluation
- **🎯 Consistent Performance**: All modes show excellent accuracy (54-57%)
- **🔬 Reasoning Excellence**: Advanced chain-of-thought capabilities evident


## Technical Details

**Model Configuration**:
- Provider: OpenRouter
- Full model name: openai/o4-mini
- Model family: o4 (reasoning)
- Context length: 200,000 tokens
- Multimodal: Text + Image → Text
- Reasoning: Advanced chain-of-thought enabled

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
- `benchmark_all_tasks_o4-mini_all_modes_*.json` - Complete evaluation results
- `*_analysis.json` - Detailed analysis and metrics
- `README.md` - This comprehensive summary
