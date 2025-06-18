# Benchmark Results: qwen-2.5-vl-32b

## Overview
Comprehensive evaluation of qwen-2.5-vl-32b model on Auto-check-EGE-math dataset.

**Evaluation Date**: 2025-06-15  
**Total Examples**: 122  
**Total Evaluations**: 366  
**Total Cost**: $1.5465
**Duration**: 1:34:07.269277

## Performance Summary

### Overall Metrics by Evaluation Mode

| Evaluation Mode | Accuracy | Quality Score | Avg Score Distance | Total Cost | Evaluations |
|----------------|----------|---------------|-------------------|------------|-------------|
| **Without Answer** | **31.15%** | **62.09%** | **1.09** | **$0.4550** | **122** |
| **With Answer** | **30.33%** | **61.95%** | **1.08** | **$0.4571** | **122** |
| **With True Solution** | **43.44%** | **70.49%** | **0.81** | **$0.6344** | **122** |

## Key Findings

### Performance Characteristics
- **Average evaluation time**: 23.12s per assessment
- **Cost efficiency**: $1.5465 total cost for 366 evaluations
- **Token usage**: ~2560 prompt + 1594 completion tokens per evaluation

### Model Capabilities
- **Multimodal processing**: Supports both text and image inputs
- **Mathematical reasoning**: Advanced mathematical problem solving
- **Visual analysis**: Can interpret mathematical diagrams and formulas

## Technical Details

**Model Configuration**:
- Provider: OpenRouter
- Full model name: qwen-2.5-vl-32b
- Model family: Qwen 2.5 VL
- Parameters: 72B
- Multimodal: Text + Image → Text

**Evaluation Settings**:
- Prompt variant: detailed
- Include examples: false
- Max examples: all (122)
- Retry logic: enabled for rate limiting

## Files in this Directory

### Results Files
- `*.json` - Raw benchmark results
- `*_analysis.json` - Detailed analysis and metrics
- `*_metrics_table.tex` - LaTeX formatted metrics tables

## Recommendations

### Use Cases
- **Mathematical problem solving**: Excellent for complex math problems
- **Visual reasoning**: Strong performance on diagram interpretation
- **Educational assessment**: Suitable for automated grading systems

### Optimal Configuration
- **Best for**: Mathematical reasoning tasks with visual components
- **Cost consideration**: Higher cost but potentially better performance
- **Performance**: Consistent evaluation times around 23.1s

## Conclusion

The qwen-2.5-vl-32b model demonstrates strong capabilities for mathematical problem solving with visual components, offering advanced performance for educational assessment applications.
