# Comprehensive Model Evaluation Results

**Generated:** 2025-06-16 16:42:59  
**Total Models Evaluated:** 7  
**Evaluation Framework:** Russian Math Exam Solutions Benchmark

## Executive Summary

This report presents a comprehensive analysis of all model evaluation results across the Russian Math Exam Solutions benchmark. The evaluation covers tasks 13-19 with three different evaluation modes: without answer images, with answer images, and with true solution images.

### Key Findings

- **Best Overall Model:** openai/o4-mini
- **Total Evaluations:** 1952
- **Total Cost:** $9.5924

## Summary Results Table

| Model | Mode | Accuracy (%) | Quality Score (%) | Avg Score Distance | Evaluations | Total Cost ($) | Avg Time (s) |
|-------|------|--------------|-------------------|-------------------|-------------|----------------|--------------|
| arcee-ai/spotlight | Without Answer | 27.87 | 64.48 | 1.041 | 122 | 0.0000 | 8.80 |
|  | With Answer | 26.23 | 63.18 | 1.090 | 122 | 0.0000 | 6.99 |
|  | With True Solution | 25.41 | 59.22 | 1.156 | 122 | 0.0000 | 6.98 |
| google/gemini-2.0-flash-001 | Without Answer | 36.89 | 71.04 | 0.844 | 122 | 0.1422 | 4.56 |
|  | With Answer | 47.54 | 74.04 | 0.754 | 122 | 0.1445 | 4.82 |
| google/gemini-2.0-flash-lite-001 | Without Answer | 31.97 | 64.96 | 1.000 | 122 | 0.0351 | 3.08 |
|  | With Answer | 35.25 | 67.83 | 0.902 | 122 | 0.0350 | 3.13 |
| google/gemini-2.5-flash-preview | Without Answer | 44.26 | 71.04 | 0.811 | 122 | 0.3214 | 16.08 |
|  | With Answer | 40.98 | 70.49 | 0.820 | 122 | 0.3048 | 14.92 |
| google/gemini-2.5-flash-preview:thinking | Without Answer | 40.16 | 64.30 | 1.046 | 122 | 0.6036 | 39.48 |
|  | With Answer | 42.62 | 66.44 | 0.991 | 122 | 0.6198 | 39.98 |
| openai/o4-mini | Without Answer | 55.74 | 75.55 | 0.664 | 122 | 2.1788 | 39.62 |
|  | With Answer | 56.56 | 78.17 | 0.595 | 122 | 2.0174 | 32.94 |
|  | With True Solution | 54.10 | 76.16 | 0.656 | 122 | 2.2779 | 58.47 |
| qwen/qwen2.5-vl-32b-instruct | Without Answer | 31.15 | 62.09 | 1.090 | 122 | 0.4550 | 22.97 |
|  | With Answer | 30.33 | 61.95 | 1.082 | 122 | 0.4571 | 23.27 |


## Detailed Model Analysis

## 1. openai/o4-mini

**Overall Performance:** 76.63% average quality score
**Total Evaluations:** 366
**Total Cost:** $6.4740

### Without Answer

- **Accuracy:** 55.74%
- **Quality Score:** 75.55%
- **Average Score Distance:** 0.664
- **Evaluations:** 122
- **Average Evaluation Time:** 39.62s
- **Total Cost:** $2.1788
- **Macro Precision:** 52.46%
- **Macro Recall:** 55.28%
- **Macro F1:** 52.69%

### With Answer

- **Accuracy:** 56.56%
- **Quality Score:** 78.17%
- **Average Score Distance:** 0.595
- **Evaluations:** 122
- **Average Evaluation Time:** 32.94s
- **Total Cost:** $2.0174
- **Macro Precision:** 57.65%
- **Macro Recall:** 57.34%
- **Macro F1:** 56.84%

### With True Solution

- **Accuracy:** 54.10%
- **Quality Score:** 76.16%
- **Average Score Distance:** 0.656
- **Evaluations:** 122
- **Average Evaluation Time:** 58.47s
- **Total Cost:** $2.2779
- **Macro Precision:** 54.96%
- **Macro Recall:** 53.31%
- **Macro F1:** 53.02%

### Performance Insights

**Best performing mode:** With Answer (Quality: 78.17%, Accuracy: 56.56%)

**Performance gap:** 2.62 percentage points between best and worst modes

**Cost Efficiency:** $0.017689 per evaluation

---

## 2. google/gemini-2.0-flash-001

**Overall Performance:** 72.54% average quality score
**Total Evaluations:** 244
**Total Cost:** $0.2867

### Without Answer

- **Accuracy:** 36.89%
- **Quality Score:** 71.04%
- **Average Score Distance:** 0.844
- **Evaluations:** 122
- **Average Evaluation Time:** 4.56s
- **Total Cost:** $0.1422
- **Macro Precision:** 31.08%
- **Macro Recall:** 27.56%
- **Macro F1:** 27.22%

### With Answer

- **Accuracy:** 47.54%
- **Quality Score:** 74.04%
- **Average Score Distance:** 0.754
- **Evaluations:** 122
- **Average Evaluation Time:** 4.82s
- **Total Cost:** $0.1445
- **Macro Precision:** 47.24%
- **Macro Recall:** 42.59%
- **Macro F1:** 43.20%

### Performance Insights

**Best performing mode:** With Answer (Quality: 74.04%, Accuracy: 47.54%)

**Performance gap:** 3.01 percentage points between best and worst modes

**Cost Efficiency:** $0.001175 per evaluation

---

## 3. google/gemini-2.5-flash-preview

**Overall Performance:** 70.77% average quality score
**Total Evaluations:** 244
**Total Cost:** $0.6261

### Without Answer

- **Accuracy:** 44.26%
- **Quality Score:** 71.04%
- **Average Score Distance:** 0.811
- **Evaluations:** 122
- **Average Evaluation Time:** 16.08s
- **Total Cost:** $0.3214
- **Macro Precision:** 41.85%
- **Macro Recall:** 36.85%
- **Macro F1:** 38.09%

### With Answer

- **Accuracy:** 40.98%
- **Quality Score:** 70.49%
- **Average Score Distance:** 0.820
- **Evaluations:** 122
- **Average Evaluation Time:** 14.92s
- **Total Cost:** $0.3048
- **Macro Precision:** 41.43%
- **Macro Recall:** 39.27%
- **Macro F1:** 40.08%

### Performance Insights

**Best performing mode:** Without Answer (Quality: 71.04%, Accuracy: 44.26%)

**Performance gap:** 0.55 percentage points between best and worst modes

**Cost Efficiency:** $0.002566 per evaluation

---

## 4. google/gemini-2.0-flash-lite-001

**Overall Performance:** 66.39% average quality score
**Total Evaluations:** 244
**Total Cost:** $0.0701

### Without Answer

- **Accuracy:** 31.97%
- **Quality Score:** 64.96%
- **Average Score Distance:** 1.000
- **Evaluations:** 122
- **Average Evaluation Time:** 3.08s
- **Total Cost:** $0.0351
- **Macro Precision:** 24.87%
- **Macro Recall:** 36.74%
- **Macro F1:** 29.07%

### With Answer

- **Accuracy:** 35.25%
- **Quality Score:** 67.83%
- **Average Score Distance:** 0.902
- **Evaluations:** 122
- **Average Evaluation Time:** 3.13s
- **Total Cost:** $0.0350
- **Macro Precision:** 28.29%
- **Macro Recall:** 38.20%
- **Macro F1:** 32.24%

### Performance Insights

**Best performing mode:** With Answer (Quality: 67.83%, Accuracy: 35.25%)

**Performance gap:** 2.87 percentage points between best and worst modes

**Cost Efficiency:** $0.000287 per evaluation

---

## 5. google/gemini-2.5-flash-preview:thinking

**Overall Performance:** 65.37% average quality score
**Total Evaluations:** 244
**Total Cost:** $1.2233

### Without Answer

- **Accuracy:** 40.16%
- **Quality Score:** 64.30%
- **Average Score Distance:** 1.046
- **Evaluations:** 122
- **Average Evaluation Time:** 39.48s
- **Total Cost:** $0.6036
- **Macro Precision:** 57.34%
- **Macro Recall:** 33.42%
- **Macro F1:** 36.11%

### With Answer

- **Accuracy:** 42.62%
- **Quality Score:** 66.44%
- **Average Score Distance:** 0.991
- **Evaluations:** 122
- **Average Evaluation Time:** 39.98s
- **Total Cost:** $0.6198
- **Macro Precision:** 38.93%
- **Macro Recall:** 33.14%
- **Macro F1:** 32.66%

### Performance Insights

**Best performing mode:** With Answer (Quality: 66.44%, Accuracy: 42.62%)

**Performance gap:** 2.14 percentage points between best and worst modes

**Cost Efficiency:** $0.005014 per evaluation

---

## 6. arcee-ai/spotlight

**Overall Performance:** 62.30% average quality score
**Total Evaluations:** 366
**Total Cost:** $0.0000

### Without Answer

- **Accuracy:** 27.87%
- **Quality Score:** 64.48%
- **Average Score Distance:** 1.041
- **Evaluations:** 122
- **Average Evaluation Time:** 8.80s
- **Total Cost:** $0.0000
- **Macro Precision:** 27.20%
- **Macro Recall:** 29.16%
- **Macro F1:** 26.83%

### With Answer

- **Accuracy:** 26.23%
- **Quality Score:** 63.18%
- **Average Score Distance:** 1.090
- **Evaluations:** 122
- **Average Evaluation Time:** 6.99s
- **Total Cost:** $0.0000
- **Macro Precision:** 25.16%
- **Macro Recall:** 27.59%
- **Macro F1:** 24.04%

### With True Solution

- **Accuracy:** 25.41%
- **Quality Score:** 59.22%
- **Average Score Distance:** 1.156
- **Evaluations:** 122
- **Average Evaluation Time:** 6.98s
- **Total Cost:** $0.0000
- **Macro Precision:** 24.58%
- **Macro Recall:** 27.23%
- **Macro F1:** 24.09%

### Performance Insights

**Best performing mode:** Without Answer (Quality: 64.48%, Accuracy: 27.87%)

**Performance gap:** 5.26 percentage points between best and worst modes

---

## 7. qwen/qwen2.5-vl-32b-instruct

**Overall Performance:** 62.02% average quality score
**Total Evaluations:** 244
**Total Cost:** $0.9121

### Without Answer

- **Accuracy:** 31.15%
- **Quality Score:** 62.09%
- **Average Score Distance:** 1.090
- **Evaluations:** 122
- **Average Evaluation Time:** 22.97s
- **Total Cost:** $0.4550
- **Macro Precision:** 27.99%
- **Macro Recall:** 33.75%
- **Macro F1:** 28.48%

### With Answer

- **Accuracy:** 30.33%
- **Quality Score:** 61.95%
- **Average Score Distance:** 1.082
- **Evaluations:** 122
- **Average Evaluation Time:** 23.27s
- **Total Cost:** $0.4571
- **Macro Precision:** 32.74%
- **Macro Recall:** 33.27%
- **Macro F1:** 27.32%

### Performance Insights

**Best performing mode:** Without Answer (Quality: 62.09%, Accuracy: 31.15%)

**Performance gap:** 0.14 percentage points between best and worst modes

**Cost Efficiency:** $0.003738 per evaluation

---


## Methodology

### Evaluation Metrics

- **Accuracy:** Percentage of exact matches between predicted and expected scores
- **Quality Score:** Normalized measure (0-100%) indicating prediction closeness relative to maximum possible error
- **Average Score Distance:** Mean absolute difference between predicted and expected scores
- **Macro Precision/Recall/F1:** Multi-class classification metrics averaged across score classes

### Evaluation Modes

1. **Without Answer:** Model evaluates solutions without seeing answer images
2. **With Answer:** Model evaluates solutions with access to answer images  
3. **With True Solution:** Model evaluates with access to true solution images

### Task Coverage

The benchmark covers Russian Math Exam tasks 13-19, each with different maximum scores:
- Task 13: Max score 2
- Task 14: Max score 3  
- Task 15: Max score 2
- Task 16: Max score 3
- Task 17: Max score 3
- Task 18: Max score 4
- Task 19: Max score 4

---

*This analysis was generated from validated evaluation results with confirmed mathematical accuracy.*
