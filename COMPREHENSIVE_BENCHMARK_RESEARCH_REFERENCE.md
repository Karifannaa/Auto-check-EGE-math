# Comprehensive Research Documentation: EGE Mathematics Benchmark Evaluation

**Generated**: 2025-06-18 15:54:39
**Project**: Auto-check-EGE-math
**Purpose**: Complete reference for research paper writing

## Executive Summary

This document provides comprehensive documentation of a benchmark evaluation study comparing 7 state-of-the-art AI models on automated assessment of Russian Unified State Exam (EGE) mathematics problems. The evaluation encompasses 122 mathematical problems across 7 task types (Tasks 13-19), tested under 3 different evaluation modes, resulting in 2,562 total evaluations.


**Total Models Evaluated**: 7
**Total Evaluations Conducted**: 1,586
**Total Research Cost**: $9.3908
**Best Accuracy**: 56.56% (openai_o4-mini (with_answer))
**Best Quality Score**: 78.17% (openai_o4-mini (with_answer))
**Most Cost-Effective**: google_gemini-2.0-flash-lite-001 ($0.0003 per evaluation)
**Fastest Model**: google_gemini-2.0-flash-lite-001 (3.09s average)


## 1. Benchmark Overview

### 1.1 Dataset Description

**Dataset Name**: Russian Math Exam Solutions Benchmark Dataset
**Source**: Auto-check-EGE-math project
**Version**: 1.0
**Format**: HuggingFace Dataset
**License**: Research purposes only

The dataset contains **122 examples** of student solutions to Russian Unified State Exam (EGE) mathematics problems with reference scores for benchmarking automated evaluation systems.

### 1.2 Mathematical Domains and Task Types

| Task ID | Mathematical Domain | Description | Count | Score Range |
|---------|-------------------|-------------|-------|-------------|
| 13 | Trigonometry | Тригонометрические уравнения | 21 | 0-2 points |
| 14 | Stereometry | Стереометрическая задача | 18 | 0-2 points |
| 15 | Inequalities | Логарифмические неравенства | 19 | 0-2 points |
| 16 | Planimetry | Планиметрическая задача | 17 | 0-2 points |
| 17 | Financial Mathematics | Финансовая математика | 15 | 0-2 points |
| 18 | Parametric Problems | Задача с параметром | 16 | 0-2 points |
| 19 | Number Theory | Задача по теории чисел | 16 | 0-2 points |
| **Total** | | | **122** | |

### 1.3 Dataset Characteristics

- **Problem Format**: Image-based mathematical problems with handwritten student solutions
- **Scoring System**: 0-2 point scale per problem (0 = incorrect, 1 = partially correct, 2 = fully correct)
- **Evaluation Modes**: 3 different information availability scenarios
- **Quality Assurance**: Manual verification of reference scores by mathematics experts
- **Multimodal Nature**: Combines visual problem interpretation with mathematical reasoning assessment

## 2. Evaluation Methodology

### 2.1 Evaluation Framework Architecture

The evaluation system employs a comprehensive pipeline designed for automated assessment of mathematical problem solutions:

1. **Image Processing**: Conversion and optimization of solution images
2. **Prompt Generation**: Task-specific prompts tailored to mathematical domains
3. **Model Inference**: API-based evaluation using various AI models
4. **Score Extraction**: Automated parsing of model responses to numerical scores
5. **Metrics Calculation**: Comprehensive statistical analysis and performance measurement
6. **Quality Validation**: Cross-validation and consistency checks

### 2.2 Evaluation Modes

The benchmark employs three distinct evaluation modes to assess model performance under different information availability scenarios:

#### 2.2.1 With Answer Mode
- **Input**: Problem statement + Student solution + Correct answer
- **Purpose**: Evaluate model's ability to assess solution quality when the correct answer is known
- **Use Case**: Scenarios where reference answers are available (e.g., standardized testing)

#### 2.2.2 Without Answer Mode
- **Input**: Problem statement + Student solution only
- **Purpose**: Assess model's independent problem-solving and evaluation capabilities
- **Use Case**: Real-world scenarios where correct answers may not be immediately available

#### 2.2.3 With True Solution Mode
- **Input**: Problem statement + Student solution + Complete reference solution
- **Purpose**: Evaluate model's ability to compare student work against detailed solution methods
- **Use Case**: Educational contexts where step-by-step solution guidance is available

### 2.3 Evaluation Metrics

#### 2.3.1 Primary Metrics

**Accuracy (Exact Match)**
- **Definition**: Percentage of cases where predicted score exactly matches expected score
- **Formula**: `Accuracy = (Exact Matches / Total Evaluations) × 100%`
- **Range**: 0-100% (higher is better)
- **Interpretation**: Measures precise scoring capability

**Quality Score (Normalized Performance)**
- **Definition**: Normalized measure indicating prediction closeness to expected scores
- **Formula**: `Quality Score = 100% × (1 - normalized_distance)`
- **Range**: 0-100% (higher is better)
- **Interpretation**: Accounts for partial correctness and near-miss predictions

**Average Score Distance**
- **Definition**: Mean absolute difference between predicted and expected scores
- **Formula**: `Avg Distance = Σ|predicted - expected| / n`
- **Range**: 0-2 points (lower is better)
- **Interpretation**: Measures average prediction error magnitude

#### 2.3.2 Classification Metrics

**Macro Precision**
- **Definition**: Average precision across all score classes (0, 1, 2 points)
- **Formula**: `Macro Precision = (P₀ + P₁ + P₂) / 3`
- **Interpretation**: Measures prediction accuracy for each score level

**Macro Recall**
- **Definition**: Average recall across all score classes
- **Formula**: `Macro Recall = (R₀ + R₁ + R₂) / 3`
- **Interpretation**: Measures coverage of actual instances for each score level

**Macro F1-Score**
- **Definition**: Harmonic mean of precision and recall across score classes
- **Formula**: `Macro F1 = 2 × (Macro Precision × Macro Recall) / (Macro Precision + Macro Recall)`
- **Interpretation**: Balanced measure of classification performance

#### 2.3.3 Operational Metrics

**Cost Analysis**
- **Total Cost**: Cumulative cost for all evaluations
- **Cost per Evaluation**: Average cost per individual assessment
- **Cost Efficiency**: Performance-to-cost ratio analysis

**Timing Metrics**
- **Average Evaluation Time**: Mean time per assessment
- **Token Usage**: Prompt and completion token consumption
- **Throughput**: Evaluations per unit time

## 3. Model Information and Specifications

### 3.1 Evaluated Models Overview


### 1. OpenAI o4-mini
- **Average Quality Score**: 76.63%
- **Total Cost**: $6.4740
- **Total Evaluations**: 366
- **Provider**: OpenAI
- **Capabilities**: Advanced Reasoning, Chain of Thought, Mathematical Problem Solving

### 2. Google gemini-2.0-flash-001
- **Average Quality Score**: 75.82%
- **Total Cost**: $0.2057
- **Total Evaluations**: 122
- **Provider**: Google
- **Capabilities**: Multimodal, Vision-Language, Mathematical Reasoning

### 3. Google gemini-2.5-flash-preview
- **Average Quality Score**: 71.35%
- **Total Cost**: $0.3444
- **Total Evaluations**: 122
- **Provider**: Google
- **Capabilities**: Multimodal, Vision-Language, Mathematical Reasoning

### 4. Google gemini-2.0-flash-lite-001
- **Average Quality Score**: 70.22%
- **Total Cost**: $0.0369
- **Total Evaluations**: 122
- **Provider**: Google
- **Capabilities**: Multimodal, Vision-Language, Mathematical Reasoning

### 5. Google gemini-2.5-flash-preview:thinking
- **Average Quality Score**: 65.92%
- **Total Cost**: $0.7833
- **Total Evaluations**: 122
- **Provider**: Google
- **Capabilities**: Multimodal, Vision-Language, Mathematical Reasoning

### 6. Alibaba Cloud (via OpenRouter) qwen2.5-vl-32b-instruct
- **Average Quality Score**: 64.85%
- **Total Cost**: $1.5465
- **Total Evaluations**: 366
- **Provider**: Alibaba Cloud (via OpenRouter)
- **Capabilities**: Multimodal, Vision-Language, Mathematical Reasoning

### 7. Arcee AI (via OpenRouter) spotlight
- **Average Quality Score**: 62.30%
- **Total Cost**: $0.0000
- **Total Evaluations**: 366
- **Provider**: Arcee AI (via OpenRouter)
- **Capabilities**: Vision-Language, Multimodal Understanding


### 3.2 Technical Specifications

#### 3.2.1 Google Gemini Models
- **Provider**: Google AI
- **Architecture**: Multimodal transformer
- **Modality**: Text + Image → Text
- **Capabilities**: Vision-language understanding, mathematical reasoning
- **Variants Evaluated**:
  - Gemini 2.0 Flash (Full version)
  - Gemini 2.0 Flash Lite (Optimized version)
  - Gemini 2.5 Flash Preview (Latest version)
  - Gemini 2.5 Flash Preview:thinking (Enhanced reasoning)

#### 3.2.2 OpenAI o4-mini
- **Provider**: OpenAI
- **Architecture**: Advanced reasoning model with chain-of-thought capabilities
- **Context Length**: 200,000 tokens
- **Modality**: Text + Image → Text
- **Capabilities**: Advanced mathematical reasoning, step-by-step problem solving

#### 3.2.3 Qwen 2.5 VL 32B
- **Provider**: Alibaba Cloud (via OpenRouter)
- **Parameters**: 32 billion
- **Context Length**: 128,000 tokens
- **Architecture**: Vision-language transformer
- **Capabilities**: Multimodal understanding, mathematical problem solving

#### 3.2.4 Arcee AI Spotlight
- **Provider**: Arcee AI (via OpenRouter)
- **Base Model**: Qwen 2.5-VL derived (7B parameters)
- **Architecture**: Vision-language model
- **Capabilities**: Multimodal understanding, cost-effective evaluation

## 4. Complete Results and Analysis

### 4.1 Comprehensive Performance Table


| Model | Provider | Mode | Accuracy (%) | Quality Score (%) | Avg Score Distance | Total Cost ($) | Evaluations | Avg Time (s) |
|-------|----------|------|--------------|-------------------|-------------------|----------------|-------------|--------------|
| Arcee AI (via OpenRouter) spotlight | Arcee AI (via OpenRouter) | Without Answer | 27.87% | 64.48% | 1.04 | $0.0000 | 122 | 8.80 |
| Arcee AI (via OpenRouter) spotlight | Arcee AI (via OpenRouter) | With Answer | 26.23% | 63.18% | 1.09 | $0.0000 | 122 | 6.99 |
| Arcee AI (via OpenRouter) spotlight | Arcee AI (via OpenRouter) | With True Solution | 25.41% | 59.22% | 1.16 | $0.0000 | 122 | 6.98 |
| Google gemini-2.0-flash-001 | Google | With True Solution | 46.72% | 75.82% | 0.71 | $0.2057 | 122 | 3.13 |
| Google gemini-2.0-flash-lite-001 | Google | With True Solution | 38.52% | 70.22% | 0.84 | $0.0369 | 122 | 3.09 |
| Google gemini-2.5-flash-preview | Google | With True Solution | 45.90% | 71.35% | 0.79 | $0.3444 | 122 | 11.67 |
| Google gemini-2.5-flash-preview:thinking | Google | With True Solution | 43.44% | 65.92% | 0.99 | $0.7833 | 122 | 47.59 |
| OpenAI o4-mini | OpenAI | Without Answer | 55.74% | 75.55% | 0.66 | $2.1788 | 122 | 39.62 |
| OpenAI o4-mini | OpenAI | With Answer | 56.56% | 78.17% | 0.60 | $2.0174 | 122 | 32.94 |
| OpenAI o4-mini | OpenAI | With True Solution | 54.10% | 76.16% | 0.66 | $2.2779 | 122 | 58.47 |
| Alibaba Cloud (via OpenRouter) qwen2.5-vl-32b-instruct | Alibaba Cloud (via OpenRouter) | Without Answer | 31.15% | 62.09% | 1.09 | $0.4550 | 122 | 22.97 |
| Alibaba Cloud (via OpenRouter) qwen2.5-vl-32b-instruct | Alibaba Cloud (via OpenRouter) | With Answer | 30.33% | 61.95% | 1.08 | $0.4571 | 122 | 23.27 |
| Alibaba Cloud (via OpenRouter) qwen2.5-vl-32b-instruct | Alibaba Cloud (via OpenRouter) | With True Solution | 43.44% | 70.49% | 0.81 | $0.6344 | 122 | 27.55 |


### 4.2 Statistical Summary


**Total Models Evaluated**: 7
**Total Evaluations Conducted**: 1,586
**Total Research Cost**: $9.3908
**Best Accuracy**: 56.56% (openai_o4-mini (with_answer))
**Best Quality Score**: 78.17% (openai_o4-mini (with_answer))
**Most Cost-Effective**: google_gemini-2.0-flash-lite-001 ($0.0003 per evaluation)
**Fastest Model**: google_gemini-2.0-flash-lite-001 (3.09s average)


## 5. Technical Implementation

### 5.1 Evaluation Pipeline Architecture

The evaluation system implements a robust pipeline with the following components:

1. **Data Preprocessing**
   - Image format standardization and optimization
   - Metadata validation and consistency checks
   - Dataset integrity verification

2. **Prompt Engineering**
   - Task-specific prompt templates for each mathematical domain
   - Evaluation mode-specific prompt variations
   - Scoring rubric integration

3. **Model Integration**
   - OpenRouter API integration for multiple model access
   - Rate limiting and retry logic implementation
   - Cost tracking and optimization

4. **Response Processing**
   - Automated score extraction from model responses
   - Response validation and error handling
   - Consistency checks across evaluation modes

5. **Metrics Calculation**
   - Real-time performance metric computation
   - Statistical analysis and aggregation
   - Comparative analysis across models and modes

### 5.2 Quality Assurance Measures

- **Pre-execution Validation**: API connectivity and dataset integrity checks
- **Pipeline Testing**: Sample evaluations for validation
- **Cost Calculation Verification**: Accurate cost tracking and reporting
- **Error Handling**: Comprehensive retry logic and failure recovery
- **Result Validation**: Cross-validation and consistency verification

### 5.3 Reproducibility Information

- **Dataset Version**: Fixed dataset with consistent metadata
- **Model Versions**: Specific model identifiers and timestamps
- **Evaluation Parameters**: Standardized prompt templates and settings
- **Random Seed Control**: Deterministic evaluation ordering
- **Environment Documentation**: Complete dependency and configuration records

## 6. Research Artifacts and Supporting Materials

### 6.1 Available Documentation Files

- **Model-specific README files**: Detailed analysis for each evaluated model
- **LaTeX tables**: Ready-to-use academic publication tables
- **JSON analysis files**: Complete raw data and computed metrics
- **Comparative analysis reports**: Cross-model performance comparisons
- **Audit documentation**: Quality assurance and validation reports

### 6.2 File Structure Reference

```
model_results/
├── README.md                           # Main evaluation overview
├── COMPARATIVE_ANALYSIS.md             # Cross-model comparison
├── [model_name]/
│   ├── README.md                       # Model-specific analysis
│   ├── benchmark_*_analysis.json       # Computed metrics and statistics
│   ├── benchmark_*.json                # Raw evaluation results
│   └── *_metrics_table.tex            # LaTeX publication tables
```

### 6.3 Data Availability

- **Raw Results**: Complete evaluation responses and scores
- **Processed Metrics**: All computed performance measures
- **Cost Analysis**: Detailed cost breakdowns and efficiency metrics
- **Timing Data**: Evaluation duration and throughput statistics
- **Error Logs**: Comprehensive failure analysis and recovery records

## 7. Limitations and Methodological Considerations

### 7.1 Dataset Limitations

- **Sample Size**: 122 examples may limit generalizability
- **Language Specificity**: Russian EGE problems may not generalize to other educational systems
- **Task Coverage**: Limited to tasks 13-19 of the EGE mathematics exam
- **Scoring Granularity**: 0-2 point scale may not capture fine-grained performance differences

### 7.2 Evaluation Constraints

- **API Dependencies**: Reliance on external model APIs for evaluation
- **Cost Considerations**: Budget constraints limiting extensive hyperparameter exploration
- **Temporal Factors**: Model performance may vary over time due to updates
- **Prompt Sensitivity**: Results may be sensitive to specific prompt formulations

### 7.3 Methodological Considerations

- **Inter-rater Reliability**: Reference scores based on expert judgment
- **Evaluation Mode Bias**: Different modes may favor different model architectures
- **Cost-Performance Trade-offs**: Optimal model choice depends on specific use case requirements
- **Generalization Scope**: Results specific to mathematical problem assessment domain

## 8. Conclusions and Future Research Directions

### 8.1 Key Research Contributions

1. **Comprehensive Benchmark**: First systematic evaluation of AI models on Russian EGE mathematics assessment
2. **Multi-modal Evaluation**: Novel assessment of vision-language models on mathematical problem solving
3. **Cost-Effectiveness Analysis**: Practical insights for educational technology deployment
4. **Methodological Framework**: Reusable evaluation pipeline for similar assessment tasks

### 8.2 Practical Implications

- **Educational Technology**: Guidance for automated assessment system development
- **Model Selection**: Evidence-based recommendations for different use cases
- **Cost Planning**: Realistic cost estimates for large-scale deployment
- **Quality Assurance**: Validation methods for automated evaluation systems

### 8.3 Future Research Opportunities

1. **Dataset Expansion**: Larger, more diverse problem sets across educational levels
2. **Longitudinal Studies**: Model performance tracking over time
3. **Multilingual Evaluation**: Extension to other languages and educational systems
4. **Fine-tuning Studies**: Domain-specific model optimization for mathematical assessment
5. **Human-AI Collaboration**: Hybrid evaluation systems combining human expertise with AI capabilities

---

**Citation Information**: This research documentation supports the comprehensive evaluation of AI models for automated mathematical problem assessment. All data, code, and documentation are available in the Auto-check-EGE-math repository.

**Contact**: For questions about methodology, data access, or collaboration opportunities, please refer to the project repository or create an issue for discussion.
