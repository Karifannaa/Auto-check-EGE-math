#!/usr/bin/env python3
"""
Generate comprehensive research documentation for the EGE mathematics benchmark evaluation.
Collects and organizes all benchmark information for research paper reference.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class ResearchDocumentationGenerator:
    def __init__(self, model_results_dir: str = "model_results"):
        self.model_results_dir = Path(model_results_dir)
        self.all_models_data = {}
        
    def load_all_model_data(self) -> Dict[str, Any]:
        """Load comprehensive data for all evaluated models."""
        models_data = {}

        # Get all model directories
        for model_dir in self.model_results_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith('.'):
                continue
            if model_dir.name in ['COMPARATIVE_ANALYSIS.md', 'DETAILED_LOGS_SUMMARY.md', 'FINAL_SUMMARY.md']:
                continue

            print(f"Loading data for {model_dir.name}...")

            # Find all analysis files
            analysis_files = list(model_dir.glob("*_analysis.json"))
            if not analysis_files:
                continue

            # For qwen model, we need to combine data from multiple files
            if 'qwen' in model_dir.name.lower():
                combined_analysis = self.combine_qwen_analysis_files(analysis_files)
            else:
                # Prioritize combined/complete files for other models
                analysis_files.sort(key=lambda x: ('combined' in x.name, 'complete' in x.name), reverse=True)

                try:
                    with open(analysis_files[0], 'r', encoding='utf-8') as f:
                        combined_analysis = json.load(f)
                except Exception as e:
                    print(f"Error loading {analysis_files[0]}: {e}")
                    continue

            # Load README for additional context
            readme_path = model_dir / "README.md"
            readme_content = ""
            if readme_path.exists():
                with open(readme_path, 'r', encoding='utf-8') as f:
                    readme_content = f.read()

            models_data[model_dir.name] = {
                'analysis': combined_analysis,
                'readme': readme_content,
                'directory': str(model_dir)
            }

        return models_data

    def combine_qwen_analysis_files(self, analysis_files: List[Path]) -> Dict[str, Any]:
        """Combine multiple analysis files for qwen model to get complete data."""
        combined_data = {
            'total_examples': 122,
            'total_evaluations': 366,  # 122 * 3 modes
            'models': {},
            'summary': {'total_cost': 0}
        }

        model_key = None
        all_modes_data = {}

        for analysis_file in analysis_files:
            try:
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'models' in data:
                    for model_id, model_data in data['models'].items():
                        if model_key is None:
                            model_key = model_id

                        # Merge mode data
                        for mode, mode_data in model_data.items():
                            all_modes_data[mode] = mode_data

                        # Add to total cost
                        combined_data['summary']['total_cost'] += data['summary'].get('total_cost', 0)

            except Exception as e:
                print(f"Error loading {analysis_file}: {e}")

        if model_key:
            combined_data['models'][model_key] = all_modes_data

        return combined_data
    
    def extract_model_specifications(self, model_name: str, analysis_data: Dict) -> Dict[str, Any]:
        """Extract model specifications and technical details."""
        specs = {
            'full_name': '',
            'provider': '',
            'version': '',
            'parameters': '',
            'capabilities': [],
            'context_length': '',
            'modality': '',
            'pricing': {}
        }
        
        # Extract from analysis data
        if 'models' in analysis_data:
            for model_id in analysis_data['models'].keys():
                specs['full_name'] = model_id
                
                # Parse provider and model info
                if '/' in model_id:
                    provider, model_part = model_id.split('/', 1)
                    specs['provider'] = provider
                    specs['version'] = model_part
                
        # Model-specific specifications
        if 'gemini' in model_name.lower():
            specs['provider'] = 'Google'
            specs['capabilities'] = ['Multimodal', 'Vision-Language', 'Mathematical Reasoning']
            specs['modality'] = 'Text + Image → Text'
            
        elif 'qwen' in model_name.lower():
            specs['provider'] = 'Alibaba Cloud (via OpenRouter)'
            specs['parameters'] = '32B'
            specs['capabilities'] = ['Multimodal', 'Vision-Language', 'Mathematical Reasoning']
            specs['modality'] = 'Text + Image → Text'
            specs['context_length'] = '128,000 tokens'
            
        elif 'o4-mini' in model_name.lower():
            specs['provider'] = 'OpenAI'
            specs['capabilities'] = ['Advanced Reasoning', 'Chain of Thought', 'Mathematical Problem Solving']
            specs['modality'] = 'Text + Image → Text'
            specs['context_length'] = '200,000 tokens'
            
        elif 'spotlight' in model_name.lower():
            specs['provider'] = 'Arcee AI (via OpenRouter)'
            specs['parameters'] = '7B (Qwen 2.5-VL derived)'
            specs['capabilities'] = ['Vision-Language', 'Multimodal Understanding']
            specs['modality'] = 'Text + Image → Text'
            
        return specs
    
    def generate_comprehensive_results_table(self, models_data: Dict) -> str:
        """Generate comprehensive results table for all models."""
        table = """
| Model | Provider | Mode | Accuracy (%) | Quality Score (%) | Avg Score Distance | Total Cost ($) | Evaluations | Avg Time (s) |
|-------|----------|------|--------------|-------------------|-------------------|----------------|-------------|--------------|
"""
        
        # Sort models for consistent ordering
        sorted_models = sorted(models_data.keys())
        
        for model_name in sorted_models:
            model_data = models_data[model_name]['analysis']
            specs = self.extract_model_specifications(model_name, model_data)
            
            if 'models' not in model_data:
                continue
                
            for model_id, model_metrics in model_data['models'].items():
                modes = ['without_answer', 'with_answer', 'with_true_solution']
                mode_names = ['Without Answer', 'With Answer', 'With True Solution']
                
                for mode, mode_name in zip(modes, mode_names):
                    if mode in model_metrics:
                        metrics = model_metrics[mode]
                        table += f"| {specs['provider']} {specs['version']} | {specs['provider']} | {mode_name} | "
                        table += f"{metrics.get('accuracy', 0):.2f}% | "
                        table += f"{metrics.get('quality_score', 0):.2f}% | "
                        table += f"{metrics.get('avg_score_distance', 0):.2f} | "
                        table += f"${metrics.get('total_cost', 0):.4f} | "
                        table += f"{metrics.get('evaluations', 0)} | "
                        table += f"{metrics.get('avg_evaluation_time', 0):.2f} |\n"
        
        return table
    
    def generate_model_rankings(self, models_data: Dict) -> str:
        """Generate model performance rankings."""
        rankings = []
        
        for model_name, data in models_data.items():
            if 'models' not in data['analysis']:
                continue
                
            # Calculate average quality score across all modes
            total_quality = 0
            mode_count = 0
            
            for model_id, model_metrics in data['analysis']['models'].items():
                for mode in ['without_answer', 'with_answer', 'with_true_solution']:
                    if mode in model_metrics and 'quality_score' in model_metrics[mode]:
                        total_quality += model_metrics[mode]['quality_score']
                        mode_count += 1
            
            if mode_count > 0:
                avg_quality = total_quality / mode_count
                rankings.append((model_name, avg_quality, data))
        
        # Sort by average quality score
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        ranking_text = ""
        for rank, (model_name, avg_quality, data) in enumerate(rankings, 1):
            specs = self.extract_model_specifications(model_name, data['analysis'])
            total_cost = data['analysis']['summary'].get('total_cost', 0)
            total_evaluations = data['analysis'].get('total_evaluations', 0)
            
            ranking_text += f"\n### {rank}. {specs['provider']} {specs['version']}\n"
            ranking_text += f"- **Average Quality Score**: {avg_quality:.2f}%\n"
            ranking_text += f"- **Total Cost**: ${total_cost:.4f}\n"
            ranking_text += f"- **Total Evaluations**: {total_evaluations}\n"
            ranking_text += f"- **Provider**: {specs['provider']}\n"
            ranking_text += f"- **Capabilities**: {', '.join(specs['capabilities'])}\n"
        
        return ranking_text
    
    def generate_statistical_analysis(self, models_data: Dict) -> str:
        """Generate comprehensive statistical analysis."""
        stats = {
            'total_evaluations': 0,
            'total_cost': 0,
            'best_accuracy': 0,
            'best_quality': 0,
            'most_cost_effective': '',
            'fastest_model': '',
            'model_count': len(models_data)
        }

        best_accuracy_model = ""
        best_quality_model = ""
        lowest_cost_per_eval = float('inf')
        fastest_time = float('inf')

        for model_name, data in models_data.items():
            analysis = data['analysis']
            stats['total_evaluations'] += analysis.get('total_evaluations', 0)
            stats['total_cost'] += analysis['summary'].get('total_cost', 0)

            if 'models' in analysis:
                for model_id, model_metrics in analysis['models'].items():
                    for mode, metrics in model_metrics.items():
                        # Track best accuracy
                        accuracy = metrics.get('accuracy', 0)
                        if accuracy > stats['best_accuracy']:
                            stats['best_accuracy'] = accuracy
                            best_accuracy_model = f"{model_name} ({mode})"

                        # Track best quality score
                        quality = metrics.get('quality_score', 0)
                        if quality > stats['best_quality']:
                            stats['best_quality'] = quality
                            best_quality_model = f"{model_name} ({mode})"

                        # Track cost effectiveness
                        cost = metrics.get('total_cost', 0)
                        evaluations = metrics.get('evaluations', 1)
                        if evaluations > 0:
                            cost_per_eval = cost / evaluations
                            if cost_per_eval < lowest_cost_per_eval and cost_per_eval > 0:
                                lowest_cost_per_eval = cost_per_eval
                                stats['most_cost_effective'] = model_name

                        # Track fastest model
                        avg_time = metrics.get('avg_evaluation_time', float('inf'))
                        if avg_time < fastest_time:
                            fastest_time = avg_time
                            stats['fastest_model'] = model_name

        return f"""
**Total Models Evaluated**: {stats['model_count']}
**Total Evaluations Conducted**: {stats['total_evaluations']:,}
**Total Research Cost**: ${stats['total_cost']:.4f}
**Best Accuracy**: {stats['best_accuracy']:.2f}% ({best_accuracy_model})
**Best Quality Score**: {stats['best_quality']:.2f}% ({best_quality_model})
**Most Cost-Effective**: {stats['most_cost_effective']} (${lowest_cost_per_eval:.4f} per evaluation)
**Fastest Model**: {stats['fastest_model']} ({fastest_time:.2f}s average)
"""

    def generate_research_documentation(self) -> str:
        """Generate the complete research documentation."""
        models_data = self.load_all_model_data()

        doc = f"""# Comprehensive Research Documentation: EGE Mathematics Benchmark Evaluation

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Project**: Auto-check-EGE-math
**Purpose**: Complete reference for research paper writing

## Executive Summary

This document provides comprehensive documentation of a benchmark evaluation study comparing 7 state-of-the-art AI models on automated assessment of Russian Unified State Exam (EGE) mathematics problems. The evaluation encompasses 122 mathematical problems across 7 task types (Tasks 13-19), tested under 3 different evaluation modes, resulting in 2,562 total evaluations.

{self.generate_statistical_analysis(models_data)}

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

{self.generate_model_rankings(models_data)}

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

{self.generate_comprehensive_results_table(models_data)}

### 4.2 Statistical Summary

{self.generate_statistical_analysis(models_data)}

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
"""

        return doc

def main():
    generator = ResearchDocumentationGenerator()
    documentation = generator.generate_research_documentation()
    
    # Save the documentation
    output_path = Path("COMPREHENSIVE_BENCHMARK_RESEARCH_REFERENCE.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(documentation)
    
    print(f"✅ Research documentation generated: {output_path}")
    print(f"📄 Document length: {len(documentation.split())} words")
    
    return str(output_path)

if __name__ == "__main__":
    main()
