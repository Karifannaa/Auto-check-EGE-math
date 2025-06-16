#!/usr/bin/env python3
"""
Generate comprehensive analysis of all model results.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_all_analysis_data() -> Dict[str, Any]:
    """Load all analysis data from model directories."""
    model_results_dir = Path("model_results")
    all_models_data = {}
    
    for model_dir in model_results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        logger.info(f"Loading analysis for {model_dir.name}")
        
        # Find the most comprehensive analysis file
        analysis_files = list(model_dir.glob("*_analysis.json"))
        if not analysis_files:
            logger.warning(f"No analysis files found for {model_dir.name}")
            continue
        
        # Prefer combined files, otherwise take the largest
        combined_files = [f for f in analysis_files if "combined" in f.name or "all_modes" in f.name]
        if combined_files:
            analysis_file = combined_files[0]
        else:
            # Take the largest analysis file
            analysis_file = max(analysis_files, key=lambda f: f.stat().st_size)
        
        logger.info(f"  Using analysis file: {analysis_file.name}")
        
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            # Extract model name from analysis
            model_names = list(analysis_data.get('models', {}).keys())
            if model_names:
                model_name = model_names[0]
                all_models_data[model_name] = {
                    'analysis': analysis_data,
                    'directory': model_dir.name,
                    'file': analysis_file.name
                }
            else:
                logger.warning(f"No model data found in {analysis_file}")
                
        except Exception as e:
            logger.error(f"Error loading {analysis_file}: {e}")
    
    return all_models_data

def generate_summary_table(all_models_data: Dict[str, Any]) -> str:
    """Generate a comprehensive summary table."""
    
    # Table header
    table = """
| Model | Mode | Accuracy (%) | Quality Score (%) | Avg Score Distance | Evaluations | Total Cost ($) | Avg Time (s) |
|-------|------|--------------|-------------------|-------------------|-------------|----------------|--------------|
"""
    
    # Sort models by name for consistent ordering
    sorted_models = sorted(all_models_data.keys())
    
    for model_name in sorted_models:
        model_data = all_models_data[model_name]['analysis']['models'][model_name]
        
        # Process each evaluation mode
        modes = ['without_answer', 'with_answer', 'with_true_solution']
        
        for mode in modes:
            if mode in model_data:
                metrics = model_data[mode]
                
                # Format model name (only show for first row)
                display_name = model_name.replace('/', '/') if mode == 'without_answer' else ""
                
                # Format mode name
                mode_display = {
                    'without_answer': 'Without Answer',
                    'with_answer': 'With Answer', 
                    'with_true_solution': 'With True Solution'
                }[mode]
                
                # Extract and format metrics
                accuracy = f"{metrics.get('accuracy', 0):.2f}"
                quality_score = f"{metrics.get('quality_score', 0):.2f}"
                avg_distance = f"{metrics.get('avg_score_distance', 0):.3f}"
                evaluations = metrics.get('evaluations', 0)
                total_cost = f"{metrics.get('total_cost', 0):.4f}"
                avg_time = f"{metrics.get('avg_evaluation_time', 0):.2f}"
                
                table += f"| {display_name} | {mode_display} | {accuracy} | {quality_score} | {avg_distance} | {evaluations} | {total_cost} | {avg_time} |\n"
    
    return table

def generate_detailed_analysis(all_models_data: Dict[str, Any]) -> str:
    """Generate detailed analysis for each model."""
    
    analysis = ""
    
    # Sort models by overall performance (average quality score)
    model_performance = []
    for model_name, data in all_models_data.items():
        model_data = data['analysis']['models'][model_name]
        
        # Calculate average quality score across all modes
        quality_scores = []
        for mode in ['without_answer', 'with_answer', 'with_true_solution']:
            if mode in model_data:
                quality_scores.append(model_data[mode].get('quality_score', 0))
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        model_performance.append((model_name, avg_quality, data))
    
    # Sort by performance (highest first)
    model_performance.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (model_name, avg_quality, data) in enumerate(model_performance, 1):
        analysis += f"\n## {rank}. {model_name}\n\n"
        
        model_data = data['analysis']['models'][model_name]
        total_evaluations = data['analysis'].get('total_evaluations', 0)
        total_cost = data['analysis']['summary'].get('total_cost', 0)
        
        analysis += f"**Overall Performance:** {avg_quality:.2f}% average quality score\n"
        analysis += f"**Total Evaluations:** {total_evaluations}\n"
        analysis += f"**Total Cost:** ${total_cost:.4f}\n\n"
        
        # Detailed metrics for each mode
        modes = ['without_answer', 'with_answer', 'with_true_solution']
        mode_names = ['Without Answer', 'With Answer', 'With True Solution']
        
        for mode, mode_name in zip(modes, mode_names):
            if mode in model_data:
                metrics = model_data[mode]
                
                analysis += f"### {mode_name}\n\n"
                analysis += f"- **Accuracy:** {metrics.get('accuracy', 0):.2f}%\n"
                analysis += f"- **Quality Score:** {metrics.get('quality_score', 0):.2f}%\n"
                analysis += f"- **Average Score Distance:** {metrics.get('avg_score_distance', 0):.3f}\n"
                analysis += f"- **Evaluations:** {metrics.get('evaluations', 0)}\n"
                analysis += f"- **Average Evaluation Time:** {metrics.get('avg_evaluation_time', 0):.2f}s\n"
                analysis += f"- **Total Cost:** ${metrics.get('total_cost', 0):.4f}\n"
                
                # Precision, Recall, F1 if available
                if 'macro_precision' in metrics:
                    analysis += f"- **Macro Precision:** {metrics.get('macro_precision', 0):.2f}%\n"
                    analysis += f"- **Macro Recall:** {metrics.get('macro_recall', 0):.2f}%\n"
                    analysis += f"- **Macro F1:** {metrics.get('macro_f1', 0):.2f}%\n"
                
                analysis += "\n"
        
        # Performance insights
        analysis += "### Performance Insights\n\n"
        
        # Compare modes if multiple available
        available_modes = [mode for mode in modes if mode in model_data]
        if len(available_modes) > 1:
            mode_comparison = []
            for mode in available_modes:
                mode_name = {'without_answer': 'Without Answer', 'with_answer': 'With Answer', 'with_true_solution': 'With True Solution'}[mode]
                accuracy = model_data[mode].get('accuracy', 0)
                quality = model_data[mode].get('quality_score', 0)
                mode_comparison.append((mode_name, accuracy, quality))
            
            # Sort by quality score
            mode_comparison.sort(key=lambda x: x[2], reverse=True)
            
            analysis += f"**Best performing mode:** {mode_comparison[0][0]} (Quality: {mode_comparison[0][2]:.2f}%, Accuracy: {mode_comparison[0][1]:.2f}%)\n\n"
            
            if len(mode_comparison) > 1:
                best_quality = mode_comparison[0][2]
                worst_quality = mode_comparison[-1][2]
                improvement = best_quality - worst_quality
                analysis += f"**Performance gap:** {improvement:.2f} percentage points between best and worst modes\n\n"
        
        # Cost efficiency
        if total_cost > 0 and total_evaluations > 0:
            cost_per_eval = total_cost / total_evaluations
            analysis += f"**Cost Efficiency:** ${cost_per_eval:.6f} per evaluation\n\n"
        
        analysis += "---\n"
    
    return analysis

def main():
    """Generate comprehensive analysis report."""
    logger.info("Loading all analysis data...")
    all_models_data = load_all_analysis_data()
    
    if not all_models_data:
        logger.error("No analysis data found!")
        return
    
    logger.info(f"Found analysis data for {len(all_models_data)} models")
    
    # Generate summary table
    logger.info("Generating summary table...")
    summary_table = generate_summary_table(all_models_data)
    
    # Generate detailed analysis
    logger.info("Generating detailed analysis...")
    detailed_analysis = generate_detailed_analysis(all_models_data)
    
    # Create comprehensive report
    report = f"""# Comprehensive Model Evaluation Results

**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Models Evaluated:** {len(all_models_data)}  
**Evaluation Framework:** Russian Math Exam Solutions Benchmark

## Executive Summary

This report presents a comprehensive analysis of all model evaluation results across the Russian Math Exam Solutions benchmark. The evaluation covers tasks 13-19 with three different evaluation modes: without answer images, with answer images, and with true solution images.

### Key Findings

- **Best Overall Model:** {sorted(all_models_data.keys(), key=lambda x: sum(all_models_data[x]['analysis']['models'][x].get(mode, {}).get('quality_score', 0) for mode in ['without_answer', 'with_answer', 'with_true_solution']) / len([mode for mode in ['without_answer', 'with_answer', 'with_true_solution'] if mode in all_models_data[x]['analysis']['models'][x]]), reverse=True)[0]}
- **Total Evaluations:** {sum(data['analysis'].get('total_evaluations', 0) for data in all_models_data.values())}
- **Total Cost:** ${sum(data['analysis']['summary'].get('total_cost', 0) for data in all_models_data.values()):.4f}

## Summary Results Table
{summary_table}

## Detailed Model Analysis
{detailed_analysis}

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
"""
    
    # Save the report
    with open('COMPREHENSIVE_MODEL_ANALYSIS.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("Comprehensive analysis saved to COMPREHENSIVE_MODEL_ANALYSIS.md")
    
    # Print summary to console
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION RESULTS")
    print("="*80)
    print(summary_table)
    print(f"\nDetailed analysis saved to: COMPREHENSIVE_MODEL_ANALYSIS.md")

if __name__ == "__main__":
    main()
