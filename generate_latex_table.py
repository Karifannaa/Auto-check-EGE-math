#!/usr/bin/env python3
"""
Generate comprehensive LaTeX table for all model evaluation results.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
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

def calculate_model_metrics(model_data: Dict[str, Any]) -> Tuple[float, str, float, float, str]:
    """Calculate key metrics for a model."""
    model_analysis = model_data['analysis']['models']
    model_name = list(model_analysis.keys())[0]
    model_metrics = model_analysis[model_name]
    
    # Calculate average quality score across all modes
    quality_scores = []
    accuracies = []
    costs = []
    mode_performance = []
    
    modes = ['without_answer', 'with_answer', 'with_true_solution']
    mode_names = ['Without Answer', 'With Answer', 'With True Solution']
    
    for mode, mode_name in zip(modes, mode_names):
        if mode in model_metrics:
            metrics = model_metrics[mode]
            quality_score = metrics.get('quality_score', 0)
            accuracy = metrics.get('accuracy', 0)
            cost = metrics.get('total_cost', 0)
            evaluations = metrics.get('evaluations', 1)
            
            quality_scores.append(quality_score)
            accuracies.append(accuracy)
            costs.append(cost)
            
            mode_performance.append((mode_name, quality_score, accuracy))
    
    # Calculate averages
    avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    total_cost = sum(costs)
    total_evaluations = model_data['analysis'].get('total_evaluations', 1)
    avg_cost_per_eval = total_cost / total_evaluations if total_evaluations > 0 else 0
    
    # Find best performing mode
    if mode_performance:
        best_mode = max(mode_performance, key=lambda x: x[1])
        best_mode_name = best_mode[0]
        best_mode_accuracy = best_mode[2]
    else:
        best_mode_name = "N/A"
        best_mode_accuracy = 0
    
    return avg_quality_score, best_mode_name, best_mode_accuracy, avg_cost_per_eval, model_name

def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    # Replace common special characters
    text = text.replace('&', '\\&')
    text = text.replace('%', '\\%')
    text = text.replace('$', '\\$')
    text = text.replace('#', '\\#')
    text = text.replace('_', '\\_')
    text = text.replace('{', '\\{')
    text = text.replace('}', '\\}')
    text = text.replace('^', '\\textasciicircum{}')
    text = text.replace('~', '\\textasciitilde{}')
    text = text.replace('\\', '\\textbackslash{}')
    return text

def format_model_name(model_name: str) -> str:
    """Format model name for LaTeX display."""
    # Clean up model names for better display
    name_mapping = {
        'openai/o4-mini': 'OpenAI O4-mini',
        'google/gemini-2.0-flash-001': 'Google Gemini 2.0 Flash',
        'google/gemini-2.0-flash-lite-001': 'Google Gemini 2.0 Flash Lite',
        'google/gemini-2.5-flash-preview': 'Google Gemini 2.5 Flash Preview',
        'google/gemini-2.5-flash-preview:thinking': 'Google Gemini 2.5 Flash Preview (Thinking)',
        'arcee-ai/spotlight': 'Arcee-AI Spotlight',
        'qwen/qwen2.5-vl-32b-instruct': 'Qwen 2.5-VL 32B Instruct'
    }
    
    display_name = name_mapping.get(model_name, model_name)
    return escape_latex(display_name)

def generate_comprehensive_latex_table(all_models_data: Dict[str, Any]) -> str:
    """Generate comprehensive LaTeX table."""
    
    # Calculate metrics for all models
    model_metrics = []
    for model_name, data in all_models_data.items():
        avg_quality, best_mode, best_accuracy, avg_cost, original_name = calculate_model_metrics(data)
        model_metrics.append({
            'original_name': original_name,
            'display_name': format_model_name(original_name),
            'avg_quality_score': avg_quality,
            'best_mode': best_mode,
            'best_accuracy': best_accuracy,
            'avg_cost_per_eval': avg_cost,
            'total_evaluations': data['analysis'].get('total_evaluations', 0)
        })
    
    # Sort by average quality score (descending)
    model_metrics.sort(key=lambda x: x['avg_quality_score'], reverse=True)
    
    # Generate LaTeX table
    latex_content = []
    
    # Table header with caption and setup
    latex_content.extend([
        "% Comprehensive Model Evaluation Results Table",
        "% Generated automatically from validated benchmark data",
        "% Use \\input{comprehensive_model_comparison_table.tex} to include in documents",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Comprehensive Model Performance Comparison on Russian Math Exam Solutions Benchmark}",
        "\\label{tab:model_comparison}",
        "\\begin{tabular}{@{}clccccr@{}}",
        "\\toprule",
        "\\textbf{Rank} & \\textbf{Model} & \\textbf{Avg Quality} & \\textbf{Best Mode} & \\textbf{Best Accuracy} & \\textbf{Cost/Eval} & \\textbf{Evaluations} \\\\",
        "& & \\textbf{Score (\\%)} & & \\textbf{(\\%)} & \\textbf{(USD)} & \\\\",
        "\\midrule"
    ])
    
    # Add data rows
    for rank, model in enumerate(model_metrics, 1):
        # Format rank with medal emojis for top 3
        if rank == 1:
            rank_display = "\\textbf{1st}"
        elif rank == 2:
            rank_display = "\\textbf{2nd}"
        elif rank == 3:
            rank_display = "\\textbf{3rd}"
        else:
            rank_display = f"{rank}th"
        
        # Format model name (bold for top 3)
        if rank <= 3:
            model_name = f"\\textbf{{{model['display_name']}}}"
        else:
            model_name = model['display_name']
        
        # Format metrics
        quality_score = f"{model['avg_quality_score']:.2f}"
        best_mode = escape_latex(model['best_mode'])
        best_accuracy = f"{model['best_accuracy']:.2f}"
        cost_per_eval = f"{model['avg_cost_per_eval']:.6f}" if model['avg_cost_per_eval'] > 0 else "0.000000"
        evaluations = f"{model['total_evaluations']}"
        
        # Bold formatting for top performer
        if rank == 1:
            quality_score = f"\\textbf{{{quality_score}}}"
            best_accuracy = f"\\textbf{{{best_accuracy}}}"
        
        # Add table row
        latex_content.append(
            f"{rank_display} & {model_name} & {quality_score} & {best_mode} & {best_accuracy} & {cost_per_eval} & {evaluations} \\\\"
        )
        
        # Add spacing after top 3
        if rank == 3:
            latex_content.append("\\addlinespace")
    
    # Table footer
    latex_content.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{tablenotes}",
        "\\small",
        "\\item \\textbf{Quality Score:} Normalized measure (0-100\\%) indicating prediction closeness relative to maximum possible error.",
        "\\item \\textbf{Best Mode:} Evaluation mode with highest quality score for each model.",
        "\\item \\textbf{Best Accuracy:} Accuracy percentage in the best performing evaluation mode.",
        "\\item \\textbf{Cost/Eval:} Average cost in USD per evaluation across all modes.",
        "\\item \\textbf{Evaluations:} Total number of evaluations performed for the model.",
        "\\item Benchmark covers Russian Math Exam tasks 13-19 with three evaluation modes:",
        "\\item \\quad Without Answer, With Answer, and With True Solution.",
        "\\end{tablenotes}",
        "\\end{table}",
        "",
        "% Additional summary statistics",
        f"% Total models evaluated: {len(model_metrics)}",
        f"% Total evaluations: {sum(m['total_evaluations'] for m in model_metrics)}",
        f"% Best performing model: {model_metrics[0]['display_name']} ({model_metrics[0]['avg_quality_score']:.2f}% avg quality)",
        f"% Most cost-effective: {min(model_metrics, key=lambda x: x['avg_cost_per_eval'] if x['avg_cost_per_eval'] > 0 else float('inf'))['display_name']}"
    ])
    
    return "\n".join(latex_content)

def main():
    """Generate comprehensive LaTeX table."""
    logger.info("Loading all analysis data...")
    all_models_data = load_all_analysis_data()
    
    if not all_models_data:
        logger.error("No analysis data found!")
        return
    
    logger.info(f"Found analysis data for {len(all_models_data)} models")
    
    # Generate LaTeX table
    logger.info("Generating comprehensive LaTeX table...")
    latex_table = generate_comprehensive_latex_table(all_models_data)
    
    # Ensure output directory exists
    output_dir = Path("dataset_benchmark/benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the table
    output_file = output_dir / "comprehensive_model_comparison_table.tex"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    
    logger.info(f"LaTeX table saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE LATEX TABLE GENERATED")
    print("="*80)
    print(f"File: {output_file}")
    print(f"Models: {len(all_models_data)}")
    print("\nTo use in LaTeX documents:")
    print("\\usepackage{booktabs}")
    print("\\usepackage{threeparttable}")
    print("\\input{comprehensive_model_comparison_table.tex}")
    print("\nTable preview:")
    print("-" * 80)
    
    # Show first few lines of the table
    lines = latex_table.split('\n')
    for line in lines[4:20]:  # Skip comments, show table structure
        if line.strip():
            print(line)
    print("...")
    print("-" * 80)

if __name__ == "__main__":
    main()
