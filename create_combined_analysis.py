#!/usr/bin/env python3
"""
Script to create a combined analysis of all three evaluation modes for arcee-ai/spotlight.
"""

import json
import os
from datetime import datetime

def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """Save JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def combine_evaluations():
    """Combine all three evaluation modes into a single comprehensive analysis."""
    
    base_dir = "model_results/arcee-ai_spotlight"
    
    # Load the three evaluation files
    with_without_file = f"{base_dir}/benchmark_all_tasks_spotlight_20250615_121720.json"
    true_solution_file = f"{base_dir}/benchmark_all_tasks_spotlight_20250615_140030.json"
    
    print("Loading evaluation results...")
    with_without_data = load_json(with_without_file)
    true_solution_data = load_json(true_solution_file)
    
    # Combine all results
    combined_results = []
    
    # Add with_answer and without_answer results
    for result in with_without_data:
        combined_results.append(result)
    
    # Add with_true_solution results
    for result in true_solution_data:
        combined_results.append(result)
    
    print(f"Combined {len(combined_results)} total evaluations")
    
    # Create comprehensive analysis
    analysis = {
        "evaluation_date": "2025-06-15",
        "model": "arcee-ai/spotlight",
        "total_examples": 122,
        "total_evaluations": len(combined_results),
        "evaluation_modes": ["with_answer", "without_answer", "with_true_solution"],
        "summary": {},
        "models": {},
        "task_breakdown": {}
    }
    
    # Group results by model and answer type
    model_results = {}
    
    for result in combined_results:
        model_id = result["model_id"]
        
        # Determine answer type based on flags
        if result.get("use_true_solution", False):
            answer_type = "with_true_solution"
        elif result.get("use_answer", False):
            answer_type = "with_answer"
        else:
            answer_type = "without_answer"
        
        if model_id not in model_results:
            model_results[model_id] = {}
        
        if answer_type not in model_results[model_id]:
            model_results[model_id][answer_type] = []
        
        model_results[model_id][answer_type].append(result)
    
    # Calculate metrics for each mode
    total_cost = 0
    total_time = 0
    total_evaluations = 0
    all_quality_scores = []
    
    for model_id, answer_types in model_results.items():
        analysis["models"][model_id] = {}
        
        for answer_type, results in answer_types.items():
            metrics = calculate_metrics(results)
            analysis["models"][model_id][answer_type] = metrics
            
            total_cost += metrics["total_cost"]
            total_time += metrics["total_time"]
            total_evaluations += metrics["evaluations"]
            
            if metrics.get("quality_score"):
                all_quality_scores.extend([r.get("quality_score", 0) for r in results if r.get("quality_score") is not None])
    
    # Calculate overall summary
    analysis["summary"] = {
        "total_cost": total_cost,
        "avg_evaluation_time": total_time / total_evaluations if total_evaluations > 0 else 0,
        "avg_quality_score": sum(all_quality_scores) / len(all_quality_scores) if all_quality_scores else 0
    }
    
    # Task breakdown
    task_results = {}
    for result in combined_results:
        task_id = result["task_id"]
        if task_id not in task_results:
            task_results[task_id] = []
        task_results[task_id].append(result)
    
    for task_id, results in task_results.items():
        # Determine answer types for this task
        answer_types = []
        for r in results:
            if r.get("use_true_solution", False):
                answer_types.append("with_true_solution")
            elif r.get("use_answer", False):
                answer_types.append("with_answer")
            else:
                answer_types.append("without_answer")
        
        analysis["task_breakdown"][task_id] = {
            "total_evaluations": len(results),
            "examples": len(set(r["solution_id"] for r in results)),
            "modes": list(set(answer_types))
        }
    
    # Save combined results and analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_file = f"{base_dir}/benchmark_all_tasks_spotlight_combined_{timestamp}.json"
    analysis_file = f"{base_dir}/benchmark_all_tasks_spotlight_combined_{timestamp}_analysis.json"
    
    save_json(combined_results, combined_file)
    save_json(analysis, analysis_file)
    
    print(f"Combined results saved to: {combined_file}")
    print(f"Combined analysis saved to: {analysis_file}")
    
    return analysis, combined_file, analysis_file

def calculate_metrics(results):
    """Calculate metrics for a set of results."""
    if not results:
        return {}
    
    # Use 'score' instead of 'predicted_score'
    total_score = sum(r.get("score", 0) for r in results)
    total_expected = sum(r.get("expected_score", 0) for r in results)
    correct_predictions = sum(1 for r in results if r.get("score") == r.get("expected_score"))
    
    # Calculate quality scores (score / expected_score)
    quality_scores = []
    for r in results:
        if r.get("expected_score", 0) > 0:
            quality_scores.append(r.get("score", 0) / r.get("expected_score", 1))
    
    score_distances = [abs(r.get("score", 0) - r.get("expected_score", 0)) for r in results]
    evaluation_times = [r.get("evaluation_time", 0) for r in results]
    costs = [r.get("cost", 0) for r in results]
    
    return {
        "evaluations": len(results),
        "accuracy": (correct_predictions / len(results)) * 100 if results else 0,
        "quality_score": (sum(quality_scores) / len(quality_scores)) * 100 if quality_scores else None,
        "avg_score_distance": sum(score_distances) / len(score_distances) if score_distances else 0,
        "avg_evaluation_time": sum(evaluation_times) / len(evaluation_times) if evaluation_times else 0,
        "total_cost": sum(costs),
        "total_time": sum(evaluation_times)
    }

def print_summary(analysis):
    """Print a summary of the combined analysis."""
    print("\n" + "="*60)
    print("ARCEE AI SPOTLIGHT - COMPLETE EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {analysis['model']}")
    print(f"Evaluation Date: {analysis['evaluation_date']}")
    print(f"Total Examples: {analysis['total_examples']}")
    print(f"Total Evaluations: {analysis['total_evaluations']}")
    print(f"Evaluation Modes: {', '.join(analysis['evaluation_modes'])}")
    print(f"Average Quality Score: {analysis['summary']['avg_quality_score']:.2f}%")
    print(f"Average Evaluation Time: {analysis['summary']['avg_evaluation_time']:.2f}s")
    print(f"Total Cost: ${analysis['summary']['total_cost']:.4f}")
    
    print(f"\nPerformance by Mode:")
    for model_id, model_data in analysis["models"].items():
        print(f"\nModel: {model_id}")
        for answer_type, metrics in model_data.items():
            print(f"  {answer_type.upper()}:")
            print(f"    Accuracy: {metrics['accuracy']:.2f}%")
            if metrics.get('quality_score'):
                print(f"    Quality Score: {metrics['quality_score']:.2f}%")
            print(f"    Avg Score Distance: {metrics['avg_score_distance']:.2f}")
            print(f"    Evaluations: {metrics['evaluations']}")
    
    print(f"\nTask Coverage:")
    for task_id, task_data in analysis["task_breakdown"].items():
        print(f"  Task {task_id}: {task_data['examples']} examples, {task_data['total_evaluations']} evaluations")

if __name__ == "__main__":
    print("Creating combined analysis for arcee-ai/spotlight...")
    analysis, combined_file, analysis_file = combine_evaluations()
    print_summary(analysis)
    print("\n✓ Combined analysis completed successfully!")
    print(f"Files created: {combined_file}, {analysis_file}")
