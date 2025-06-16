#!/usr/bin/env python3
"""
Detailed Metrics Validation Script

This script performs detailed validation of specific metric calculations
and investigates the anomalies found in the comprehensive audit.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DetailedValidator:
    """Detailed validator for specific metric calculations."""
    
    def __init__(self):
        self.max_scores = {
            'task_13': 2,
            'task_14': 3,
            'task_15': 2,
            'task_16': 3,
            'task_17': 3,
            'task_18': 4,
            'task_19': 4
        }
    
    def validate_sample_calculations(self, results: List[Dict[str, Any]], sample_size: int = 20) -> Dict[str, Any]:
        """Manually validate a random sample of metric calculations."""
        logger.info(f"Validating {sample_size} random sample calculations...")
        
        # Filter valid results
        valid_results = [r for r in results if r.get('score') is not None and r.get('expected_score') is not None]
        
        if len(valid_results) < sample_size:
            sample_size = len(valid_results)
        
        # Random sample
        sample = random.sample(valid_results, sample_size)
        
        validation_results = {
            'total_validated': sample_size,
            'calculation_errors': [],
            'verified_correct': 0
        }
        
        for i, result in enumerate(sample):
            score = result['score']
            expected_score = result['expected_score']
            task_type = result.get('task_type', '')
            solution_id = result.get('solution_id', 'unknown')
            
            # Get max score for task type
            max_score = self.max_scores.get(task_type, 4)
            
            # Calculate expected metrics
            expected_accuracy = score == expected_score
            expected_score_distance = abs(score - expected_score)
            expected_normalized_distance = expected_score_distance / max_score
            expected_quality_score = 1.0 - expected_normalized_distance
            
            # Validate ranges
            errors = []
            
            if score < 0 or score > max_score:
                errors.append(f"Score {score} out of range [0, {max_score}]")
            
            if expected_score < 0 or expected_score > max_score:
                errors.append(f"Expected score {expected_score} out of range [0, {max_score}]")
            
            if expected_quality_score < 0 or expected_quality_score > 1:
                errors.append(f"Quality score {expected_quality_score} out of range [0, 1]")
            
            if expected_normalized_distance < 0 or expected_normalized_distance > 1:
                errors.append(f"Normalized distance {expected_normalized_distance} out of range [0, 1]")
            
            if errors:
                validation_results['calculation_errors'].append({
                    'solution_id': solution_id,
                    'task_type': task_type,
                    'score': score,
                    'expected_score': expected_score,
                    'errors': errors
                })
            else:
                validation_results['verified_correct'] += 1
                
            # Log detailed calculation for first few samples
            if i < 5:
                logger.info(f"Sample {i+1}: {solution_id}")
                logger.info(f"  Score: {score}, Expected: {expected_score}, Task: {task_type}")
                logger.info(f"  Accuracy: {expected_accuracy}")
                logger.info(f"  Score Distance: {expected_score_distance}")
                logger.info(f"  Normalized Distance: {expected_normalized_distance:.4f}")
                logger.info(f"  Quality Score: {expected_quality_score:.4f}")
        
        return validation_results
    
    def investigate_evaluation_time_outliers(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Investigate evaluation time outliers in detail."""
        logger.info("Investigating evaluation time outliers...")
        
        # Get all evaluation times
        eval_times = []
        time_by_model = {}
        time_by_task = {}
        
        for result in results:
            eval_time = result.get('evaluation_time')
            if eval_time is not None:
                eval_times.append(eval_time)
                
                model_id = result.get('model_id', 'unknown')
                task_type = result.get('task_type', 'unknown')
                
                if model_id not in time_by_model:
                    time_by_model[model_id] = []
                time_by_model[model_id].append(eval_time)
                
                if task_type not in time_by_task:
                    time_by_task[task_type] = []
                time_by_task[task_type].append(eval_time)
        
        if not eval_times:
            return {'error': 'No evaluation times found'}
        
        eval_times.sort()
        n = len(eval_times)
        
        # Calculate statistics
        min_time = min(eval_times)
        max_time = max(eval_times)
        median_time = eval_times[n // 2]
        mean_time = sum(eval_times) / len(eval_times)
        
        # Calculate quartiles
        q1 = eval_times[n // 4]
        q3 = eval_times[3 * n // 4]
        iqr = q3 - q1
        
        # Outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Count outliers
        outliers = [t for t in eval_times if t < lower_bound or t > upper_bound]
        
        # Model statistics
        model_stats = {}
        for model_id, times in time_by_model.items():
            if times:
                model_stats[model_id] = {
                    'count': len(times),
                    'mean': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times)
                }
        
        # Task statistics
        task_stats = {}
        for task_type, times in time_by_task.items():
            if times:
                task_stats[task_type] = {
                    'count': len(times),
                    'mean': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times)
                }
        
        return {
            'total_evaluations': len(eval_times),
            'time_statistics': {
                'min': min_time,
                'max': max_time,
                'mean': mean_time,
                'median': median_time,
                'q1': q1,
                'q3': q3,
                'iqr': iqr
            },
            'outlier_analysis': {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(eval_times)) * 100
            },
            'model_statistics': model_stats,
            'task_statistics': task_stats
        }
    
    def investigate_cost_outliers(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Investigate cost outliers in detail."""
        logger.info("Investigating cost outliers...")
        
        costs = []
        cost_by_model = {}
        
        for result in results:
            cost = result.get('cost')
            if cost is not None:
                costs.append(cost)
                
                model_id = result.get('model_id', 'unknown')
                if model_id not in cost_by_model:
                    cost_by_model[model_id] = []
                cost_by_model[model_id].append(cost)
        
        if not costs:
            return {'error': 'No costs found'}
        
        costs.sort()
        
        # Calculate statistics
        min_cost = min(costs)
        max_cost = max(costs)
        mean_cost = sum(costs) / len(costs)
        
        # Calculate standard deviation
        variance = sum((c - mean_cost) ** 2 for c in costs) / len(costs)
        std_cost = variance ** 0.5
        
        # High cost threshold (3 standard deviations)
        high_cost_threshold = mean_cost + 3 * std_cost
        high_cost_outliers = [c for c in costs if c > high_cost_threshold]
        
        # Model cost statistics
        model_stats = {}
        for model_id, model_costs in cost_by_model.items():
            if model_costs:
                model_stats[model_id] = {
                    'count': len(model_costs),
                    'total': sum(model_costs),
                    'mean': sum(model_costs) / len(model_costs),
                    'min': min(model_costs),
                    'max': max(model_costs)
                }
        
        return {
            'total_evaluations': len(costs),
            'cost_statistics': {
                'min': min_cost,
                'max': max_cost,
                'mean': mean_cost,
                'std': std_cost,
                'total': sum(costs)
            },
            'outlier_analysis': {
                'high_cost_threshold': high_cost_threshold,
                'high_cost_outliers': len(high_cost_outliers),
                'outlier_percentage': (len(high_cost_outliers) / len(costs)) * 100
            },
            'model_statistics': model_stats
        }
    
    def check_duplicate_evaluations(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Investigate duplicate evaluations in detail."""
        logger.info("Investigating duplicate evaluations...")
        
        evaluation_groups = {}
        
        for result in results:
            # Create evaluation key
            key = (
                result.get('solution_id', ''),
                result.get('model_id', ''),
                result.get('use_answer', False),
                result.get('use_true_solution', False)
            )
            
            if key not in evaluation_groups:
                evaluation_groups[key] = []
            evaluation_groups[key].append(result)
        
        # Find duplicates
        duplicates = {k: v for k, v in evaluation_groups.items() if len(v) > 1}
        
        # Analyze duplicate patterns
        duplicate_analysis = {
            'total_unique_evaluations': len(evaluation_groups),
            'duplicate_groups': len(duplicates),
            'total_duplicate_entries': sum(len(v) - 1 for v in duplicates.values()),
            'examples': []
        }
        
        # Add examples of duplicates
        for key, group in list(duplicates.items())[:5]:
            solution_id, model_id, use_answer, use_true_solution = key
            duplicate_analysis['examples'].append({
                'solution_id': solution_id,
                'model_id': model_id,
                'use_answer': use_answer,
                'use_true_solution': use_true_solution,
                'duplicate_count': len(group),
                'scores': [r.get('score') for r in group],
                'evaluation_times': [r.get('evaluation_time') for r in group]
            })
        
        return duplicate_analysis

def main():
    """Main function to run detailed validation."""
    # Load all results
    model_results_dir = Path("model_results")
    all_results = []
    
    for model_dir in model_results_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        for json_file in model_dir.glob("*.json"):
            if "_analysis.json" in json_file.name or "_metadata.json" in json_file.name:
                continue
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    
                if isinstance(results, list):
                    all_results.extend(results)
                    
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
    
    logger.info(f"Loaded {len(all_results)} total evaluations for detailed validation")
    
    # Run detailed validations
    validator = DetailedValidator()
    
    # 1. Validate sample calculations
    sample_validation = validator.validate_sample_calculations(all_results, 50)
    
    # 2. Investigate evaluation time outliers
    time_analysis = validator.investigate_evaluation_time_outliers(all_results)
    
    # 3. Investigate cost outliers
    cost_analysis = validator.investigate_cost_outliers(all_results)
    
    # 4. Check duplicate evaluations
    duplicate_analysis = validator.check_duplicate_evaluations(all_results)
    
    # Generate detailed report
    detailed_report = {
        'sample_validation': sample_validation,
        'evaluation_time_analysis': time_analysis,
        'cost_analysis': cost_analysis,
        'duplicate_analysis': duplicate_analysis
    }
    
    # Save detailed report
    with open('detailed_metrics_validation_report.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_report, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*80)
    print("DETAILED METRICS VALIDATION REPORT")
    print("="*80)
    
    print(f"\nSAMPLE VALIDATION:")
    print(f"  Total validated: {sample_validation['total_validated']}")
    print(f"  Verified correct: {sample_validation['verified_correct']}")
    print(f"  Calculation errors: {len(sample_validation['calculation_errors'])}")
    
    print(f"\nEVALUATION TIME ANALYSIS:")
    if 'error' not in time_analysis:
        stats = time_analysis['time_statistics']
        print(f"  Mean time: {stats['mean']:.2f}s")
        print(f"  Median time: {stats['median']:.2f}s")
        print(f"  Range: {stats['min']:.2f}s - {stats['max']:.2f}s")
        print(f"  Outliers: {time_analysis['outlier_analysis']['outlier_count']} ({time_analysis['outlier_analysis']['outlier_percentage']:.1f}%)")
    
    print(f"\nCOST ANALYSIS:")
    if 'error' not in cost_analysis:
        stats = cost_analysis['cost_statistics']
        print(f"  Total cost: ${stats['total']:.4f}")
        print(f"  Mean cost: ${stats['mean']:.6f}")
        print(f"  Range: ${stats['min']:.6f} - ${stats['max']:.6f}")
        print(f"  High cost outliers: {cost_analysis['outlier_analysis']['high_cost_outliers']} ({cost_analysis['outlier_analysis']['outlier_percentage']:.1f}%)")
    
    print(f"\nDUPLICATE ANALYSIS:")
    print(f"  Unique evaluations: {duplicate_analysis['total_unique_evaluations']}")
    print(f"  Duplicate groups: {duplicate_analysis['duplicate_groups']}")
    print(f"  Total duplicate entries: {duplicate_analysis['total_duplicate_entries']}")
    
    print(f"\nDetailed report saved to: detailed_metrics_validation_report.json")

if __name__ == "__main__":
    main()
