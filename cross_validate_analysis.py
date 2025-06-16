#!/usr/bin/env python3
"""
Cross-validation script to verify analysis files match individual results.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_results_and_analysis(model_dir: Path) -> tuple:
    """Load both results and analysis files for a model."""
    # Find the most comprehensive result file (usually the combined one or the largest)
    result_files = []
    analysis_files = []

    for json_file in model_dir.glob("*.json"):
        if "_analysis.json" in json_file.name:
            analysis_files.append(json_file)
        elif "_metadata.json" not in json_file.name:
            result_files.append(json_file)

    if not result_files:
        return [], None

    # Prefer combined files, otherwise take the largest file
    combined_files = [f for f in result_files if "combined" in f.name]
    if combined_files:
        result_file = combined_files[0]  # Take first combined file
    else:
        # Take the largest file by size
        result_file = max(result_files, key=lambda f: f.stat().st_size)

    # Load the selected result file
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
        if not isinstance(results, list):
            results = []

    # Find corresponding analysis file
    analysis = None
    analysis_file_name = result_file.name.replace('.json', '_analysis.json')
    analysis_file_path = model_dir / analysis_file_name

    if analysis_file_path.exists():
        with open(analysis_file_path, 'r', encoding='utf-8') as f:
            analysis = json.load(f)

    return results, analysis

def validate_model_analysis(model_name: str, results: List[Dict], analysis: Dict) -> Dict[str, Any]:
    """Validate that analysis matches individual results for a model."""
    logger.info(f"Validating analysis for {model_name}")
    
    validation_results = {
        'model_name': model_name,
        'errors': [],
        'warnings': [],
        'verified_metrics': {}
    }
    
    max_scores = {
        'task_13': 2, 'task_14': 3, 'task_15': 2, 'task_16': 3,
        'task_17': 3, 'task_18': 4, 'task_19': 4
    }
    
    # Group results by mode
    results_by_mode = {
        'without_answer': [],
        'with_answer': [],
        'with_true_solution': []
    }
    
    for result in results:
        use_answer = result.get('use_answer', False)
        use_true_solution = result.get('use_true_solution', False)
        
        if use_true_solution:
            mode = 'with_true_solution'
        elif use_answer:
            mode = 'with_answer'
        else:
            mode = 'without_answer'
        
        results_by_mode[mode].append(result)
    
    # Validate each mode
    for mode, mode_results in results_by_mode.items():
        if not mode_results:
            continue
            
        # Calculate expected metrics
        valid_results = [r for r in mode_results if r.get('score') is not None and r.get('expected_score') is not None]
        
        if not valid_results:
            continue
        
        # Calculate accuracy
        correct_predictions = sum(1 for r in valid_results if r['score'] == r['expected_score'])
        expected_accuracy = (correct_predictions / len(valid_results)) * 100
        
        # Calculate quality scores and score distances
        quality_scores = []
        score_distances = []
        
        for result in valid_results:
            score = result['score']
            expected_score = result['expected_score']
            task_type = result.get('task_type', '')
            max_score = max_scores.get(task_type, 4)
            
            score_distance = abs(score - expected_score)
            normalized_distance = score_distance / max_score
            quality_score = 1 - normalized_distance
            
            quality_scores.append(quality_score)
            score_distances.append(score_distance)
        
        expected_avg_quality_score = (sum(quality_scores) / len(quality_scores)) * 100
        expected_avg_score_distance = sum(score_distances) / len(score_distances)
        
        # Get analysis metrics for this mode
        if model_name in analysis.get('models', {}):
            model_analysis = analysis['models'][model_name]
            if mode in model_analysis:
                mode_analysis = model_analysis[mode]
                
                # Compare metrics
                analysis_accuracy = mode_analysis.get('accuracy', 0)
                analysis_quality_score = mode_analysis.get('quality_score', 0)
                analysis_score_distance = mode_analysis.get('avg_score_distance', 0)
                analysis_evaluations = mode_analysis.get('evaluations', 0)
                
                # Tolerance for floating point comparison
                tolerance = 0.01
                
                # Validate accuracy
                if abs(expected_accuracy - analysis_accuracy) > tolerance:
                    validation_results['errors'].append({
                        'mode': mode,
                        'metric': 'accuracy',
                        'expected': expected_accuracy,
                        'analysis': analysis_accuracy,
                        'difference': abs(expected_accuracy - analysis_accuracy)
                    })
                
                # Validate quality score
                if abs(expected_avg_quality_score - analysis_quality_score) > tolerance:
                    validation_results['errors'].append({
                        'mode': mode,
                        'metric': 'quality_score',
                        'expected': expected_avg_quality_score,
                        'analysis': analysis_quality_score,
                        'difference': abs(expected_avg_quality_score - analysis_quality_score)
                    })
                
                # Validate score distance
                if abs(expected_avg_score_distance - analysis_score_distance) > tolerance:
                    validation_results['errors'].append({
                        'mode': mode,
                        'metric': 'avg_score_distance',
                        'expected': expected_avg_score_distance,
                        'analysis': analysis_score_distance,
                        'difference': abs(expected_avg_score_distance - analysis_score_distance)
                    })
                
                # Validate evaluation count
                if len(valid_results) != analysis_evaluations:
                    validation_results['errors'].append({
                        'mode': mode,
                        'metric': 'evaluations',
                        'expected': len(valid_results),
                        'analysis': analysis_evaluations,
                        'difference': abs(len(valid_results) - analysis_evaluations)
                    })
                
                # Store verified metrics
                validation_results['verified_metrics'][mode] = {
                    'calculated_accuracy': expected_accuracy,
                    'analysis_accuracy': analysis_accuracy,
                    'calculated_quality_score': expected_avg_quality_score,
                    'analysis_quality_score': analysis_quality_score,
                    'calculated_score_distance': expected_avg_score_distance,
                    'analysis_score_distance': analysis_score_distance,
                    'evaluations': len(valid_results)
                }
    
    return validation_results

def main():
    """Main function to cross-validate all model analyses."""
    model_results_dir = Path("model_results")
    all_validation_results = []
    
    for model_dir in model_results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        try:
            results, analysis = load_model_results_and_analysis(model_dir)
            
            if not results or not analysis:
                logger.warning(f"Missing results or analysis for {model_dir.name}")
                continue
            
            # Extract model name from analysis
            model_names = list(analysis.get('models', {}).keys())
            if not model_names:
                logger.warning(f"No model found in analysis for {model_dir.name}")
                continue
            
            model_name = model_names[0]  # Take first model name
            
            validation_result = validate_model_analysis(model_name, results, analysis)
            all_validation_results.append(validation_result)
            
        except Exception as e:
            logger.error(f"Error validating {model_dir.name}: {e}")
    
    # Generate summary report
    total_errors = sum(len(vr['errors']) for vr in all_validation_results)
    total_warnings = sum(len(vr['warnings']) for vr in all_validation_results)
    
    print("\n" + "="*80)
    print("CROSS-VALIDATION ANALYSIS REPORT")
    print("="*80)
    
    print(f"\nSUMMARY:")
    print(f"  Models validated: {len(all_validation_results)}")
    print(f"  Total errors: {total_errors}")
    print(f"  Total warnings: {total_warnings}")
    
    if total_errors == 0:
        print("  ✓ All analysis files match individual results calculations")
    else:
        print("  ✗ Discrepancies found between analysis and individual results")
    
    # Show errors by model
    for validation_result in all_validation_results:
        model_name = validation_result['model_name']
        errors = validation_result['errors']
        
        if errors:
            print(f"\nERRORS for {model_name}:")
            for error in errors:
                print(f"  {error['mode']} - {error['metric']}: "
                      f"Expected {error['expected']:.4f}, "
                      f"Analysis {error['analysis']:.4f}, "
                      f"Diff {error['difference']:.4f}")
        else:
            print(f"\n✓ {model_name}: All metrics validated correctly")
    
    # Save detailed results
    with open('cross_validation_report.json', 'w', encoding='utf-8') as f:
        json.dump(all_validation_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed report saved to: cross_validation_report.json")

if __name__ == "__main__":
    main()
