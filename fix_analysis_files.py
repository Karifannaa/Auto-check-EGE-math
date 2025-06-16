#!/usr/bin/env python3
"""
Script to fix analysis files by re-running analysis with corrected grouping logic.
"""

import os
import sys
import json
from pathlib import Path
import logging

# Add the dataset_benchmark directory to the path
sys.path.append('dataset_benchmark')

from analyze_existing_results import ResultsAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_all_analysis_files():
    """Re-analyze all result files to fix the grouping issue."""
    model_results_dir = Path("model_results")
    analyzer = ResultsAnalyzer()
    
    fixed_count = 0
    error_count = 0
    
    for model_dir in model_results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        logger.info(f"Processing {model_dir.name}")
        
        # Find result files (not analysis files)
        result_files = []
        for json_file in model_dir.glob("*.json"):
            if "_analysis.json" not in json_file.name and "_metadata.json" not in json_file.name:
                result_files.append(json_file)
        
        for result_file in result_files:
            try:
                logger.info(f"  Re-analyzing {result_file.name}")
                
                # Load results
                results = analyzer.load_results(str(result_file))
                
                # Re-analyze with fixed grouping logic
                analysis = analyzer.analyze_results(results)
                
                # Add filename to analysis for reference
                analysis["filename"] = result_file.name
                
                # Save corrected analysis
                analysis_file = analyzer.save_analysis(analysis, str(result_file))
                
                logger.info(f"  ✓ Fixed analysis saved to {analysis_file}")
                fixed_count += 1
                
            except Exception as e:
                logger.error(f"  ✗ Error processing {result_file}: {e}")
                error_count += 1
    
    logger.info(f"\nSummary:")
    logger.info(f"  Fixed: {fixed_count} analysis files")
    logger.info(f"  Errors: {error_count} files")
    
    return fixed_count, error_count

def validate_fixes():
    """Validate that the fixes worked by running cross-validation again."""
    logger.info("Validating fixes...")
    
    # Import and run the cross-validation
    import cross_validate_analysis
    
    # Run cross-validation
    model_results_dir = Path("model_results")
    all_validation_results = []
    
    for model_dir in model_results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        try:
            results, analysis = cross_validate_analysis.load_model_results_and_analysis(model_dir)
            
            if not results or not analysis:
                logger.warning(f"Missing results or analysis for {model_dir.name}")
                continue
            
            # Extract model name from analysis
            model_names = list(analysis.get('models', {}).keys())
            if not model_names:
                logger.warning(f"No model found in analysis for {model_dir.name}")
                continue
            
            model_name = model_names[0]  # Take first model name
            
            validation_result = cross_validate_analysis.validate_model_analysis(model_name, results, analysis)
            all_validation_results.append(validation_result)
            
        except Exception as e:
            logger.error(f"Error validating {model_dir.name}: {e}")
    
    # Check if fixes worked
    total_errors = sum(len(vr['errors']) for vr in all_validation_results)
    
    if total_errors == 0:
        logger.info("✓ All fixes validated successfully - no more discrepancies!")
        return True
    else:
        logger.warning(f"✗ Still {total_errors} errors remaining after fixes")
        
        # Show remaining errors
        for validation_result in all_validation_results:
            model_name = validation_result['model_name']
            errors = validation_result['errors']
            
            if errors:
                logger.warning(f"Remaining errors for {model_name}:")
                for error in errors:
                    logger.warning(f"  {error['mode']} - {error['metric']}: "
                                 f"Expected {error['expected']:.4f}, "
                                 f"Analysis {error['analysis']:.4f}, "
                                 f"Diff {error['difference']:.4f}")
        
        return False

def main():
    """Main function to fix analysis files."""
    logger.info("Starting analysis file fixes...")
    
    # Fix all analysis files
    fixed_count, error_count = fix_all_analysis_files()
    
    if error_count > 0:
        logger.error(f"Encountered {error_count} errors during fixing")
        return False
    
    # Validate the fixes
    validation_success = validate_fixes()
    
    if validation_success:
        logger.info("✓ All analysis files have been successfully fixed!")
        return True
    else:
        logger.error("✗ Some issues remain after fixing")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
