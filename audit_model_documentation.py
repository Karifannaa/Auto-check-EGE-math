#!/usr/bin/env python3
"""
Comprehensive audit script for model evaluation documentation inconsistencies.
Compares README documentation against actual JSON evaluation files.
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import re

class ModelDocumentationAuditor:
    def __init__(self, model_results_dir: str = "model_results"):
        self.model_results_dir = Path(model_results_dir)
        self.audit_results = {}
        self.discrepancies = []
        
    def get_model_directories(self) -> List[Path]:
        """Get all model directories to audit."""
        model_dirs = []
        for item in self.model_results_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Skip summary files directories
                if item.name not in ['COMPARATIVE_ANALYSIS.md', 'DETAILED_LOGS_SUMMARY.md', 'FINAL_SUMMARY.md']:
                    model_dirs.append(item)
        return model_dirs
    
    def parse_json_files(self, model_dir: Path) -> Dict[str, Any]:
        """Parse all JSON files in a model directory to extract evaluation modes."""
        json_files = list(model_dir.glob("*.json"))
        evaluation_data = {
            'modes_found': set(),
            'total_evaluations': 0,
            'files_analyzed': [],
            'evaluation_details': {}
        }

        # Track modes we've already seen to avoid double counting
        modes_processed = set()

        # Sort files to prioritize combined/complete files
        json_files.sort(key=lambda x: ('combined' in x.name, 'complete' in x.name), reverse=True)

        for json_file in json_files:
            if json_file.name.endswith('_analysis.json'):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    evaluation_data['files_analyzed'].append(json_file.name)

                    # Extract evaluation modes from the analysis file
                    if 'models' in data:
                        for model_name, model_data in data['models'].items():
                            for mode, mode_data in model_data.items():
                                # Only count each mode once (use the first/best file we encounter)
                                if mode not in modes_processed:
                                    evaluation_data['modes_found'].add(mode)
                                    modes_processed.add(mode)

                                    if 'evaluations' in mode_data:
                                        evaluation_data['total_evaluations'] += mode_data['evaluations']

                                        # Store detailed metrics for each mode
                                        evaluation_data['evaluation_details'][mode] = {
                                            'evaluations': mode_data.get('evaluations', 0),
                                            'accuracy': mode_data.get('accuracy', 0),
                                            'quality_score': mode_data.get('quality_score', 0),
                                            'avg_score_distance': mode_data.get('avg_score_distance', 0),
                                            'total_cost': mode_data.get('total_cost', 0)
                                        }

                except Exception as e:
                    print(f"Error parsing {json_file}: {e}")

        return evaluation_data
    
    def parse_readme_documentation(self, model_dir: Path) -> Dict[str, Any]:
        """Parse README.md to extract documented evaluation modes."""
        readme_path = model_dir / "README.md"
        readme_data = {
            'modes_documented': set(),
            'total_evaluations_documented': 0,
            'evaluation_table_found': False,
            'readme_exists': False
        }
        
        if not readme_path.exists():
            return readme_data
            
        readme_data['readme_exists'] = True
        
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for evaluation mode table
            lines = content.split('\n')
            in_table = False
            
            for line in lines:
                # Check for total evaluations in overview
                if 'Total Evaluations' in line:
                    match = re.search(r'(\d+)', line)
                    if match:
                        readme_data['total_evaluations_documented'] = int(match.group(1))
                
                # Look for evaluation mode table
                if '| Evaluation Mode |' in line or '|----------------|' in line:
                    in_table = True
                    readme_data['evaluation_table_found'] = True
                    continue
                    
                if in_table and line.strip().startswith('|') and not line.strip().startswith('|---'):
                    # Parse table row
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if len(parts) > 0 and parts[0] not in ['Evaluation Mode', '']:
                        mode_name = parts[0].replace('**', '').strip()
                        # Normalize mode names
                        if 'without' in mode_name.lower() and 'answer' in mode_name.lower():
                            readme_data['modes_documented'].add('without_answer')
                        elif 'with' in mode_name.lower() and 'answer' in mode_name.lower() and 'true' not in mode_name.lower():
                            readme_data['modes_documented'].add('with_answer')
                        elif 'true' in mode_name.lower() and 'solution' in mode_name.lower():
                            readme_data['modes_documented'].add('with_true_solution')
                
                # Stop parsing table when we hit an empty line or non-table content
                if in_table and not line.strip().startswith('|') and line.strip() != '':
                    in_table = False
                    
        except Exception as e:
            print(f"Error parsing README for {model_dir.name}: {e}")
        
        return readme_data
    
    def audit_single_model(self, model_dir: Path) -> Dict[str, Any]:
        """Audit a single model directory."""
        print(f"\n🔍 Auditing {model_dir.name}...")

        # Parse JSON files to get actual evaluation data
        json_data = self.parse_json_files(model_dir)
        print(f"   JSON files found: {json_data['files_analyzed']}")
        print(f"   Total evaluations from JSON: {json_data['total_evaluations']}")
        print(f"   Modes found: {json_data['modes_found']}")

        # Parse README to get documented evaluation data
        readme_data = self.parse_readme_documentation(model_dir)
        
        # Compare and identify discrepancies
        audit_result = {
            'model_name': model_dir.name,
            'json_data': json_data,
            'readme_data': readme_data,
            'discrepancies': [],
            'status': 'PASS'
        }
        
        # Check for missing modes in documentation
        missing_modes = json_data['modes_found'] - readme_data['modes_documented']
        if missing_modes:
            audit_result['discrepancies'].append(f"Missing modes in README: {missing_modes}")
            audit_result['status'] = 'FAIL'
        
        # Check for extra modes in documentation
        extra_modes = readme_data['modes_documented'] - json_data['modes_found']
        if extra_modes:
            audit_result['discrepancies'].append(f"Extra modes in README: {extra_modes}")
            audit_result['status'] = 'FAIL'
        
        # Check total evaluations count
        if readme_data['total_evaluations_documented'] != json_data['total_evaluations'] and readme_data['total_evaluations_documented'] > 0:
            audit_result['discrepancies'].append(
                f"Total evaluations mismatch: README shows {readme_data['total_evaluations_documented']}, "
                f"actual is {json_data['total_evaluations']}"
            )
            audit_result['status'] = 'FAIL'
        
        # Check if README exists
        if not readme_data['readme_exists']:
            audit_result['discrepancies'].append("README.md file missing")
            audit_result['status'] = 'FAIL'
        
        return audit_result
    
    def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run comprehensive audit of all models."""
        print("🚀 Starting comprehensive model documentation audit...")
        
        model_dirs = self.get_model_directories()
        print(f"Found {len(model_dirs)} model directories to audit")
        
        audit_summary = {
            'total_models': len(model_dirs),
            'models_passed': 0,
            'models_failed': 0,
            'detailed_results': {},
            'all_discrepancies': []
        }
        
        for model_dir in model_dirs:
            result = self.audit_single_model(model_dir)
            audit_summary['detailed_results'][model_dir.name] = result
            
            if result['status'] == 'PASS':
                audit_summary['models_passed'] += 1
                print(f"✅ {model_dir.name}: PASS")
            else:
                audit_summary['models_failed'] += 1
                print(f"❌ {model_dir.name}: FAIL")
                for discrepancy in result['discrepancies']:
                    print(f"   - {discrepancy}")
                    audit_summary['all_discrepancies'].append(f"{model_dir.name}: {discrepancy}")
        
        self.audit_results = audit_summary
        return audit_summary

def main():
    auditor = ModelDocumentationAuditor()
    results = auditor.run_comprehensive_audit()
    
    print(f"\n📊 AUDIT SUMMARY")
    print(f"================")
    print(f"Total models audited: {results['total_models']}")
    print(f"Models passed: {results['models_passed']}")
    print(f"Models failed: {results['models_failed']}")
    
    if results['all_discrepancies']:
        print(f"\n🚨 ALL DISCREPANCIES FOUND:")
        for discrepancy in results['all_discrepancies']:
            print(f"  - {discrepancy}")
    
    return results

if __name__ == "__main__":
    main()
