# Model Evaluation Results

This directory contains comprehensive benchmark results for various models evaluated on the Auto-check-EGE-math dataset.

## Directory Structure

Each model has its own subdirectory containing:
- **Results files** (JSON format with detailed evaluation data)
- **Analysis files** (JSON format with computed metrics and statistics)
- **LaTeX tables** (Ready-to-use tables for academic papers)
- **README.md** (Comprehensive summary and analysis)

## Evaluation Methodology

### Dataset
- **Source**: Auto-check-EGE-math dataset
- **Tasks**: Mathematical problems from Russian EGE exam (tasks 13-19)
- **Total examples**: 122 across 7 task types
- **Format**: Image-based mathematical problems with scoring rubrics

### Evaluation Modes
1. **With Answer**: Model sees the problem and student's answer
2. **Without Answer**: Model sees only the problem
3. **With True Solution**: Model sees problem, answer, and correct solution

### Metrics Collected
- **Accuracy**: Percentage of exact score matches
- **Quality Score**: Normalized performance metric (0-100%)
- **Average Score Distance**: Mean absolute difference from expected scores
- **Macro Precision/Recall/F1**: Classification metrics across score levels
- **Cost Analysis**: Total cost and cost per evaluation
- **Timing**: Average evaluation time per example

## Available Models

### google/gemini-2.0-flash-lite-001
- **Status**: ✅ Complete
- **Evaluation Date**: 2025-06-14
- **Best Accuracy**: 38.52% (With True Solution mode)
- **Best Quality Score**: 70.22%
- **Total Cost**: $0.107 for 366 evaluations
- **Key Strength**: Extremely cost-effective

## Comparative Analysis

### Performance Ranking (by Quality Score)
1. **google/gemini-2.0-flash-lite-001**: 70.22% (With True Solution)

### Cost Efficiency Ranking (by cost per evaluation)
1. **google/gemini-2.0-flash-lite-001**: $0.0003 per evaluation

## Usage Guidelines

### For Researchers
- Each model directory contains LaTeX tables ready for academic papers
- Analysis files provide detailed statistical breakdowns
- README files offer comprehensive performance summaries

### For Developers
- JSON result files contain raw evaluation data for further analysis
- Consistent directory structure enables automated comparison scripts
- Cost and timing data helps with resource planning

### For Educators
- Quality scores indicate suitability for educational assessment
- Task-by-task breakdowns show subject-specific performance
- Accuracy metrics help understand reliability for grading

## File Naming Convention

### Results Files
- `benchmark_all_tasks_{model_name}_{timestamp}.json`
- `benchmark_all_tasks_{model_name}_{timestamp}_analysis.json`

### LaTeX Tables
- `benchmark_all_tasks_{model_name}_{timestamp}_metrics_table.tex`

### Documentation
- `README.md` - Model-specific comprehensive summary

## Adding New Models

When evaluating new models:
1. Create directory: `model_results/{provider}_{model_name}/`
2. Run comprehensive benchmark (all 3 modes)
3. Copy result files to model directory
4. Generate model-specific README.md
5. Update this main README.md with new model entry

## Quality Assurance

All evaluations include:
- ✅ Pre-execution validation (API connectivity, dataset integrity)
- ✅ Pipeline testing with sample evaluations
- ✅ Cost calculation verification
- ✅ Error handling and retry logic
- ✅ Comprehensive result analysis and validation

## Contact

For questions about evaluation methodology or results interpretation, please refer to the main project documentation or create an issue in the repository.
