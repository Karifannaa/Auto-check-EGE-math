# Russian Math Exam Solutions Benchmark Dataset

## Dataset Description

This dataset contains student solutions to Russian Unified State Exam (EGE) mathematics problems, with reference scores for benchmarking automated evaluation systems.

### Task Types

| Task ID | Description | Count |
|---------|-------------|-------|
| 13 | Trigonometric equations | 21 |
| 14 | Stereometric problem | 18 |
| 15 | Logarithmic inequalities | 19 |
| 16 | Planimetric problem | 17 |
| 17 | Financial mathematics | 15 |
| 18 | Problem with parameters | 16 |
| 19 | Number theory problem | 16 |

### Score Distribution

| Score | Count | Percentage |
|-------|-------|------------|
| 0 | 28 | 22.95% |
| 1 | 40 | 32.79% |
| 2 | 35 | 28.69% |
| 3 | 11 | 9.02% |
| 4 | 8 | 6.56% |

## Dataset Structure

Each example contains:

- `solution_id`: Unique identifier for the solution
- `task_id`: Task type ID (13-19)
- `example_id`: Specific example identifier
- `task_type`: Description of the task type
- `score`: Reference score (0-4)
- `parts_count`: Number of parts in the solution
- `images_with_answer`: List of image paths containing student solution with correct answer
- `images_without_answer`: List of image paths containing only student solution
- `images_with_true_solution`: List of image paths containing task with true solution

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('username/russian-math-exam-benchmark')

# Access an example
example = dataset['train'][0]
print(example['solution_id'], example['score'])
```
