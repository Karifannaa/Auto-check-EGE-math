# Russian Math Exam Solutions Benchmark Dataset

## Dataset Description

This dataset contains student solutions to Russian Unified State Exam (EGE) mathematics problems, with reference scores for benchmarking automated evaluation systems.

### Task Types

| Task ID | Description | Count |
|---------|-------------|-------|
| 13 | Тригонометрические уравнения | 21 |
| 14 | Стереометрическая задача | 18 |
| 15 | Логарифмические неравенства | 19 |
| 16 | Планиметрическая задача | 17 |
| 17 | Финансовая математика | 15 |
| 18 | Задача с параметром | 16 |
| 19 | Задача по теории чисел | 16 |

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

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('username/russian-math-exam-benchmark')

# Access an example
example = dataset['train'][0]
print(example['solution_id'], example['score'])
```

## License

This dataset is provided for research purposes only.
