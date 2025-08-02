# Auto-check-EGE-math

Automated solution checking system for Russian Unified State Exam (EGE) mathematics problems using AI models.

## What it does

This system automatically evaluates student solutions to EGE mathematics problems (tasks 13-19) by analyzing images of handwritten solutions. It uses AI models through the OpenRouter API to assign scores according to official EGE criteria.

**Supported tasks:** Trigonometry, stereometry, logarithms, planimetry, financial mathematics, parameters, and number theory problems.

# EGE Math Solutions Assessment Benchmark

This repository provides a link to the **EGE Math Solutions Assessment Benchmark** dataset, available on Hugging Face.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/Karifannaa/EGE_Math_Solutions_Assessment_Benchmark)

## Key Features

- Upload images of student solutions and get automated scores
- Web interface for easy solution checking (maybe is not working)
- Benchmarking system to test model performance
- Cost tracking for API usage

## Requirements

- Python 3.8+
- Node.js 16+ (for web interface)
- OpenRouter API key ([get one here](https://openrouter.ai))

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Karifannaa/Auto-check-EGE-math.git
   cd Auto-check-EGE-math
   ```

2. **Set up the backend**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  
   pip install -r requirements.txt

   # Add your OpenRouter API key to .env file
   cp .env.example .env
   # Edit .env and add: OPENROUTER_API_KEY=your_key_here
   ```

## Testing Models

You can test different AI models on the included dataset of 122 student solutions:

```bash
# Quick test with 5 examples
cd dataset_benchmark
python run_full_benchmark.py --max-examples 5

# Test specific task type
python run_task13_benchmark.py

# Full evaluation (costs $5-20)
python run_full_evaluation.py
```

## Dataset

The project includes a dataset of 122 real student solutions from EGE mathematics exams, covering tasks 13-19. Each solution has been evaluated by experts according to official criteria.

**Dataset structure:**
- 122 total examples across 7 task types
- Images in PNG format with high resolution
- Reference scores from 0-4 points
- "with answer", "without answer", "with true solution" versions

## License

The source code and dataset for this research are available under the MIT License. This permissive license allows for reuse, modification, and distribution, both in academic and commercial settings, provided that the original copyright and license notice are included.


## Citation

If you use this work in your research, please consider citing it.


