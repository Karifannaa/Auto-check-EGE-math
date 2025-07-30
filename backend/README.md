# EGE Math Solution Checker - Backend

Backend for the automated checking system of EGE mathematics problem solutions using reasoning models through the OpenRouter API.

## Requirements

- Python 3.8+
- FastAPI
- OpenRouter API key

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-folder>/backend
```

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create `.env` file based on `.env.example` and add your OpenRouter API key:

```bash
cp .env.example .env
# Edit the .env file, adding your API key
```

## Running

Start the development server:

```bash
uvicorn app.main:app --reload
```

The server will be available at http://localhost:8000.

## API Endpoints

### Solution Checking

- `POST /api/v1/solutions/evaluate` - Evaluate problem solution

### Model Information

- `GET /api/v1/models/available` - Get list of available models
  - Parameter `category` - filter by model category (`reasoning` or `non_reasoning`)
- `GET /api/v1/models/task-types` - Get list of task types
- `GET /api/v1/models/openrouter-models` - Get list of all models from OpenRouter
- `GET /api/v1/models/credits` - Get account credits information
- `GET /api/v1/models/cost-estimate/{model_id}` - Get cost estimate for using model on entire dataset
- `GET /api/v1/models/compare-costs` - Compare costs of using different models

## Testing

Run tests using pytest:

```bash
pytest
```

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   └── openrouter_client.py  # Client for working with OpenRouter API
│   ├── core/
│   │   └── config.py             # Application configuration
│   ├── models/
│   │   └── solution.py           # Data models
│   ├── routers/
│   │   ├── models.py             # Routes for working with models
│   │   └── solutions.py          # Routes for checking solutions
│   ├── utils/
│   │   ├── image_utils.py        # Utilities for working with images
│   │   ├── prompt_utils.py       # Utilities for forming prompts
│   │   └── cost_calculator.py    # Utilities for cost calculation
│   └── main.py                   # Main application file
├── tests/
│   └── test_openrouter_client.py # Tests for OpenRouter client
├── .env.example                  # Example environment variables file
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```
