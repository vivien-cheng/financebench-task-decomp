# Project Summary

## Project Structure
```
.
├── config/
│   ├── config.py         # Configuration settings
│   └── .env             # Environment variables
├── data/
│   ├── financebench_document_information.jsonl
│   └── financebench_open_source.jsonl
├── src/
│   ├── framework.py     # Core task decomposition framework
│   ├── models.py        # Model implementations
│   └── evaluate.py      # Evaluation script
├── results/            # Evaluation results
├── report/            # Documentation and reports
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Setup Instructions

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
1. Copy `.env.template` to `.env`
2. Add your API keys:
   - OpenAI API key
   - Anthropic API key
   - Google API key (for Gemini)

### 3. Running Evaluations
```bash
# Run the evaluation script
python src/evaluate.py
```

## Key Components

### Framework
- `TaskNode`: Represents individual tasks in decomposition
- `TaskDecomposition`: Manages task decomposition DAGs
- `FinanceTaskDecomposer`: Handles financial question decomposition

### Models
- GPT-4 implementation
- Claude implementation
- Llama implementation
- Gemini implementation

### Evaluation
- Question loading and selection
- Model evaluation
- Results analysis and visualization

## Data
The project uses the FinanceBench dataset, which includes:
- Financial document information
- Open-source financial questions

## Results
Evaluation results are stored in the `results/` directory, including:
- JSON files with detailed metrics
- CSV files for analysis
- Visualization plots

## License
This project is licensed under the MIT License - see the LICENSE file for details. 