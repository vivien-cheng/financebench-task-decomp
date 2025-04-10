# FinanceBench Task Decomposition

A framework for evaluating language models on financial task decomposition using the FinanceBench dataset.

## Overview

This project evaluates how different language models (GPT-4, Claude, Llama, and Gemini) decompose complex financial questions into structured tasks. The system implements a DAG-based task decomposition approach and provides comprehensive evaluation metrics.

## Features

- Support for multiple language models (GPT-4, Claude, Llama, Gemini)
- DAG-based task decomposition framework
- Comprehensive evaluation metrics
- Visualization capabilities
- Detailed reporting system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financebench-task-decomp.git
cd financebench-task-decomp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.template .env
# Edit .env with your API keys
```

## Usage

1. Run the evaluation:
```bash
python src/evaluate.py
```

2. View results:
- Results are saved in the `results/` directory
- Reports are generated in the `report/` directory

## Project Structure

```
.
├── src/                    # Source code
│   ├── evaluate.py        # Evaluation script
│   ├── framework.py       # Core framework
│   └── models.py          # Model implementations
├── data/                   # Data files
├── results/               # Evaluation results
├── report/                # Generated reports
├── config/                # Configuration files
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Results

Detailed evaluation results and recommendations can be found in the `report/` directory:
- `executive_summary.md`: High-level overview
- `results_summary.md`: Detailed results
- `recommendations.md`: Improvement suggestions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 