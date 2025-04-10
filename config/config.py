"""
Configuration settings for the FinanceBench Task Decomposition project.
"""

import os
from pathlib import Path

# Project paths
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
SRC_DIR = Path("src")

# API Keys
API_KEYS = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
}

# Evaluation settings
EVALUATION_CONFIG = {
    "num_questions": 5,  # Number of questions to evaluate
    "models": ["gpt4", "claude", "gemini"],  # Models to evaluate
    "metrics": ["num_tasks", "task_types"],  # Metrics to calculate
    "output_formats": ["json"],  # Output formats for results
}

# Model settings
MODEL_CONFIG = {
    "gpt4": {
        "model": "gpt-4-turbo-preview",
        "temperature": 0.7,
        "max_tokens": 2000,
    },
    "claude": {
        "model": "claude-3-opus-20240229",
        "temperature": 0.7,
        "max_tokens": 2000,
    },
    "gemini": {
        "model": "gemini-pro",
        "temperature": 0.7,
        "max_tokens": 2000,
    }
}

# Create necessary directories
for directory in [DATA_DIR, RESULTS_DIR, SRC_DIR]:
    directory.mkdir(exist_ok=True) 