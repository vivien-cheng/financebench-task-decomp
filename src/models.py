"""
Model implementations for different LLMs to generate task decompositions from financial questions.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai

from framework import (
    TaskDecomposition,
    TaskNode,
    FinanceTaskDecomposer,
    ReasoningType,
    DecompositionStrategy
)

# Load environment variables from .env file
env_path = find_dotenv()
if env_path:
    load_dotenv(env_path)
    print(f"Loaded environment variables from {env_path}")
else:
    print("Warning: .env file not found")

def get_api_key(service: str) -> str:
    """Get API key from environment variables with fallback."""
    key = os.getenv(f"{service.upper()}_API_KEY")
    if not key:
        raise ValueError(f"{service} API key not found in environment variables. Please check your .env file.")
    return key

# Initialize API clients with error handling
try:
    api_key = get_api_key("openai")
    print(f"OpenAI API key found: {'*' * len(api_key)}")
    openai_client = OpenAI(api_key=api_key)
except Exception as e:
    print(f"Error initializing OpenAI client: {str(e)}")
    openai_client = None

try:
    api_key = get_api_key("anthropic")
    print(f"Anthropic API key found: {'*' * len(api_key)}")
    anthropic_client = Anthropic(api_key=api_key)
except Exception as e:
    print(f"Error initializing Anthropic client: {str(e)}")
    anthropic_client = None

try:
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
except Exception as e:
    print(f"Error initializing Google Generative AI client: {str(e)}")

def create_task_node(task_data: Dict[str, Any]) -> TaskNode:
    """Create a TaskNode from task data with validation."""
    required_fields = ["id", "description", "task_type", "required_metrics", "dependencies"]
    for field in required_fields:
        if field not in task_data:
            raise ValueError(f"Missing required field: {field}")
    
    return TaskNode(
        id=task_data["id"],
        description=task_data["description"],
        task_type=task_data["task_type"],
        required_metrics=task_data["required_metrics"],
        dependencies=task_data["dependencies"],
        output_format=task_data.get("output_format", "json")
    )

def gpt4_decomposition_function(question: str, reasoning_type: str) -> TaskDecomposition:
    """Generate a task decomposition using GPT-4."""
    if not openai_client:
        raise RuntimeError("OpenAI client not initialized. Please check your .env file and API key.")
    
    try:
        decomposition = TaskDecomposition(
            question_text=question,
            reasoning_type=reasoning_type
        )
        
        prompt = f"""Given the following financial question and reasoning type, create a detailed task decomposition.
        Question: {question}
        Reasoning Type: {reasoning_type}
        
        Create a JSON structure with the following format:
        {{
            "tasks": [
                {{
                    "id": "task1",
                    "description": "Task description",
                    "task_type": "Type of task",
                    "required_metrics": ["metric1", "metric2"],
                    "dependencies": [],
                    "output_format": "Format of output"
                }}
            ]
        }}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial analysis expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        # Extract JSON from response
        content = response.choices[0].message.content
        json_str = re.search(r'\{.*\}', content, re.DOTALL).group()
        task_data = json.loads(json_str)
        
        # Validate and add nodes
        if "tasks" not in task_data:
            raise ValueError("Response missing 'tasks' field")
            
        for task in task_data["tasks"]:
            node = create_task_node(task)
            decomposition.add_node(node)
        
        return decomposition
        
    except Exception as e:
        print(f"Error in GPT-4 decomposition: {str(e)}")
        raise

def claude_decomposition_function(question: str, reasoning_type: str) -> TaskDecomposition:
    """
    Generate a task decomposition using Claude.
    
    Args:
        question: The financial question to decompose
        reasoning_type: The type of reasoning required
        
    Returns:
        TaskDecomposition object containing the decomposition
    """
    try:
        # Create a new decomposition object
        decomposition = TaskDecomposition(
            question_text=question,
            reasoning_type=reasoning_type
        )
        
        prompt = f"""Given the following financial question and reasoning type, create a detailed task decomposition.
        Question: {question}
        Reasoning Type: {reasoning_type}
        
        Create a JSON structure with the following format:
        {{
            "tasks": [
                {{
                    "id": "task1",
                    "description": "Task description",
                    "task_type": "Type of task",
                    "required_metrics": ["metric1", "metric2"],
                    "dependencies": [],
                    "output_format": "Format of output"
                }}
            ]
        }}
        """
        
        response = anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract JSON from response
        content = response.content[0].text
        json_str = re.search(r'\{.*\}', content, re.DOTALL).group()
        task_data = json.loads(json_str)
        
        # Add nodes to the decomposition
        for task in task_data["tasks"]:
            node = TaskNode(
                id=task["id"],
                description=task["description"],
                task_type=task["task_type"],
                required_metrics=task["required_metrics"],
                dependencies=task["dependencies"],
                output_format=task.get("output_format", "json")
            )
            decomposition.add_node(node)
        
        return decomposition
        
    except Exception as e:
        print(f"Error in Claude decomposition: {str(e)}")
        # Return an empty decomposition on error
        return TaskDecomposition(question_text=question, reasoning_type=reasoning_type)

def llama_decomposition_function(question: str, reasoning_type: str) -> TaskDecomposition:
    """Simple implementation for testing."""
    decomposition = TaskDecomposition(
        question_text=question,
        reasoning_type=reasoning_type
    )
    
    # Add basic tasks
    node1 = TaskNode(
        id="task1",
        description="Extract relevant financial metrics",
        task_type="information_extraction",
        required_metrics=["revenue", "profit"],
        dependencies=[],
        output_format="json"
    )
    decomposition.add_node(node1)
    
    node2 = TaskNode(
        id="task2",
        description="Perform calculations",
        task_type="calculation",
        required_metrics=["growth_rate", "margin"],
        dependencies=["task1"],
        output_format="json"
    )
    decomposition.add_node(node2)
    
    return decomposition

def small_model_decomposition_function(question: str, reasoning_type: str) -> TaskDecomposition:
    """Simple implementation for testing."""
    decomposition = TaskDecomposition(
        question_text=question,
        reasoning_type=reasoning_type
    )
    
    # Add basic task
    node = TaskNode(
        id="task1",
        description="Extract key information",
        task_type="information_extraction",
        required_metrics=["key_metrics"],
        dependencies=[],
        output_format="json"
    )
    decomposition.add_node(node)
    
    return decomposition

def gemini_decomposition_function(question: str) -> dict:
    """
    Generate a task decomposition using Gemini 2.5 Pro.
    
    Args:
        question (str): The financial question to decompose
        
    Returns:
        dict: Task decomposition in the required format
    """
    try:
        # Create the prompt
        prompt = f"""Given the following financial question, break it down into a sequence of tasks that would be needed to answer it.
        Each task should be clear, specific, and executable. The tasks should build upon each other logically.

        Question: {question}

        Please provide the decomposition in the following JSON format:
        {{
            "tasks": [
                {{
                    "task_id": "task_1",
                    "description": "First task description",
                    "task_type": "task_type_1",
                    "dependencies": []
                }},
                {{
                    "task_id": "task_2",
                    "description": "Second task description",
                    "task_type": "task_type_2",
                    "dependencies": ["task_1"]
                }}
            ]
        }}

        Task types should be one of: information_extraction, calculation, analysis, comparison, or other specific types as needed.
        """
        
        # Generate response using Gemini
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        try:
            # Try to find JSON in the response
            json_str = re.search(r'\{.*\}', response.text, re.DOTALL).group()
            decomposition = json.loads(json_str)
            return decomposition
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Error parsing Gemini response: {e}")
            return None
            
    except Exception as e:
        print(f"Error in Gemini decomposition: {e}")
        return None 