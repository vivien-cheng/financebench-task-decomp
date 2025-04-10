"""
Enhanced evaluation script for task decomposition models on FinanceBench.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

from models import (
    gpt4_decomposition_function,
    claude_decomposition_function,
    llama_decomposition_function,
    gemini_decomposition_function
)
from framework import TaskDecomposition, TaskNode

def load_test_questions(jsonl_path: str, num_questions: int = 8) -> List[Dict[str, Any]]:
    """Load test questions from JSONL file with diverse reasoning types."""
    questions = []
    reasoning_types = defaultdict(int)
    
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                reasoning_type = data.get("question_reasoning", "general")
                
                # Ensure diversity in reasoning types
                if reasoning_types[reasoning_type] < 2:  # Max 2 questions per reasoning type
                    questions.append({
                        "id": data.get("financebench_id"),
                        "question": data.get("question"),
                        "reasoning": reasoning_type,
                        "answer": data.get("answer", ""),
                        "context": data.get("context", {})
                    })
                    reasoning_types[reasoning_type] += 1
                
                if len(questions) >= num_questions:
                    break
                    
        if not questions:
            raise ValueError(f"No questions found in {jsonl_path}")
        return questions
    except FileNotFoundError:
        print(f"Error: Data file not found at {jsonl_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {jsonl_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading questions: {str(e)}")
        sys.exit(1)

def calculate_task_similarity(task1: TaskNode, task2: TaskNode) -> float:
    """Calculate similarity between two tasks based on their attributes."""
    similarity = 0.0
    total_weights = 0
    
    # Compare task types
    if task1.task_type.lower() == task2.task_type.lower():
        similarity += 0.4
    total_weights += 0.4
    
    # Compare required metrics
    metrics1 = set(task1.required_metrics)
    metrics2 = set(task2.required_metrics)
    if metrics1 and metrics2:
        jaccard_similarity = len(metrics1.intersection(metrics2)) / len(metrics1.union(metrics2))
        similarity += 0.3 * jaccard_similarity
    total_weights += 0.3
    
    # Compare dependencies
    deps1 = set(task1.dependencies)
    deps2 = set(task2.dependencies)
    if deps1 and deps2:
        jaccard_similarity = len(deps1.intersection(deps2)) / len(deps1.union(deps2))
        similarity += 0.3 * jaccard_similarity
    total_weights += 0.3
    
    return similarity / total_weights if total_weights > 0 else 0.0

def compare_with_gold_standard(model_decomp: TaskDecomposition, gold_decomp: TaskDecomposition) -> Dict[str, float]:
    """Compare model decomposition with gold standard."""
    metrics = {
        "task_count_similarity": 0.0,
        "task_type_coverage": 0.0,
        "dependency_similarity": 0.0,
        "overall_similarity": 0.0
    }
    
    if not model_decomp.nodes or not gold_decomp.nodes:
        return metrics
    
    # Task count similarity
    model_count = len(model_decomp.nodes)
    gold_count = len(gold_decomp.nodes)
    metrics["task_count_similarity"] = 1 - abs(model_count - gold_count) / max(model_count, gold_count)
    
    # Task type coverage
    model_types = set(node.task_type.lower() for node in model_decomp.nodes.values())
    gold_types = set(node.task_type.lower() for node in gold_decomp.nodes.values())
    metrics["task_type_coverage"] = len(model_types.intersection(gold_types)) / len(gold_types)
    
    # Dependency similarity
    model_deps = set()
    gold_deps = set()
    for node in model_decomp.nodes.values():
        model_deps.update((node.id, dep) for dep in node.dependencies)
    for node in gold_decomp.nodes.values():
        gold_deps.update((node.id, dep) for dep in node.dependencies)
    metrics["dependency_similarity"] = len(model_deps.intersection(gold_deps)) / len(gold_deps) if gold_deps else 0.0
    
    # Overall similarity (weighted average)
    weights = {
        "task_count_similarity": 0.3,
        "task_type_coverage": 0.4,
        "dependency_similarity": 0.3
    }
    metrics["overall_similarity"] = sum(
        metrics[metric] * weight 
        for metric, weight in weights.items()
    )
    
    return metrics

def evaluate_model(questions: List[Dict[str, Any]], model_name: str, decomposition_function) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Evaluate a model's decomposition on questions with enhanced metrics."""
    results = {}
    total_tasks = 0
    model_metrics = defaultdict(float)
    
    print(f"\nTesting {model_name} decomposition...")
    for q in questions:
        print(f"\nProcessing question {q['id']}...")
        try:
            # Try to decompose the question
            decomp = decomposition_function(
                question=q["question"],
                reasoning_type=q["reasoning"]
            )
            
            # Create gold standard decomposition
            gold_decomp = create_gold_standard(q)
            
            # Compare with gold standard
            comparison_metrics = compare_with_gold_standard(decomp, gold_decomp)
            
            # Record detailed results
            results[q["id"]] = {
                "success": True,
                "num_tasks": len(decomp.nodes),
                "task_types": list(set(node.task_type for node in decomp.nodes.values())),
                "question": q["question"],
                "reasoning_type": q["reasoning"],
                "metrics": comparison_metrics
            }
            
            # Update aggregate metrics
            for metric, value in comparison_metrics.items():
                model_metrics[metric] += value
            
            total_tasks += len(decomp.nodes)
            print(f"Success! Created {len(decomp.nodes)} tasks")
            print(f"Task types: {', '.join(results[q['id']]['task_types'])}")
            print(f"Similarity to gold standard: {comparison_metrics['overall_similarity']:.2f}")
            
        except Exception as e:
            results[q["id"]] = {
                "success": False,
                "error": str(e),
                "question": q["question"],
                "reasoning_type": q["reasoning"]
            }
            print(f"Failed: {str(e)}")
    
    # Calculate average metrics
    num_successful = sum(1 for r in results.values() if r["success"])
    if num_successful > 0:
        for metric in model_metrics:
            model_metrics[metric] /= num_successful
    
    return results, total_tasks, model_metrics

def create_gold_standard(question: Dict[str, Any]) -> TaskDecomposition:
    """Create a gold standard decomposition for a question."""
    decomp = TaskDecomposition(
        question_text=question["question"],
        reasoning_type=question["reasoning"]
    )
    
    # Add tasks based on reasoning type and question context
    if "Information extraction" in question["reasoning"]:
        decomp.add_node(TaskNode(
            id="extract_info",
            description="Extract relevant information from the document",
            task_type="Information Extraction",
            required_metrics=["document_text"],
            dependencies=[],
            output_format="json"
        ))
        decomp.add_node(TaskNode(
            id="format_answer",
            description="Format the extracted information into the required answer format",
            task_type="Data Presentation",
            required_metrics=["extracted_info"],
            dependencies=["extract_info"],
            output_format=question.get("answer_format", "text")
        ))
    elif "Numerical reasoning" in question["reasoning"]:
        decomp.add_node(TaskNode(
            id="extract_numbers",
            description="Extract numerical values from the document",
            task_type="Data Extraction",
            required_metrics=["document_text"],
            dependencies=[],
            output_format="json"
        ))
        decomp.add_node(TaskNode(
            id="perform_calculation",
            description="Perform the required calculation",
            task_type="Calculation",
            required_metrics=["extracted_numbers"],
            dependencies=["extract_numbers"],
            output_format="number"
        ))
        decomp.add_node(TaskNode(
            id="format_result",
            description="Format the calculation result",
            task_type="Data Presentation",
            required_metrics=["calculation_result"],
            dependencies=["perform_calculation"],
            output_format=question.get("answer_format", "text")
        ))
    else:  # Default decomposition for other reasoning types
        decomp.add_node(TaskNode(
            id="analyze_question",
            description="Analyze the question requirements",
            task_type="Analysis",
            required_metrics=["question_text"],
            dependencies=[],
            output_format="json"
        ))
        decomp.add_node(TaskNode(
            id="gather_info",
            description="Gather required information",
            task_type="Information Retrieval",
            required_metrics=["analysis_result"],
            dependencies=["analyze_question"],
            output_format="json"
        ))
        decomp.add_node(TaskNode(
            id="process_info",
            description="Process the gathered information",
            task_type="Data Processing",
            required_metrics=["gathered_info"],
            dependencies=["gather_info"],
            output_format="json"
        ))
        decomp.add_node(TaskNode(
            id="form_answer",
            description="Formulate the final answer",
            task_type="Answer Formation",
            required_metrics=["processed_info"],
            dependencies=["process_info"],
            output_format=question.get("answer_format", "text")
        ))
    
    return decomp

def save_results(results: Dict[str, Any], total_tasks: int, metrics: Dict[str, float], output_file: Path) -> None:
    """Save evaluation results to file with enhanced metrics."""
    try:
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                "results": results,
                "summary": {
                    "total_questions": len(results),
                    "successful_decompositions": sum(1 for r in results.values() if r["success"]),
                    "total_tasks": total_tasks,
                    "avg_tasks_per_question": total_tasks / len(results) if results else 0,
                    "metrics": metrics
                }
            }, f, indent=2)
        print(f"\nResults saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        sys.exit(1)

def run_model_evaluations(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run evaluations for all models on the provided questions.
    
    Args:
        questions (List[Dict[str, Any]]): List of questions to evaluate
        
    Returns:
        Dict[str, Any]: Evaluation results for all models
    """
    results = {}
    
    # Define models to evaluate
    models = {
        "GPT-4": gpt4_decomposition_function,
        "Claude": claude_decomposition_function,
        "Llama": llama_decomposition_function,
        "Gemini": gemini_decomposition_function
    }
    
    # Evaluate each model
    for model_name, model_func in models.items():
        print(f"\nEvaluating {model_name}...")
        results[model_name] = evaluate_model(questions, model_name, model_func)
        
    return results

def main():
    """Run enhanced evaluation for all models."""
    try:
        # Setup paths
        data_file = Path("../data/financebench_open_source.jsonl")
        
        # Load test questions
        print(f"Loading test questions from {data_file}")
        questions = load_test_questions(data_file, num_questions=8)
        print(f"Loaded {len(questions)} questions")
        
        # Run evaluations for all models
        results = run_model_evaluations(questions)
        
        # Save results
        for model_name, (model_results, total_tasks, metrics) in results.items():
            save_results(model_results, total_tasks, metrics, Path(f"../results/{model_name.lower()}_test_results.json"))
            
            # Print summary
            successful = sum(1 for r in model_results.values() if r["success"])
            print(f"\n{model_name} Evaluation Summary:")
            print(f"Total questions: {len(questions)}")
            print(f"Successful decompositions: {successful}/{len(questions)}")
            print(f"Total tasks created: {total_tasks}")
            print(f"Average tasks per question: {total_tasks/len(questions):.1f}")
            print("\nQuality Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.2f}")
        
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 