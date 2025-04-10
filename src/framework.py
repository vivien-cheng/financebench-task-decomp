"""
Finance Task Decomposition Framework for FinanceBench

This module provides a framework for creating and evaluating task decompositions
for financial questions from the FinanceBench dataset.
"""

from typing import List, Dict, Optional, Any, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
import json
import enum
from collections import defaultdict

class ReasoningType(enum.Enum):
    INFORMATION_EXTRACTION = "Information extraction"
    NUMERICAL_REASONING = "Numerical reasoning"
    LOGICAL_REASONING = "Logical reasoning"
    LOGICAL_NUMERICAL = "Logical reasoning (based on numerical reasoning)"
    MIXED_REASONING = "Mixed reasoning"

class DecompositionStrategy(enum.Enum):
    # Information extraction strategies
    DIRECT_EXTRACTION = "direct_extraction"  # Simple extraction from document
    MULTI_DOCUMENT_EXTRACTION = "multi_document_extraction"  # Extract from multiple docs
    CONTEXTUAL_EXTRACTION = "contextual_extraction"  # Extract with context awareness
    
    # Numerical reasoning strategies
    BASIC_CALCULATION = "basic_calculation"  # Simple arithmetic operations
    FINANCIAL_RATIO_ANALYSIS = "financial_ratio_analysis"  # Calculate financial ratios
    TIME_SERIES_ANALYSIS = "time_series_analysis"  # Analyze trends over time
    COMPARATIVE_ANALYSIS = "comparative_analysis"  # Compare different metrics
    
    # Logical reasoning strategies
    CONDITION_EVALUATION = "condition_evaluation"  # Evaluate based on conditions
    TREND_INTERPRETATION = "trend_interpretation"  # Interpret financial trends
    CAUSAL_ANALYSIS = "causal_analysis"  # Analyze cause-effect relationships
    MULTI_FACTOR_EVALUATION = "multi_factor_evaluation"  # Consider multiple factors
    
    # Mixed reasoning strategies
    FINANCIAL_HEALTH_ASSESSMENT = "financial_health_assessment"  # Overall financial health
    RISK_ASSESSMENT = "risk_assessment"  # Evaluate financial risks
    PERFORMANCE_EVALUATION = "performance_evaluation"  # Evaluate financial performance

class FinancialDocumentType(enum.Enum):
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW_STATEMENT = "cash_flow_statement"
    NOTES_TO_FINANCIAL_STATEMENTS = "notes_to_financial_statements"
    MANAGEMENT_DISCUSSION = "management_discussion"
    ANNUAL_REPORT = "annual_report"
    QUARTERLY_REPORT = "quarterly_report"
    EARNINGS_CALL = "earnings_call"

@dataclass
class TaskNode:
    """Represents a single node in the task decomposition DAG."""
    id: str
    description: str
    task_type: str  # e.g., "information_extraction", "numerical_reasoning", "logical_reasoning"
    required_metrics: List[str]  # List of metrics needed to solve this task
    dependencies: List[str]  # List of node IDs this task depends on
    output_format: Optional[str] = None  # Expected format of the output
    document_sections: Optional[List[FinancialDocumentType]] = None  # Relevant document sections
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata about the task

class TaskDecomposition:
    """A class to represent and manage task decompositions as DAGs."""
    
    def __init__(self, question_id: str = None, question_text: str = None, 
                 reasoning_type: Union[ReasoningType, str] = None, 
                 company: str = None, source_doc: str = None):
        self.nodes: Dict[str, TaskNode] = {}
        self.edges: Dict[str, List[str]] = {}  # Maps node ID to its dependencies
        self.question_id = question_id
        self.question_text = question_text
        self.reasoning_type = reasoning_type if isinstance(reasoning_type, ReasoningType) else self._parse_reasoning_type(reasoning_type)
        self.company = company
        self.source_doc = source_doc
        self.strategy = None
        
    def _parse_reasoning_type(self, reasoning_str: str) -> ReasoningType:
        """Convert string reasoning type to enum."""
        if not reasoning_str:
            return ReasoningType.MIXED_REASONING
            
        reasoning_str = reasoning_str.lower()
        if "information extraction" in reasoning_str:
            return ReasoningType.INFORMATION_EXTRACTION
        elif "numerical reasoning" in reasoning_str:
            return ReasoningType.NUMERICAL_REASONING
        elif "logical reasoning based on numerical" in reasoning_str:
            return ReasoningType.LOGICAL_NUMERICAL
        elif "logical reasoning" in reasoning_str:
            return ReasoningType.LOGICAL_REASONING
        else:
            return ReasoningType.MIXED_REASONING
        
    def add_node(self, node: TaskNode) -> None:
        """Add a new node to the DAG."""
        self.nodes[node.id] = node
        self.edges[node.id] = node.dependencies
        
    def get_node(self, node_id: str) -> Optional[TaskNode]:
        """Retrieve a node by its ID."""
        return self.nodes.get(node_id)
    
    def get_dependencies(self, node_id: str) -> List[str]:
        """Get all dependencies for a given node."""
        return self.edges.get(node_id, [])
    
    def get_dependent_nodes(self, node_id: str) -> List[str]:
        """Get all nodes that depend on the given node."""
        return [nid for nid, deps in self.edges.items() if node_id in deps]
    
    def is_valid_dag(self) -> bool:
        """Check if the current structure forms a valid DAG (no cycles)."""
        visited = set()
        recursion_stack = set()
        
        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            recursion_stack.add(node_id)
            
            for dep in self.get_dependencies(node_id):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in recursion_stack:
                    return True
            
            recursion_stack.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                if has_cycle(node_id):
                    return False
        return True
    
    def get_execution_order(self) -> List[str]:
        """Get a topological sort of the DAG for execution order."""
        if not self.is_valid_dag():
            raise ValueError("Graph contains cycles")
            
        visited = set()
        order = []
        
        def visit(node_id: str):
            if node_id not in visited:
                visited.add(node_id)
                for dep in self.get_dependencies(node_id):
                    visit(dep)
                order.append(node_id)
        
        for node_id in self.nodes:
            visit(node_id)
            
        return order[::-1]  # Reverse to get correct order
    
    def to_json(self) -> str:
        """Convert the DAG to a JSON representation."""
        dag_data = {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "reasoning_type": self.reasoning_type.value if isinstance(self.reasoning_type, ReasoningType) else self.reasoning_type,
            "company": self.company,
            "source_doc": self.source_doc,
            "strategy": self.strategy.value if self.strategy else None,
            "nodes": {
                node_id: {
                    "description": node.description,
                    "task_type": node.task_type,
                    "required_metrics": node.required_metrics,
                    "dependencies": node.dependencies,
                    "output_format": node.output_format,
                    "document_sections": [ds.value for ds in node.document_sections] if node.document_sections else None,
                    "metadata": node.metadata
                }
                for node_id, node in self.nodes.items()
            }
        }
        return json.dumps(dag_data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TaskDecomposition':
        """Create a TaskDecomposition instance from a JSON string."""
        data = json.loads(json_str)
        
        dag = cls(
            question_id=data.get("question_id"),
            question_text=data.get("question_text"),
            reasoning_type=data.get("reasoning_type"),
            company=data.get("company"),
            source_doc=data.get("source_doc")
        )
        
        if data.get("strategy"):
            dag.strategy = DecompositionStrategy(data["strategy"])
        
        for node_id, node_data in data["nodes"].items():
            document_sections = None
            if node_data.get("document_sections"):
                document_sections = [FinancialDocumentType(ds) for ds in node_data["document_sections"]]
                
            node = TaskNode(
                id=node_id,
                description=node_data["description"],
                task_type=node_data["task_type"],
                required_metrics=node_data["required_metrics"],
                dependencies=node_data["dependencies"],
                output_format=node_data.get("output_format"),
                document_sections=document_sections,
                metadata=node_data.get("metadata")
            )
            dag.add_node(node)
            
        return dag

class FinanceTaskDecomposer:
    """A class to create task decompositions for financial questions."""
    
    @staticmethod
    def get_strategy_for_question(question_text: str, reasoning_type: ReasoningType) -> DecompositionStrategy:
        """Determine the appropriate decomposition strategy based on the question and reasoning type."""
        question_lower = question_text.lower()
        
        # Information extraction strategies
        if reasoning_type == ReasoningType.INFORMATION_EXTRACTION:
            if "compare" in question_lower or "difference" in question_lower:
                return DecompositionStrategy.MULTI_DOCUMENT_EXTRACTION
            
            if any(term in question_lower for term in ["context", "explain", "describe"]):
                return DecompositionStrategy.CONTEXTUAL_EXTRACTION
            
            return DecompositionStrategy.DIRECT_EXTRACTION
        
        # Numerical reasoning strategies
        if reasoning_type == ReasoningType.NUMERICAL_REASONING:
            if any(term in question_lower for term in ["ratio", "percentage", "proportion"]):
                return DecompositionStrategy.FINANCIAL_RATIO_ANALYSIS
            
            if any(term in question_lower for term in ["trend", "over time", "growth"]):
                return DecompositionStrategy.TIME_SERIES_ANALYSIS
            
            if "compare" in question_lower or "difference" in question_lower:
                return DecompositionStrategy.COMPARATIVE_ANALYSIS
            
            return DecompositionStrategy.BASIC_CALCULATION
        
        # Logical reasoning strategies
        if reasoning_type == ReasoningType.LOGICAL_REASONING:
            if any(term in question_lower for term in ["if", "when", "condition"]):
                return DecompositionStrategy.CONDITION_EVALUATION
                
            if any(term in question_lower for term in ["trend", "pattern", "change"]):
                return DecompositionStrategy.TREND_INTERPRETATION
                
            if any(term in question_lower for term in ["cause", "effect", "impact", "result"]):
                return DecompositionStrategy.CAUSAL_ANALYSIS
                
            return DecompositionStrategy.MULTI_FACTOR_EVALUATION
        
        # Mixed reasoning strategies
        return DecompositionStrategy.FINANCIAL_HEALTH_ASSESSMENT
    
    @staticmethod
    def create_decomposition(question_id: str, question_text: str, 
                            reasoning_type: Union[ReasoningType, str],
                            company: str = None, source_doc: str = None,
                            strategy: DecompositionStrategy = None) -> TaskDecomposition:
        """Create a task decomposition for a financial question."""
        decomposition = TaskDecomposition(
            question_id=question_id,
            question_text=question_text,
            reasoning_type=reasoning_type,
            company=company,
            source_doc=source_doc
        )
        
        if not strategy:
            strategy = FinanceTaskDecomposer.get_strategy_for_question(
                question_text, 
                decomposition.reasoning_type
            )
        
        decomposition.strategy = strategy
        
        # Create decomposition based on strategy
        if strategy == DecompositionStrategy.DIRECT_EXTRACTION:
            FinanceTaskDecomposer._create_direct_extraction_decomposition(decomposition)
        elif strategy == DecompositionStrategy.MULTI_DOCUMENT_EXTRACTION:
            FinanceTaskDecomposer._create_multi_document_extraction_decomposition(decomposition)
        elif strategy == DecompositionStrategy.CONTEXTUAL_EXTRACTION:
            FinanceTaskDecomposer._create_contextual_extraction_decomposition(decomposition)
        elif strategy == DecompositionStrategy.BASIC_CALCULATION:
            FinanceTaskDecomposer._create_basic_calculation_decomposition(decomposition)
        elif strategy == DecompositionStrategy.FINANCIAL_RATIO_ANALYSIS:
            FinanceTaskDecomposer._create_financial_ratio_analysis_decomposition(decomposition)
        elif strategy == DecompositionStrategy.TIME_SERIES_ANALYSIS:
            FinanceTaskDecomposer._create_time_series_analysis_decomposition(decomposition)
        elif strategy == DecompositionStrategy.COMPARATIVE_ANALYSIS:
            FinanceTaskDecomposer._create_comparative_analysis_decomposition(decomposition)
        elif strategy == DecompositionStrategy.CONDITION_EVALUATION:
            FinanceTaskDecomposer._create_condition_evaluation_decomposition(decomposition)
        elif strategy == DecompositionStrategy.TREND_INTERPRETATION:
            FinanceTaskDecomposer._create_trend_interpretation_decomposition(decomposition)
        elif strategy == DecompositionStrategy.CAUSAL_ANALYSIS:
            FinanceTaskDecomposer._create_causal_analysis_decomposition(decomposition)
        elif strategy == DecompositionStrategy.MULTI_FACTOR_EVALUATION:
            FinanceTaskDecomposer._create_multi_factor_evaluation_decomposition(decomposition)
        elif strategy == DecompositionStrategy.FINANCIAL_HEALTH_ASSESSMENT:
            FinanceTaskDecomposer._create_financial_health_assessment_decomposition(decomposition)
        elif strategy == DecompositionStrategy.RISK_ASSESSMENT:
            FinanceTaskDecomposer._create_risk_assessment_decomposition(decomposition)
        elif strategy == DecompositionStrategy.PERFORMANCE_EVALUATION:
            FinanceTaskDecomposer._create_performance_evaluation_decomposition(decomposition)
        
        return decomposition

    @staticmethod
    def _create_direct_extraction_decomposition(decomposition: TaskDecomposition) -> None:
        """Create a decomposition for direct information extraction."""
        # Task 1: Analyze the question
        decomposition.add_node(TaskNode(
            id="task1",
            description="Analyze the question to identify the required information",
            task_type="question_analysis",
            required_metrics=["information_type", "document_section"],
            dependencies=[],
            output_format="dict"
        ))
        
        # Task 2: Locate information
        decomposition.add_node(TaskNode(
            id="task2",
            description="Locate the relevant information in the document",
            task_type="information_extraction",
            required_metrics=["location", "context"],
            dependencies=["task1"],
            output_format="dict"
        ))
        
        # Task 3: Extract and format
        decomposition.add_node(TaskNode(
            id="task3",
            description="Extract and format the required information",
            task_type="information_extraction",
            required_metrics=["value", "unit", "time_period"],
            dependencies=["task2"],
            output_format="text"
        ))

    @staticmethod
    def _create_multi_document_extraction_decomposition(decomposition: TaskDecomposition) -> None:
        """Create a decomposition for extracting information from multiple documents."""
        # Task 1: Analyze the question
        decomposition.add_node(TaskNode(
            id="task1",
            description="Analyze the question to identify required information from multiple documents",
            task_type="question_analysis",
            required_metrics=["information_types", "document_sections"],
            dependencies=[],
            output_format="dict"
        ))
        
        # Task 2: Identify documents
        decomposition.add_node(TaskNode(
            id="task2",
            description="Identify relevant documents and sections",
            task_type="document_navigation",
            required_metrics=["document_list", "section_list"],
            dependencies=["task1"],
            output_format="dict"
        ))
        
        # Task 3: Extract from each document
        decomposition.add_node(TaskNode(
            id="task3",
            description="Extract information from each identified document",
            task_type="information_extraction",
            required_metrics=["values", "units", "time_periods"],
            dependencies=["task2"],
            output_format="dict"
        ))
        
        # Task 4: Combine and format
        decomposition.add_node(TaskNode(
            id="task4",
            description="Combine information from multiple sources and format the answer",
            task_type="information_synthesis",
            required_metrics=["combined_value", "context"],
            dependencies=["task3"],
            output_format="text"
        ))

    @staticmethod
    def _create_contextual_extraction_decomposition(decomposition: TaskDecomposition) -> None:
        """Create a decomposition for contextual information extraction."""
        # Task 1: Analyze the question
        decomposition.add_node(TaskNode(
            id="task1",
            description="Analyze the question to identify required information and context",
            task_type="question_analysis",
            required_metrics=["information_type", "context_type"],
            dependencies=[],
            output_format="dict"
        ))
        
        # Task 2: Extract context
        decomposition.add_node(TaskNode(
            id="task2",
            description="Extract relevant contextual information",
            task_type="context_extraction",
            required_metrics=["context_info", "time_period"],
            dependencies=["task1"],
            output_format="dict"
        ))
        
        # Task 3: Extract target information
        decomposition.add_node(TaskNode(
            id="task3",
            description="Extract target information with context",
            task_type="information_extraction",
            required_metrics=["value", "unit", "context"],
            dependencies=["task2"],
            output_format="dict"
        ))
        
        # Task 4: Synthesize and explain
        decomposition.add_node(TaskNode(
            id="task4",
            description="Synthesize information and provide contextual explanation",
            task_type="explanation_generation",
            required_metrics=["explanation", "supporting_facts"],
            dependencies=["task3"],
            output_format="text"
        ))

    @staticmethod
    def _create_basic_calculation_decomposition(decomposition: TaskDecomposition) -> None:
        """Create a decomposition for basic numerical calculations."""
        # Task 1: Analyze calculation requirements
        decomposition.add_node(TaskNode(
            id="task1",
            description="Analyze the question to identify required calculations",
            task_type="calculation_analysis",
            required_metrics=["operation_type", "required_values"],
            dependencies=[],
            output_format="dict"
        ))
        
        # Task 2: Extract values
        decomposition.add_node(TaskNode(
            id="task2",
            description="Extract required numerical values",
            task_type="information_extraction",
            required_metrics=["values", "units"],
            dependencies=["task1"],
            output_format="dict"
        ))
        
        # Task 3: Perform calculation
        decomposition.add_node(TaskNode(
            id="task3",
            description="Perform the required calculation",
            task_type="numerical_calculation",
            required_metrics=["result", "precision"],
            dependencies=["task2"],
            output_format="dict"
        ))
        
        # Task 4: Format result
        decomposition.add_node(TaskNode(
            id="task4",
            description="Format the calculation result",
            task_type="result_formatting",
            required_metrics=["formatted_result", "unit"],
            dependencies=["task3"],
            output_format="text"
        ))

    @staticmethod
    def _create_financial_ratio_analysis_decomposition(decomposition: TaskDecomposition) -> None:
        """Create a decomposition for financial ratio analysis."""
        # Task 1: Identify ratio components
        decomposition.add_node(TaskNode(
            id="task1",
            description="Identify the required financial ratio and its components",
            task_type="ratio_analysis",
            required_metrics=["ratio_type", "components"],
            dependencies=[],
            output_format="dict"
        ))
        
        # Task 2: Extract component values
        decomposition.add_node(TaskNode(
            id="task2",
            description="Extract values for ratio components",
            task_type="information_extraction",
            required_metrics=["component_values", "time_period"],
            dependencies=["task1"],
            output_format="dict"
        ))
        
        # Task 3: Calculate ratio
        decomposition.add_node(TaskNode(
            id="task3",
            description="Calculate the financial ratio",
            task_type="ratio_calculation",
            required_metrics=["ratio_value", "precision"],
            dependencies=["task2"],
            output_format="dict"
        ))
        
        # Task 4: Interpret ratio
        decomposition.add_node(TaskNode(
            id="task4",
            description="Interpret the ratio in context",
            task_type="ratio_interpretation",
            required_metrics=["interpretation", "benchmark_comparison"],
            dependencies=["task3"],
            output_format="text"
        ))

    @staticmethod
    def _create_time_series_analysis_decomposition(decomposition: TaskDecomposition) -> None:
        """Create a decomposition for time series analysis."""
        # Task 1: Define time series parameters
        decomposition.add_node(TaskNode(
            id="task1",
            description="Define the time series parameters and metrics",
            task_type="time_series_definition",
            required_metrics=["metric", "time_range", "frequency"],
            dependencies=[],
            output_format="dict"
        ))
        
        # Task 2: Extract time series data
        decomposition.add_node(TaskNode(
            id="task2",
            description="Extract data points for the time series",
            task_type="information_extraction",
            required_metrics=["values", "timestamps"],
            dependencies=["task1"],
            output_format="dict"
        ))
        
        # Task 3: Analyze trends
        decomposition.add_node(TaskNode(
            id="task3",
            description="Analyze trends in the time series",
            task_type="trend_analysis",
            required_metrics=["trend_type", "growth_rate"],
            dependencies=["task2"],
            output_format="dict"
        ))
        
        # Task 4: Summarize findings
        decomposition.add_node(TaskNode(
            id="task4",
            description="Summarize time series analysis findings",
            task_type="trend_summary",
            required_metrics=["summary", "key_points"],
            dependencies=["task3"],
            output_format="text"
        ))

    @staticmethod
    def _create_comparative_analysis_decomposition(decomposition: TaskDecomposition) -> None:
        """Create a decomposition for comparative analysis."""
        # Task 1: Define comparison parameters
        decomposition.add_node(TaskNode(
            id="task1",
            description="Define the comparison parameters",
            task_type="comparison_definition",
            required_metrics=["metrics", "entities", "time_periods"],
            dependencies=[],
            output_format="dict"
        ))
        
        # Task 2: Extract comparison data
        decomposition.add_node(TaskNode(
            id="task2",
            description="Extract data for comparison",
            task_type="information_extraction",
            required_metrics=["values", "contexts"],
            dependencies=["task1"],
            output_format="dict"
        ))
        
        # Task 3: Perform comparison
        decomposition.add_node(TaskNode(
            id="task3",
            description="Compare the extracted values",
            task_type="comparative_analysis",
            required_metrics=["differences", "percentages"],
            dependencies=["task2"],
            output_format="dict"
        ))
        
        # Task 4: Interpret comparison
        decomposition.add_node(TaskNode(
            id="task4",
            description="Interpret the comparison results",
            task_type="comparison_interpretation",
            required_metrics=["interpretation", "key_findings"],
            dependencies=["task3"],
            output_format="text"
        ))

    @staticmethod
    def _create_condition_evaluation_decomposition(decomposition: TaskDecomposition) -> None:
        """Create a decomposition for condition evaluation."""
        # Task 1: Identify conditions
        decomposition.add_node(TaskNode(
            id="task1",
            description="Identify the conditions to evaluate",
            task_type="condition_identification",
            required_metrics=["conditions", "thresholds"],
            dependencies=[],
            output_format="dict"
        ))
        
        # Task 2: Extract relevant data
        decomposition.add_node(TaskNode(
            id="task2",
            description="Extract data needed for condition evaluation",
            task_type="information_extraction",
            required_metrics=["values", "contexts"],
            dependencies=["task1"],
            output_format="dict"
        ))
        
        # Task 3: Evaluate conditions
        decomposition.add_node(TaskNode(
            id="task3",
            description="Evaluate each condition",
            task_type="condition_evaluation",
            required_metrics=["evaluation_results", "confidence"],
            dependencies=["task2"],
            output_format="dict"
        ))
        
        # Task 4: Summarize evaluation
        decomposition.add_node(TaskNode(
            id="task4",
            description="Summarize the condition evaluation results",
            task_type="evaluation_summary",
            required_metrics=["summary", "implications"],
            dependencies=["task3"],
            output_format="text"
        ))

    @staticmethod
    def _create_trend_interpretation_decomposition(decomposition: TaskDecomposition) -> None:
        """Create a decomposition for trend interpretation."""
        # Task 1: Define trend parameters
        decomposition.add_node(TaskNode(
            id="task1",
            description="Define the trend parameters to analyze",
            task_type="trend_definition",
            required_metrics=["metrics", "time_range"],
            dependencies=[],
            output_format="dict"
        ))
        
        # Task 2: Extract trend data
        decomposition.add_node(TaskNode(
            id="task2",
            description="Extract data points for trend analysis",
            task_type="information_extraction",
            required_metrics=["values", "timestamps"],
            dependencies=["task1"],
            output_format="dict"
        ))
        
        # Task 3: Analyze trend patterns
        decomposition.add_node(TaskNode(
            id="task3",
            description="Analyze patterns in the trend",
            task_type="pattern_analysis",
            required_metrics=["pattern_type", "characteristics"],
            dependencies=["task2"],
            output_format="dict"
        ))
        
        # Task 4: Interpret implications
        decomposition.add_node(TaskNode(
            id="task4",
            description="Interpret the implications of the trend",
            task_type="trend_interpretation",
            required_metrics=["interpretation", "significance"],
            dependencies=["task3"],
            output_format="text"
        ))

    @staticmethod
    def _create_causal_analysis_decomposition(decomposition: TaskDecomposition) -> None:
        """Create a decomposition for causal analysis."""
        # Task 1: Identify cause-effect relationship
        decomposition.add_node(TaskNode(
            id="task1",
            description="Identify the cause-effect relationship to analyze",
            task_type="relationship_identification",
            required_metrics=["cause_factors", "effect_metrics"],
            dependencies=[],
            output_format="dict"
        ))
        
        # Task 2: Extract relevant data
        decomposition.add_node(TaskNode(
            id="task2",
            description="Extract data for causal analysis",
            task_type="information_extraction",
            required_metrics=["cause_data", "effect_data"],
            dependencies=["task1"],
            output_format="dict"
        ))
        
        # Task 3: Analyze causation
        decomposition.add_node(TaskNode(
            id="task3",
            description="Analyze the causal relationship",
            task_type="causation_analysis",
            required_metrics=["correlation", "causation_evidence"],
            dependencies=["task2"],
            output_format="dict"
        ))
        
        # Task 4: Explain relationship
        decomposition.add_node(TaskNode(
            id="task4",
            description="Explain the causal relationship",
            task_type="causal_explanation",
            required_metrics=["explanation", "supporting_evidence"],
            dependencies=["task3"],
            output_format="text"
        ))

    @staticmethod
    def _create_multi_factor_evaluation_decomposition(decomposition: TaskDecomposition) -> None:
        """Create a decomposition for multi-factor evaluation."""
        # Task 1: Identify factors
        decomposition.add_node(TaskNode(
            id="task1",
            description="Identify the factors to evaluate",
            task_type="factor_identification",
            required_metrics=["factors", "weights"],
            dependencies=[],
            output_format="dict"
        ))
        
        # Task 2: Extract factor data
        decomposition.add_node(TaskNode(
            id="task2",
            description="Extract data for each factor",
            task_type="information_extraction",
            required_metrics=["factor_values", "contexts"],
            dependencies=["task1"],
            output_format="dict"
        ))
        
        # Task 3: Evaluate factors
        decomposition.add_node(TaskNode(
            id="task3",
            description="Evaluate each factor",
            task_type="factor_evaluation",
            required_metrics=["evaluations", "relative_importance"],
            dependencies=["task2"],
            output_format="dict"
        ))
        
        # Task 4: Synthesize evaluation
        decomposition.add_node(TaskNode(
            id="task4",
            description="Synthesize the multi-factor evaluation",
            task_type="evaluation_synthesis",
            required_metrics=["overall_assessment", "key_factors"],
            dependencies=["task3"],
            output_format="text"
        ))

    @staticmethod
    def _create_financial_health_assessment_decomposition(decomposition: TaskDecomposition) -> None:
        """Create a decomposition for financial health assessment."""
        # Task 1: Define assessment metrics
        decomposition.add_node(TaskNode(
            id="task1",
            description="Define the financial health metrics to assess",
            task_type="metric_definition",
            required_metrics=["metrics", "benchmarks"],
            dependencies=[],
            output_format="dict"
        ))
        
        # Task 2: Extract financial data
        decomposition.add_node(TaskNode(
            id="task2",
            description="Extract relevant financial data",
            task_type="information_extraction",
            required_metrics=["metric_values", "time_periods"],
            dependencies=["task1"],
            output_format="dict"
        ))
        
        # Task 3: Calculate health indicators
        decomposition.add_node(TaskNode(
            id="task3",
            description="Calculate financial health indicators",
            task_type="health_calculation",
            required_metrics=["indicators", "trends"],
            dependencies=["task2"],
            output_format="dict"
        ))
        
        # Task 4: Assess overall health
        decomposition.add_node(TaskNode(
            id="task4",
            description="Assess overall financial health",
            task_type="health_assessment",
            required_metrics=["assessment", "recommendations"],
            dependencies=["task3"],
            output_format="text"
        ))

    @staticmethod
    def _create_risk_assessment_decomposition(decomposition: TaskDecomposition) -> None:
        """Create a decomposition for risk assessment."""
        # Task 1: Identify risk factors
        decomposition.add_node(TaskNode(
            id="task1",
            description="Identify relevant risk factors",
            task_type="risk_identification",
            required_metrics=["risk_factors", "risk_types"],
            dependencies=[],
            output_format="dict"
        ))
        
        # Task 2: Extract risk data
        decomposition.add_node(TaskNode(
            id="task2",
            description="Extract data for risk analysis",
            task_type="information_extraction",
            required_metrics=["risk_metrics", "historical_data"],
            dependencies=["task1"],
            output_format="dict"
        ))
        
        # Task 3: Analyze risks
        decomposition.add_node(TaskNode(
            id="task3",
            description="Analyze identified risks",
            task_type="risk_analysis",
            required_metrics=["risk_levels", "probabilities"],
            dependencies=["task2"],
            output_format="dict"
        ))
        
        # Task 4: Assess risk impact
        decomposition.add_node(TaskNode(
            id="task4",
            description="Assess potential risk impact",
            task_type="impact_assessment",
            required_metrics=["impact_assessment", "mitigation_strategies"],
            dependencies=["task3"],
            output_format="text"
        ))

    @staticmethod
    def _create_performance_evaluation_decomposition(decomposition: TaskDecomposition) -> None:
        """Create a decomposition for performance evaluation."""
        # Task 1: Define performance metrics
        decomposition.add_node(TaskNode(
            id="task1",
            description="Define performance metrics to evaluate",
            task_type="metric_definition",
            required_metrics=["metrics", "targets"],
            dependencies=[],
            output_format="dict"
        ))
        
        # Task 2: Extract performance data
        decomposition.add_node(TaskNode(
            id="task2",
            description="Extract performance data",
            task_type="information_extraction",
            required_metrics=["metric_values", "benchmarks"],
            dependencies=["task1"],
            output_format="dict"
        ))
        
        # Task 3: Analyze performance
        decomposition.add_node(TaskNode(
            id="task3",
            description="Analyze performance against targets",
            task_type="performance_analysis",
            required_metrics=["achievements", "gaps"],
            dependencies=["task2"],
            output_format="dict"
        ))
        
        # Task 4: Evaluate overall performance
        decomposition.add_node(TaskNode(
            id="task4",
            description="Evaluate overall performance",
            task_type="performance_evaluation",
            required_metrics=["evaluation", "recommendations"],
            dependencies=["task3"],
            output_format="text"
        )) 