from typing import Dict, Optional, List
from loguru import logger

from src.interfaces.diagnosis_interfaces import IDiagnosticRule
from src.domain.models import DiagnosisObject

# ==============================================================================
# Concrete Diagnostic Rules (The "Experts")
# ==============================================================================

class RetrievalQualityRule(IDiagnosticRule):
    """
    Diagnoses issues related to the Retrieval phase of RAG.
    Triggers when the system fails to pull relevant context.
    """
    def __init__(self, threshold: float = 3.5):
        self.threshold = threshold

    def analyze(self, metric_averages: Dict[str, float]) -> Optional[DiagnosisObject]:
        # If the metric wasn't computed (e.g. Case 1 only run), skip this rule
        if "context_relevance" not in metric_averages or metric_averages["context_relevance"] == 0.0:
            return None
            
        score = metric_averages["context_relevance"]
        
        if score < self.threshold:
            logger.debug(f"[Rule Triggered] RetrievalQualityRule: score {score:.2f} < {self.threshold}")
            return DiagnosisObject(
                issue="Low Retrieval Quality",
                evidence=[f"avg_context_relevance ({score:.2f}) is below the acceptable threshold of {self.threshold}."],
                likely_root_causes=[
                    "Chunk size too small causing semantic fragmentation.",
                    "Dense embedding model lacks domain-specific vocabulary sensitivity.",
                    "Top-K parameter is too restrictive to capture the necessary answer span."
                ],
                recommended_actions=[
                    "Switch chunking strategy to 'recursive_section_aware' to preserve context boundaries.",
                    "Upgrade indexing to 'hybrid_bm25_dense' to capture exact keyword matches alongside semantic meaning.",
                    "Increase retrieval 'top_k' parameter from 5 to 10."
                ]
            )
        return None


class HallucinationRule(IDiagnosticRule):
    """
    Diagnoses issues related to Generator hallucination.
    Triggers when the LLM invents facts not found in the retrieved context.
    """
    def __init__(self, threshold: float = 4.0):
        self.threshold = threshold

    def analyze(self, metric_averages: Dict[str, float]) -> Optional[DiagnosisObject]:
        if "faithfulness" not in metric_averages or metric_averages["faithfulness"] == 0.0:
            return None
            
        score = metric_averages["faithfulness"]
        
        if score < self.threshold:
            logger.debug(f"[Rule Triggered] HallucinationRule: score {score:.2f} < {self.threshold}")
            return DiagnosisObject(
                issue="High Generator Hallucination",
                evidence=[f"avg_faithfulness ({score:.2f}) is below the acceptable threshold of {self.threshold}."],
                likely_root_causes=[
                    "LLM temperature is too high, encouraging creative divergence.",
                    "System prompt lacks strict instructions forcing the LLM to rely solely on provided context.",
                    "The LLM is relying on its internal parametric memory rather than the retrieved chunks."
                ],
                recommended_actions=[
                    "Switch generation config to 'strict_citation_low_temp'.",
                    "Lower LLM generation temperature to 0.0 or 0.1.",
                    "Implement a post-generation self-correction LLM pass to verify citations."
                ]
            )
        return None


class AnswerRelevanceRule(IDiagnosticRule):
    """
    Diagnoses issues where the retrieval is good, but the LLM fails to answer the actual question.
    """
    def __init__(self, context_threshold: float = 3.5, answer_threshold: float = 3.5):
        self.context_threshold = context_threshold
        self.answer_threshold = answer_threshold

    def analyze(self, metric_averages: Dict[str, float]) -> Optional[DiagnosisObject]:
        if "context_relevance" not in metric_averages or "answer_relevance" not in metric_averages:
            return None
        if metric_averages["context_relevance"] == 0.0 or metric_averages["answer_relevance"] == 0.0:
            return None
            
        ctx_score = metric_averages["context_relevance"]
        ans_score = metric_averages["answer_relevance"]
        
        # Trigger: Good Context, but Bad Answer
        if ctx_score >= self.context_threshold and ans_score < self.answer_threshold:
            logger.debug(f"[Rule Triggered] AnswerRelevanceRule: ans {ans_score:.2f} < {self.answer_threshold} while ctx is {ctx_score:.2f}")
            return DiagnosisObject(
                issue="Poor Answer Relevance despite Good Retrieval",
                evidence=[
                    f"avg_context_relevance ({ctx_score:.2f}) is healthy (>= {self.context_threshold}).",
                    f"However, avg_answer_relevance ({ans_score:.2f}) is poor (< {self.answer_threshold})."
                ],
                likely_root_causes=[
                    "The user's initial query was too vague or poorly phrased.",
                    "The LLM was confused by 'semantic noise' (distracting information) present in the retrieved chunks.",
                    "The LLM generator is too weak to synthesize a coherent answer from multiple distinct chunks."
                ],
                recommended_actions=[
                    "Enable query_prompting strategy: 'query_rewrite' or 'step_back' to clarify the user's intent before retrieval.",
                    "Inject a reranker (e.g., 'cross_encoder_miniLM') to aggressively filter out middle-ranked noise before generation."
                ]
            )
        return None


class BenchmarkCorrectnessRule(IDiagnosticRule):
    """
    Diagnoses issues specific to Case 1 (Benchmark Evaluation).
    Triggers when the system's output fundamentally diverges from the Ground Truth.
    """
    def __init__(self, threshold: float = 4.0):
        self.threshold = threshold

    def analyze(self, metric_averages: Dict[str, float]) -> Optional[DiagnosisObject]:
        if "correctness" not in metric_averages or metric_averages["correctness"] == 0.0:
            return None
            
        score = metric_averages["correctness"]
        
        if score < self.threshold:
            logger.debug(f"[Rule Triggered] BenchmarkCorrectnessRule: score {score:.2f} < {self.threshold}")
            return DiagnosisObject(
                issue="Low Accuracy against Benchmark Ground Truth",
                evidence=[f"avg_correctness ({score:.2f}) is below the acceptable threshold of {self.threshold}."],
                likely_root_causes=[
                    "The system is failing on complex, multi-hop reasoning questions present in the benchmark dataset.",
                    "Crucial information required by the Ground Truth is completely missing from the retrieval corpus."
                ],
                recommended_actions=[
                    "Upgrade query_prompting strategy to 'decompose_then_merge' to handle complex multi-part questions.",
                    "Verify that the necessary source documents for the benchmark batch are actually ingested in the database."
                ]
            )
        return None

# ==============================================================================
# The Rule Engine Orchestrator
# ==============================================================================

class DiagnosticEngine:
    """
    The core Expert System.
    Loads registered rules and sequentially evaluates them against metric averages.
    """
    def __init__(self):
        self.rules: List[IDiagnosticRule] = []
        
    def register_rule(self, rule: IDiagnosticRule) -> None:
        """Dynamically add a new heuristic rule to the engine (OCP compliant)."""
        self.rules.append(rule)
        
    def diagnose(self, metric_averages: Dict[str, float]) -> List[DiagnosisObject]:
        """
        Executes all registered rules against the provided metrics.
        Returns a list of all triggered diagnoses.
        """
        logger.info(f"Expert Engine beginning diagnosis across {len(self.rules)} rules...")
        diagnoses = []
        
        for rule in self.rules:
            # Polymorphic execution
            result = rule.analyze(metric_averages)
            if result:
                diagnoses.append(result)
                
        logger.info(f"Diagnosis complete. Engine identified {len(diagnoses)} critical issues.")
        return diagnoses