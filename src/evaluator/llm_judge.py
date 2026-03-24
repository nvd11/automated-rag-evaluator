import asyncio
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Sequence
from loguru import logger

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from src.interfaces.evaluator_interfaces import ILLMJudge
from src.domain.models import QueryEvaluationDTO, ScoreWithReasoning
from src.configs.settings import settings

class Case1EvaluationResult(BaseModel):
    """Structured output for Case 1 Evaluation."""
    correctness_score: float = Field(description="Numerical score for correctness (0.0 to 5.0).")
    correctness_reasoning: str = Field(description="Detailed explanation for the correctness score.")

class Case2EvaluationResult(BaseModel):
    """Structured output for Case 2 Evaluation (RAG Triad)."""
    context_relevance_score: float = Field(description="Score for context relevance (0.0 to 5.0).")
    context_relevance_reasoning: str = Field(description="Reasoning for the context relevance score.")
    faithfulness_score: float = Field(description="Score for faithfulness / hallucination (0.0 to 5.0).")
    faithfulness_reasoning: str = Field(description="Reasoning for the faithfulness score.")
    answer_relevance_score: float = Field(description="Score for answer relevance (0.0 to 5.0).")
    answer_relevance_reasoning: str = Field(description="Reasoning for the answer relevance score.")

class GoldenBaselineJudge(ILLMJudge):
    """
    Concrete evaluator for Case 1 (Benchmark Dataset).
    Compares the RAG-generated answer directly against the Human/Teacher Ground Truth.
    """
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.structured_llm = self.llm.with_structured_output(Case1EvaluationResult)
        
        template = """You are an impartial and rigorous Evaluation Judge grading an AI assistant's answer.
        
Your task is to compare the "Generated Answer" against the "Ground Truth" answer for the given "Question".
You must evaluate the 'correctness' of the Generated Answer.

Criteria for 'correctness' (Scale 0.0 to 5.0):
5.0: The Generated Answer is perfectly accurate, complete, and aligns entirely with the Ground Truth.
4.0: The Generated Answer is correct and covers the main points, but might miss minor nuances from the Ground Truth.
3.0: The Generated Answer is partially correct but misses significant details or includes slight inaccuracies.
1.0-2.0: The Generated Answer is mostly incorrect or fails to address the core of the Ground Truth.
0.0: The Generated Answer is completely wrong, entirely fabricated, or states it cannot answer when the Ground Truth provides the answer.

You MUST return your evaluation in the requested JSON format.

---
Question: {question}

Ground Truth (The Perfect Answer):
{ground_truth}

Generated Answer (The Student's Answer):
{generated_answer}
"""
        self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = self.prompt | self.structured_llm

    async def evaluate_query(self, dto: QueryEvaluationDTO) -> List[ScoreWithReasoning]:
        if not dto.has_ground_truth:
            raise ValueError(f"GoldenBaselineJudge requires a ground_truth, but query_id {dto.query_id} does not have one.")
            
        logger.debug(f"GoldenBaselineJudge evaluating query_id: {dto.query_id}")
        
        try:
            if settings.ENABLE_PROXY:
                logger.debug(f"GoldenBaselineJudge executing via Sync Thread for Proxy...")
                result: Case1EvaluationResult = await asyncio.to_thread(self.chain.invoke, {
                    "question": dto.question,
                    "ground_truth": dto.ground_truth,
                    "generated_answer": dto.generated_answer
                })
            else:
                result: Case1EvaluationResult = await self.chain.ainvoke({
                    "question": dto.question,
                    "ground_truth": dto.ground_truth,
                    "generated_answer": dto.generated_answer
                })
            
            return [
                ScoreWithReasoning(
                    metric_name="correctness",
                    score=result.correctness_score,
                    reasoning=result.correctness_reasoning
                )
            ]
        except Exception as e:
            logger.error(f"Failed to evaluate correctness for query_id {dto.query_id}: {e}")
            return []


class RagTriadJudge(ILLMJudge):
    """
    Concrete evaluator for Case 2 (Blind / Production Testing).
    Calculates the RAG Triad (Faithfulness, Answer Relevance, Context Relevance)
    without needing a ground truth.
    """
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.structured_llm = self.llm.with_structured_output(Case2EvaluationResult)
        
        template = """You are an impartial and rigorous Evaluation Judge auditing an enterprise RAG (Retrieval-Augmented Generation) system.
        
You must evaluate the system based on the "RAG Triad". You will be provided with the user's "Question", the "Retrieved Context" (the documents the system found), and the "Generated Answer" (what the system replied).

You must evaluate and return exactly three metrics. Score each on a scale of 0.0 to 5.0, and provide detailed reasoning for each.

Metric 1: 'context_relevance'
- Does the Retrieved Context actually contain information relevant to answering the Question?
- 5.0: The context perfectly contains the exact information needed.
- 0.0: The context is completely irrelevant garbage.

Metric 2: 'faithfulness' (Hallucination Check)
- Is the Generated Answer completely supported by the Retrieved Context? 
- 5.0: The answer is 100% derived from the context with no outside hallucinations.
- 0.0: The answer invents facts not found in the context, or hallucinates entirely.
- Note: If the answer correctly states "I cannot answer based on the context", faithfulness is 5.0.

Metric 3: 'answer_relevance'
- Does the Generated Answer directly and usefully answer the user's Question?
- 5.0: Direct, concise, and perfectly answers the specific question.
- 0.0: Completely dodges the question or rambles about unrelated topics.

You MUST return your evaluation in the requested JSON format containing these three scores and their reasoning.

---
Question: {question}

Retrieved Context:
{contexts}

Generated Answer:
{generated_answer}
"""
        self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = self.prompt | self.structured_llm

    def _format_contexts(self, retrieved_contexts: List[dict]) -> str:
        formatted = []
        for i, ctx in enumerate(retrieved_contexts):
            text = ctx.get("text", "")
            formatted.append(f"[Document {i+1}]:\n{text}")
        return "\n\n".join(formatted)

    async def evaluate_query(self, dto: QueryEvaluationDTO) -> List[ScoreWithReasoning]:
        logger.debug(f"RagTriadJudge evaluating query_id: {dto.query_id}")
        
        contexts_str = self._format_contexts(dto.retrieved_contexts)
        
        if not contexts_str.strip():
            logger.warning(f"RagTriadJudge received empty context for query_id {dto.query_id}. Faithfulness may be skewed.")
            contexts_str = "<NO CONTEXT RETRIEVED>"
            
        try:
            if settings.ENABLE_PROXY:
                logger.debug(f"RagTriadJudge executing via Sync Thread for Proxy...")
                result: Case2EvaluationResult = await asyncio.to_thread(self.chain.invoke, {
                    "question": dto.question,
                    "contexts": contexts_str,
                    "generated_answer": dto.generated_answer
                })
            else:
                result: Case2EvaluationResult = await self.chain.ainvoke({
                    "question": dto.question,
                    "contexts": contexts_str,
                    "generated_answer": dto.generated_answer
                })
            
            return [
                ScoreWithReasoning(
                    metric_name="context_relevance",
                    score=result.context_relevance_score,
                    reasoning=result.context_relevance_reasoning
                ),
                ScoreWithReasoning(
                    metric_name="faithfulness",
                    score=result.faithfulness_score,
                    reasoning=result.faithfulness_reasoning
                ),
                ScoreWithReasoning(
                    metric_name="answer_relevance",
                    score=result.answer_relevance_score,
                    reasoning=result.answer_relevance_reasoning
                )
            ]
            
        except Exception as e:
            logger.error(f"Failed to evaluate RAG Triad for query_id {dto.query_id}: {e}")
            return []