from abc import ABC, abstractmethod
from typing import List, Optional
from src.domain.models import Chunk, GoldenRecord, QueryEvaluationDTO, EvaluationMetricRecord, ScoreWithReasoning

class IGoldenRecordDAO(ABC):
    """
    Contract for data persistence related to Benchmark Evaluation datasets.
    """
    
    @abstractmethod
    async def get_random_seed_chunks(self, limit: int, topics: Optional[List[str]] = None) -> List[Chunk]:
        """
        Retrieves N random, high-quality document chunks to be used as factual seeds
        for generating synthetic Q&A pairs.
        
        Args:
            limit: The exact number of chunks to randomly sample.
            topics: Optional list of domains to restrict the sampling (e.g., ['credit_risk']).
        """
        pass

    @abstractmethod
    async def bulk_insert_golden_records(self, batch_name: str, records: List[GoldenRecord], created_by: str) -> None:
        """
        Persists a batch of freshly generated evaluation questions to the database.
        Must ensure idempotency by soft-deleting any existing records with the identical `batch_name`.
        """
        pass


class IDatasetGenerator(ABC):
    """
    Contract for the LLM component responsible for "reverse-engineering" questions
    from raw text chunks to construct the Golden Dataset.
    """
    
    @abstractmethod
    async def agenerate_qa_from_chunk(self, chunk: Chunk, batch_name: str) -> GoldenRecord:
        """
        Consumes a single raw text chunk, prompts the Teacher LLM to formulate a 
        professional question and extract its exact answer, and returns a fully
        structured GoldenRecord entity.
        
        This method is designed to be highly concurrent (e.g., via asyncio.gather).
        """
        pass

class IEvaluationDAO(ABC):
    """
    Contract for data persistence related to the LLM-as-a-Judge Evaluation Pipeline.
    Manages loading historical queries and saving the granular EAV metric scores.
    """
    
    @abstractmethod
    async def fetch_queries_for_evaluation(self, run_id: str) -> List[QueryEvaluationDTO]:
        """
        Extracts all queries executed during a specific inference run.
        Must seamlessly combine Case 1 (with ground truth) and Case 2 (without) queries.
        """
        pass
        
    @abstractmethod
    async def bulk_insert_evaluation_metrics(self, metrics: List[EvaluationMetricRecord], created_by: str) -> None:
        """
        Persists the LLM Judge's scoring results into the EAV database table transactionally.
        Must handle UNIQUE constraint violations gracefully to avoid duplicate scoring.
        """
        pass

class ILLMJudge(ABC):
    """
    Contract for the LLM component acting as an impartial judge/evaluator.
    Uses Structured Output to enforce the ScoreWithReasoning schema.
    Polymorphic design: Concrete implementations (e.g., GoldenBaselineJudge, RagTriadJudge) 
    will implement their specific evaluation strategies.
    """
    
    @abstractmethod
    async def evaluate_query(self, dto: QueryEvaluationDTO) -> List[ScoreWithReasoning]:
        """
        Evaluates a single query based on the concrete judge's specific strategy.
        
        Args:
            dto: The QueryEvaluationDTO containing all context (question, answer, retrieved chunks, and optionally ground truth).
            
        Returns:
            A list of scored metrics with reasoning, formatted as ScoreWithReasoning.
        """
        pass
