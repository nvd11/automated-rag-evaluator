from abc import ABC, abstractmethod
from typing import List, Optional
from src.domain.models import Chunk, GoldenRecord

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
