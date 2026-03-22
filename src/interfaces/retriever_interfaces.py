from abc import ABC, abstractmethod
from typing import List, Optional
from src.domain.models import SearchQuery, RetrievedContext

class IRetrieverDAO(ABC):
    """Contract for the Data Access Object that handles vector similarity search."""
    @abstractmethod
    async def semantic_search(self, query: SearchQuery) -> List[RetrievedContext]:
        """
        Executes a vector search in the database based on the provided SearchQuery.
        Must apply metadata filters and cosine similarity thresholding.
        """
        pass

class ILLMGenerator(ABC):
    """Contract for the LLM component that synthesizes the final answer (using LCEL)."""
    @abstractmethod
    async def generate_answer(self, context_texts: List[str], query: str) -> str:
        """
        Combines retrieved context chunks with the user's query into a prompt,
        then calls the LLM to generate the final answer.
        """
        pass

class BaseRetriever(ABC):
    """Contract for the Retriever Orchestrator that combines Embedder and DAO."""
    @abstractmethod
    async def retrieve(
        self, 
        text: str, 
        top_k: int, 
        similarity_threshold: float, 
        filters: Optional[List[str]] = None
    ) -> List[RetrievedContext]:
        """
        Takes raw text, generates its embedding, and returns the top-K relevant chunks
        that meet the similarity threshold and metadata filters.
        """
        pass
