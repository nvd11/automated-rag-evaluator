from abc import ABC, abstractmethod
from typing import List
from src.domain.models import Document, Chunk

class BaseLoader(ABC):
    @abstractmethod
    async def load(self, file_path: str) -> Document:
        """Parses a file and returns a structured Document domain object."""
        pass

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Splits the raw text of a Document into smaller Chunks."""
        pass

class BaseEmbedder(ABC):
    @abstractmethod
    async def embed_batch(self, chunks: List[Chunk]) -> List[Chunk]:
        """Generates vector embeddings for a batch of Chunks and mutates the objects."""
        pass

class BaseDAO(ABC):
    @abstractmethod
    async def upsert_document_transactionally(self, document: Document, created_by: str) -> int:
        """
        Persists the Document, its Topics, and its embedded Chunks into the database
        within a single ACID transaction. Performs idempotency cleanup if needed.
        Returns the new or existing document_id.
        """
        pass
