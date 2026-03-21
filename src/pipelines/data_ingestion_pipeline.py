import asyncio
from loguru import logger
from src.interfaces.ingestion_interfaces import BaseLoader, BaseChunker, BaseEmbedder, BaseDAO
from src.domain.models import Document

class DataIngestionPipeline:
    """
    Orchestrator for the RAG Data Ingestion Process.
    Applies Dependency Injection for maximal testability and flexibility.
    """
    def __init__(
        self,
        loader: BaseLoader,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        dao: BaseDAO,
        created_by_user: str = "automated_ingestion_pipeline"
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.dao = dao
        self.created_by_user = created_by_user

    async def run(self, file_path: str, topics: list[str]) -> int:
        """
        Executes the full ingestion pipeline: Load -> Chunk -> Embed -> Save (ACID)
        """
        logger.info(f"🚀 Starting Ingestion Pipeline for: {file_path}")
        logger.info(f"Assigning metadata topics: {topics}")
        
        # 1. Parsing the PDF
        logger.debug(f"Stage 1: Loading Document via {self.loader.__class__.__name__}")
        document: Document = await self.loader.load(file_path)
        document.topics = topics
        logger.info(f"Document Loaded: {document.document_name} | {document.total_pages} Pages | MD5: {document.md5_hash[:8]}...")
        
        # 2. Chunking
        logger.debug(f"Stage 2: Chunking Document via {self.chunker.__class__.__name__}")
        document.chunks = self.chunker.chunk(document)
        logger.info(f"Chunking Complete: Generated {len(document.chunks)} chunks.")
        
        if not document.chunks:
            logger.warning("No text was extracted or chunked. Aborting pipeline.")
            return -1

        # 3. Embedding
        logger.debug(f"Stage 3: Embedding {len(document.chunks)} chunks via {self.embedder.__class__.__name__}")
        document.chunks = await self.embedder.embed_batch(document.chunks)
        logger.info("Embedding Complete: Vectors successfully generated.")
        
        # 4. Transactional Upsert
        logger.debug(f"Stage 4: Transactional Database Upsert via {self.dao.__class__.__name__}")
        document_id = await self.dao.upsert_document_transactionally(
            document=document,
            created_by=self.created_by_user
        )
        
        logger.info(f"✅ Ingestion Pipeline Finished. Document ID: {document_id}")
        return document_id

if __name__ == "__main__":
    logger.info("This module defines the DataIngestionPipeline class.")
    logger.info("To run the ingestion process, please execute the main runner script (e.g., run_ingestion.py).")
