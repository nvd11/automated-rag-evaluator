import asyncio
import uuid
import datetime
import time
from typing import List, Optional
from loguru import logger

from src.configs.settings import settings
from src.domain.models import InferenceRun, QueryHistoryRecord
from src.dao.inference_dao import PgVectorInferenceDAO

# Components for Agent
from src.llm.gemini_factory import GeminiLLMFactory
from src.ingestion.embedders.gemini_embedder import GeminiEmbedder
from src.dao.pgvector_retriever_dao import PgVectorRetrieverDAO
from src.retrieval.semantic_retriever import SemanticRetriever
from src.retrieval.langchain_generator import LangchainRAGGenerator
from src.agents.rag_agent import RAGAgent

class InferenceRunner:
    """
    Simulates high-throughput production RAG traffic and benchmarks.
    Supports both Case 1 (Golden Records) and Case 2 (Blind Testing) 
    utilizing the Orphaned Query Pattern.
    """
    
    def __init__(self):
        # Initialize Data Access
        self.inference_dao = PgVectorInferenceDAO()
        
        # Initialize RAG Agent components
        embedder = GeminiEmbedder()
        retriever_dao = PgVectorRetrieverDAO()
        retriever = SemanticRetriever(embedder=embedder, dao=retriever_dao)
        
        llm_factory = GeminiLLMFactory()
        inference_llm = llm_factory.create_llm(model_name=settings.LLM_INFERENCE_MODEL, temperature=0.0)
        generator = LangchainRAGGenerator(llm=inference_llm)
        
        self.agent = RAGAgent(retriever=retriever, generator=generator)

    async def _process_single_query(self, question: str, golden_record_id: Optional[str] = None) -> QueryHistoryRecord:
        """Runs a single query through the RAG Agent and formats the result."""
        query_id = str(uuid.uuid4())
        
        # Start timing
        # We assume query and retrieval timings are instantaneous for this demo tracking,
        # but in a production system we'd track these from the Agent's internal telemetry.
        now = datetime.datetime.now(datetime.UTC).isoformat()
        
        response = await self.agent.ask(
            question=question,
            top_k=settings.RETRIEVAL_TOP_K,
            similarity_threshold=settings.RETRIEVAL_SIMILARITY_THRESHOLD
        )
        
        end_now = datetime.datetime.now(datetime.UTC).isoformat()
        
        return QueryHistoryRecord(
            query_id=query_id,
            question=question,
            generated_answer=response.generated_answer,
            retrieved_contexts=[ctx.model_dump() for ctx in response.retrieved_contexts],
            query_time=now,
            retrieval_time=now,
            response_time=end_now,
            golden_record_id=golden_record_id
        )

    async def run(
        self, 
        dataset_mode: str, 
        batch_name: Optional[str] = None, 
        limit: Optional[int] = None, 
        source_queries: Optional[List[str]] = None, 
        created_by: str = "inference_runner"
    ) -> str:
        """
        Executes the Inference Runner pipeline.
        
        Args:
            dataset_mode: 'case1' (Golden Records) or 'case2' (Blind Proxy Test).
            batch_name: Name of the Golden Record batch (Required for case1).
            limit: Optional limit of queries to process.
            source_queries: List of raw question strings (Required for case2).
            created_by: Identifier for auditing.
            
        Returns:
            str: The generated run_id.
        """
        run_id = str(uuid.uuid4())
        logger.info(f"Starting Inference Run [{run_id}] - Mode: {dataset_mode.upper()}")
        
        # 1. Setup InferenceRun Metadata
        inference_run = InferenceRun(
            run_id=run_id,
            chunking_config="default", 
            indexing_config="pgvector_hnsw",
            reranking_config="none",
            prompting_config="standard_rag",
            generation_config=settings.LLM_INFERENCE_MODEL
        )
        
        queries_to_process = [] # List of tuples: (question_str, golden_record_id_str_or_None)
        
        # 2. Branching Logic: Case 1 vs Case 2
        if dataset_mode == "case1":
            if not batch_name:
                raise ValueError("batch_name is required for case1")
                
            golden_records = await self.inference_dao.fetch_golden_records(batch_name=batch_name, limit=limit)
            if not golden_records:
                logger.warning(f"No Golden Records found for batch '{batch_name}'.")
                return run_id
                
            for record in golden_records:
                queries_to_process.append((record.question, record.id))
                
        elif dataset_mode == "case2":
            if not source_queries:
                raise ValueError("source_queries are required for case2")
                
            # Truncate if limit provided
            if limit and limit < len(source_queries):
                source_queries = source_queries[:limit]
                
            for q in source_queries:
                queries_to_process.append((q, None)) # Orphaned Query Pattern: No Ground Truth ID
                
        else:
            raise ValueError(f"Unknown dataset_mode: {dataset_mode}. Use 'case1' or 'case2'.")
            
        logger.info(f"Loaded {len(queries_to_process)} queries. Beginning inference execution...")
        
        # 3. Concurrent Inference
        # We use asyncio.gather to process queries in parallel, wrapped in a Semaphore 
        # to respect API rate limits (e.g., Gemini limits).
        tasks = [self._process_single_query(q, g_id) for q, g_id in queries_to_process]
        
        sem = asyncio.Semaphore(3) # Safe concurrency limit
        
        async def sem_task(task):
            async with sem:
                return await task
                
        results: List[QueryHistoryRecord] = await asyncio.gather(*(sem_task(t) for t in tasks))
        
        # 4. Transactional Persistence
        logger.info(f"Inference complete for {len(results)} queries. Persisting results to database...")
        await self.inference_dao.persist_inference_run(
            run=inference_run,
            queries=results,
            created_by=created_by
        )
        
        logger.info(f"Inference Run [{run_id}] successfully completed and saved.")
        return run_id

if __name__ == "__main__":
    # A simple entry point for quick manual testing if run directly
    import sys
    from src.configs.db import init_db_pool, close_db_pool
    
    async def main():
        await init_db_pool()
        try:
            runner = InferenceRunner()
            # Demo Case 2
            run_id = await runner.run(
                dataset_mode="case2", 
                source_queries=["What is HSBC's total profit in 2025?"]
            )
            logger.info(f"Demo run completed with run_id: {run_id}")
        finally:
            await close_db_pool()
            
    asyncio.run(main())