import os
import sys
import asyncio
import uuid
import datetime
import time
from typing import List, Optional
from loguru import logger

# Add project root to sys.path so we can run this script directly
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
  sys.path.insert(0, project_root)

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
    
    # 5. Output Summary Report
    logger.info("=" * 60)
    logger.info("INFERENCE RUN SUMMARY REPORT")
    logger.info("=" * 60)
    logger.info(f"Mode:      {dataset_mode.upper()}")
    logger.info(f"Run ID:     {run_id}")
    logger.info("-" * 60)
    logger.info("Database Records Generated:")
    logger.info(f" - inference_run_history     : 1")
    logger.info(f" - query_history         : {len(results)}")
    logger.info(f" - inference_run_query_mapping  : {len(results)}")
    
    case1_count = sum(1 for q in results if q.golden_record_id is not None)
    if dataset_mode == "case1":
      logger.info(f" - golden_record_query_mapping  : {case1_count}")
    else:
      logger.info(f" - golden_record_query_mapping  : 0 (Orphaned Query Pattern)")
    logger.info("=" * 60)
    
    return run_id

if __name__ == "__main__":
  # A simple entry point for executing Inference Runs against the Database
  import sys
  from src.configs.db import init_db_pool, close_db_pool
  
  async def main():
    await init_db_pool()
    try:
      runner = InferenceRunner()
      
      # --- Execute Case 1 (Golden Records Benchmark) ---
      # Make sure 'hsbc_2025_eval_v1' was already generated by the GoldenDatasetRunner
      # logger.info("Executing Phase 4 - Case 1: Evaluating against Golden Records")
      # case1_run_id = await runner.run(
      #   dataset_mode="case1", 
      #   batch_name="hsbc_2025_eval_v1",
      #   limit=None # Remove limit to process all 48 golden records
      # )
      
      # --- Execute Case 2 (Blind Proxy Test) ---
      logger.info("Executing Phase 4 - Case 2: Blind Proxy Test (No Ground Truth)")
      blind_questions = [
        # Financial & Performance
        "What is the total comprehensive income for the year ended 31 December 2025?",
        "How much did operating expenses increase compared to the previous year?",
        "What was the reported profit before tax for the UK region?",
        "Can you summarize the net interest income trends for 2025?",
        "What is the total allowance for expected credit losses?",
        "How much dividends were paid out to ordinary shareholders in 2025?",
        "What is the bank's common equity tier 1 (CET1) ratio?",
        "How did the wealth and personal banking division perform?",
        "What was the impact of foreign exchange fluctuations on the balance sheet?",
        "What are the total assets reported at the end of 2025?",
        
        # Risk & Compliance
        "How is the matrix management structure used to identify Material Risk Takers?",
        "What are the primary operational risks identified by the board?",
        "How does the bank manage its exposure to liquidity risks?",
        "What stress testing scenarios were applied to the credit portfolio?",
        "How are non-performing loans classified?",
        "What is the bank's policy on anti-money laundering?",
        "How does the bank hedge against interest rate risk?",
        "What were the major regulatory fines or settlements in 2025?",
        "How are cybersecurity risks mitigated?",
        "What is the role of the Group Risk Committee?",
        
        # ESG, Strategy & Corporate Governance
        "What are the specific climate change and sustainability goals mentioned for the future?",
        "What is the bank's target for financed emissions by 2030?",
        "How does the new Group-wide leadership framework support the refreshed strategy?",
        "Who are the new members appointed to the Board of Directors in 2025?",
        "What are the diversity and inclusion targets for senior management?",
        "How much was invested in digital transformation and technology?",
        "What is the company's policy on executive remuneration?",
        "How is the bank supporting the transition to a low-carbon economy?",
        "What are the key priorities outlined by the new Chairman?",
        "How does the bank measure customer satisfaction?"
      ]
      case2_run_id = await runner.run(
        dataset_mode="case2", 
        source_queries=blind_questions
      )
      
      logger.info("="*60)
      logger.info("ALL INFERENCE RUNS COMPLETED SUCCESSFULLY")
      # logger.info(f"Case 1 Run ID: {case1_run_id}")
      logger.info(f"Case 2 Run ID: {case2_run_id}")
      logger.info("Data is now ready for Phase 5: The Evaluation Runner.")
      logger.info("="*60)
      
    finally:
      await close_db_pool()
      
  asyncio.run(main())