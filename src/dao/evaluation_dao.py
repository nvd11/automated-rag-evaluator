import json
from typing import List
from loguru import logger

from src.interfaces.evaluator_interfaces import IEvaluationDAO
from src.domain.models import QueryEvaluationDTO, EvaluationMetricRecord, EvaluationJobHistory
from src.configs.db import get_db_connection

class PgVectorEvaluationDAO(IEvaluationDAO):
  """
  Concrete implementation for the LLM-as-a-Judge Evaluation Pipeline Data Access.
  """

  async def fetch_queries_for_evaluation(self, run_id: str) -> List[QueryEvaluationDTO]:
    logger.info(f"Fetching historical queries for inference run '{run_id}'...")
    
    # The ultimate LEFT JOIN:
    # We start with the specific run mapping, get the query details,
    # then LEFT JOIN on the golden record mapping. If it matches, we pull the ground truth!
    query = """
      SELECT 
        qh.query_id,
        qh.question,
        qh.generated_answer,
        qh.retrieved_contexts,
        gr.ground_truth
      FROM inference_run_query_mapping irm
      JOIN query_history qh 
        ON irm.query_id = qh.query_id
      LEFT JOIN golden_record_query_mapping grm 
        ON qh.query_id = grm.query_id
      LEFT JOIN golden_records gr 
        ON grm.golden_record_id = gr.id AND gr.is_deleted = FALSE
      WHERE irm.run_id = %s 
       AND irm.is_deleted = FALSE 
       AND qh.is_deleted = FALSE
    """
    
    results: List[QueryEvaluationDTO] = []
    
    async with get_db_connection() as conn:
      async with conn.cursor() as cur:
        await cur.execute(query, (run_id,))
        rows = await cur.fetchall()
        
        for row in rows:
          query_id, question, generated_answer, retrieved_contexts, ground_truth = row
          
          results.append(
            QueryEvaluationDTO(
              query_id=str(query_id),
              question=question,
              generated_answer=generated_answer if generated_answer else "",
              retrieved_contexts=retrieved_contexts if isinstance(retrieved_contexts, list) else json.loads(retrieved_contexts or "[]"),
              ground_truth=ground_truth
            )
          )
    
    case1_count = sum(1 for q in results if q.has_ground_truth)
    case2_count = len(results) - case1_count
    logger.debug(f"Fetched {len(results)} queries for evaluation (Case 1: {case1_count}, Case 2: {case2_count})")
    
    return results

  async def create_evaluation_job(self, job: EvaluationJobHistory, created_by: str) -> None:
    logger.info(f"Creating evaluation job history record for job '{job.job_id}'...")
    async with get_db_connection() as conn:
      async with conn.cursor() as cur:
        query = """
          INSERT INTO evaluation_job_history 
          (job_id, inference_run_id, start_time, end_time, evaluator_model, evaluator_prompt_version, created_by)
          VALUES (%s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, %s, %s, %s)
        """
        await cur.execute(query, (
          job.job_id,
          job.inference_run_id,
          job.evaluator_model,
          job.evaluator_prompt_version,
          created_by
        ))
      await conn.commit()

  async def bulk_insert_evaluation_metrics(self, metrics: List[EvaluationMetricRecord], created_by: str) -> None:
    if not metrics:
      logger.warning("No evaluation metrics provided for bulk insert.")
      return
      
    logger.info(f"Initiating transactional bulk insert for {len(metrics)} metric records...")
    
    query = """
      INSERT INTO evaluation_metrics 
      (id, job_id, query_id, evaluation_strategy, metric_category, metric_name, metric_value, reasoning, created_by)
      VALUES (gen_random_uuid(), %s, %s, %s, %s, %s, %s, %s, %s)
      ON CONFLICT ON CONSTRAINT uq_query_strategy_metric_job DO NOTHING;
    """
    
    tuples = [
      (
        m.job_id,
        m.query_id,
        m.evaluation_strategy,
        m.metric_category,
        m.metric_name,
        m.metric_value,
        m.reasoning,
        created_by
      )
      for m in metrics
    ]
    
    async with get_db_connection() as conn:
      try:
        async with conn.cursor() as cur:
          await cur.executemany(query, tuples)
          inserted_count = cur.rowcount
        await conn.commit()
        logger.info(f"Successfully persisted {inserted_count} new metric records (Ignored duplicates if any).")
      except Exception as e:
        await conn.rollback()
        logger.error(f"Failed to persist evaluation metrics: {e}")
        raise