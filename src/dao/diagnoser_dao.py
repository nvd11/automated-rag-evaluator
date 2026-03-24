from typing import Dict
from loguru import logger

from src.interfaces.diagnosis_interfaces import IDiagnoserDAO
from src.configs.db import get_db_connection

class PgVectorDiagnoserDAO(IDiagnoserDAO):
  """
  Concrete implementation for the Read-Only Diagnoser Pipeline Data Access.
  """

  async def fetch_metric_averages(self, job_id: str) -> Dict[str, float]:
    logger.info(f"Fetching average evaluation metrics for job_id '{job_id}'...")
    
    # Leverage the Pivot View to compute averages across the entire job execution
    query = """
      SELECT 
        AVG(context_relevance_score) as avg_context_relevance,
        AVG(faithfulness_score) as avg_faithfulness,
        AVG(answer_relevance_score) as avg_answer_relevance,
        AVG(correctness_score) as avg_correctness,
        AVG(semantic_similarity_score) as avg_semantic_similarity
      FROM v_evaluation_metrics_pivot
      WHERE job_id = %s
    """
    
    averages = {}
    
    async with get_db_connection() as conn:
      async with conn.cursor() as cur:
        await cur.execute(query, (job_id,))
        row = await cur.fetchone()
        
        if row:
          averages["context_relevance"] = float(row[0]) if row[0] is not None else 0.0
          averages["faithfulness"] = float(row[1]) if row[1] is not None else 0.0
          averages["answer_relevance"] = float(row[2]) if row[2] is not None else 0.0
          averages["correctness"] = float(row[3]) if row[3] is not None else 0.0
          averages["semantic_similarity"] = float(row[4]) if row[4] is not None else 0.0
        else:
          logger.warning(f"No metric data found for job_id '{job_id}'. Defaulting averages to 0.0.")
          averages = {
            "context_relevance": 0.0,
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "correctness": 0.0,
            "semantic_similarity": 0.0
          }
          
    logger.debug(f"Computed metric averages: {averages}")
    return averages

  async def fetch_job_metadata(self, job_id: str) -> Dict[str, str]:
    logger.info(f"Fetching metadata for evaluation job '{job_id}'...")
    
    # We need the original inference run ID, and if it was a Case 1 run, the batch name.
    # We join the evaluation_job_history -> inference_run_query_mapping -> query_history -> golden_record_query_mapping -> golden_records
    # to determine the dataset_name. If it yields NULL, it's a Case 2 blind test.
    query = """
      SELECT 
        ejh.inference_run_id,
        MAX(gr.batch_name) as dataset_name
      FROM evaluation_job_history ejh
      LEFT JOIN inference_run_query_mapping irm ON ejh.inference_run_id = irm.run_id
      LEFT JOIN golden_record_query_mapping grm ON irm.query_id = grm.query_id
      LEFT JOIN golden_records gr ON grm.golden_record_id = gr.id
      WHERE ejh.job_id = %s
      GROUP BY ejh.inference_run_id
    """
    
    metadata = {}
    
    async with get_db_connection() as conn:
      async with conn.cursor() as cur:
        await cur.execute(query, (job_id,))
        row = await cur.fetchone()
        
        if row:
          inference_run_id = str(row[0]) if row[0] else "UNKNOWN_RUN"
          dataset_name = row[1] if row[1] else "case2_blind_test_dataset"
          
          metadata["setting_id"] = inference_run_id
          metadata["dataset_name"] = dataset_name
        else:
          logger.warning(f"No job metadata found for job_id '{job_id}'.")
          metadata = {"setting_id": "NOT_FOUND", "dataset_name": "NOT_FOUND"}
          
    logger.debug(f"Retrieved job metadata: {metadata}")
    return metadata