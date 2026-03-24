import json
from typing import List, Optional
from loguru import logger

from src.interfaces.inference_interfaces import IInferenceDAO
from src.domain.models import GoldenRecord, InferenceRun, QueryHistoryRecord
from src.configs.db import get_db_connection

class PgVectorInferenceDAO(IInferenceDAO):
  """
  Concrete implementation of IInferenceDAO utilizing async psycopg3.
  Executes the 'Orphaned Query Pattern' architecture for isolating Case 1 and Case 2 evaluations.
  """
  
  async def fetch_golden_records(self, batch_name: str, limit: Optional[int] = None) -> List[GoldenRecord]:
    logger.info(f"Fetching Golden Records for batch '{batch_name}'...")
    
    query = """
      SELECT id, batch_name, question, ground_truth, expected_topics, complexity 
      FROM golden_records 
      WHERE batch_name = %s AND is_deleted = FALSE
    """
    params = [batch_name]
    
    if limit:
      query += " LIMIT %s"
      params.append(limit)
      
    records: List[GoldenRecord] = []
    
    async with get_db_connection() as conn:
      async with conn.cursor() as cur:
        await cur.execute(query, params)
        rows = await cur.fetchall()
        
        for row in rows:
          records.append(
            GoldenRecord(
              id=str(row[0]),
              batch_name=row[1],
              question=row[2],
              ground_truth=row[3],
              expected_topics=row[4] if isinstance(row[4], list) else json.loads(row[4] or "[]"),
              complexity=row[5]
            )
          )
          
    logger.debug(f"Retrieved {len(records)} Golden Records.")
    return records

  async def persist_inference_run(self, run: InferenceRun, queries: List[QueryHistoryRecord], created_by: str) -> None:
    if not queries:
      logger.warning(f"No queries provided for inference run '{run.run_id}'. Aborting insert.")
      return
      
    logger.info(f"Initiating transactional bulk insert for inference run '{run.run_id}' with {len(queries)} queries.")
    
    async with get_db_connection() as conn:
      try:
        async with conn.cursor() as cur:
          # 1. Insert Inference Run Metadata
          run_query = """
            INSERT INTO inference_run_history 
            (run_id, start_time, end_time, chunking_config, indexing_config, reranking_config, prompting_config, generation_config, cost_estimate, created_by)
            VALUES (%s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, %s, %s, %s, %s, %s, %s, %s);
          """
          # We default start_time and end_time to DB current_timestamp for simplicity in this demo.
          await cur.execute(run_query, (
            run.run_id,
            run.chunking_config,
            run.indexing_config,
            run.reranking_config,
            run.prompting_config,
            run.generation_config,
            0.0, # Dummy cost
            created_by
          ))

          # 2. Bulk Insert Query History
          qh_query = """
            INSERT INTO query_history 
            (query_id, queried_by, question, retrieved_contexts, generated_answer, query_time, retrieval_time, response_time, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
          """
          
          qh_tuples = [
            (
              q.query_id,
              created_by, # queried_by (who initiated the run)
              q.question,
              json.dumps(q.retrieved_contexts),
              q.generated_answer,
              q.query_time,
              q.retrieval_time,
              q.response_time,
              created_by
            ) 
            for q in queries
          ]
          await cur.executemany(qh_query, qh_tuples)

          # 3. Bulk Insert Mappings to the Inference Run
          run_map_query = """
            INSERT INTO inference_run_query_mapping (run_id, query_id, created_by)
            VALUES (%s, %s, %s);
          """
          run_map_tuples = [(run.run_id, q.query_id, created_by) for q in queries]
          await cur.executemany(run_map_query, run_map_tuples)
          
          # 4. (The Orphaned Query Pattern) Bulk Insert Mappings to Golden Records if Case 1
          # We filter the list to only include queries that HAVE a golden_record_id attached
          case1_queries = [q for q in queries if q.golden_record_id is not None]
          
          if case1_queries:
            logger.info(f"Linking {len(case1_queries)} queries to their Ground Truth (Case 1 Setup).")
            golden_map_query = """
              INSERT INTO golden_record_query_mapping (query_id, golden_record_id, created_by)
              VALUES (%s, %s, %s);
            """
            golden_map_tuples = [(q.query_id, q.golden_record_id, created_by) for q in case1_queries]
            await cur.executemany(golden_map_query, golden_map_tuples)
          else:
            logger.info("No Ground Truth links detected. Proceeding as Case 2 (Blind Test / RAG Triad).")
            
        # 5. Commit Transaction
        await conn.commit()
        logger.info(f"Successfully persisted inference run '{run.run_id}'.")
        
      except Exception as e:
        await conn.rollback()
        logger.error(f"Failed to persist inference run. Transaction rolled back. Error: {e}")
        raise
