import pytest
import pytest_asyncio
import uuid
import datetime
from loguru import logger
import json

from src.configs.db import init_db_pool, close_db_pool, get_db_connection
from src.dao.evaluation_dao import PgVectorEvaluationDAO
from src.domain.models import EvaluationJobHistory, EvaluationMetricRecord, QueryEvaluationDTO

@pytest_asyncio.fixture(scope="function", autouse=True)
async def db_pool_setup_teardown():
  logger.info("Initializing DB Pool for Evaluation DAO Tests...")
  await init_db_pool()
  yield
  logger.info("Closing DB Pool after Evaluation DAO Tests...")
  await close_db_pool()

@pytest.mark.asyncio
@pytest.mark.integration
class TestPgVectorEvaluationDAO:
  """
  Validates the data access layer for the LLM-as-a-Judge evaluation pipeline.
  Crucially, ensures that the LEFT JOIN logic correctly categorizes Case 1 (Golden)
  and Case 2 (Orphaned) queries, and that the Upgraded EAV model metrics
  are persisted idempotently.
  """

  async def _setup_dummy_inference_data(self, test_run_id: str):
    """Helper to inject synthetic data mimicking a completed Inference Run into the DB."""
    q_id_case1 = str(uuid.uuid4())
    q_id_case2 = str(uuid.uuid4())
    gr_id = str(uuid.uuid4())
    
    async with get_db_connection() as conn:
      async with conn.cursor() as cur:
        # 1. Insert dummy inference run history
        await cur.execute(
          "INSERT INTO inference_run_history (run_id, start_time, end_time, created_by) VALUES (%s, NOW(), NOW(), 'pytest')", 
          (test_run_id,)
        )
        
        # 2. Insert dummy query history (Case 1 & Case 2)
        await cur.execute("""
          INSERT INTO query_history (query_id, queried_by, question, generated_answer, retrieved_contexts, query_time, retrieval_time, response_time, created_by)
          VALUES 
          (%s, 'pytest', 'Q Case 1', 'A Case 1', %s, NOW(), NOW(), NOW(), 'pytest'),
          (%s, 'pytest', 'Q Case 2', 'A Case 2', %s, NOW(), NOW(), NOW(), 'pytest')
        """, (
          q_id_case1, json.dumps([{"text": "context1"}]), 
          q_id_case2, json.dumps([{"text": "context2"}])
        ))
        
        # 3. Map queries to the inference run
        await cur.execute("""
          INSERT INTO inference_run_query_mapping (run_id, query_id, created_by)
          VALUES (%s, %s, 'pytest'), (%s, %s, 'pytest')
        """, (test_run_id, q_id_case1, test_run_id, q_id_case2))
        
        # 4. Create a dummy golden record for Case 1
        await cur.execute(
          "INSERT INTO golden_records (id, batch_name, question, ground_truth, complexity, created_by) VALUES (%s, 'pytest_batch', 'Q Case 1', 'GT Case 1', 'Factoid', 'pytest')",
          (gr_id,)
        )
        
        # 5. Map ONLY Case 1 to the golden record (This defines it as Case 1)
        await cur.execute(
          "INSERT INTO golden_record_query_mapping (query_id, golden_record_id, created_by) VALUES (%s, %s, 'pytest')",
          (q_id_case1, gr_id)
        )
        
      await conn.commit()
      
    return q_id_case1, q_id_case2, gr_id

  async def _cleanup_dummy_data(self, test_run_id: str, q_id_case1: str, q_id_case2: str, gr_id: str, job_id: str = None):
    """Helper to cleanly delete all data associated with the test run."""
    async with get_db_connection() as conn:
      async with conn.cursor() as cur:
        await cur.execute("DELETE FROM golden_record_query_mapping WHERE query_id IN (%s, %s)", (q_id_case1, q_id_case2))
        await cur.execute("DELETE FROM golden_records WHERE id = %s", (gr_id,))
        await cur.execute("DELETE FROM inference_run_query_mapping WHERE run_id = %s", (test_run_id,))
        await cur.execute("DELETE FROM query_history WHERE query_id IN (%s, %s)", (q_id_case1, q_id_case2))
        await cur.execute("DELETE FROM inference_run_history WHERE run_id = %s", (test_run_id,))
        if job_id:
          await cur.execute("DELETE FROM evaluation_metrics WHERE job_id = %s", (job_id,))
          await cur.execute("DELETE FROM evaluation_job_history WHERE job_id = %s", (job_id,))
      await conn.commit()
      logger.info(f"Cleaned up synthetic data for test_run_id: {test_run_id}")

  async def test_fetch_queries_and_insert_metrics(self):
    """
    Tests the full DAO lifecycle:
    1. Fetch queries via complex LEFT JOIN, ensuring ground_truth is mapped for Case 1 and None for Case 2.
    2. Create an Evaluation Job history record.
    3. Insert metrics into the EAV table.
    4. Ensure idempotency (Unique Constraints) prevents duplicate metric insertion.
    """
    dao = PgVectorEvaluationDAO()
    test_run_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    
    q_id_case1, q_id_case2, gr_id = await self._setup_dummy_inference_data(test_run_id)
    
    try:
      # --- 1. Test Fetching (The LEFT JOIN) ---
      dtos = await dao.fetch_queries_for_evaluation(test_run_id)
      
      assert len(dtos) == 2, "Should fetch exactly 2 queries mapped to this run"
      
      case1_dto = next((d for d in dtos if d.query_id == q_id_case1), None)
      case2_dto = next((d for d in dtos if d.query_id == q_id_case2), None)
      
      assert case1_dto is not None
      assert case2_dto is not None
      
      # Assert properties map correctly
      assert case1_dto.has_ground_truth is True
      assert case1_dto.ground_truth == 'GT Case 1'
      
      assert case2_dto.has_ground_truth is False
      assert case2_dto.ground_truth is None
      
      # --- 2. Test Job Creation ---
      job = EvaluationJobHistory(
        job_id=job_id,
        inference_run_id=test_run_id,
        evaluator_model="pytest-judge-v1",
        evaluator_prompt_version="1.0"
      )
      await dao.create_evaluation_job(job, "pytest")
      
      # --- 3. Test Bulk Metric Insertion (EAV 2.0) ---
      metrics = [
        EvaluationMetricRecord(
          job_id=job_id,
          query_id=q_id_case1,
          evaluation_strategy="CASE1_GROUND_TRUTH",
          metric_category="generation",
          metric_name="correctness",
          metric_value=4.5,
          reasoning="The answer matched the GT closely.",
          judge_model="pytest-judge-v1"
        ),
        EvaluationMetricRecord(
          job_id=job_id,
          query_id=q_id_case2,
          evaluation_strategy="CASE2_RAG_TRIAD",
          metric_category="generation",
          metric_name="faithfulness",
          metric_value=5.0,
          reasoning="No hallucinations detected against context.",
          judge_model="pytest-judge-v1"
        )
      ]
      
      # Initial insert should succeed
      await dao.bulk_insert_evaluation_metrics(metrics, "pytest")
      
      # Assert they exist
      async with get_db_connection() as conn:
        async with conn.cursor() as cur:
          await cur.execute("SELECT COUNT(*) FROM evaluation_metrics WHERE job_id = %s", (job_id,))
          count = (await cur.fetchone())[0]
          assert count == 2
      
      # --- 4. Test Idempotency (UNIQUE Constraint) ---
      # Trying to insert the exact same metrics (same query, strategy, metric, job) again
      # should trigger DO NOTHING, NOT raise an error, and NOT duplicate the rows.
      await dao.bulk_insert_evaluation_metrics(metrics, "pytest")
      
      async with get_db_connection() as conn:
        async with conn.cursor() as cur:
          await cur.execute("SELECT COUNT(*) FROM evaluation_metrics WHERE job_id = %s", (job_id,))
          count = (await cur.fetchone())[0]
          assert count == 2, "Idempotency failed: Row count should remain 2"
      
    finally:
      await self._cleanup_dummy_data(test_run_id, q_id_case1, q_id_case2, gr_id, job_id)