import pytest
import pytest_asyncio
import uuid
import json
from loguru import logger

from src.configs.db import init_db_pool, close_db_pool, get_db_connection
from src.dao.diagnoser_dao import PgVectorDiagnoserDAO

@pytest_asyncio.fixture(scope="function", autouse=True)
async def db_pool_setup_teardown():
  logger.info("Initializing DB Pool for Diagnoser DAO Tests...")
  await init_db_pool()
  yield
  logger.info("Closing DB Pool after Diagnoser DAO Tests...")
  await close_db_pool()

@pytest.mark.asyncio
@pytest.mark.integration
class TestPgVectorDiagnoserDAO:
  """
  Validates the Read-Only analytical capabilities of the Diagnoser DAO.
  Tests aggregation (AVG) across the EAV wide view and complex metadata JOINs.
  """

  async def _setup_dummy_eval_data(self) -> str:
    """Helper to inject synthetic data mimicking a completed Evaluation Job."""
    test_run_id = str(uuid.uuid4())
    test_job_id = str(uuid.uuid4())
    q_id_case1 = str(uuid.uuid4())
    q_id_case2 = str(uuid.uuid4())
    gr_id = str(uuid.uuid4())
    
    async with get_db_connection() as conn:
      async with conn.cursor() as cur:
        # 1. Inference Run
        await cur.execute(
          "INSERT INTO inference_run_history (run_id, start_time, end_time, created_by) VALUES (%s, NOW(), NOW(), 'pytest')", 
          (test_run_id,)
        )
        
        # 2. Queries
        await cur.execute("""
          INSERT INTO query_history (query_id, queried_by, question, generated_answer, query_time, retrieval_time, response_time, created_by)
          VALUES 
          (%s, 'pytest', 'Q1', 'A1', NOW(), NOW(), NOW(), 'pytest'),
          (%s, 'pytest', 'Q2', 'A2', NOW(), NOW(), NOW(), 'pytest')
        """, (q_id_case1, q_id_case2))
        
        # 3. Map Queries to Run
        await cur.execute("""
          INSERT INTO inference_run_query_mapping (run_id, query_id, created_by)
          VALUES (%s, %s, 'pytest'), (%s, %s, 'pytest')
        """, (test_run_id, q_id_case1, test_run_id, q_id_case2))
        
        # 4. Golden Record mapping (Case 1 Only)
        await cur.execute(
          "INSERT INTO golden_records (id, batch_name, question, ground_truth, created_by) VALUES (%s, 'pytest_eval_batch_name', 'Q1', 'GT1', 'pytest')",
          (gr_id,)
        )
        await cur.execute(
          "INSERT INTO golden_record_query_mapping (query_id, golden_record_id, created_by) VALUES (%s, %s, 'pytest')",
          (q_id_case1, gr_id)
        )
        
        # 5. Evaluation Job
        await cur.execute(
          "INSERT INTO evaluation_job_history (job_id, inference_run_id, start_time, end_time, evaluator_model, created_by) VALUES (%s, %s, NOW(), NOW(), 'pytest-judge', 'pytest')",
          (test_job_id, test_run_id)
        )
        
        # 6. Evaluation Metrics
        # Let's mock a scenario:
        # q1 (Case 1) correctness: 4.0
        # q2 (Case 2) context_relevance: 2.0, faithfulness: 3.0, answer_relevance: 1.0
        await cur.execute("""
          INSERT INTO evaluation_metrics (id, job_id, query_id, evaluation_strategy, metric_category, metric_name, metric_value, reasoning, created_by)
          VALUES 
          (gen_random_uuid(), %s, %s, 'CASE1', 'gen', 'correctness', 4.0, 'ok', 'pytest'),
          (gen_random_uuid(), %s, %s, 'CASE2', 'ret', 'context_relevance', 2.0, 'bad', 'pytest'),
          (gen_random_uuid(), %s, %s, 'CASE2', 'gen', 'faithfulness', 3.0, 'hallucinated a bit', 'pytest'),
          (gen_random_uuid(), %s, %s, 'CASE2', 'gen', 'answer_relevance', 1.0, 'off topic', 'pytest')
        """, (test_job_id, q_id_case1, test_job_id, q_id_case2, test_job_id, q_id_case2, test_job_id, q_id_case2))
        
      await conn.commit()
      
    return test_job_id, test_run_id, q_id_case1, q_id_case2, gr_id

  async def _cleanup_dummy_data(self, job_id, run_id, q1, q2, gr):
    async with get_db_connection() as conn:
      async with conn.cursor() as cur:
        await cur.execute("DELETE FROM evaluation_metrics WHERE job_id = %s", (job_id,))
        await cur.execute("DELETE FROM evaluation_job_history WHERE job_id = %s", (job_id,))
        await cur.execute("DELETE FROM golden_record_query_mapping WHERE query_id = %s", (q1,))
        await cur.execute("DELETE FROM golden_records WHERE id = %s", (gr,))
        await cur.execute("DELETE FROM inference_run_query_mapping WHERE run_id = %s", (run_id,))
        await cur.execute("DELETE FROM query_history WHERE query_id IN (%s, %s)", (q1, q2))
        await cur.execute("DELETE FROM inference_run_history WHERE run_id = %s", (run_id,))
      await conn.commit()

  async def test_fetch_metric_averages(self):
    """Verify the DAO correctly calculates the AVG() over the Pivot View."""
    dao = PgVectorDiagnoserDAO()
    job_id, run_id, q1, q2, gr = await self._setup_dummy_eval_data()
    
    try:
      averages = await dao.fetch_metric_averages(job_id)
      
      # Assert correct mathematical aggregation
      assert averages["correctness"] == 4.0
      assert averages["context_relevance"] == 2.0
      assert averages["faithfulness"] == 3.0
      assert averages["answer_relevance"] == 1.0
      
    finally:
      await self._cleanup_dummy_data(job_id, run_id, q1, q2, gr)

  async def test_fetch_job_metadata(self):
    """Verify the DAO resolves the complex JOIN back to the Golden Record batch name."""
    dao = PgVectorDiagnoserDAO()
    job_id, run_id, q1, q2, gr = await self._setup_dummy_eval_data()
    
    try:
      metadata = await dao.fetch_job_metadata(job_id)
      
      assert metadata["setting_id"] == run_id
      assert metadata["dataset_name"] == "pytest_eval_batch_name"
      
    finally:
      await self._cleanup_dummy_data(job_id, run_id, q1, q2, gr)