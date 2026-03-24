import pytest
import pytest_asyncio
import uuid
import datetime
from loguru import logger

from src.configs.db import init_db_pool, close_db_pool, get_db_connection
from src.dao.inference_dao import PgVectorInferenceDAO
from src.domain.models import InferenceRun, QueryHistoryRecord, GoldenRecord
from src.dao.golden_record_dao import PgVectorGoldenRecordDAO

@pytest_asyncio.fixture(scope="function", autouse=True)
async def db_pool_setup_teardown():
  logger.info("Initializing DB Pool for Inference DAO Tests...")
  await init_db_pool()
  yield
  logger.info("Closing DB Pool after Inference DAO Tests...")
  await close_db_pool()

@pytest.mark.asyncio
@pytest.mark.integration
class TestPgVectorInferenceDAO:
  """
  Validates the Orphaned Query Pattern and transactional boundaries
  for the Inference Runs phase.
  """
  
  async def test_fetch_golden_records(self):
    """
    Creates a temporary golden record batch, fetches it via InferenceDAO,
    verifies correct mapping, and cleans up.
    """
    golden_dao = PgVectorGoldenRecordDAO()
    inference_dao = PgVectorInferenceDAO()
    
    test_batch = "pytest_inference_fetch_batch"
    rec_id = str(uuid.uuid4())
    
    dummy_record = GoldenRecord(
      id=rec_id,
      batch_name=test_batch,
      question="What is the test question?",
      ground_truth="The test answer.",
      expected_topics=["Testing"],
      complexity="Factoid"
    )
    
    # Insert
    await golden_dao.bulk_insert_golden_records(test_batch, [dummy_record], "pytest")
    
    # Fetch
    results = await inference_dao.fetch_golden_records(test_batch)
    
    assert len(results) == 1
    assert results[0].id == rec_id
    assert results[0].question == "What is the test question?"
    
    # Cleanup
    async with get_db_connection() as conn:
      async with conn.cursor() as cur:
        await cur.execute("DELETE FROM golden_records WHERE batch_name = %s", (test_batch,))
      await conn.commit()

  async def test_persist_inference_run_case1_and_case2_patterns(self):
    """
    Crucial architectural test: 
    Inserts a Case 1 query (with Golden Record ID) and a Case 2 query (without).
    Validates that ONLY the Case 1 query gets written to golden_record_query_mapping,
    proving the Orphaned Query Pattern works for blind/proxy evaluations.
    """
    dao = PgVectorInferenceDAO()
    
    run_id = str(uuid.uuid4())
    case1_query_id = str(uuid.uuid4())
    case2_query_id = str(uuid.uuid4())
    fake_golden_id = str(uuid.uuid4()) # We don't enforce hard FKs anymore, so this is fine
    
    test_run = InferenceRun(
      run_id=run_id,
      chunking_config="fixed_100",
      indexing_config="dense",
      reranking_config="none",
      prompting_config="raw",
      generation_config="temp_0"
    )
    
    now = datetime.datetime.now(datetime.UTC).isoformat()
    
    # Case 1 Query (Has golden_record_id)
    query_case1 = QueryHistoryRecord(
      query_id=case1_query_id,
      question="Case 1 Question",
      generated_answer="Case 1 Answer",
      retrieved_contexts=[{"chunk_id": "c1", "text": "foo"}],
      query_time=now,
      retrieval_time=now,
      response_time=now,
      golden_record_id=fake_golden_id # <--- CRITICAL LINK
    )
    
    # Case 2 Query (No golden_record_id - Orphaned)
    query_case2 = QueryHistoryRecord(
      query_id=case2_query_id,
      question="Case 2 Question",
      generated_answer="Case 2 Answer",
      retrieved_contexts=[{"chunk_id": "c2", "text": "bar"}],
      query_time=now,
      retrieval_time=now,
      response_time=now,
      golden_record_id=None # <--- CRITICAL OMISSION
    )
    
    # Execute persistence
    await dao.persist_inference_run(test_run, [query_case1, query_case2], "pytest_user")
    
    # --- Assertions ---
    async with get_db_connection() as conn:
      async with conn.cursor() as cur:
        # 1. Assert both queries are in query_history
        await cur.execute("SELECT query_id FROM query_history WHERE query_id IN (%s, %s)", (case1_query_id, case2_query_id))
        assert len(await cur.fetchall()) == 2
        
        # 2. Assert both are mapped to the inference run
        await cur.execute("SELECT query_id FROM inference_run_query_mapping WHERE run_id = %s", (run_id,))
        assert len(await cur.fetchall()) == 2
        
        # 3. Assert ONLY Case 1 is mapped to the golden record
        await cur.execute("SELECT query_id FROM golden_record_query_mapping WHERE query_id IN (%s, %s)", (case1_query_id, case2_query_id))
        golden_mapped = await cur.fetchall()
        assert len(golden_mapped) == 1, "Only one query should be mapped to the golden records"
        assert str(golden_mapped[0][0]) == case1_query_id, "The mapped query MUST be Case 1"
        
        # Cleanup everything from this test
        await cur.execute("DELETE FROM golden_record_query_mapping WHERE query_id IN (%s, %s)", (case1_query_id, case2_query_id))
        await cur.execute("DELETE FROM inference_run_query_mapping WHERE run_id = %s", (run_id,))
        await cur.execute("DELETE FROM query_history WHERE query_id IN (%s, %s)", (case1_query_id, case2_query_id))
        await cur.execute("DELETE FROM inference_run_history WHERE run_id = %s", (run_id,))
      await conn.commit()
