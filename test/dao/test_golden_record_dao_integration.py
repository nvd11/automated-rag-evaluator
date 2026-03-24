import pytest
import pytest_asyncio
import uuid
import uuid_utils # just in case standard uuid isn't used
from loguru import logger

from src.configs.db import init_db_pool, close_db_pool, get_db_connection
from src.dao.golden_record_dao import PgVectorGoldenRecordDAO
from src.domain.models import GoldenRecord

@pytest_asyncio.fixture(scope="function", autouse=True)
async def db_pool_setup_teardown():
  """
  Ensure the DB pool is initialized before any tests in this module run,
  and cleanly closed afterwards.
  """
  logger.info("Initializing DB Pool for DAO Integration Tests...")
  await init_db_pool()
  yield
  logger.info("Closing DB Pool after DAO Integration Tests...")
  await close_db_pool()

@pytest.mark.asyncio
@pytest.mark.integration
class TestPgVectorGoldenRecordDAO:
  """
  Integration tests for PgVectorGoldenRecordDAO against the real Cloud SQL database.
  Verifies SQL syntax correctness, transaction boundaries, and idempotency logic.
  """

  async def test_get_random_seed_chunks_empty_or_valid(self):
    """
    Tests the SELECT extraction query.
    Since the DB might be empty after a fresh schema rebuild, we just verify
    that the complex SQL query (with JOINs and RANDOM) executes without syntax errors.
    """
    dao = PgVectorGoldenRecordDAO()
    
    # 1. Test without topics filter
    chunks = await dao.get_random_seed_chunks(limit=2)
    assert isinstance(chunks, list), "Should return a list"
    
    # 2. Test with topics filter (validates dynamic JOIN syntax)
    chunks_with_topics = await dao.get_random_seed_chunks(limit=2, topics=["Financial Performance", "Risk Management"])
    assert isinstance(chunks_with_topics, list), "Should return a list even with topic filters"

  async def test_bulk_insert_idempotency_and_soft_delete(self):
    """
    Tests the core idempotency logic:
    Inserting a batch with a name should soft-delete any previously 
    existing records with that exact batch name.
    """
    dao = PgVectorGoldenRecordDAO()
    test_batch_name = "pytest_eval_batch_v1"
    test_user = "pytest_runner"
    
    # Create a dummy record for Version 1
    record_v1 = GoldenRecord(
      id=str(uuid.uuid4()),
      batch_name=test_batch_name,
      question="What is the net profit?",
      ground_truth="10 Billion USD",
      expected_topics=["Financial"],
      complexity="Factoid"
    )
    
    # 1. Insert Version 1
    await dao.bulk_insert_golden_records(
      batch_name=test_batch_name, 
      records=[record_v1], 
      created_by=test_user
    )
    
    # Verify it was inserted and is active
    async with get_db_connection() as conn:
      async with conn.cursor() as cur:
        await cur.execute(
          "SELECT id, is_deleted FROM golden_records WHERE id = %s", 
          (record_v1.id,)
        )
        row_v1 = await cur.fetchone()
        assert row_v1 is not None, "Record V1 should exist in DB"
        assert row_v1[1] is False, "Record V1 should be active (is_deleted=FALSE)"
        
    # Create a dummy record for Version 2 (same batch name!)
    record_v2 = GoldenRecord(
      id=str(uuid.uuid4()),
      batch_name=test_batch_name,
      question="What are the main risks?",
      ground_truth="Credit and Market Risk",
      expected_topics=["Risk"],
      complexity="Reasoning"
    )
    
    # 2. Insert Version 2
    await dao.bulk_insert_golden_records(
      batch_name=test_batch_name, 
      records=[record_v2], 
      created_by=test_user
    )
    
    # 3. Verify Version 1 is now soft-deleted and Version 2 is active
    async with get_db_connection() as conn:
      async with conn.cursor() as cur:
        # Check V1
        await cur.execute(
          "SELECT is_deleted FROM golden_records WHERE id = %s", 
          (record_v1.id,)
        )
        row_v1_after = await cur.fetchone()
        assert row_v1_after[0] is True, "Idempotency failed: Prior record V1 was NOT soft-deleted!"
        
        # Check V2
        await cur.execute(
          "SELECT is_deleted FROM golden_records WHERE id = %s", 
          (record_v2.id,)
        )
        row_v2_after = await cur.fetchone()
        assert row_v2_after[0] is False, "New record V2 should be active (is_deleted=FALSE)"
        
    # 4. Cleanup test data
    async with get_db_connection() as conn:
      async with conn.cursor() as cur:
        await cur.execute("DELETE FROM golden_records WHERE batch_name = %s", (test_batch_name,))
      await conn.commit()
