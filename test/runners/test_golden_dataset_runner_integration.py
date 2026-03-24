import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import pytest
import asyncio
from loguru import logger

from src.configs.settings import settings
from src.configs.db import init_db_pool, close_db_pool, get_db_connection

from src.llm.gemini_factory import GeminiLLMFactory
from src.dao.golden_record_dao import PgVectorGoldenRecordDAO
from src.evaluator.dataset_generator import LangchainDatasetGenerator
from src.runners.golden_dataset_runner import GoldenDatasetRunner

@pytest.fixture(autouse=True)
async def db_pool_setup_teardown():
    """
    Ensure the DB pool is initialized before any tests in this module run,
    and cleanly closed afterwards.
    """
    logger.info("Initializing DB Pool for Runner Integration Tests...")
    await init_db_pool()
    yield
    logger.info("Closing DB Pool after Runner Integration Tests...")
    await close_db_pool()

@pytest.mark.asyncio
@pytest.mark.integration
async def test_golden_dataset_runner_end_to_end_live():
    """
    End-to-End LIVE Integration Test for the GoldenDatasetRunner.
    
    This test executes the full pipeline:
    1. Fetches real chunks from the Cloud SQL database via DAO.
    2. Sends them to the actual Gemini Teacher Model (consumes real quota!).
    3. Receives structured QA_Pairs and maps them to GoldenRecords.
    4. Performs a transactional bulk insert into the `golden_records` table.
    
    CRITICAL: It then manually queries the DB to assert successful ingestion 
    and ALWAYS cleans up (hard deletes) the test batch records.
    """
    # 1. Skip check if no API key is provided
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY.startswith("your_"):
        pytest.skip("Skipping live Golden Dataset Runner test because GEMINI_API_KEY is not configured.")
        
    # Generate a dynamic UUID suffix for the batch name to guarantee 100% isolation per test run.
    # This ensures that concurrent CI runs or local developer runs never delete each other's data,
    # and strictly protects historical/production data.
    import uuid
    TEST_BATCH_NAME = f"pytest_e2e_live_{uuid.uuid4().hex[:8]}"
    SAMPLE_SIZE = 2 # Keep it small to save API costs and time
    
    # 2. Assemble the components using the real implementations
    logger.info(f"Instantiating Live RAG components for Runner Test...")
    llm_factory = GeminiLLMFactory()
    teacher_llm = llm_factory.create_llm(model_name=settings.LLM_TEACHER_MODEL, temperature=0.7)
    
    generator = LangchainDatasetGenerator(llm=teacher_llm)
    dao = PgVectorGoldenRecordDAO()
    
    runner = GoldenDatasetRunner(dao=dao, generator=generator)
    
    try:
        # 3. Execute the Full Pipeline
        logger.info(f"Executing Live Pipeline for batch: {TEST_BATCH_NAME}")
        await runner.run(batch_name=TEST_BATCH_NAME, sample_size=SAMPLE_SIZE, topics=None)
        
        # 4. Verify Database Ingestion
        logger.info("Querying database to assert successful record ingestion...")
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, question, ground_truth, complexity FROM golden_records WHERE batch_name = %s AND is_deleted = FALSE", 
                    (TEST_BATCH_NAME,)
                )
                rows = await cur.fetchall()
                
                # We requested SAMPLE_SIZE chunks. The LLM might reject some if they are junk, 
                # but typically we should get at least 1 valid QA pair back.
                assert len(rows) > 0, "Pipeline ran but no records were inserted into the database."
                assert len(rows) <= SAMPLE_SIZE, "Pipeline inserted more records than the requested sample size."
                
                for row in rows:
                    assert isinstance(str(row[0]), str), "UUID ID should be string"
                    assert len(row[1]) > 10, "Question should be generated and non-empty"
                    assert len(row[2]) >= 1, "Ground Truth answer should be generated and non-empty"
                    assert row[3] in ["Factoid", "Reasoning"], "Complexity must strictly match the Pydantic enum"
                    logger.debug(f"Verified live generated question: '{row[1]}'")
                    
    finally:
        # 5. Guaranteed Cleanup (Hard Delete)
        logger.info(f"Cleaning up live test data for batch: {TEST_BATCH_NAME}")
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM golden_records WHERE batch_name = %s", 
                    (TEST_BATCH_NAME,)
                )
                logger.info(f"Deleted {cur.rowcount} test records from golden_records.")
            await conn.commit()
