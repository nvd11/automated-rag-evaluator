import pytest
import pytest_asyncio
import uuid
from loguru import logger

from src.configs.db import init_db_pool, close_db_pool, get_db_connection
from src.dao.golden_record_dao import PgVectorGoldenRecordDAO
from src.domain.models import GoldenRecord

# Assuming the runner will be imported from here:
from src.runners.inference_runner import InferenceRunner

@pytest_asyncio.fixture(scope="function", autouse=True)
async def db_pool_setup_teardown():
    logger.info("Initializing DB Pool for Inference Runner Tests...")
    await init_db_pool()
    yield
    logger.info("Closing DB Pool after Inference Runner Tests...")
    await close_db_pool()

@pytest.mark.asyncio
@pytest.mark.integration
class TestInferenceRunnerIntegration:
    """
    End-to-End integration test for InferenceRunner.
    Requires real LLM connectivity (GEMINI_API_KEY) and real DB connection.
    Tests the architecture of Case 1 (Golden Records) vs Case 2 (Blind Proxy).
    """

    async def _cleanup_run_data(self, run_id: str):
        """
        Helper to cleanly delete all data associated with a specific inference run.
        
        DESIGN DECISION 1: Targeted Deletion to Protect Historical Data
        We ensure that this pytest ONLY deletes data from the current run by:
        1. Using the unique, dynamically generated `run_id` as the anchor.
        2. Querying `inference_run_query_mapping` to find only the `query_id`s created during this test.
        3. Using `ANY(query_ids)` to strictly delete only those specific mappings and history records.
        This prevents any accidental deletion of real production or baseline data (no full table scans or TRUNCATEs).
        """
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                # 1. Get all query_ids for this run
                await cur.execute("SELECT query_id FROM inference_run_query_mapping WHERE run_id = %s", (run_id,))
                rows = await cur.fetchall()
                query_ids = [row[0] for row in rows]

                if query_ids:
                    # 2. Clean up Case 1 mappings (Golden Records)
                    await cur.execute("DELETE FROM golden_record_query_mapping WHERE query_id = ANY(%s)", (query_ids,))
                    # 3. Clean up run mappings
                    await cur.execute("DELETE FROM inference_run_query_mapping WHERE run_id = %s", (run_id,))
                    # 4. Clean up the actual query history
                    await cur.execute("DELETE FROM query_history WHERE query_id = ANY(%s)", (query_ids,))
                
                # 5. Clean up the run metadata
                await cur.execute("DELETE FROM inference_run_history WHERE run_id = %s", (run_id,))
            await conn.commit()
            logger.info(f"Cleaned up data for run_id: {run_id}")

    async def test_inference_runner_case1_golden_records(self):
        """
        Tests Case 1: Evaluation driven by Golden Records.
        Validates real LLM generation, writing to query_history, 
        and the explicit linkage via golden_record_query_mapping.
        
        DESIGN DECISION 2: Limit the Scope to 1 Record in Testing
        Though the InferenceRunner naturally processes entire datasets (e.g., all 48 Golden Records) in production, 
        here we enforce `limit=1` or supply only 1 dummy record to:
        1. Avoid unnecessary API usage costs and rate limits from real LLMs (Gemini).
        2. Keep the integration test fast.
        The underlying system architecture correctly handles bulk concurrency (Semaphore etc.), 
        so testing 1 query is sufficient to prove the mapping logic works securely.
        """
        golden_dao = PgVectorGoldenRecordDAO()
        test_batch = "pytest_inference_runner_batch_case1"
        rec_id = str(uuid.uuid4())
        
        # 1. Setup Dummy Golden Record
        dummy_record = GoldenRecord(
            id=rec_id,
            batch_name=test_batch,
            question="What is the test question about HSBC?",
            ground_truth="This is a test answer for HSBC.",
            expected_topics=["Testing"],
            complexity="Factoid"
        )
        await golden_dao.bulk_insert_golden_records(test_batch, [dummy_record], "pytest_user")

        run_id = None
        try:
            # 2. Execute Runner (Real LLM Call & DB Write)
            runner = InferenceRunner()
            # Contract: run() should execute the pipeline and return the generated UUID string (run_id)
            run_id = await runner.run(dataset_mode="case1", batch_name=test_batch, limit=1)
            
            assert run_id is not None, "Runner must return the generated run_id"

            # 3. Validate DB Records
            async with get_db_connection() as conn:
                async with conn.cursor() as cur:
                    # Assert Inference Run was created
                    await cur.execute("SELECT chunking_config FROM inference_run_history WHERE run_id = %s", (run_id,))
                    assert await cur.fetchone() is not None, "Inference run history record missing"

                    # Assert queries mapped to run
                    await cur.execute("SELECT query_id FROM inference_run_query_mapping WHERE run_id = %s", (run_id,))
                    query_ids = [row[0] for row in await cur.fetchall()]
                    assert len(query_ids) == 1, "Should have exactly 1 mapped query"
                    q_id = query_ids[0]

                    # Assert query history content
                    await cur.execute("SELECT question, generated_answer FROM query_history WHERE query_id = %s", (q_id,))
                    qh_row = await cur.fetchone()
                    assert qh_row is not None
                    assert qh_row[0] == dummy_record.question, "Question mismatch"
                    assert len(qh_row[1]) > 5, "LLM should have generated a real, substantive answer"

                    # Assert Golden Record linkage (Case 1 Specific)
                    await cur.execute("SELECT golden_record_id FROM golden_record_query_mapping WHERE query_id = %s", (q_id,))
                    gr_mapping = await cur.fetchone()
                    assert gr_mapping is not None, "Case 1 query MUST be linked to a Golden Record"
                    assert str(gr_mapping[0]) == rec_id

        finally:
            # 4. Cleanup
            if run_id:
                await self._cleanup_run_data(run_id)
            
            async with get_db_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("DELETE FROM golden_records WHERE batch_name = %s", (test_batch,))
                await conn.commit()

    async def test_inference_runner_case2_blind_test(self):
        """
        Tests Case 2: Proxy/Blind evaluation with no Golden Records.
        Validates real LLM generation, writing to query_history, 
        and confirms NO linkage in golden_record_query_mapping (Orphaned Query Pattern).
        
        Like Case 1, we pass only 1 raw question to simulate production traffic, 
        saving test time and LLM costs while still proving the Orphaned Pattern writes perfectly.
        """
        run_id = None
        # Passing raw string questions to simulate production traffic
        test_source_queries = ["How does climate change impact global banking operations?"]

        try:
            # 1. Execute Runner (Real LLM Call & DB Write)
            runner = InferenceRunner()
            run_id = await runner.run(dataset_mode="case2", source_queries=test_source_queries)
            
            assert run_id is not None, "Runner must return the generated run_id"

            # 2. Validate DB Records
            async with get_db_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT query_id FROM inference_run_query_mapping WHERE run_id = %s", (run_id,))
                    query_ids = [row[0] for row in await cur.fetchall()]
                    assert len(query_ids) == 1
                    q_id = query_ids[0]

                    await cur.execute("SELECT generated_answer FROM query_history WHERE query_id = %s", (q_id,))
                    qh_row = await cur.fetchone()
                    assert qh_row is not None
                    assert len(qh_row[0]) > 5, "LLM should have generated a real answer"

                    # Assert NO Golden Record linkage (Case 2 Specific - Orphaned Query)
                    await cur.execute("SELECT * FROM golden_record_query_mapping WHERE query_id = %s", (q_id,))
                    gr_mapping = await cur.fetchone()
                    assert gr_mapping is None, "Case 2 (Blind Test) queries MUST NOT be linked to Golden Records"

        finally:
            # 3. Cleanup
            if run_id:
                await self._cleanup_run_data(run_id)
