import asyncio
import os
import sys
from loguru import logger

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.configs.db import init_db_pool, close_db_pool, get_db_connection

async def update_view():
    await init_db_pool()
    try:
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DROP VIEW IF EXISTS v_evaluation_metrics_pivot;")
                await cur.execute("""
                    CREATE OR REPLACE VIEW v_evaluation_metrics_pivot AS
                    SELECT 
                        e.job_id,
                        e.query_id,
                        qh.question,
                        qh.generated_answer,
                        qh.retrieved_contexts,
                        gr.ground_truth,
                        MAX(CASE WHEN e.metric_name = 'context_relevance' THEN e.metric_value END) AS context_relevance_score,
                        MAX(CASE WHEN e.metric_name = 'faithfulness' THEN e.metric_value END) AS faithfulness_score,
                        MAX(CASE WHEN e.metric_name = 'answer_relevance' THEN e.metric_value END) AS answer_relevance_score,
                        MAX(CASE WHEN e.metric_name = 'correctness' THEN e.metric_value END) AS correctness_score,
                        MAX(CASE WHEN e.metric_name = 'semantic_similarity' THEN e.metric_value END) AS semantic_similarity_score
                    FROM evaluation_metrics e
                    JOIN query_history qh ON e.query_id = qh.query_id
                    LEFT JOIN golden_record_query_mapping grm ON qh.query_id = grm.query_id
                    LEFT JOIN golden_records gr ON grm.golden_record_id = gr.id AND gr.is_deleted = FALSE
                    WHERE e.is_deleted = FALSE AND qh.is_deleted = FALSE
                    GROUP BY e.job_id, e.query_id, qh.question, qh.generated_answer, qh.retrieved_contexts, gr.ground_truth;
                """)
                await conn.commit()
                logger.info("Successfully updated view v_evaluation_metrics_pivot")
    finally:
        await close_db_pool()

if __name__ == "__main__":
    asyncio.run(update_view())
