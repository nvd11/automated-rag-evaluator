import asyncio
import csv
import os
import sys
from loguru import logger

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.configs.db import init_db_pool, close_db_pool, get_db_connection

async def run():
    await init_db_pool()
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                # 1. Update the View to include reasoning
                logger.info("Updating v_evaluation_metrics_pivot to include reasoning columns...")
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
                        MAX(CASE WHEN e.metric_name = 'context_relevance' THEN e.reasoning END) AS context_relevance_reasoning,
                        MAX(CASE WHEN e.metric_name = 'faithfulness' THEN e.metric_value END) AS faithfulness_score,
                        MAX(CASE WHEN e.metric_name = 'faithfulness' THEN e.reasoning END) AS faithfulness_reasoning,
                        MAX(CASE WHEN e.metric_name = 'answer_relevance' THEN e.metric_value END) AS answer_relevance_score,
                        MAX(CASE WHEN e.metric_name = 'answer_relevance' THEN e.reasoning END) AS answer_relevance_reasoning,
                        MAX(CASE WHEN e.metric_name = 'correctness' THEN e.metric_value END) AS correctness_score,
                        MAX(CASE WHEN e.metric_name = 'correctness' THEN e.reasoning END) AS correctness_reasoning,
                        MAX(CASE WHEN e.metric_name = 'semantic_similarity' THEN e.metric_value END) AS semantic_similarity_score,
                        MAX(CASE WHEN e.metric_name = 'semantic_similarity' THEN e.reasoning END) AS semantic_similarity_reasoning
                    FROM evaluation_metrics e
                    JOIN query_history qh ON e.query_id = qh.query_id
                    LEFT JOIN golden_record_query_mapping grm ON qh.query_id = grm.query_id
                    LEFT JOIN golden_records gr ON grm.golden_record_id = gr.id AND gr.is_deleted = FALSE
                    WHERE e.is_deleted = FALSE AND qh.is_deleted = FALSE
                    GROUP BY e.job_id, e.query_id, qh.question, qh.generated_answer, qh.retrieved_contexts, gr.ground_truth;
                """)
                await conn.commit()
                
                # 2. Get job IDs for Case 1 and Case 2
                await cur.execute("""
                    SELECT DISTINCT job_id FROM v_evaluation_metrics_pivot 
                    WHERE correctness_score IS NOT NULL 
                    ORDER BY job_id DESC LIMIT 1
                """)
                row1 = await cur.fetchone()
                job_id_case1 = row1[0] if row1 else None

                await cur.execute("""
                    SELECT job_id FROM evaluation_job_history 
                    WHERE inference_run_id = '57c7466a-46f2-49b4-af9f-14d4352a25ef'
                    ORDER BY start_time DESC LIMIT 1
                """)
                row2 = await cur.fetchone()
                job_id_case2 = row2[0] if row2 else None
                
                if not job_id_case2:
                    await cur.execute("""
                        SELECT job_id FROM v_evaluation_metrics_pivot 
                        WHERE faithfulness_score IS NOT NULL 
                        GROUP BY job_id 
                        ORDER BY COUNT(query_id) DESC LIMIT 1
                    """)
                    row2_fallback = await cur.fetchone()
                    job_id_case2 = row2_fallback[0] if row2_fallback else None

                jobs_to_export = [
                    ("case1_benchmark_evaluation.csv", job_id_case1),
                    ("case2_blind_test_evaluation.csv", job_id_case2)
                ]

                # 3. Export CSVs
                for filename, j_id in jobs_to_export:
                    if not j_id:
                        logger.warning(f"No job found for {filename}, skipping.")
                        continue
                        
                    logger.info(f"Exporting data for Job ID: {j_id} to {filename}...")
                    query = """
                        SELECT 
                            query_id,
                            question,
                            retrieved_contexts,
                            generated_answer,
                            ground_truth,
                            context_relevance_score,
                            context_relevance_reasoning,
                            faithfulness_score,
                            faithfulness_reasoning,
                            answer_relevance_score,
                            answer_relevance_reasoning,
                            correctness_score,
                            correctness_reasoning
                        FROM v_evaluation_metrics_pivot
                        WHERE job_id = %s
                    """
                    await cur.execute(query, (j_id,))
                    rows = await cur.fetchall()
                    
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            "Query ID", 
                            "Question", 
                            "Retrieved Contexts",
                            "Generated Answer", 
                            "Golden Answer (Ground Truth)",
                            "Context Relevance Score", 
                            "Context Relevance Reasoning", 
                            "Faithfulness Score", 
                            "Faithfulness Reasoning", 
                            "Answer Relevance Score", 
                            "Answer Relevance Reasoning", 
                            "Correctness Score",
                            "Correctness Reasoning"
                        ])
                        
                        for r in rows:
                            contexts_str = str(r[2]) if r[2] is not None else ""
                            writer.writerow([
                                str(r[0]),      # query_id
                                str(r[1]),      # question
                                contexts_str,   # retrieved_contexts
                                str(r[3]),      # generated_answer
                                str(r[4]) if r[4] is not None else "", # ground_truth
                                f"{r[5]:.2f}" if r[5] is not None else "", # context_relevance
                                str(r[6]) if r[6] is not None else "",     # context_relevance_reasoning
                                f"{r[7]:.2f}" if r[7] is not None else "", # faithfulness
                                str(r[8]) if r[8] is not None else "",     # faithfulness_reasoning
                                f"{r[9]:.2f}" if r[9] is not None else "", # answer_relevance
                                str(r[10]) if r[10] is not None else "",   # answer_relevance_reasoning
                                f"{r[11]:.2f}" if r[11] is not None else "", # correctness
                                str(r[12]) if r[12] is not None else ""      # correctness_reasoning
                            ])
                            
                    logger.info(f"Successfully exported {len(rows)} records to {filepath}")

    finally:
        await close_db_pool()

if __name__ == "__main__":
    asyncio.run(run())
