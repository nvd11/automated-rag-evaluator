import asyncio
import csv
import json
import os
import sys
from loguru import logger

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.configs.db import init_db_pool, close_db_pool, get_db_connection

async def export_pivot_view_to_csv():
    await init_db_pool()
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                # Get the most recent Job IDs (one for Case 1, one for Case 2)
                # Case 1 has correctness_score
                await cur.execute("""
                    SELECT DISTINCT job_id FROM v_evaluation_metrics_pivot 
                    WHERE correctness_score IS NOT NULL 
                    ORDER BY job_id DESC LIMIT 1
                """)
                row1 = await cur.fetchone()
                job_id_case1 = row1[0] if row1 else None

                # Case 2 has faithfulness_score. We want the one with 30 queries
                # The run_id for 30 queries was: 57c7466a-46f2-49b4-af9f-14d4352a25ef
                await cur.execute("""
                    SELECT job_id FROM evaluation_job_history 
                    WHERE inference_run_id = '57c7466a-46f2-49b4-af9f-14d4352a25ef'
                    ORDER BY start_time DESC LIMIT 1
                """)
                row2 = await cur.fetchone()
                job_id_case2 = row2[0] if row2 else None
                
                # If the specific 30-query run doesn't have an eval job, fallback to the latest case 2 job
                if not job_id_case2:
                    await cur.execute("""
                        SELECT job_id FROM v_evaluation_metrics_pivot 
                        WHERE faithfulness_score IS NOT NULL 
                        GROUP BY job_id 
                        ORDER BY COUNT(query_id) DESC LIMIT 1
                    """)
                    try:
                        row2_fallback = await cur.fetchone()
                        job_id_case2 = row2_fallback[0] if row2_fallback else None
                    except Exception:
                        pass

                jobs_to_export = [
                    ("case1_benchmark_evaluation.csv", job_id_case1),
                    ("case2_blind_test_evaluation.csv", job_id_case2)
                ]

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
                            context_relevance_score,
                            faithfulness_score,
                            answer_relevance_score,
                            correctness_score
                        FROM v_evaluation_metrics_pivot
                        WHERE job_id = %s
                    """
                    await cur.execute(query, (j_id,))
                    rows = await cur.fetchall()
                    
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        # Write Header
                        writer.writerow([
                            "Query ID", 
                            "Question", 
                            "Retrieved Contexts",
                            "Generated Answer", 
                            "Context Relevance (0-5)", 
                            "Faithfulness (0-5)", 
                            "Answer Relevance (0-5)", 
                            "Correctness (0-5)"
                        ])
                        
                        # Write Data
                        for r in rows:
                            # `retrieved_contexts` is likely a list or JSON string. Convert it nicely.
                            contexts_str = str(r[2]) if r[2] is not None else ""
                            writer.writerow([
                                str(r[0]),      # query_id
                                str(r[1]),      # question
                                contexts_str,   # retrieved_contexts
                                str(r[3]),      # generated_answer
                                f"{r[4]:.2f}" if r[4] is not None else "", # context_relevance
                                f"{r[5]:.2f}" if r[5] is not None else "", # faithfulness
                                f"{r[6]:.2f}" if r[6] is not None else "", # answer_relevance
                                f"{r[7]:.2f}" if r[7] is not None else ""  # correctness
                            ])
                            
                    logger.info(f"Successfully exported {len(rows)} records to {filepath}")

    finally:
        await close_db_pool()

if __name__ == "__main__":
    asyncio.run(export_pivot_view_to_csv())