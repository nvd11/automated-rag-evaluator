import json
from typing import List, Optional
from loguru import logger

from src.interfaces.evaluator_interfaces import IGoldenRecordDAO
from src.domain.models import Chunk, GoldenRecord
from src.configs.db import get_db_connection

class PgVectorGoldenRecordDAO(IGoldenRecordDAO):
    """
    Concrete implementation of IGoldenRecordDAO utilizing async psycopg3.
    Interacts with the `document_chunks` table for reading seeds and 
    the `golden_records` table for writing generated datasets.
    """
    
    async def get_random_seed_chunks(self, limit: int, topics: Optional[List[str]] = None) -> List[Chunk]:
        """
        Samples random text chunks directly from the parsed corpus.
        Optionally filters by explicit topics to ensure the generated questions 
        cover specific domains (e.g., 'Risk Management').
        """
        logger.info(f"Extracting {limit} random seed chunks for dataset synthesis.")
        
        # We enforce a basic length filter to prevent the LLM from trying 
        # to generate questions from uselessly short fragments (like headers or disclaimers).
        base_query = """
            SELECT 
                dc.content, 
                dc.metadata 
            FROM document_chunks dc
        """
        
        where_clauses = [
            "dc.is_deleted = FALSE",
            "LENGTH(dc.content) > 300"
        ]
        query_params = []
        
        if topics:
            logger.info(f"Applying domain filters to sampling: {topics}")
            base_query += """
                JOIN document_topics dt ON dc.doc_id = dt.doc_id
                JOIN topics t ON dt.topic_id = t.topic_id
            """
            where_clauses.append("t.topic_name = ANY(%s)")
            where_clauses.append("t.is_deleted = FALSE")
            where_clauses.append("dt.is_deleted = FALSE")
            query_params.append(topics)
            
        base_query += " WHERE " + " AND ".join(where_clauses)
        
        # Core mechanism: Random ordering for diverse sampling
        base_query += """
            ORDER BY RANDOM()
            LIMIT %s;
        """
        query_params.append(limit)
        
        chunks: List[Chunk] = []
        
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(base_query, query_params)
                rows = await cur.fetchall()
                
                for row in rows:
                    content = row[0]
                    # Parse metadata jsonb, handling potential string format variations
                    metadata_dict = row[1] if isinstance(row[1], dict) else json.loads(row[1] or "{}")
                    
                    # Reconstruct into the Domain Model 
                    # Note: We omit embedding, chunk_index, etc., since they aren't needed for the Generator prompt.
                    chunks.append(
                        Chunk(
                            text=content,
                            page_number=metadata_dict.get("page_number", 0),
                            chunk_index=0
                        )
                    )
                    
        logger.debug(f"Successfully sampled {len(chunks)} diverse chunks from the database.")
        return chunks

    async def bulk_insert_golden_records(self, batch_name: str, records: List[GoldenRecord], created_by: str) -> None:
        """
        Executes an ACID transaction to write a complete benchmark batch.
        Enforces idempotency (and the 'Control Variable' evaluation pattern) by 
        soft-deleting any prior benchmark execution bearing the same `batch_name`.
        """
        if not records:
            logger.warning(f"No records provided for batch_name '{batch_name}'. Aborting insert.")
            return
            
        logger.info(f"Initiating transactional bulk insert for {len(records)} golden records in batch '{batch_name}'.")
        
        async with get_db_connection() as conn:
            try:
                async with conn.cursor() as cur:
                    # 1. Idempotency Check / Clean Slate: Soft-delete old batch
                    await cur.execute("""
                        UPDATE golden_records 
                        SET is_deleted = TRUE, updated_at = CURRENT_TIMESTAMP, updated_by = %s
                        WHERE batch_name = %s AND is_deleted = FALSE;
                    """, (created_by, batch_name))
                    
                    if cur.rowcount > 0:
                        logger.warning(f"Overwrote (soft-deleted) {cur.rowcount} prior records for batch '{batch_name}'.")

                    # 2. Bulk Insert New Records
                    insert_query = """
                        INSERT INTO golden_records 
                        (id, batch_name, question, ground_truth, expected_topics, complexity, created_by)
                        VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """
                    
                    # Transform Pydantic models into native tuples for executemany
                    data_tuples = [
                        (
                            r.id, 
                            r.batch_name, 
                            r.question, 
                            r.ground_truth, 
                            json.dumps(r.expected_topics), 
                            r.complexity, 
                            created_by
                        ) 
                        for r in records
                    ]
                    
                    await cur.executemany(insert_query, data_tuples)
                    
                # 3. Commit Transaction
                await conn.commit()
                logger.info(f"Successfully persisted benchmark dataset '{batch_name}'.")
                
            except Exception as e:
                await conn.rollback()
                logger.error(f"Failed to persist golden records. Transaction rolled back. Error: {e}")
                raise
