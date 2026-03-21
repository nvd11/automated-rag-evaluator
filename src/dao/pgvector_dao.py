from src.interfaces.ingestion_interfaces import BaseDAO
from src.domain.models import Document
from src.configs.db import get_db_connection
from loguru import logger

class PgVectorDAO(BaseDAO):
    """
    Implementation of BaseDAO using Psycopg v3.
    Provides strict ACID transactional upserts, idempotency cleanup, and pgvector bulk inserts.
    """
    
    async def clean_document_data(self, cursor, document_id: int) -> None:
        """
        Deletes all existing chunks and topic mappings for a specific document_id.
        This is an internal helper executed within an active transaction cursor to ensure idempotency.
        """
        logger.debug(f"Executing clean_document_data() for doc_id: {document_id}")
        await cursor.execute("DELETE FROM document_chunks WHERE document_id = %s;", (document_id,))
        await cursor.execute("DELETE FROM document_topics_map WHERE document_id = %s;", (document_id,))

    async def upsert_document_transactionally(self, document: Document, created_by: str) -> int:
        logger.info(f"Persisting document {document.document_name} to database transactionally...")
        
        async with get_db_connection() as conn:
            try:
                async with conn.cursor() as cur:
                    # 1. Upsert Document Core Record
                    await cur.execute("""
                        INSERT INTO documents (document_name, file_path, md5_hash, total_pages, created_by)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (document_name) 
                        DO UPDATE SET 
                            file_path = EXCLUDED.file_path,
                            md5_hash = EXCLUDED.md5_hash,
                            total_pages = EXCLUDED.total_pages,
                            updated_by = %s,
                            updated_at = CURRENT_TIMESTAMP,
                            is_deleted = false
                        RETURNING id;
                    """, (document.document_name, document.file_path, document.md5_hash, document.total_pages, created_by, created_by))
                    
                    doc_id = (await cur.fetchone())[0]
                    logger.debug(f"Document ID resolved: {doc_id}")

                    # 2. Cleanup old chunks (Idempotency Enforcement)
                    await self.clean_document_data(cursor=cur, document_id=doc_id)
                    
                    # 3. Handle Metadata Topics Upsert
                    topic_ids = []
                    for topic_name in document.topics:
                        await cur.execute("""
                            INSERT INTO topics (topic_name, created_by)
                            VALUES (%s, %s)
                            ON CONFLICT (topic_name) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                            RETURNING id;
                        """, (topic_name, created_by))
                        topic_ids.append((await cur.fetchone())[0])
                    
                    # 4. Map Topics to Document (N-to-N)
                    for topic_id in topic_ids:
                        await cur.execute("""
                            INSERT INTO document_topics_map (document_id, topic_id, created_by)
                            VALUES (%s, %s, %s)
                            ON CONFLICT DO NOTHING;
                        """, (doc_id, topic_id, created_by))
                        
                    # 5. Bulk Insert Text Chunks and Vectors
                    if document.chunks:
                        query = """
                            INSERT INTO document_chunks 
                            (document_id, chunk_index, page_number, chunk_text, token_count, embedding, created_by)
                            VALUES (%s, %s, %s, %s, %s, %s, %s);
                        """
                        # Psycopg3 executemany is highly optimized
                        chunk_data = [
                            (doc_id, c.chunk_index, c.page_number, c.text, c.token_count, c.embedding, created_by)
                            for c in document.chunks
                        ]
                        await cur.executemany(query, chunk_data)
                        logger.debug(f"Bulk inserted {len(chunk_data)} chunks for document {doc_id}")

                # Commit transaction explicitly
                await conn.commit()
                logger.info("Transaction committed successfully. Data safely persisted.")
                return doc_id
                
            except Exception as e:
                # Catch-all Rollback
                await conn.rollback()
                logger.error(f"Transaction failed, rolled back immediately. Error: {e}")
                raise
