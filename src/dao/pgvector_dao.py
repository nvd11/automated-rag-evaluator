from src.interfaces.ingestion_interfaces import BaseDAO
from src.domain.models import Document
from src.configs.db import get_db_connection
from loguru import logger
import uuid
import json

class PgVectorDAO(BaseDAO):
    """
    Implementation of BaseDAO using Psycopg v3.
    Provides strict ACID transactional upserts, idempotency cleanup, and pgvector bulk inserts.
    """
    
    async def clean_document_data(self, cursor, doc_id: str, created_by: str) -> None:
        """
        Soft-deletes all existing chunks and topic mappings for a specific doc_id.
        This respects the enterprise audit trail requirement (is_deleted = TRUE).
        """
        logger.debug(f"Soft-deleting historical data for doc_id: {doc_id}")
        # Soft delete chunks
        await cursor.execute("""
            UPDATE document_chunks 
            SET is_deleted = TRUE, updated_by = %s, updated_at = CURRENT_TIMESTAMP
            WHERE doc_id = %s AND is_deleted = FALSE;
        """, (created_by, doc_id))
        
        # Soft delete topic mappings
        await cursor.execute("""
            UPDATE document_topics 
            SET is_deleted = TRUE, updated_by = %s, updated_at = CURRENT_TIMESTAMP
            WHERE doc_id = %s AND is_deleted = FALSE;
        """, (created_by, doc_id))

    async def upsert_document_transactionally(self, document: Document, created_by: str) -> str:
        logger.info(f"Persisting document {document.document_name} to database transactionally...")
        
        # In our schema, doc_id is a VARCHAR. We'll use the MD5 hash as the deterministic doc_id.
        doc_id = document.md5_hash
        
        # Prepare metadata JSON
        doc_metadata = json.dumps({
            "file_path": document.file_path,
            "total_pages": document.total_pages
        })
        
        async with get_db_connection() as conn:
            try:
                async with conn.cursor() as cur:
                    # 1. Upsert Document Core Record (Using doc_id as primary key)
                    await cur.execute("""
                        INSERT INTO documents (doc_id, doc_name, metadata, created_by)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (doc_id) 
                        DO UPDATE SET 
                            doc_name = EXCLUDED.doc_name,
                            metadata = EXCLUDED.metadata,
                            updated_by = %s,
                            updated_at = CURRENT_TIMESTAMP,
                            is_deleted = FALSE
                        RETURNING doc_id;
                    """, (doc_id, document.document_name, doc_metadata, created_by, created_by))
                    
                    returned_doc_id = (await cur.fetchone())[0]
                    logger.debug(f"Document ID resolved: {returned_doc_id}")

                    # 2. Cleanup old chunks and mappings (Soft Delete for Idempotency)
                    await self.clean_document_data(cursor=cur, doc_id=returned_doc_id, created_by=created_by)
                    
                    # 3. Handle Metadata Topics Upsert
                    topic_ids = []
                    for topic_name in document.topics:
                        # Generate deterministic UUID for topic based on name
                        topic_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, topic_name))
                        await cur.execute("""
                            INSERT INTO topics (topic_id, topic_name, created_by)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (topic_id) DO UPDATE SET 
                                updated_at = CURRENT_TIMESTAMP,
                                is_deleted = FALSE
                            RETURNING topic_id;
                        """, (topic_uuid, topic_name, created_by))
                        topic_ids.append((await cur.fetchone())[0])
                    
                    # 4. Map Topics to Document (N-to-N)
                    for topic_id in topic_ids:
                        await cur.execute("""
                            INSERT INTO document_topics (doc_id, topic_id, created_by)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (doc_id, topic_id) DO UPDATE SET 
                                is_deleted = FALSE,
                                updated_by = %s,
                                updated_at = CURRENT_TIMESTAMP;
                        """, (returned_doc_id, topic_id, created_by, created_by))
                        
                    # 5. Bulk Insert Text Chunks and Vectors
                    if document.chunks:
                        query = """
                            INSERT INTO document_chunks 
                            (id, doc_id, chunking_strategy, chunk_index, content, metadata, embedding, created_by)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                        """
                        # Psycopg3 executemany is highly optimized
                        chunk_data = []
                        for c in document.chunks:
                            chunk_uuid = str(uuid.uuid4())
                            chunk_meta = json.dumps({"page_number": c.page_number, "token_count": c.token_count})
                            chunk_data.append((
                                chunk_uuid, returned_doc_id, "RecursiveCharacterTextSplitter",
                                c.chunk_index, c.text, chunk_meta, c.embedding, created_by
                            ))
                            
                        await cur.executemany(query, chunk_data)
                        logger.debug(f"Bulk inserted {len(chunk_data)} chunks for document {returned_doc_id}")

                # Commit transaction explicitly
                await conn.commit()
                logger.info("Transaction committed successfully. Data safely persisted.")
                return returned_doc_id
                
            except Exception as e:
                # Catch-all Rollback
                await conn.rollback()
                logger.error(f"Transaction failed, rolled back immediately. Error: {e}")
                raise
