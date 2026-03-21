from src.interfaces.ingestion_interfaces import BaseDAO
from src.domain.models import Document
from src.configs.db import get_db_connection
from loguru import logger
import uuid
import json

class PgVectorDAO(BaseDAO):
    
    async def clean_document_data(self, cursor, doc_name: str, created_by: str) -> None:
        """
        Soft-deletes all existing document records, chunks, and topic mappings 
        for a specific document_name. This ensures immutable historical snapshots.
        """
        logger.debug(f"Soft-deleting historical data for doc_name: {doc_name}")
        
        # 1. Find all active document IDs for this name
        await cursor.execute("SELECT doc_id FROM documents WHERE doc_name = %s AND is_deleted = FALSE;", (doc_name,))
        rows = await cursor.fetchall()
        old_doc_ids = [row[0] for row in rows]

        for old_id in old_doc_ids:
            # Soft delete chunks
            await cursor.execute("""
                UPDATE document_chunks 
                SET is_deleted = TRUE, updated_by = %s, updated_at = CURRENT_TIMESTAMP
                WHERE doc_id = %s AND is_deleted = FALSE;
            """, (created_by, old_id))
            
            # Soft delete topic mappings
            await cursor.execute("""
                UPDATE document_topics 
                SET is_deleted = TRUE, updated_by = %s, updated_at = CURRENT_TIMESTAMP
                WHERE doc_id = %s AND is_deleted = FALSE;
            """, (created_by, old_id))
            
            # Soft delete the document core record
            await cursor.execute("""
                UPDATE documents
                SET is_deleted = TRUE, updated_by = %s, updated_at = CURRENT_TIMESTAMP
                WHERE doc_id = %s AND is_deleted = FALSE;
            """, (created_by, old_id))
            
        if old_doc_ids:
            logger.info(f"Soft-deleted {len(old_doc_ids)} previous versions of document: {doc_name}")

    async def upsert_document_transactionally(self, document: Document, created_by: str) -> str:
        logger.info(f"Persisting NEW version of document {document.document_name} to database transactionally...")
        
        # Generate a completely new UUID for this ingestion version
        new_doc_id = str(uuid.uuid4())
        
        doc_metadata = json.dumps({
            "file_path": document.file_path,
            "total_pages": document.total_pages,
            "md5_hash": document.md5_hash
        })
        
        async with get_db_connection() as conn:
            try:
                async with conn.cursor() as cur:
                    # 1. Clean up (Soft Delete) any existing versions of this document
                    await self.clean_document_data(cursor=cur, doc_name=document.document_name, created_by=created_by)

                    # 2. Insert Brand New Document Record
                    await cur.execute("""
                        INSERT INTO documents (doc_id, doc_name, metadata, created_by)
                        VALUES (%s, %s, %s, %s)
                        RETURNING doc_id;
                    """, (new_doc_id, document.document_name, doc_metadata, created_by))
                    
                    returned_doc_id = (await cur.fetchone())[0]
                    logger.debug(f"New Document ID created: {returned_doc_id}")
                    
                    # 3. Handle Metadata Topics Upsert
                    topic_ids = []
                    for topic_name in document.topics:
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
                    
                    # 4. Map Topics to the NEW Document (N-to-N)
                    for topic_id in topic_ids:
                        await cur.execute("""
                            INSERT INTO document_topics (doc_id, topic_id, created_by)
                            VALUES (%s, %s, %s);
                        """, (returned_doc_id, topic_id, created_by))
                        
                    # 5. Bulk Insert Text Chunks and Vectors to the NEW Document
                    if document.chunks:
                        query = """
                            INSERT INTO document_chunks 
                            (id, doc_id, chunking_strategy, chunk_index, content, metadata, embedding, created_by)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                        """
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
