from src.interfaces.ingestion_interfaces import BaseDAO
from src.domain.models import Document
from src.configs.db import get_db_connection
from loguru import logger
import uuid
import json

class PgVectorDAO(BaseDAO):
  
  async def clean_document_data(self, cursor, doc_name: str, created_by: str) -> dict:
    """
    Soft-deletes all existing document records, chunks, and topic mappings 
    for a specific document_name. This ensures immutable historical snapshots.
    """
    logger.debug(f"Soft-deleting historical data for doc_name: {doc_name}")
    
    deleted_stats = {"documents": 0, "document_chunks": 0, "document_topics": 0}
    
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
      deleted_stats["document_chunks"] += cursor.rowcount
      
      # Soft delete topic mappings
      await cursor.execute("""
        UPDATE document_topics 
        SET is_deleted = TRUE, updated_by = %s, updated_at = CURRENT_TIMESTAMP
        WHERE doc_id = %s AND is_deleted = FALSE;
      """, (created_by, old_id))
      deleted_stats["document_topics"] += cursor.rowcount
      
      # Soft delete the document core record
      await cursor.execute("""
        UPDATE documents
        SET is_deleted = TRUE, updated_by = %s, updated_at = CURRENT_TIMESTAMP
        WHERE doc_id = %s AND is_deleted = FALSE;
      """, (created_by, old_id))
      deleted_stats["documents"] += cursor.rowcount
      
    if old_doc_ids:
      logger.info(f"Soft-deleted {len(old_doc_ids)} previous versions of document: {doc_name}")
      
    return deleted_stats

  async def _insert_document_record(self, cursor, document: Document, new_doc_id: str, created_by: str) -> str:
    doc_metadata = json.dumps({
      "file_path": document.file_path,
      "total_pages": document.total_pages,
      "md5_hash": document.md5_hash
    })
    await cursor.execute("""
      INSERT INTO documents (doc_id, doc_name, metadata, created_by)
      VALUES (%s, %s, %s, %s)
      RETURNING doc_id;
    """, (new_doc_id, document.document_name, doc_metadata, created_by))
    
    returned_doc_id = (await cursor.fetchone())[0]
    logger.debug(f"New Document ID created: {returned_doc_id}")
    return returned_doc_id

  async def _upsert_topics(self, cursor, topics: list, created_by: str) -> tuple[list, int]:
    topic_ids = []
    inserted_count = 0
    for topic_name in topics:
      topic_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, topic_name))
      
      # Use PostgreSQL xmax system column to detect if a row was actually inserted vs updated.
      # If xmax = 0, it was inserted. If xmax != 0, it was updated.
      await cursor.execute("""
        INSERT INTO topics (topic_id, topic_name, created_by)
        VALUES (%s, %s, %s)
        ON CONFLICT (topic_id) DO UPDATE SET 
          updated_at = CURRENT_TIMESTAMP,
          is_deleted = FALSE
        RETURNING topic_id, (xmax = 0) AS is_inserted;
      """, (topic_uuid, topic_name, created_by))
      
      row = await cursor.fetchone()
      topic_ids.append(row[0])
      if row[1]: # is_inserted
        inserted_count += 1
        
    return topic_ids, inserted_count

  async def _map_document_topics(self, cursor, doc_id: str, topic_ids: list, created_by: str) -> None:
    for topic_id in topic_ids:
      await cursor.execute("""
        INSERT INTO document_topics (doc_id, topic_id, created_by)
        VALUES (%s, %s, %s);
      """, (doc_id, topic_id, created_by))

  async def _bulk_insert_chunks(self, cursor, document: Document, doc_id: str, created_by: str) -> None:
    if not document.chunks:
      return
      
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
        chunk_uuid, doc_id, "RecursiveCharacterTextSplitter",
        c.chunk_index, c.text, chunk_meta, c.embedding, created_by
      ))
      
    await cursor.executemany(query, chunk_data)
    logger.debug(f"Bulk inserted {len(chunk_data)} chunks for document {doc_id}")

  async def upsert_document_transactionally(self, document: Document, created_by: str) -> dict:
    """
    Main entry point for transactionally inserting a document and its components.
    
    Note on Transaction Context:
    Uses a raw `cursor` instead of an ORM `Session`. The cursor is scoped to a single
    `connection`, ensuring all DML operations within this function belong to the same 
    ACID transaction block. Any failure triggers a full connection rollback.
    """
    logger.info(f"Persisting NEW version of document {document.document_name} to database transactionally...")
    
    new_doc_id = str(uuid.uuid4())
    
    async with get_db_connection() as conn:
      try:
        async with conn.cursor() as cur:
          # 1. Clean up (Soft Delete) any existing versions
          deleted_stats = await self.clean_document_data(cursor=cur, doc_name=document.document_name, created_by=created_by)

          # 2. Insert Brand New Document Record
          returned_doc_id = await self._insert_document_record(
            cursor=cur, document=document, new_doc_id=new_doc_id, created_by=created_by
          )
          
          # 3. Handle Metadata Topics Upsert
          topic_ids, inserted_topic_count = await self._upsert_topics(
            cursor=cur, topics=document.topics, created_by=created_by
          )
          
          # 4. Map Topics to the NEW Document (N-to-N)
          await self._map_document_topics(
            cursor=cur, doc_id=returned_doc_id, topic_ids=topic_ids, created_by=created_by
          )
            
          # 5. Bulk Insert Text Chunks and Vectors
          await self._bulk_insert_chunks(
            cursor=cur, document=document, doc_id=returned_doc_id, created_by=created_by
          )

        # Commit transaction explicitly
        await conn.commit()
        logger.info("Transaction committed successfully. Data safely persisted.")
        
        # Format final return dict containing all operation statistics
        return {
          "doc_id": returned_doc_id,
          "stats": {
            "soft_deleted": deleted_stats,
            "inserted": {
              "documents": 1,
              "topics": inserted_topic_count,
              "document_topics": len(topic_ids),
              "document_chunks": len(document.chunks) if document.chunks else 0
            }
          }
        }
        
      except Exception as e:
        # Catch-all Rollback
        await conn.rollback()
        logger.error(f"Transaction failed, rolled back immediately. Error: {e}")
        raise
