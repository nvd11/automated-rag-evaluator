import json
from typing import List
from loguru import logger

from src.interfaces.retriever_interfaces import IRetrieverDAO
from src.domain.models import SearchQuery, RetrievedContext
from src.configs.db import get_db_connection

class PgVectorRetrieverDAO(IRetrieverDAO):
    """
    Concrete implementation of the Retriever DAO using native psycopg3 and pgvector.
    Executes high-performance semantic search with optional metadata pre-filtering.
    """
    
    async def semantic_search(self, query: SearchQuery) -> List[RetrievedContext]:
        logger.debug(f"Executing semantic search. Top_K: {query.top_k}, Threshold: {query.similarity_threshold}")
        
        retrieved_contexts: List[RetrievedContext] = []
        
        # pgvector uses `<=>` for Cosine Distance. 
        # Cosine Similarity = 1 - Cosine Distance
        base_query = """
            SELECT 
                dc.id as chunk_id,
                dc.doc_id,
                dc.content as text,
                dc.metadata as chunk_metadata,
                (1 - (dc.embedding <=> %s::vector)) as similarity_score
            FROM document_chunks dc
            JOIN documents d ON dc.doc_id = d.doc_id
        """
        
        query_params = [query.embedding]
        where_clauses = ["dc.is_deleted = FALSE", "d.is_deleted = FALSE"]
        
        # 1. Dynamic Metadata Pre-filtering (Topic JOIN)
        if query.topic_filters:
            logger.info(f"Applying metadata pre-filtering for topics: {query.topic_filters}")
            
            # To avoid N+1 queries or complex subqueries, we JOIN with the mapping tables
            base_query += """
                JOIN document_topics dt ON d.doc_id = dt.doc_id
                JOIN topics t ON dt.topic_id = t.topic_id
            """
            
            # Using ANY for efficient array matching in PostgreSQL
            where_clauses.append("t.topic_name = ANY(%s)")
            where_clauses.append("t.is_deleted = FALSE")
            where_clauses.append("dt.is_deleted = FALSE")
            
            query_params.append(query.topic_filters)
            
        # 2. Assemble WHERE clauses
        if where_clauses:
            base_query += " WHERE " + " AND ".join(where_clauses)
            
        # 3. Filter by similarity threshold & Sort by exact Cosine Distance limit K
        # Note: We enforce the threshold directly in SQL to minimize data transfer across the network!
        base_query += """
            AND (1 - (dc.embedding <=> %s::vector)) >= %s
            ORDER BY dc.embedding <=> %s::vector
            LIMIT %s;
        """
        
        # Add the remaining parameters for the WHERE and ORDER BY clauses
        query_params.extend([query.embedding, query.similarity_threshold, query.embedding, query.top_k])
        
        logger.debug("Executing compiled Retriever SQL query...")
        
        # Execute the transaction
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(base_query, query_params)
                rows = await cur.fetchall()
                
                for row in rows:
                    chunk_id = row[0]
                    doc_id = row[1]
                    text = row[2]
                    metadata_dict = row[3] if isinstance(row[3], dict) else json.loads(row[3] or "{}")
                    similarity_score = float(row[4])
                    
                    retrieved_contexts.append(
                        RetrievedContext(
                            doc_id=doc_id,
                            chunk_id=chunk_id,
                            text=text,
                            metadata=metadata_dict,
                            similarity_score=similarity_score
                        )
                    )
                    
        logger.info(f"Semantic search completed. Found {len(retrieved_contexts)} relevant chunks above threshold.")
        return retrieved_contexts
