import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from src.dao.pgvector_retriever_dao import PgVectorRetrieverDAO
from src.domain.models import SearchQuery, RetrievedContext

@pytest.fixture
def mock_db_connection():
    """
    Creates a deeply mocked asynchronous psycopg3 connection.
    Simulates the context manager behavior of `get_db_connection()`.
    """
    mock_conn = MagicMock()
    mock_cursor = AsyncMock()
    
    # conn.cursor() returns an async context manager
    mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor
    
    mock_get_db_conn = AsyncMock()
    mock_get_db_conn.__aenter__.return_value = mock_conn
    
    return mock_get_db_conn, mock_cursor

@pytest.fixture
def sample_query():
    return SearchQuery(
        query_text="What is the profit?",
        embedding=[0.1, 0.2, 0.3], # Shortened for mock testing
        top_k=3,
        similarity_threshold=0.75
    )

@pytest.mark.asyncio
@patch("src.dao.pgvector_retriever_dao.get_db_connection")
async def test_semantic_search_without_topic_filters(mock_get_db_connection_decorator, mock_db_connection, sample_query):
    """
    Test standard semantic search without metadata pre-filtering.
    Verifies that the SQL constructs correctly and maps DB rows to DTOs.
    """
    # Arrange
    mock_get_db_conn, mock_cursor = mock_db_connection
    mock_get_db_connection_decorator.return_value = mock_get_db_conn
    
    # Mocking database row returns: chunk_id, doc_id, text, metadata(json string), similarity_score
    mock_cursor.fetchall.return_value = [
        ("chunk-123", "doc-456", "HSBC profit was $12B", '{"page_number": 5}', 0.85),
        ("chunk-124", "doc-456", "Revenue increased by 10%", '{"page_number": 6}', 0.76)
    ]
    
    dao = PgVectorRetrieverDAO()
    
    # Act
    results = await dao.semantic_search(sample_query)
    
    # Assert
    assert len(results) == 2
    assert isinstance(results[0], RetrievedContext)
    assert results[0].chunk_id == "chunk-123"
    assert results[0].doc_id == "doc-456"
    assert results[0].text == "HSBC profit was $12B"
    assert results[0].metadata == {"page_number": 5}
    assert results[0].similarity_score == 0.85
    
    # Verify the SQL was executed
    mock_cursor.execute.assert_called_once()
    args, kwargs = mock_cursor.execute.call_args
    sql_executed = args[0]
    query_params = args[1]
    
    # Verify exact SQL structures (no JOIN on topics)
    assert "JOIN document_topics" not in sql_executed
    assert "AND (1 - (dc.embedding <=> %s::vector)) >= %s" in sql_executed
    
    # Verify params passed to the query
    # Should be: [embedding, embedding, threshold, embedding, top_k]
    assert query_params[0] == sample_query.embedding
    assert query_params[-1] == sample_query.top_k

@pytest.mark.asyncio
@patch("src.dao.pgvector_retriever_dao.get_db_connection")
async def test_semantic_search_with_topic_filters(mock_get_db_connection_decorator, mock_db_connection, sample_query):
    """
    Test semantic search with metadata pre-filtering.
    Verifies that the correct JOINs and ANY array filters are added to the SQL.
    """
    # Arrange
    sample_query.topic_filters = ["Financial Performance", "Risk Management"]
    
    mock_get_db_conn, mock_cursor = mock_db_connection
    mock_get_db_connection_decorator.return_value = mock_get_db_conn
    mock_cursor.fetchall.return_value = [] # Empty result is fine, we just care about the SQL construction
    
    dao = PgVectorRetrieverDAO()
    
    # Act
    await dao.semantic_search(sample_query)
    
    # Assert
    mock_cursor.execute.assert_called_once()
    args, kwargs = mock_cursor.execute.call_args
    sql_executed = args[0]
    query_params = args[1]
    
    # Verify exact SQL structures (JOIN on topics must exist)
    assert "JOIN document_topics dt ON d.doc_id = dt.doc_id" in sql_executed
    assert "JOIN topics t ON dt.topic_id = t.topic_id" in sql_executed
    assert "t.topic_name = ANY(%s)" in sql_executed
    
    # Verify the topic array was correctly injected into the parameters
    assert ["Financial Performance", "Risk Management"] in query_params

@pytest.mark.asyncio
@patch("src.dao.pgvector_retriever_dao.get_db_connection")
async def test_semantic_search_empty_results(mock_get_db_connection_decorator, mock_db_connection, sample_query):
    """
    Test behavior when the vector DB returns no matches above the threshold.
    """
    # Arrange
    mock_get_db_conn, mock_cursor = mock_db_connection
    mock_get_db_connection_decorator.return_value = mock_get_db_conn
    mock_cursor.fetchall.return_value = [] # DB returns empty
    
    dao = PgVectorRetrieverDAO()
    
    # Act
    results = await dao.semantic_search(sample_query)
    
    # Assert
    assert isinstance(results, list)
    assert len(results) == 0

