import pytest
from unittest.mock import AsyncMock, MagicMock
from src.retrieval.semantic_retriever import SemanticRetriever
from src.domain.models import SearchQuery, RetrievedContext

@pytest.mark.asyncio
async def test_semantic_retriever_ainvoke():
    mock_embedder = AsyncMock()
    mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    
    mock_dao = AsyncMock()
    mock_context = RetrievedContext(
        doc_id="test_doc",
        chunk_id="test_chunk",
        text="Sample text",
        similarity_score=0.9
    )
    mock_dao.semantic_search.return_value = [mock_context]
    
    retriever = SemanticRetriever(embedder=mock_embedder, dao=mock_dao)
    
    # Test ainvoke
    results = await retriever.ainvoke("HSBC profit")
    
    assert len(results) == 1
    assert results[0].text == "Sample text"
    
    mock_embedder.embed_query.assert_called_once_with("HSBC profit")
    
    # Check that search_query was correctly built
    args, kwargs = mock_dao.semantic_search.call_args
    search_query = args[0]
    
    assert isinstance(search_query, SearchQuery)
    assert search_query.query_text == "HSBC profit"
    assert search_query.embedding == [0.1, 0.2, 0.3]
