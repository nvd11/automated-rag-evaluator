import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agents.rag_agent import RAGAgent
from src.domain.models import RAGResponse, RetrievedContext

@pytest.mark.asyncio
async def test_rag_agent_ask_with_results():
    """Test RAGAgent asking a question that returns context chunks."""
    # Arrange
    mock_retriever = AsyncMock()
    mock_context = RetrievedContext(
        doc_id="doc1", chunk_id="chunk1", text="HSBC profit is high.", similarity_score=0.88
    )
    mock_retriever.ainvoke.return_value = [mock_context]
    
    mock_generator = AsyncMock()
    mock_generator.ainvoke.return_value = "The generated mock answer."
    
    agent = RAGAgent(retriever=mock_retriever, generator=mock_generator)
    
    # Act
    response = await agent.ask("What is the profit?", top_k=3)
    
    # Assert
    assert isinstance(response, RAGResponse)
    assert response.query == "What is the profit?"
    assert response.generated_answer == "The generated mock answer."
    assert len(response.retrieved_contexts) == 1
    assert response.retrieved_contexts[0].similarity_score == 0.88
    
    # Verify method calls
    mock_retriever.ainvoke.assert_called_once()
    args, kwargs = mock_retriever.ainvoke.call_args
    assert kwargs["input"] == "What is the profit?"
    assert kwargs["config"]["configurable"]["top_k"] == 3
    
    mock_generator.ainvoke.assert_called_once()

@pytest.mark.asyncio
async def test_rag_agent_ask_empty_context():
    """Test RAGAgent behavior when no context is retrieved."""
    # Arrange
    mock_retriever = AsyncMock()
    mock_retriever.ainvoke.return_value = [] # Database returns nothing above threshold
    
    mock_generator = AsyncMock()
    
    agent = RAGAgent(retriever=mock_retriever, generator=mock_generator)
    
    # Act
    response = await agent.ask("Unknown question")
    
    # Assert
    assert response.retrieved_contexts == []
    assert "I'm sorry" in response.generated_answer
    
    # Generator should NOT be called if context is empty
    mock_generator.ainvoke.assert_not_called()
