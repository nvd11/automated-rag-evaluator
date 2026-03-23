import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langchain_core.messages import AIMessage
from src.retrieval.langchain_generator import LangchainGeminiGenerator
from src.domain.models import RetrievedContext
from src.configs.settings import settings

@pytest.mark.asyncio
@patch("src.retrieval.langchain_generator.settings")
async def test_generator_ainvoke(mock_settings):
    # Mock settings so we can test both ENABLE_PROXY logic paths without hitting real API
    mock_settings.ENABLE_PROXY = False
    mock_settings.LLM_INFERENCE_MODEL = "gemini-2.5-pro"
    mock_settings.GEMINI_API_KEY = "test_key"
    
    # We patch the actual chain's ainvoke method since LCEL internals wrap the chat model's ainvoke
    with patch("langchain_core.runnables.RunnableSequence.ainvoke", new_callable=AsyncMock) as mock_chain_ainvoke:
        mock_chain_ainvoke.return_value = "Generated mock answer"
        
        generator = LangchainGeminiGenerator()
        
        # Setup mock context
        context = RetrievedContext(
            doc_id="doc1",
            chunk_id="chunk1",
            text="HSBC profit is $12B.",
            similarity_score=0.99
        )
        
        # Act
        input_data = {
            "context": [context],
            "question": "What is the profit?"
        }
        result = await generator.ainvoke(input_data)
        
        # Assert
        assert result == "Generated mock answer"
        mock_chain_ainvoke.assert_called_once()
        
        # Check if the prompt was formatted correctly into the chain input
        args, kwargs = mock_chain_ainvoke.call_args
        chain_input = args[0]
        
        assert "HSBC profit is $12B." in chain_input["context"]
        assert "What is the profit?" in chain_input["question"]
        assert "Score: 0.9900" in chain_input["context"]
