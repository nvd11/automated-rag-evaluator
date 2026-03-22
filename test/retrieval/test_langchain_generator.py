import pytest
from unittest.mock import patch, AsyncMock
from langchain_core.messages import AIMessage
from src.retrieval.langchain_generator import LangchainGeminiGenerator
from src.domain.models import RetrievedContext

@pytest.mark.asyncio
@patch("langchain_google_genai.ChatGoogleGenerativeAI.ainvoke", new_callable=AsyncMock)
async def test_generator_ainvoke(mock_ainvoke):
    generator = LangchainGeminiGenerator()
    mock_ainvoke.return_value = AIMessage(content="Generated mock answer")
    
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
    mock_ainvoke.assert_called_once()
    
    # Check if the prompt was formatted correctly
    args, kwargs = mock_ainvoke.call_args
    prompt_value = args[0]
    prompt_string = prompt_value.to_string()
    
    assert "HSBC profit is $12B." in prompt_string
    assert "What is the profit?" in prompt_string
    assert "Score: 0.9900" in prompt_string
