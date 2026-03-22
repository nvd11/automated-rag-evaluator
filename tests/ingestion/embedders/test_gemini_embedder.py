import pytest
import asyncio
from unittest.mock import patch, MagicMock

from src.ingestion.embedders.gemini_embedder import GeminiEmbedder
from src.domain.models import Chunk

@pytest.fixture
def dummy_chunks():
    return [
        Chunk(text="This is chunk 1", page_number=1, chunk_index=0),
        Chunk(text="This is chunk 2", page_number=1, chunk_index=1),
        Chunk(text="This is chunk 3", page_number=2, chunk_index=2),
    ]

@pytest.mark.asyncio
@patch("src.ingestion.embedders.gemini_embedder.genai.Client")
async def test_embed_batch_success(mock_client_class, dummy_chunks):
    """
    Test embedding a batch of chunks successfully using Gemini.
    It verifies that the embeddings are correctly attached to the Chunks.
    """
    mock_client = MagicMock()
    
    # We set side_effect to return different responses per batch call
    response_batch_1 = MagicMock()
    response_batch_1.embeddings = [
        MagicMock(values=[0.1, 0.2, 0.3]),
        MagicMock(values=[0.4, 0.5, 0.6])
    ]
    
    response_batch_2 = MagicMock()
    response_batch_2.embeddings = [
        MagicMock(values=[0.7, 0.8, 0.9])
    ]
    
    mock_client.models.embed_content.side_effect = [response_batch_1, response_batch_2]
    mock_client_class.return_value = mock_client

    embedder = GeminiEmbedder(batch_size=2)
    embedded_chunks = await embedder.embed_batch(dummy_chunks)

    assert len(embedded_chunks) == 3
    assert embedded_chunks[0].embedding == [0.1, 0.2, 0.3]
    assert embedded_chunks[1].embedding == [0.4, 0.5, 0.6]
    assert embedded_chunks[2].embedding == [0.7, 0.8, 0.9]
    assert mock_client.models.embed_content.call_count == 2

@pytest.mark.asyncio
@patch("src.ingestion.embedders.gemini_embedder.genai.Client")
async def test_embed_batch_retries(mock_client_class, dummy_chunks):
    """
    Test embedding exponential backoff retry mechanism by simulating an API failure.
    """
    mock_client = MagicMock()
    # Force the first call to fail, the second call to succeed
    mock_client.models.embed_content.side_effect = [
        Exception("API Rate Limit Error"),
        MagicMock(embeddings=[
            MagicMock(values=[1.0] * 768),
            MagicMock(values=[1.0] * 768),
            MagicMock(values=[1.0] * 768)
        ])
    ]
    mock_client_class.return_value = mock_client

    embedder = GeminiEmbedder(batch_size=3)
    embedded_chunks = await embedder.embed_batch(dummy_chunks)

    assert len(embedded_chunks) == 3
    assert mock_client.models.embed_content.call_count == 2
    assert embedded_chunks[0].embedding == [1.0] * 768

@pytest.mark.asyncio
@patch("src.ingestion.embedders.gemini_embedder.genai.Client")
async def test_embed_batch_fatal_error(mock_client_class, dummy_chunks):
    """
    Test what happens when the API persistently fails and exhausts retries.
    It should raise an exception upwards.
    """
    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = Exception("Fatal API Error")
    mock_client_class.return_value = mock_client

    embedder = GeminiEmbedder(batch_size=3)
    
    # Fast-forward retries by replacing wait
    embedder._embed_text_batch.retry.wait = MagicMock(return_value=0)

    # We catch generic Exception because Tenacity throws a RetryError 
    with pytest.raises(Exception):
        await embedder.embed_batch(dummy_chunks)
