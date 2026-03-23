import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import pytest
from unittest.mock import patch, MagicMock
from src.domain.models import Chunk, QA_Pair, GoldenRecord
from src.evaluator.dataset_generator import LangchainDatasetGenerator
from loguru import logger

# --- Unit Tests ---
@pytest.mark.asyncio
async def test_generator_ainvoke_success():
    """
    Unit test to verify the LangchainDatasetGenerator constructs a valid GoldenRecord
    when the underlying LLM successfully outputs a valid QA_Pair structured object.
    We mock the LCEL chain to return a pre-fab QA_Pair object directly.
    """
    fake_llm = MagicMock()
    generator = LangchainDatasetGenerator(llm=fake_llm)

    # 1. Mock the LCEL Chain's final output (which is already a Pydantic QA_Pair object)
    mock_qa = QA_Pair(
        question="What was the net profit for 2025?",
        answer="10 Billion USD",
        complexity="Factoid"
    )
    
    # We patch the 'ainvoke' of the chain object itself, because the chain is a complex RunnableSequence 
    # created via self.prompt | self.structured_llm in the __init__
    with patch("langchain_core.runnables.RunnableSequence.ainvoke", return_value=mock_qa) as mock_chain_ainvoke:
        
        # 2. Setup a dummy input Chunk
        dummy_chunk = Chunk(
            text="HSBC reported a net profit of 10 Billion USD in 2025.",
            page_number=10,
            chunk_index=1,
            token_count=15
        )
        
        # 3. Execute the Generator
        result = await generator.agenerate_qa_from_chunk(chunk=dummy_chunk, batch_name="test_batch_123")
        
        # 4. Assertions
        mock_chain_ainvoke.assert_called_once()
        
        # The result must be a fully formed GoldenRecord Domain Model
        assert result is not None
        assert isinstance(result, GoldenRecord)
        
        # Assert the data mappings are correct
        assert result.batch_name == "test_batch_123"
        assert result.question == "What was the net profit for 2025?"
        assert result.ground_truth == "10 Billion USD"
        assert result.complexity == "Factoid"
        assert isinstance(result.id, str) # UUID was generated
        
@pytest.mark.asyncio
async def test_generator_ainvoke_rejects_junk_chunk():
    """
    Unit test to verify that if the LLM determines the text is useless and returns 
    the "INVALID_CHUNK" magic string (as instructed in the Prompt), the Generator gracefully 
    returns None instead of polluting the database with a bad question.
    """
    fake_llm = MagicMock()
    generator = LangchainDatasetGenerator(llm=fake_llm)

    # 1. Mock the LCEL Chain returning a rejection
    mock_qa = QA_Pair(
        question="INVALID_CHUNK",
        answer="REJECTED",
        complexity="Factoid"
    )
    
    with patch("langchain_core.runnables.RunnableSequence.ainvoke", return_value=mock_qa) as mock_chain_ainvoke:
        
        # 2. Setup a useless dummy input Chunk (e.g. just a page number or disclaimer)
        junk_chunk = Chunk(
            text="Page 45 | Confidential Document",
            page_number=45,
            chunk_index=1,
            token_count=5
        )
        
        # 3. Execute the Generator
        result = await generator.agenerate_qa_from_chunk(chunk=junk_chunk, batch_name="test_batch_123")
        
        # 4. Assertions
        mock_chain_ainvoke.assert_called_once()
        
        # The result MUST be None, preventing database pollution
        assert result is None

