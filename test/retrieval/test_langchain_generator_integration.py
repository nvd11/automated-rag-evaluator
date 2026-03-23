import pytest
import os
from loguru import logger
from src.retrieval.langchain_generator import LangchainRAGGenerator
from src.domain.models import RetrievedContext
from src.configs.settings import settings

@pytest.mark.asyncio
@pytest.mark.integration
async def test_gemini_25_pro_live_connection():
    """
    Live integration test to verify connectivity and inference with Gemini 2.5 Pro.
    This test DOES NOT use any mocks and will consume real API quota.
    """
    # 1. Ensure API key is configured
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY.startswith("your_"):
        pytest.skip("Skipping live Gemini test because GEMINI_API_KEY is not configured.")

    logger.info(f"Testing live connection to {settings.LLM_JUDGE_MODEL}...")

    # 2. Instantiate our actual LLM Generator using the Abstract Factory Pattern
    from src.llm.llm_factory import ILLMFactory
    from src.llm.gemini_factory import GeminiLLMFactory
    
    llm_factory: ILLMFactory = GeminiLLMFactory()
    live_llm = llm_factory.create_llm(model_name=settings.LLM_INFERENCE_MODEL, temperature=0.0)
    generator = LangchainRAGGenerator(llm=live_llm)

    # 3. Create a minimal dummy context to feed the prompt template
    dummy_context = RetrievedContext(
        doc_id="test-doc-001",
        chunk_id="test-chunk-001",
        text="The capital of France is London. Wait, no, the capital of France is Paris. The population is about 2.1 million.",
        similarity_score=0.99,
        metadata={"page_number": 1}
    )

    # 4. Create the exact input dictionary expected by the ILLMGenerator.ainvoke contract
    input_data = {
        "context": [dummy_context],
        "question": "According to the provided text, what is the capital of France?"
    }

    # 5. Execute the live call
    try:
        answer = await generator.ainvoke(input_data)
        logger.info(f"Received answer from Gemini: '{answer}'")
        
        # 6. Assertions
        assert answer is not None
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert "Paris" in answer
        assert "London" not in answer # It should be smart enough to extract the right fact
        
    except Exception as e:
        pytest.fail(f"Live call to Gemini 2.5 Pro failed: {e}")
