import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from src.evaluator.llm_judge import GoldenBaselineJudge, RagTriadJudge, Case1EvaluationResult, Case2EvaluationResult
from src.domain.models import QueryEvaluationDTO

@pytest.fixture
def dummy_case1_dto():
    return QueryEvaluationDTO(
        query_id="case1-query-123",
        question="What is the capital of France?",
        generated_answer="The capital of France is Paris.",
        retrieved_contexts=[],
        ground_truth="Paris is the capital of France."
    )

@pytest.fixture
def dummy_case2_dto():
    return QueryEvaluationDTO(
        query_id="case2-query-456",
        question="What is HSBC's 2025 profit?",
        generated_answer="HSBC's profit was 22.6 billion.",
        retrieved_contexts=[{"text": "HSBC Holdings statement shows a profit of 22,611 million for 2025."}],
        ground_truth=None
    )

@pytest.mark.asyncio
async def test_golden_baseline_judge_success(dummy_case1_dto):
    """
    Test that GoldenBaselineJudge correctly extracts the single 'correctness' metric
    from the structured LLM output (Case1EvaluationResult) and maps it to a ScoreWithReasoning object.
    """
    fake_llm = MagicMock()
    # Mock the structured output returned by the LLM
    mock_result = Case1EvaluationResult(
        correctness_score=5.0,
        correctness_reasoning="Perfectly matches ground truth."
    )
    
    judge = GoldenBaselineJudge(llm=fake_llm)
    
    # We patch both ainvoke and invoke to cover proxy scenarios
    with patch("langchain_core.runnables.RunnableSequence.ainvoke", return_value=mock_result) as mock_ainvoke:
        with patch("langchain_core.runnables.RunnableSequence.invoke", return_value=mock_result) as mock_invoke:
            
            scores = await judge.evaluate_query(dummy_case1_dto)
            
            # Assertions
            assert len(scores) == 1
            assert scores[0].metric_name == "correctness"
            assert scores[0].score == 5.0
            assert scores[0].reasoning == "Perfectly matches ground truth."
            assert mock_ainvoke.called or mock_invoke.called

@pytest.mark.asyncio
async def test_golden_baseline_judge_missing_ground_truth(dummy_case2_dto):
    """
    Test that GoldenBaselineJudge safely aborts if given a query without ground_truth.
    """
    fake_llm = MagicMock()
    judge = GoldenBaselineJudge(llm=fake_llm)
    
    with pytest.raises(ValueError, match="requires a ground_truth"):
        await judge.evaluate_query(dummy_case2_dto)

@pytest.mark.asyncio
async def test_rag_triad_judge_success(dummy_case2_dto):
    """
    Test that RagTriadJudge correctly extracts the three RAG Triad metrics
    from the structured LLM output (Case2EvaluationResult) and maps them to a list.
    """
    fake_llm = MagicMock()
    # Mock the structured output returned by the LLM
    mock_result = Case2EvaluationResult(
        context_relevance_score=4.0,
        context_relevance_reasoning="Context contains the profit figure.",
        faithfulness_score=5.0,
        faithfulness_reasoning="Answer is derived entirely from context.",
        answer_relevance_score=4.5,
        answer_relevance_reasoning="Directly answers the user's question."
    )
    
    judge = RagTriadJudge(llm=fake_llm)
    
    with patch("langchain_core.runnables.RunnableSequence.ainvoke", return_value=mock_result) as mock_ainvoke:
        with patch("langchain_core.runnables.RunnableSequence.invoke", return_value=mock_result) as mock_invoke:
            
            scores = await judge.evaluate_query(dummy_case2_dto)
            
            # Assertions
            assert len(scores) == 3
            
            # Verify extraction of all 3 metrics
            metric_names = [s.metric_name for s in scores]
            assert "context_relevance" in metric_names
            assert "faithfulness" in metric_names
            assert "answer_relevance" in metric_names
            
            # Check a specific score mapping
            faithfulness = next(s for s in scores if s.metric_name == "faithfulness")
            assert faithfulness.score == 5.0
            assert faithfulness.reasoning == "Answer is derived entirely from context."
            
            assert mock_ainvoke.called or mock_invoke.called