import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.pipelines.evaluation_pipeline import EvaluationPipeline
from src.domain.models import QueryEvaluationDTO, ScoreWithReasoning, EvaluationMetricRecord
from src.interfaces.evaluator_interfaces import IEvaluationDAO, ILLMJudge

@pytest.fixture
def mock_dao():
  dao = MagicMock(spec=IEvaluationDAO)
  dao.fetch_queries_for_evaluation = AsyncMock()
  dao.create_evaluation_job = AsyncMock()
  dao.bulk_insert_evaluation_metrics = AsyncMock()
  return dao

@pytest.fixture
def mock_case1_judge():
  judge = MagicMock(spec=ILLMJudge)
  judge.evaluate_query = AsyncMock()
  return judge

@pytest.fixture
def mock_case2_judge():
  judge = MagicMock(spec=ILLMJudge)
  judge.evaluate_query = AsyncMock()
  return judge

@pytest.fixture
def sample_queries():
  return [
    QueryEvaluationDTO(
      query_id="q1",
      question="What is HSBC?",
      generated_answer="A bank.",
      retrieved_contexts=[],
      ground_truth="A global bank." # Has Ground Truth -> Case 1
    ),
    QueryEvaluationDTO(
      query_id="q2",
      question="What is the profit?",
      generated_answer="22.6 billion.",
      retrieved_contexts=[],
      ground_truth=None # No Ground Truth -> Case 2
    )
  ]

@pytest.mark.asyncio
async def test_evaluation_pipeline_success(mock_dao, mock_case1_judge, mock_case2_judge, sample_queries):
  """
  Unit test to verify the EvaluationPipeline correctly routes queries based on
  the presence of ground truth, extracts the structured LLM judge scores, 
  maps them to EAV records, and saves them to the DAO.
  """
  
  # 1. Setup Mock DAO
  mock_dao.fetch_queries_for_evaluation.return_value = sample_queries
  
  # 2. Setup Mock Judges
  mock_case1_judge.evaluate_query.return_value = [
    ScoreWithReasoning(metric_name="correctness", score=4.5, reasoning="Close enough.")
  ]
  
  mock_case2_judge.evaluate_query.return_value = [
    ScoreWithReasoning(metric_name="faithfulness", score=5.0, reasoning="No hallucinations."),
    ScoreWithReasoning(metric_name="answer_relevance", score=4.0, reasoning="Answered the question."),
    ScoreWithReasoning(metric_name="context_relevance", score=3.0, reasoning="Context was okay.")
  ]
  
  pipeline = EvaluationPipeline(
    dao=mock_dao,
    case1_judge=mock_case1_judge,
    case2_judge=mock_case2_judge,
    evaluator_model_name="mock-judge-v1"
  )
  
  # 3. Execute Pipeline
  run_id = "test-inference-run-123"
  result = await pipeline.run(inference_run_id=run_id, created_by="pytest_runner")
  
  # 4. Assertions on DAO interactions
  mock_dao.fetch_queries_for_evaluation.assert_called_once_with(run_id)
  mock_dao.create_evaluation_job.assert_called_once()
  mock_dao.bulk_insert_evaluation_metrics.assert_called_once()
  
  # 5. Assertions on Polymorphic Judge Routing
  # Case 1 Judge should be called exactly once for 'q1'
  mock_case1_judge.evaluate_query.assert_called_once()
  assert mock_case1_judge.evaluate_query.call_args[0][0].query_id == "q1"
  
  # Case 2 Judge should be called exactly once for 'q2'
  mock_case2_judge.evaluate_query.assert_called_once()
  assert mock_case2_judge.evaluate_query.call_args[0][0].query_id == "q2"
  
  # 6. Assertions on Metric Flattening (EAV construction)
  inserted_metrics = mock_dao.bulk_insert_evaluation_metrics.call_args[0][0]
  
  # We expect 1 metric from Case 1 and 3 metrics from Case 2 = 4 total rows
  assert len(inserted_metrics) == 4
  
  # Check that the evaluation_strategy was correctly assigned during the flatten process
  case1_metrics = [m for m in inserted_metrics if m.query_id == "q1"]
  assert len(case1_metrics) == 1
  assert case1_metrics[0].evaluation_strategy == "CASE1_GROUND_TRUTH"
  assert case1_metrics[0].metric_name == "correctness"
  
  case2_metrics = [m for m in inserted_metrics if m.query_id == "q2"]
  assert len(case2_metrics) == 3
  assert all(m.evaluation_strategy == "CASE2_RAG_TRIAD" for m in case2_metrics)
  
  # Check return dictionary summary
  assert result["total_queries_evaluated"] == 2
  assert result["metrics_generated"] == 4
  assert "job_id" in result

@pytest.mark.asyncio
async def test_evaluation_pipeline_empty_queries(mock_dao, mock_case1_judge, mock_case2_judge):
  """
  Test that the pipeline gracefully aborts if the DAO returns no historical queries.
  """
  mock_dao.fetch_queries_for_evaluation.return_value = []
  
  pipeline = EvaluationPipeline(
    dao=mock_dao,
    case1_judge=mock_case1_judge,
    case2_judge=mock_case2_judge,
    evaluator_model_name="mock-judge-v1"
  )
  
  result = await pipeline.run(inference_run_id="empty-run")
  
  mock_dao.fetch_queries_for_evaluation.assert_called_once()
  # It should abort before creating a job or inserting metrics
  mock_dao.create_evaluation_job.assert_not_called()
  mock_dao.bulk_insert_evaluation_metrics.assert_not_called()
  mock_case1_judge.evaluate_query.assert_not_called()
  
  assert result["total_queries"] == 0
  assert result["metrics_inserted"] == 0