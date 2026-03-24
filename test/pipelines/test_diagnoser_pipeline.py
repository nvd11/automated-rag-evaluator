import pytest
import asyncio
import os
import json
from unittest.mock import AsyncMock, MagicMock, patch

from src.pipelines.diagnoser_pipeline import DiagnoserPipeline
from src.interfaces.diagnosis_interfaces import IDiagnoserDAO
from src.diagnosis.rules import DiagnosticEngine
from src.domain.models import DiagnosisObject

@pytest.fixture
def mock_dao():
    dao = MagicMock(spec=IDiagnoserDAO)
    dao.fetch_job_metadata = AsyncMock()
    dao.fetch_metric_averages = AsyncMock()
    return dao

@pytest.fixture
def mock_engine():
    engine = MagicMock(spec=DiagnosticEngine)
    # The diagnose method is synchronous
    engine.diagnose = MagicMock()
    return engine

@pytest.mark.asyncio
async def test_diagnoser_pipeline_success(mock_dao, mock_engine, tmp_path):
    """
    Unit test to verify the DiagnoserPipeline correctly fetches data from DAO,
    delegates to the Rule Engine, formats the result into the required JSON schema,
    and successfully exports it to disk.
    """
    # 1. Setup Mocks
    job_id = "test-job-1234"
    
    mock_dao.fetch_job_metadata.return_value = {
        "setting_id": "test-inference-run-001",
        "dataset_name": "pytest_golden_dataset"
    }
    
    mock_dao.fetch_metric_averages.return_value = {
        "context_relevance": 3.2, # Should trigger RetrievalQualityRule
        "faithfulness": 4.8,
        "answer_relevance": 3.0,
        "correctness": 3.5 # Should trigger BenchmarkCorrectnessRule
    }
    
    # Mock the engine returning two diagnoses
    mock_engine.diagnose.return_value = [
        DiagnosisObject(
            issue="Low Retrieval Quality",
            evidence=["avg_context_relevance (3.20) is below 3.5"],
            likely_root_causes=["Chunk size too small"],
            recommended_actions=["Switch to recursive_section_aware"]
        ),
        DiagnosisObject(
            issue="Low Accuracy against Benchmark Ground Truth",
            evidence=["avg_correctness (3.50) is below 4.0"],
            likely_root_causes=["Failing on multi-hop questions"],
            recommended_actions=["Upgrade to decompose_then_merge"]
        )
    ]
    
    # 2. Instantiate Pipeline
    pipeline = DiagnoserPipeline(dao=mock_dao, engine=mock_engine)
    
    # 3. Execute Pipeline
    # We use pytest's built-in tmp_path fixture to write the file safely
    output_dir = str(tmp_path)
    file_path = await pipeline.generate_report(evaluation_job_id=job_id, output_dir=output_dir)
    
    # 4. Assertions on dependencies
    mock_dao.fetch_job_metadata.assert_called_once_with(job_id)
    mock_dao.fetch_metric_averages.assert_called_once_with(job_id)
    mock_engine.diagnose.assert_called_once_with(mock_dao.fetch_metric_averages.return_value)
    
    # 5. Assertions on the output file
    assert file_path is not None
    assert os.path.exists(file_path)
    assert file_path.endswith(f"diagnosis_report_{job_id[:8]}.json")
    
    # Parse the generated JSON file to verify schema adherence
    with open(file_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)
        
    assert report_data["setting_id"] == "test-inference-run-001"
    assert report_data["dataset_name"] == "pytest_golden_dataset"
    
    # Check overall quality calculation (Average of 3.2, 4.8, 3.0, 3.5)
    # (3.2 + 4.8 + 3.0 + 3.5) / 4 = 14.5 / 4 = 3.625 -> rounded to 3.62
    assert report_data["overall_summary"]["quality_score"] == 3.62
    
    # Check nested stage formatting
    assert report_data["stage_metrics"]["retrieval"]["context_relevance"] == 3.2
    assert report_data["stage_metrics"]["generation"]["faithfulness"] == 4.8
    assert report_data["stage_metrics"]["end_to_end"]["correctness"] == 3.5
    
    # Check diagnosis list
    assert len(report_data["diagnosis"]) == 2
    assert report_data["diagnosis"][0]["issue"] == "Low Retrieval Quality"
    assert report_data["diagnosis"][1]["issue"] == "Low Accuracy against Benchmark Ground Truth"


@pytest.mark.asyncio
async def test_diagnoser_pipeline_empty_metrics(mock_dao, mock_engine, tmp_path):
    """
    Test that the pipeline gracefully aborts if the DAO returns empty or all-zero metrics.
    """
    mock_dao.fetch_job_metadata.return_value = {}
    mock_dao.fetch_metric_averages.return_value = {"correctness": 0.0, "faithfulness": 0.0}
    
    pipeline = DiagnoserPipeline(dao=mock_dao, engine=mock_engine)
    
    file_path = await pipeline.generate_report("empty-job-123", str(tmp_path))
    
    # Should abort before calling engine or writing file
    mock_engine.diagnose.assert_not_called()
    assert file_path == ""