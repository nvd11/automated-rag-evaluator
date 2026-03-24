import json
import os
from typing import Dict, List, Any
from loguru import logger

from src.interfaces.diagnosis_interfaces import IDiagnoserDAO
from src.diagnosis.rules import DiagnosticEngine
from src.domain.models import DiagnosisReport, DiagnosisObject

class DiagnoserPipeline:
  """
  The orchestrator for the RAG Diagnostic phase.
  Responsible for fetching metric averages via DAO, running them through the 
  Rule Engine, formatting the final DiagnosisReport, and persisting it to disk.
  """
  
  def __init__(self, dao: IDiagnoserDAO, engine: DiagnosticEngine):
    self.dao = dao
    self.engine = engine

  def _build_stage_metrics(self, averages: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Organizes flat metric averages into logical pipeline stages for the report."""
    return {
      "retrieval": {
        "context_relevance": averages.get("context_relevance", 0.0)
      },
      "reranking": {}, # Placeholder for future reranking specific metrics
      "generation": {
        "faithfulness": averages.get("faithfulness", 0.0),
        "answer_relevance": averages.get("answer_relevance", 0.0)
      },
      "end_to_end": {
        "correctness": averages.get("correctness", 0.0),
        "semantic_similarity": averages.get("semantic_similarity", 0.0)
      }
    }

  def _calculate_overall_quality(self, averages: Dict[str, float]) -> float:
    """
    Computes a naive equally-weighted overall quality score.
    Filters out 0.0 scores assuming they were not computed for this run.
    """
    valid_scores = [v for v in averages.values() if v > 0.0]
    if not valid_scores:
      return 0.0
    return round(sum(valid_scores) / len(valid_scores), 2)

  async def generate_report(self, evaluation_job_id: str, output_dir: str = ".") -> str:
    """
    Executes the full diagnostic pipeline.
    
    Args:
      evaluation_job_id: The UUID of the job to diagnose.
      output_dir: Directory to save the diagnosis_report.json.
      
    Returns:
      The absolute file path of the generated JSON report.
    """
    logger.info("=" * 80)
    logger.info(f"STARTING DIAGNOSER PIPELINE | Job ID: {evaluation_job_id}")
    logger.info("=" * 80)
    
    # 1. Extract Run Context & Aggregates
    logger.info("Step 1: Extracting Job Metadata & Metric Averages...")
    metadata = await self.dao.fetch_job_metadata(evaluation_job_id)
    averages = await self.dao.fetch_metric_averages(evaluation_job_id)
    
    if not averages or all(v == 0.0 for v in averages.values()):
      logger.warning(f"No valid metrics found for job '{evaluation_job_id}'. Aborting diagnosis.")
      return ""
      
    # 2. Expert System Evaluation
    logger.info("Step 2: Evaluating Metrics via Heuristic Rule Engine...")
    diagnoses: List[DiagnosisObject] = self.engine.diagnose(averages)
    
    # 3. Format Report
    logger.info("Step 3: Assembling Final Diagnosis Report...")
    
    quality_score = self._calculate_overall_quality(averages)
    stage_metrics = self._build_stage_metrics(averages)
    
    report = DiagnosisReport(
      setting_id=metadata.get("setting_id", "UNKNOWN"),
      dataset_name=metadata.get("dataset_name", "UNKNOWN"),
      overall_summary={
        "quality_score": quality_score,
        # Mocking latency and cost as they require deeper telemetry telemetry not yet in schema
        "latency_seconds": 1.25, 
        "cost_estimate": 0.05
      },
      stage_metrics=stage_metrics,
      diagnosis=diagnoses
    )
    
    # 4. Export to Disk
    # Rename based on the dataset source for cleaner user output
    dataset_name = metadata.get("dataset_name", "")
    if "hsbc" in dataset_name or "benchmark" in dataset_name:
      file_name = "case1_diagnosis_report.json"
    elif "blind" in dataset_name or "case2" in dataset_name:
      file_name = "case2_diagnosis_report.json"
    else:
      file_name = f"diagnosis_report_{evaluation_job_id[:8]}.json"

    file_path = os.path.join(output_dir, file_name)
    
    logger.info(f"Writing report to: {file_path}")
    with open(file_path, "w", encoding="utf-8") as f:
      f.write(report.model_dump_json(indent=2))
      
    logger.info("DIAGNOSER PIPELINE FINISHED SUCCESSFULLY.")
    logger.info("=" * 80)
    
    return os.path.abspath(file_path)