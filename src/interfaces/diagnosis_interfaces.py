from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from src.domain.models import DiagnosisObject, DiagnosisReport

class IDiagnoserDAO(ABC):
  """
  Contract for extracting aggregated metric data required by the Expert Rule Engine.
  Operates strictly as a Read-Only analytical consumer.
  """
  
  @abstractmethod
  async def fetch_metric_averages(self, job_id: str) -> Dict[str, float]:
    """
    Retrieves the average score for each metric (e.g., 'faithfulness', 'correctness')
    associated with a specific evaluation job.
    
    Returns:
      A dictionary mapping metric names to their average scores.
    """
    pass
    
  @abstractmethod
  async def fetch_job_metadata(self, job_id: str) -> Dict[str, str]:
    """
    Retrieves high-level context (e.g., inference_run_id, dataset_name) 
    needed to populate the final DiagnosisReport.
    """
    pass

class IDiagnosticRule(ABC):
  """
  Contract for a single Heuristic Diagnostic Rule.
  Follows the Open-Closed Principle (OCP) and Strategy Pattern.
  """
  
  @abstractmethod
  def analyze(self, metric_averages: Dict[str, float]) -> Optional[DiagnosisObject]:
    """
    Analyzes the aggregated metrics against predefined thresholds.
    
    Returns:
      A populated DiagnosisObject if the rule's conditions trigger.
      Returns None if the metrics pass this rule's health check.
    """
    pass