from abc import ABC, abstractmethod
from typing import List, Optional
from src.domain.models import GoldenRecord, InferenceRun, QueryHistoryRecord

class IInferenceDAO(ABC):
  """
  Contract for data persistence related to Inference Runs (the RAG Agent taking the test).
  Handles fetching test questions and persisting the generated answers with their latencies.
  """
  
  @abstractmethod
  async def fetch_golden_records(self, batch_name: str, limit: Optional[int] = None) -> List[GoldenRecord]:
    """
    Retrieves a set of benchmark questions (Case 1) that have established Ground Truths.
    """
    pass

  @abstractmethod
  async def persist_inference_run(self, run: InferenceRun, queries: List[QueryHistoryRecord], created_by: str) -> None:
    """
    Persists an entire inference batch transactionally.
    This includes:
    1. Creating the InferenceRun record.
    2. Bulk inserting the QueryHistoryRecord answers.
    3. Linking the queries to the run in `inference_run_query_mapping`.
    4. (CRITICAL) If a query has a `golden_record_id`, linking it in `golden_record_query_mapping` (Case 1 vs Case 2 mechanism).
    """
    pass
