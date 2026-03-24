import asyncio
import uuid
from typing import List, Dict
from loguru import logger

from src.interfaces.evaluator_interfaces import IEvaluationDAO, ILLMJudge
from src.domain.models import QueryEvaluationDTO, EvaluationMetricRecord, EvaluationJobHistory, ScoreWithReasoning

class EvaluationPipeline:
    """
    The orchestrator for the LLM-as-a-Judge RAG Evaluation process.
    Responsible for fetching historical queries, categorizing them (Case 1 vs Case 2),
    dispatching them concurrently to the appropriate judges, and persisting the results.
    """
    
    def __init__(
        self, 
        dao: IEvaluationDAO, 
        case1_judge: ILLMJudge, 
        case2_judge: ILLMJudge,
        evaluator_model_name: str,
        evaluator_prompt_version: str = "v1.0"
    ):
        self.dao = dao
        self.case1_judge = case1_judge
        self.case2_judge = case2_judge
        self.evaluator_model_name = evaluator_model_name
        self.evaluator_prompt_version = evaluator_prompt_version
        
    async def _evaluate_single_query(
        self, 
        dto: QueryEvaluationDTO, 
        job_id: str, 
        sem: asyncio.Semaphore
    ) -> List[EvaluationMetricRecord]:
        """
        Evaluates a single query using the correct judge based on the presence of ground truth.
        Wraps the execution in a Semaphore to prevent API rate limiting.
        """
        async with sem:
            metrics: List[EvaluationMetricRecord] = []
            
            if dto.has_ground_truth:
                strategy = "CASE1_GROUND_TRUTH"
                judge = self.case1_judge
            else:
                strategy = "CASE2_RAG_TRIAD"
                judge = self.case2_judge
                
            try:
                # Polymorphic call to the correct judge
                scores: List[ScoreWithReasoning] = await judge.evaluate_query(dto)
                
                # Map structured LLM output to Database EAV records
                for s in scores:
                    record = EvaluationMetricRecord(
                        query_id=dto.query_id,
                        job_id=job_id,
                        evaluation_strategy=strategy,
                        metric_category="generation",  # All current metrics are generation-focused
                        metric_name=s.metric_name,
                        metric_value=s.score,
                        reasoning=s.reasoning,
                        judge_model=self.evaluator_model_name
                    )
                    metrics.append(record)
                    
            except Exception as e:
                logger.error(f"Pipeline failed to evaluate query {dto.query_id} using {strategy}: {e}")
                
            return metrics

    async def run(self, inference_run_id: str, created_by: str = "evaluation_pipeline") -> Dict[str, int]:
        """
        Executes the full evaluation pipeline for a given historical inference run.
        
        Returns:
            Dict containing summary statistics of the evaluation job.
        """
        job_id = str(uuid.uuid4())
        logger.info("=" * 80)
        logger.info(f"STARTING EVALUATION PIPELINE | Job ID: {job_id}")
        logger.info(f"Target Inference Run ID: {inference_run_id}")
        logger.info(f"Evaluator Model: {self.evaluator_model_name}")
        logger.info("=" * 80)
        
        # 1. Fetch data
        queries = await self.dao.fetch_queries_for_evaluation(inference_run_id)
        if not queries:
            logger.warning(f"No queries found for inference run {inference_run_id}. Aborting evaluation.")
            return {"total_queries": 0, "metrics_inserted": 0}
            
        case1_count = sum(1 for q in queries if q.has_ground_truth)
        case2_count = len(queries) - case1_count
        logger.info(f"Loaded {len(queries)} queries (Case 1: {case1_count}, Case 2: {case2_count}).")
        
        # 2. Record Job History
        job_history = EvaluationJobHistory(
            job_id=job_id,
            inference_run_id=inference_run_id,
            evaluator_model=self.evaluator_model_name,
            evaluator_prompt_version=self.evaluator_prompt_version
        )
        await self.dao.create_evaluation_job(job_history, created_by)
        
        # 3. Concurrent Evaluation
        logger.info("Dispatching queries to LLM Judges (Concurrency Limit: 3)...")
        sem = asyncio.Semaphore(3) # Protects Gemini API rate limits
        
        tasks = [self._evaluate_single_query(dto=q, job_id=job_id, sem=sem) for q in queries]
        
        # results is a list of lists of EvaluationMetricRecord
        results_nested: List[List[EvaluationMetricRecord]] = await asyncio.gather(*tasks)
        
        # Flatten the list
        all_metrics: List[EvaluationMetricRecord] = [metric for sublist in results_nested for metric in sublist]
        
        logger.info(f"Evaluation complete. Generated {len(all_metrics)} individual metric scores.")
        
        # 4. Transactional Bulk Persistence
        if all_metrics:
            await self.dao.bulk_insert_evaluation_metrics(all_metrics, created_by)
        else:
            logger.warning("No metrics were generated (possibly all failed). Nothing to insert.")
            
        logger.info("EVALUATION PIPELINE FINISHED SUCCESSFULLY.")
        logger.info("=" * 80)
        
        return {
            "total_queries_evaluated": len(queries),
            "metrics_generated": len(all_metrics),
            "job_id": job_id
        }