import asyncio
import os
import sys
from loguru import logger

# Add project root to sys.path so we can run this script directly
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.configs.settings import settings
from src.configs.db import init_db_pool, close_db_pool

# Components for Dependency Injection
from src.llm.gemini_factory import GeminiLLMFactory
from src.evaluator.llm_judge import GoldenBaselineJudge, RagTriadJudge
from src.dao.evaluation_dao import PgVectorEvaluationDAO
from src.pipelines.evaluation_pipeline import EvaluationPipeline

class EvaluationRunner:
    """
    Entrypoint orchestrator for the Phase 5 Evaluation Pipeline.
    Loads configurations, instantiates concrete dependencies (DAO, LLMs, Judges),
    and triggers the EvaluationPipeline to score historical inference runs.
    """
    
    def __init__(self):
        logger.info("Initializing Evaluation Runner dependencies...")
        
        # 1. Instantiate the Factory and the common Judge LLM
        # We explicitly use the LLM_JUDGE_MODEL (e.g., gemini-3.1-pro-preview) for highest reasoning capability
        llm_factory = GeminiLLMFactory()
        judge_llm = llm_factory.create_llm(model_name=settings.LLM_JUDGE_MODEL, temperature=0.0)
        
        # 2. Instantiate Concrete Judges (Polymorphism)
        case1_judge = GoldenBaselineJudge(llm=judge_llm)
        case2_judge = RagTriadJudge(llm=judge_llm)
        
        # 3. Instantiate Data Access Layer
        dao = PgVectorEvaluationDAO()
        
        # 4. Assemble the Pipeline
        self.pipeline = EvaluationPipeline(
            dao=dao,
            case1_judge=case1_judge,
            case2_judge=case2_judge,
            evaluator_model_name=settings.LLM_JUDGE_MODEL,
            evaluator_prompt_version="v2.0_upgraded_eav"
        )
        
    async def run_evaluation(self, inference_run_id: str) -> None:
        """
        Triggers the evaluation pipeline for a specific inference run ID.
        """
        if not inference_run_id:
            logger.error("A valid inference_run_id must be provided to start evaluation.")
            return
            
        try:
            logger.info(f"Triggering Evaluation Pipeline for Inference Run: {inference_run_id}")
            # Delegate all business logic to the pipeline
            summary = await self.pipeline.run(inference_run_id=inference_run_id, created_by="evaluation_runner_script")
            
            logger.info("=" * 60)
            logger.info("EVALUATION RUNNER JOB SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Evaluated Run ID : {inference_run_id}")
            logger.info(f"New Job ID       : {summary.get('job_id')}")
            logger.info(f"Total Queries    : {summary.get('total_queries_evaluated')}")
            logger.info(f"Metrics Generated: {summary.get('metrics_generated')}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.exception(f"A fatal error occurred during evaluation execution: {e}")


async def main():
    # Validate API Key
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY.startswith("your_"):
        logger.error("Invalid GEMINI_API_KEY. Please set a valid key in .env")
        sys.exit(1)

    try:
        await init_db_pool()
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        sys.exit(1)

    try:
        runner = EvaluationRunner()
        
        # --- HARDCODED DEMO FOR PHASE 5 Execution ---
        # Note: In a real system, you would pass these run IDs via CLI arguments (e.g., argparse)
        
        # 1. Provide the Run ID from your Phase 4 Case 1 execution (Benchmark)
        # Please replace this string with the actual Run ID printed during your Inference execution!
        TARGET_RUN_ID_CASE1 = "32f23ad6-8d00-4a7f-a1ae-29b7b50dfc91" # The 48-record Golden run from earlier
        # await runner.run_evaluation(inference_run_id=TARGET_RUN_ID_CASE1)
        
        # 2. Provide the Run ID from your Phase 4 Case 2 execution (Blind Test)
        TARGET_RUN_ID_CASE2 = "57c7466a-46f2-49b4-af9f-14d4352a25ef" # The actual 30-record Blind run from earlier
        await runner.run_evaluation(inference_run_id=TARGET_RUN_ID_CASE2)

    finally:
        await close_db_pool()
        logger.info("Evaluation Runner shutdown complete.")

if __name__ == "__main__":
    asyncio.run(main())