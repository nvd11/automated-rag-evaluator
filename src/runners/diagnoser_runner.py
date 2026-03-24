import asyncio
import os
import sys
from loguru import logger

# Add project root to sys.path so we can run this script directly
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.configs.db import init_db_pool, close_db_pool

# Components for Dependency Injection
from src.dao.diagnoser_dao import PgVectorDiagnoserDAO
from src.diagnosis.rules import (
    DiagnosticEngine,
    RetrievalQualityRule,
    HallucinationRule,
    AnswerRelevanceRule,
    BenchmarkCorrectnessRule
)
from src.pipelines.diagnoser_pipeline import DiagnoserPipeline

class DiagnoserRunner:
    """
    Entrypoint orchestrator for the Phase 6 Diagnostic Pipeline.
    Loads configurations, instantiates concrete dependencies (DAO, Rule Engine, Rules),
    and triggers the DiagnoserPipeline to generate the final optimization report.
    """
    
    def __init__(self):
        logger.info("Initializing Diagnoser Runner dependencies...")
        
        # 1. Instantiate Data Access Layer (Read-Only)
        dao = PgVectorDiagnoserDAO()
        
        # 2. Instantiate and Configure the Expert Rule Engine
        # We explicitly load the thresholds defined in the assignment's optimizer_config_template.json
        # In a dynamic system, these could be parsed from the JSON file directly.
        engine = DiagnosticEngine()
        engine.register_rule(RetrievalQualityRule(threshold=4.0)) # Stricter threshold for better retrieval
        engine.register_rule(HallucinationRule(threshold=4.5))    # High threshold to heavily penalize hallucination
        engine.register_rule(AnswerRelevanceRule(context_threshold=3.5, answer_threshold=3.5))
        engine.register_rule(BenchmarkCorrectnessRule(threshold=4.0))
        
        # 3. Assemble the Pipeline
        self.pipeline = DiagnoserPipeline(dao=dao, engine=engine)
        
    async def run_diagnosis(self, evaluation_job_id: str, output_dir: str = ".") -> None:
        """
        Triggers the diagnostic pipeline for a specific evaluation job ID.
        """
        if not evaluation_job_id:
            logger.error("A valid evaluation_job_id must be provided to start diagnosis.")
            return
            
        try:
            logger.info(f"Triggering Diagnoser Pipeline for Job ID: {evaluation_job_id}")
            # Delegate all business logic to the pipeline
            report_path = await self.pipeline.generate_report(
                evaluation_job_id=evaluation_job_id,
                output_dir=output_dir
            )
            
            if report_path:
                logger.info(f"Diagnostic Report successfully generated at: {report_path}")
            else:
                logger.warning(f"Diagnosis aborted or failed to generate a report for job '{evaluation_job_id}'.")
            
        except Exception as e:
            logger.exception(f"A fatal error occurred during diagnosis execution: {e}")

async def main():
    try:
        await init_db_pool()
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        sys.exit(1)

    try:
        runner = DiagnoserRunner()
        
        # Create output directory for reports if it doesn't exist
        reports_dir = os.path.join(project_root, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # --- HARDCODED DEMO FOR PHASE 6 Execution ---
        # Note: In a real system, you would pass these job IDs via CLI arguments
        
        # 1. Provide the Job ID from your Phase 5 Case 1 execution (Benchmark)
        TARGET_JOB_ID_CASE1 = "9f63801e-edba-49b3-9d14-f925071956f1" # Replace with actual job_id
        await runner.run_diagnosis(evaluation_job_id=TARGET_JOB_ID_CASE1, output_dir=reports_dir)
        
        # 2. Provide the Job ID from your Phase 5 Case 2 execution (Blind Test - 30 queries)
        TARGET_JOB_ID_CASE2 = "9ee6047d-4e63-44cc-9386-298f1154d162" # The job id matching 30 queries
        await runner.run_diagnosis(evaluation_job_id=TARGET_JOB_ID_CASE2, output_dir=reports_dir)

    finally:
        await close_db_pool()
        logger.info("Diagnoser Runner shutdown complete.")

if __name__ == "__main__":
    asyncio.run(main())