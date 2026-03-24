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

from src.ingestion.loaders.pdf_loader import PyMuPDFLoader
from src.ingestion.chunkers.langchain_chunker import LangchainRecursiveChunker
from src.ingestion.embedders.gemini_embedder import GeminiEmbedder
from src.dao.pgvector_dao import PgVectorDAO
from src.pipelines.data_ingestion_pipeline import DataIngestionPipeline

def print_human_readable_summary(result: dict):
  doc_id = result.get("doc_id")
  stats = result.get("stats", {})
  deleted = stats.get("soft_deleted", {})
  inserted = stats.get("inserted", {})
  
  border = "=" * 60
  logger.info(border)
  logger.info("DATA INGESTION PIPELINE SUMMARY")
  logger.info(border)
  logger.info(f"Generated Document ID : {doc_id}")
  logger.info("-" * 60)
  logger.info("SOFT DELETED (Historical Data cleanup)")
  logger.info(f"  • `documents` table    : {deleted.get('documents', 0):>5} rows")
  logger.info(f"  • `document_topics` table : {deleted.get('document_topics', 0):>5} rows")
  logger.info(f"  • `document_chunks` table : {deleted.get('document_chunks', 0):>5} rows")
  logger.info("-" * 60)
  logger.info("INSERTED (New Data)")
  logger.info(f"  • `documents` table    : {inserted.get('documents', 0):>5} rows")
  logger.info(f"  • `topics` table     : {inserted.get('topics', 0):>5} rows")
  logger.info(f"  • `document_topics` table : {inserted.get('document_topics', 0):>5} rows")
  logger.info(f"  • `document_chunks` table : {inserted.get('document_chunks', 0):>5} rows (with Embeddings)")
  logger.info(border)

async def main():
  # 1. Initialize Logging
  logger.info("Initializing Data Ingestion Runner...")

  # Validate API Key
  if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY.startswith("your_") or settings.GEMINI_API_KEY.startswith("AIzaSy_dummy"):
    logger.error("Invalid GEMINI_API_KEY detected. Please update your .env file with a real Google AI Studio Key.")
    sys.exit(1)

  # Define the input file and topics
  pdf_file_path = os.path.join(project_root, "data", "HSBC_Annual_Report_2025.pdf")
  topics = ["Financial Performance", "Risk Management", "Sustainability", "Corporate Governance"]

  if not os.path.exists(pdf_file_path):
    logger.error(f"Target PDF file not found at: {pdf_file_path}")
    sys.exit(1)

  # 2. Initialize Database Connection Pool
  try:
    await init_db_pool()
  except Exception as e:
    logger.error(f"Failed to initialize database pool: {e}")
    sys.exit(1)

  try:
    # 3. Assemble the Pipeline (Dependency Injection)
    loader = PyMuPDFLoader()
    chunker = LangchainRecursiveChunker()
    embedder = GeminiEmbedder()
    dao = PgVectorDAO()

    pipeline = DataIngestionPipeline(
      loader=loader,
      chunker=chunker,
      embedder=embedder,
      dao=dao,
      created_by_user="ingestion_runner_script"
    )

    # 4. Execute the Pipeline
    result = await pipeline.run(file_path=pdf_file_path, topics=topics)
    
    if result and result.get("doc_id"):
      print_human_readable_summary(result)
    else:
      logger.error("Ingestion pipeline failed or aborted.")

  except Exception as e:
    logger.exception(f"A fatal error occurred during the ingestion pipeline: {e}")
    
  finally:
    # 5. Graceful Shutdown
    await close_db_pool()
    logger.info("Shutdown complete.")

if __name__ == "__main__":
  asyncio.run(main())
