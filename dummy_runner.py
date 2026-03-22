import asyncio
import os
import sys
from loguru import logger
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path: sys.path.insert(0, project_root)
from src.configs.settings import settings
from src.configs.db import init_db_pool, close_db_pool
from src.ingestion.loaders.pdf_loader import PyMuPDFLoader
from src.ingestion.chunkers.langchain_chunker import LangchainRecursiveChunker
from src.ingestion.embedders.gemini_embedder import GeminiEmbedder
from src.dao.pgvector_dao import PgVectorDAO
from src.pipelines.data_ingestion_pipeline import DataIngestionPipeline

async def main():
    await init_db_pool()
    loader = PyMuPDFLoader()
    chunker = LangchainRecursiveChunker(chunk_size=1000, chunk_overlap=200)
    embedder = GeminiEmbedder(batch_size=100)
    dao = PgVectorDAO()
    pipeline = DataIngestionPipeline(loader, chunker, embedder, dao)
    
    # We create a fake 1 page PDF to test end-to-end integration and dimensions
    import fitz
    pdf_path = "data/dummy_test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "HSBC is a global bank. We test 768 dimensions.")
    doc.save(pdf_path)
    
    doc_id = await pipeline.run(file_path=pdf_path, topics=["Test"])
    logger.info(f"Done! Doc ID: {doc_id}")
    await close_db_pool()

asyncio.run(main())
