import asyncio
import os
import sys
from loguru import logger
import time

# Add project root to sys.path so we can run this script directly
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.configs.settings import settings
from src.configs.db import init_db_pool, close_db_pool

# Components
from src.ingestion.embedders.gemini_embedder import GeminiEmbedder
from src.dao.pgvector_retriever_dao import PgVectorRetrieverDAO
from src.retrieval.semantic_retriever import SemanticRetriever
from src.retrieval.langchain_generator import LangchainGeminiGenerator
from src.agents.rag_agent import RAGAgent

def print_human_readable_answer(response):
    border = "=" * 80
    logger.info(border)
    logger.info(f"🤔 QUESTION: {response.query}")
    logger.info("-" * 80)
    logger.info(f"🤖 AI ANSWER:\n{response.generated_answer}")
    logger.info("-" * 80)
    
    if not response.retrieved_contexts:
        logger.warning("No context chunks were retrieved (below threshold).")
    else:
        logger.info(f"📚 CITED SOURCES ({len(response.retrieved_contexts)} chunks):")
        for i, ctx in enumerate(response.retrieved_contexts):
            meta_str = ", ".join([f"{k}={v}" for k, v in ctx.metadata.items()])
            logger.info(f"  [{i+1}] Score: {ctx.similarity_score:.4f} | {meta_str}")
            # Print a short preview of the text
            preview = ctx.text[:100].replace('\n', ' ') + "..." if len(ctx.text) > 100 else ctx.text
            logger.info(f"      Preview: {preview}")
            
    logger.info(border + "\n")

async def main():
    logger.info("Initializing RAG Agent Runner Demo...")

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
        # 1. Assemble the components (Dependency Injection)
        embedder = GeminiEmbedder()
        dao = PgVectorRetrieverDAO()
        retriever = SemanticRetriever(embedder=embedder, dao=dao)
        generator = LangchainGeminiGenerator()
        
        agent = RAGAgent(retriever=retriever, generator=generator)

        # 2. Define test questions
        test_questions = [
            # A direct question about the HSBC Annual Report
            "What was the total profit of HSBC according to the annual report?",
            # A targeted question using a metadata filter
            "What are the specific climate change and sustainability goals mentioned for the future?"
        ]

        # 3. Execute queries
        for q in test_questions:
            start_time = time.time()
            
            # If asking about sustainability, let's pretend the system decided to pre-filter by topic
            topic_filter = ["Sustainability"] if "sustainability" in q.lower() else None
            
            response = await agent.ask(
                question=q, 
                top_k=settings.RETRIEVAL_TOP_K, 
                similarity_threshold=settings.RETRIEVAL_SIMILARITY_THRESHOLD,
                topic_filters=topic_filter
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Query completed in {elapsed_time:.2f} seconds.")
            
            print_human_readable_answer(response)

    except Exception as e:
        logger.exception(f"A fatal error occurred during the RAG pipeline: {e}")
        
    finally:
        await close_db_pool()
        logger.info("Runner shutdown complete.")

if __name__ == "__main__":
    asyncio.run(main())
