from typing import List, Optional, Any, Dict
from loguru import logger

from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch

from src.interfaces.retriever_interfaces import BaseRetriever, ILLMGenerator
from src.domain.models import RAGResponse, RetrievedContext

class RAGAgent:
  """
  The top-level orchestrator for the Retrieval-Augmented Generation workflow.
  Combines a Semantic Retriever and an LLM Generator into a unified LCEL pipeline.
  """
  def __init__(self, retriever: BaseRetriever, generator: ILLMGenerator):
    self.retriever = retriever
    self.generator = generator

  async def ask(
    self, 
    question: str, 
    top_k: Optional[int] = None, 
    similarity_threshold: Optional[float] = None,
    topic_filters: Optional[List[str]] = None
  ) -> RAGResponse:
    """
    Executes the end-to-end RAG pipeline for a given question.
    Returns a structured RAGResponse containing the generated answer and retrieved contexts.
    """
    logger.info(f"RAGAgent processing query: '{question}'")
    
    # 1. Prepare dynamic retrieval configuration
    retriever_config = {}
    if top_k is not None: retriever_config["top_k"] = top_k
    if similarity_threshold is not None: retriever_config["similarity_threshold"] = similarity_threshold
    if topic_filters is not None: retriever_config["topic_filters"] = topic_filters
    
    config_wrapper = {"configurable": retriever_config} if retriever_config else None

    # 2. Phase 1: Retrieval (The "R" in RAG)
    logger.debug("Delegating to SemanticRetriever...")
    # ARCHITECTURE NOTE: Imperative Orchestration vs. Pure LCEL
    # We explicitly orchestrate these two `Runnable` components sequentially in Python instead of building a monolithic LCEL chain (e.g., `chain = retriever | generator`). 
    # This trade-off is evaluated across 5 key dimensions:
    # 1. Lines of Code: While LCEL is more concise, our imperative approach adds only a few lines to handle the crucial `RAGResponse` packaging.
    # 2. Readability: The linear flow of fetching context, conditionally checking it, and calling the LLM is universally understood and transparent compared to implicit LCEL routing (e.g., `RunnableParallel.assign()`).
    # 3. Bug Risk & Debugging: Debugging intermediate state in a long LCEL chain is notoriously difficult when Pydantic validation or async generators fail. This approach allows trivial breakpoints between major domain boundaries.
    # 4. Maintainability: Conditional business logic—like the cost-saving "Early Exit" below if no context meets the `similarity_threshold`—is trivial in Python (`if not`) but requires brittle `RunnableBranch` constructs in LCEL.
    # 5. Extensibility & Observability: Because `SemanticRetriever` and `LangchainGeminiGenerator` individually implement the LangChain `Runnable` protocol, we preserve full native observability hooks (e.g., for LangSmith) while maintaining the flexibility to extract and persist intermediate DTOs for downstream automated evaluation.
    retrieved_contexts = await self.retriever.ainvoke(input=question, config=config_wrapper)
    
    if not retrieved_contexts:
      logger.warning(f"No relevant context found for query: '{question}'")
      return RAGResponse(
        query=question,
        generated_answer="I'm sorry, I couldn't find any relevant information in the knowledge base to answer your question.",
        retrieved_contexts=[]
      )

    # 3. Phase 2: Generation (The "A & G" in RAG)
    logger.debug("Delegating to LLMGenerator...")
    generator_input = {
      "context": retrieved_contexts,
      "question": question
    }
    
    generated_answer = await self.generator.ainvoke(input=generator_input)
    
    # 4. Package the payload for downstream Evaluation/Tracking
    response = RAGResponse(
      query=question,
      generated_answer=generated_answer,
      retrieved_contexts=retrieved_contexts
    )
    
    logger.info(f"RAGAgent successfully generated answer for query: '{question}'")
    return response
