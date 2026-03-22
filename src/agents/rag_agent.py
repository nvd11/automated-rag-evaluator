from typing import List, Optional, Any, Dict
from loguru import logger

from src.interfaces.retriever_interfaces import BaseRetriever, ILLMGenerator
from src.domain.models import RAGResponse

class RAGAgent:
    """
    The top-level orchestrator for the Retrieval-Augmented Generation workflow.
    Combines a Semantic Retriever and an LLM Generator to produce a structured RAGResponse.
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
        Executes the end-to-end RAG pipeline using a unified LCEL chain.
        Returns a structured RAGResponse containing the generated answer and retrieved contexts.
        """
        logger.info(f"RAGAgent processing query: '{question}'")
        
        # 1. Prepare dynamic retrieval configuration
        retriever_config = {}
        if top_k is not None: retriever_config["top_k"] = top_k
        if similarity_threshold is not None: retriever_config["similarity_threshold"] = similarity_threshold
        if topic_filters is not None: retriever_config["topic_filters"] = topic_filters
        
        config_wrapper = {"configurable": retriever_config} if retriever_config else None

        # 2. Phase 1: Explicit Retrieval (Intercepted for DTO packaging & Early Exit)
        logger.debug("Executing SemanticRetriever to fetch raw context objects...")
        retrieved_contexts = await self.retriever.ainvoke(input=question, config=config_wrapper)
        
        if not retrieved_contexts:
            logger.warning(f"No relevant context found for query: '{question}'")
            return RAGResponse(
                query=question,
                generated_answer="I'm sorry, I couldn't find any relevant information in the knowledge base to answer your question.",
                retrieved_contexts=[]
            )

        # 3. Phase 2: Generation via LCEL
        # Note on LCEL: While we *could* pipe the retriever directly into the generator using RunnablePassthrough, 
        # we explicitly execute the retriever first in Python to intercept the `retrieved_contexts` List[RetrievedContext] objects.
        # This is critical for our Automated Evaluator: we must package the exact chunk metadata and similarity scores
        # alongside the final answer into the `RAGResponse` DTO for downstream Faithfulness and Context Relevance grading.
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
