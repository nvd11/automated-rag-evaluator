import asyncio
from typing import Dict, Any, Optional, AsyncIterator
from loguru import logger

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel

from src.interfaces.retriever_interfaces import ILLMGenerator
from src.domain.models import RetrievedContext
from src.configs.settings import settings

class LangchainRAGGenerator(ILLMGenerator):
  """
  Implements ILLMGenerator using LangChain's Expression Language (LCEL).
  Constructs an LLM chain consisting of a PromptTemplate, a provided BaseChatModel, and a String parser.
  """
  def __init__(self, llm: BaseChatModel):
    # 1. Define the Prompt Template
    # We explicitly separate Context and Question to guide the LLM's attention.
    template = """You are a very capable AI assistant specializing in corporate and financial compliance data.
    
Use the following retrieved context chunks to answer the user's question. 
If the answer is not explicitly contained in the context, do not guess; simply state that the information is not available in the provided documents.

Context:
{context}

Question:
{question}

Answer:"""
    self.prompt = ChatPromptTemplate.from_template(template)
    
    # 2. Store the injected LLM instance
    self.llm = llm
    
    # 3. Define the Output Parser
    self.output_parser = StrOutputParser()
    
    # 4. Construct the LCEL Chain
    self.chain = self.prompt | self.llm | self.output_parser
    
  def _format_context(self, contexts: list) -> str:
    """Helper to flatten RetrievedContext objects into a single readable string."""
    formatted = []
    for i, ctx in enumerate(contexts):
      # Assumes ctx is RetrievedContext
      meta_str = ", ".join([f"{k}={v}" for k, v in ctx.metadata.items()])
      formatted.append(f"[Source {i+1} | Score: {ctx.similarity_score:.4f} | {meta_str}]\n{ctx.text}")
    return "\n\n".join(formatted)

  async def ainvoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None, **kwargs) -> str:
    """
    Expects input dict containing:
    - 'context': List[RetrievedContext]
    - 'question': str
    """
    logger.debug(f"LLM Generator ainvoke triggered for question: {input.get('question')}")
    
    raw_contexts = input.get("context", [])
    formatted_context = self._format_context(raw_contexts)
    
    chain_input = {
      "context": formatted_context,
      "question": input.get("question")
    }
    
    model_name = getattr(self.llm, "model_name", getattr(self.llm, "model", "Unknown Model"))

    if settings.ENABLE_PROXY:
      logger.info(f"Generating answer using {model_name} via LCEL chain (Sync in Thread for Proxy Compatibility)...")
      # By using asyncio.to_thread with chain.invoke(), we force the REST transport 
      # to use the synchronous requests library, which correctly respects HTTP proxies
      # and avoids the proxy-ignoring behavior of aiohttp.
      result = await asyncio.to_thread(self.chain.invoke, chain_input, config=config, **kwargs)
    else:
      logger.info(f"Generating answer using {model_name} via LCEL chain (Native Async)...")
      result = await self.chain.ainvoke(chain_input, config=config, **kwargs)
      
    return result
    
  async def astream(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None, **kwargs) -> AsyncIterator[str]:
    raw_contexts = input.get("context", [])
    formatted_context = self._format_context(raw_contexts)
    
    chain_input = {
      "context": formatted_context,
      "question": input.get("question")
    }
    
    model_name = getattr(self.llm, "model_name", getattr(self.llm, "model", "Unknown Model"))
    
    if settings.ENABLE_PROXY:
      logger.warning(f"Streaming answer using {model_name} via LCEL chain (Proxy Fallback to Threaded Sync)...")
      # If we try to use native astream, it might hang via gRPC. 
      # Safest proxy workaround is to run the sync invoke in a thread and yield the final result.
      result = await asyncio.to_thread(self.chain.invoke, chain_input, config=config, **kwargs)
      yield result
    else:
      logger.info(f"Streaming answer using {model_name} via LCEL chain (Native Async)...")
      async for chunk in self.chain.astream(chain_input, config=config, **kwargs):
        yield chunk

  def invoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None, **kwargs) -> str:
    raise NotImplementedError("LangchainRAGGenerator is fully async. Use ainvoke instead.")
