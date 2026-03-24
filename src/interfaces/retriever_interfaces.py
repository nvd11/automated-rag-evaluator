from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict, AsyncIterator
from langchain_core.runnables import Runnable, RunnableConfig
from src.domain.models import SearchQuery, RetrievedContext, RAGResponse

class IRetrieverDAO(ABC):
  """Contract for the Data Access Object that handles vector similarity search."""
  @abstractmethod
  async def semantic_search(self, query: SearchQuery) -> List[RetrievedContext]:
    pass

class BaseRetriever(Runnable, ABC):
  """
  Contract for the Retriever Orchestrator. 
  By inheriting from LangChain's Runnable, this class natively supports 
  .invoke(), .ainvoke(), and can be easily piped into LCEL chains.
  """
  @abstractmethod
  async def ainvoke(self, input: str, config: Optional[RunnableConfig] = None, **kwargs) -> List[RetrievedContext]:
    """
    The standard LangChain async entry point.
    Takes a raw query string, embeds it, and returns the relevant context chunks.
    config can contain 'top_k', 'similarity_threshold', and 'topic_filters'.
    """
    pass

class ILLMGenerator(Runnable, ABC):
  """
  Contract for the LLM component.
  Inherits from Runnable to participate natively in LCEL pipelines.
  """
  @abstractmethod
  async def ainvoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None, **kwargs) -> str:
    """
    Takes a dictionary (typically containing 'context' and 'question'),
    formats the prompt, and generates the final answer.
    """
    pass
    
  @abstractmethod
  async def astream(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None, **kwargs) -> AsyncIterator[str]:
    """Supports streaming the generated answer token-by-token."""
    pass
