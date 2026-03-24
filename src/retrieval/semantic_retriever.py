from typing import List, Optional, Any, Dict
from loguru import logger
from langchain_core.runnables import RunnableConfig

from src.interfaces.retriever_interfaces import BaseRetriever, IRetrieverDAO
from src.interfaces.ingestion_interfaces import BaseEmbedder
from src.domain.models import SearchQuery, RetrievedContext
from src.configs.settings import settings

class SemanticRetriever(BaseRetriever):
  """
  Orchestrates the retrieval process by generating embeddings for the query
  and fetching relevant contexts from the DAO.
  Implements the LangChain Runnable protocol via BaseRetriever.
  """
  def __init__(self, embedder: BaseEmbedder, dao: IRetrieverDAO):
    self.embedder = embedder
    self.dao = dao

  async def ainvoke(self, input: str, config: Optional[RunnableConfig] = None, **kwargs) -> List[RetrievedContext]:
    """
    Takes a raw query string, generates its embedding, and retrieves relevant chunks.
    
    config can contain:
    - top_k: int
    - similarity_threshold: float
    - topic_filters: List[str]
    """
    logger.info(f"Retrieving context for query: '{input}'")
    
    # Parse config for overrides, fallback to global settings
    top_k = settings.RETRIEVAL_TOP_K
    similarity_threshold = settings.RETRIEVAL_SIMILARITY_THRESHOLD
    topic_filters = None
    
    if config and "configurable" in config:
      conf = config["configurable"]
      top_k = conf.get("top_k", top_k)
      similarity_threshold = conf.get("similarity_threshold", similarity_threshold)
      topic_filters = conf.get("topic_filters", topic_filters)

    # 1. Embed the query
    logger.debug("Generating query embedding...")
    query_vector = await self.embedder.embed_query(input)
    
    # 2. Construct DTO
    query_dto = SearchQuery(
      query_text=input,
      embedding=query_vector,
      top_k=top_k,
      similarity_threshold=similarity_threshold,
      topic_filters=topic_filters
    )
    
    # 3. Perform search
    logger.debug("Executing vector search via DAO...")
    results = await self.dao.semantic_search(query_dto)
    
    return results

  def invoke(self, input: str, config: Optional[RunnableConfig] = None, **kwargs) -> List[RetrievedContext]:
    raise NotImplementedError("SemanticRetriever is fully async. Use ainvoke instead.")
