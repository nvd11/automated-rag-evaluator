from typing import List
import asyncio
from google import genai
from google.genai import types
from loguru import logger
from tenacity import retry, wait_exponential, stop_after_attempt
from src.domain.models import Chunk
from src.interfaces.ingestion_interfaces import BaseEmbedder
from src.configs.settings import settings

class GeminiEmbedder(BaseEmbedder):
    """
    Implementation of BaseEmbedder using modern google-genai SDK.
    Features async thread-pool offloading and exponential backoff retries.
    """
    def __init__(self, batch_size: int = 100):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_name = settings.EMBEDDING_MODEL
        self.batch_size = batch_size

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
    async def _embed_text_batch(self, texts: List[str]) -> List[List[float]]:
        # The genai SDK's embed_content is synchronous.
        # We wrap it in run_in_executor to make it fully non-blocking in our Async flow.
        loop = asyncio.get_running_loop()
        
        def do_embed():
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            # Extracted list of float vectors
            return [emb.values for emb in response.embeddings]

        return await loop.run_in_executor(None, do_embed)

    async def embed_batch(self, chunks: List[Chunk]) -> List[Chunk]:
        logger.info(f"Generating embeddings for {len(chunks)} chunks using {self.model_name}")
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            texts = [chunk.text for chunk in batch]
            
            try:
                embeddings = await self._embed_text_batch(texts)
                for chunk, emb in zip(batch, embeddings):
                    chunk.embedding = emb
            except Exception as e:
                logger.error(f"Failed to embed batch starting at index {i}: {e}")
                raise
                
        return chunks
