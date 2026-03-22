from typing import List
import asyncio
from google import genai
from google.genai import types
from loguru import logger
from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm.asyncio import tqdm
from src.domain.models import Chunk
from src.interfaces.ingestion_interfaces import BaseEmbedder
from src.configs.settings import settings

class GeminiEmbedder(BaseEmbedder):
    """
    Gemini Embedder using modern google-genai SDK.
    
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
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768  # Force the model to truncate/project to 768 dimensions
                )
            )
            # Extracted list of float vectors
            return [emb.values for emb in response.embeddings]

        return await loop.run_in_executor(None, do_embed)

    async def embed_batch(self, chunks: List[Chunk]) -> List[Chunk]:
        logger.info(f"Generating embeddings for {len(chunks)} chunks using {self.model_name}")
        
        # Calculate total batches for the progress bar
        batches = [chunks[i:i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]
        
        # We use tqdm to wrap our async iteration
        with tqdm(total=len(chunks), desc="Embedding Chunks", unit="chunk", colour="green") as pbar:
            for batch in batches:
                texts = [chunk.text for chunk in batch]
                
                try:
                    embeddings = await self._embed_text_batch(texts)
                    for chunk, emb in zip(batch, embeddings):
                        chunk.embedding = emb
                    # Update progress bar by the number of chunks processed in this batch
                    pbar.update(len(batch))
                except Exception as e:
                    logger.error(f"Failed to embed batch: {e}")
                    raise
                
        return chunks
