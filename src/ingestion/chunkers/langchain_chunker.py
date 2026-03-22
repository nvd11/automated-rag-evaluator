from typing import List
from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.domain.models import Document, Chunk
from src.interfaces.ingestion_interfaces import BaseChunker
from src.configs.settings import settings

class LangchainRecursiveChunker(BaseChunker):
    """
    Langchain RecursiveCharacterTextSplitter wrapper.
    
    """
    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, chunk_overlap: int = settings.CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk(self, document: Document) -> List[Chunk]:
        logger.info(f"Chunking document with chunk_size={self.splitter._chunk_size}")
        
        chunks = []
        global_chunk_index = 0
        
        for page_idx, page_text in enumerate(document.raw_pages_text):
            page_number = page_idx + 1
            
            # Skip empty pages
            if not page_text.strip():
                continue
                
            texts = self.splitter.split_text(page_text)
            for text in texts:
                chunks.append(Chunk(
                    text=text,
                    page_number=page_number,
                    chunk_index=global_chunk_index
                ))
                global_chunk_index += 1
                
        logger.debug(f"Produced {len(chunks)} chunks in total.")
        return chunks
