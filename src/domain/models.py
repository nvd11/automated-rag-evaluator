from pydantic import BaseModel, Field
from typing import List, Optional

class Chunk(BaseModel):
    """Represents a chunk of text extracted from a document."""
    text: str = Field(description="The actual text content of the chunk")
    page_number: int = Field(description="The page number where this chunk originated")
    chunk_index: int = Field(description="The sequential index of this chunk within the document")
    token_count: Optional[int] = Field(default=None, description="The estimated token count of the chunk")
    embedding: Optional[List[float]] = Field(default=None, description="The vector representation of the text")

class Document(BaseModel):
    """Represents a parsed document ready for chunking and embedding."""
    document_name: str = Field(description="The name of the file (e.g., HSBC_Annual_Report_2025.pdf)")
    file_path: str = Field(description="The absolute or relative path to the file")
    md5_hash: str = Field(description="MD5 hash of the file for idempotency checks")
    total_pages: int = Field(description="Total number of pages in the document")
    raw_pages_text: List[str] = Field(description="List of text extracted per page, where index = page_number - 1")
    topics: List[str] = Field(default_factory=list, description="List of topics associated with this document for metadata pre-filtering")
    chunks: List[Chunk] = Field(default_factory=list, description="The processed chunks of this document")
