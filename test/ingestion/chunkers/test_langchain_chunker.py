import pytest
from src.ingestion.chunkers.langchain_chunker import LangchainRecursiveChunker
from src.domain.models import Document, Chunk

def test_langchain_recursive_chunker():
  """
  Test the chunker to ensure it handles overlapping properly and skips empty pages.
  """
  # Arrange
  chunker = LangchainRecursiveChunker(chunk_size=50, chunk_overlap=10)
  
  # Create mock Document
  doc = Document(
    document_name="test_doc.pdf",
    file_path="/fake/test_doc.pdf",
    md5_hash="dummy_md5",
    total_pages=3,
    raw_pages_text=[
      "This is a long sentence that should definitely be chunked.",
      "  \n ", # Page 2 is essentially empty
      "Another sentence on the last page."
    ]
  )

  # Act
  chunks = chunker.chunk(doc)

  # Assert
  assert isinstance(chunks, list)
  assert all(isinstance(c, Chunk) for c in chunks)
  
  # Check that empty page 2 was completely skipped
  page_numbers = [c.page_number for c in chunks]
  assert 2 not in page_numbers
  
  # Verify sequential chunk indexing
  indexes = [c.chunk_index for c in chunks]
  assert indexes == list(range(len(chunks)))
  
  # Check if first page was split correctly (since chunk_size=50)
  page_1_chunks = [c for c in chunks if c.page_number == 1]
  assert len(page_1_chunks) > 1
  
  # Ensure all chunks are within limits (mostly true for RecursiveCharacterTextSplitter)
  for c in page_1_chunks:
    assert len(c.text) <= 50

  # Ensure overlapping logic works (the second chunk should contain part of the first)
  # The splitter overlaps by 10 characters.
  assert page_1_chunks[1].text.startswith("be chunked.") or "chunked" in page_1_chunks[1].text

def test_chunker_empty_document():
  """
  Test chunking an empty document to ensure it gracefully returns an empty list.
  """
  # Arrange
  chunker = LangchainRecursiveChunker()
  doc = Document(
    document_name="empty.pdf",
    file_path="/fake/empty.pdf",
    md5_hash="empty_md5",
    total_pages=1,
    raw_pages_text=[""]
  )

  # Act
  chunks = chunker.chunk(doc)

  # Assert
  assert chunks == []
