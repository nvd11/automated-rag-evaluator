import hashlib
import fitz # PyMuPDF
import os
from loguru import logger
from src.domain.models import Document
from src.interfaces.ingestion_interfaces import BaseLoader

class PyMuPDFLoader(BaseLoader):
  """
  PyMuPDF loader for fast
  PDF extraction.
  """
  async def load(self, file_path: str) -> Document:
    if not os.path.exists(file_path):
      raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading PDF: {file_path}")
    
    # Calculate MD5 for idempotency
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
      for chunk in iter(lambda: f.read(4096), b""):
        md5_hash.update(chunk)
    file_md5 = md5_hash.hexdigest()

    document_name = os.path.basename(file_path)
    raw_pages_text = []

    # Extract text page by page
    doc = fitz.open(file_path)
    total_pages = len(doc)
    for page_num in range(total_pages):
      page = doc.load_page(page_num)
      text = page.get_text("text")
      raw_pages_text.append(text)
    doc.close()

    logger.debug(f"Extracted {total_pages} pages from {document_name}")

    return Document(
      document_name=document_name,
      file_path=file_path,
      md5_hash=file_md5,
      total_pages=total_pages,
      raw_pages_text=raw_pages_text
    )
