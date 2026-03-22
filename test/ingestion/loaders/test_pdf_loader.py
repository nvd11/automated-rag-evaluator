import pytest
import hashlib
from unittest.mock import patch, mock_open, MagicMock

from src.ingestion.loaders.pdf_loader import PyMuPDFLoader
from src.domain.models import Document

@pytest.fixture
def mock_file_content():
    return b"dummy pdf content"

@pytest.fixture
def mock_file_md5(mock_file_content):
    return hashlib.md5(mock_file_content).hexdigest()

@pytest.mark.asyncio
@patch("src.ingestion.loaders.pdf_loader.os.path.exists")
@patch("src.ingestion.loaders.pdf_loader.fitz.open")
async def test_load_valid_pdf(mock_fitz_open, mock_exists, mock_file_content, mock_file_md5):
    """
    Test loading a valid PDF file.
    Expects correct MD5 calculation, proper page parsing, and a well-formed Document object.
    """
    # Arrange
    file_path = "/fake/dir/dummy_report.pdf"
    mock_exists.return_value = True

    # Mock the fitz Document
    mock_pdf_doc = MagicMock()
    mock_pdf_doc.__len__.return_value = 2  # 2 pages
    
    # Mock pages
    mock_page_0 = MagicMock()
    mock_page_0.get_text.return_value = "Page 1 Content"
    mock_page_1 = MagicMock()
    mock_page_1.get_text.return_value = "Page 2 Content"
    
    # Configure load_page to return the mocked pages sequentially
    mock_pdf_doc.load_page.side_effect = [mock_page_0, mock_page_1]
    mock_fitz_open.return_value = mock_pdf_doc

    loader = PyMuPDFLoader()

    # Act
    # We patch `builtins.open` to simulate reading the file for MD5 calculation
    with patch("builtins.open", mock_open(read_data=mock_file_content)):
        document = await loader.load(file_path)

    # Assert
    mock_exists.assert_called_once_with(file_path)
    mock_fitz_open.assert_called_once_with(file_path)
    
    assert isinstance(document, Document)
    assert document.document_name == "dummy_report.pdf"
    assert document.file_path == file_path
    assert document.md5_hash == mock_file_md5
    assert document.total_pages == 2
    assert document.raw_pages_text == ["Page 1 Content", "Page 2 Content"]

@pytest.mark.asyncio
@patch("src.ingestion.loaders.pdf_loader.os.path.exists")
async def test_load_file_not_found(mock_exists):
    """
    Test loading a non-existent PDF file.
    Expects a FileNotFoundError to be raised.
    """
    # Arrange
    file_path = "/fake/dir/missing.pdf"
    mock_exists.return_value = False
    
    loader = PyMuPDFLoader()

    # Act & Assert
    with pytest.raises(FileNotFoundError, match=f"File not found: {file_path}"):
        await loader.load(file_path)
