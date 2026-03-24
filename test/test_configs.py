import pytest
from loguru import logger
from src.configs.settings import settings

def test_hello_world_config():
  """
  Test that the Pydantic settings load correctly and log a hello world message.
  """
  logger.info("Hello World! This is pytest speaking.")
  
  # We now verify the API Key setting instead of the GCP Project
  api_key_status = "Configured" if settings.GEMINI_API_KEY else "Missing"
  logger.info(f"Loaded Google AI API Key Status: {api_key_status}")
  logger.info(f"Loaded Database Host: {settings.DB_HOST}")
  
  # Simple assertion to ensure critical settings are loaded
  assert settings.GEMINI_API_KEY is not None
  assert settings.DB_HOST is not None
