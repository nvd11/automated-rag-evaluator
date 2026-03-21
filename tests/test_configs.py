import pytest
from loguru import logger
from src.configs.settings import settings

def test_hello_world_config():
    """
    Test that the Pydantic settings load correctly and log a hello world message.
    """
    logger.info("Hello World! This is pytest speaking.")
    logger.info(f"Loaded GCP Project: {settings.GCP_PROJECT}")
    logger.info(f"Loaded Database Host: {settings.DB_HOST}")
    
    # Simple assertion to ensure settings are loaded
    assert settings.GCP_PROJECT is not None
    assert settings.DB_HOST is not None
