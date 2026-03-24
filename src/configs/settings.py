import os
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource
from pydantic import Field
from .log_config import setup_logging
from loguru import logger

class Settings(BaseSettings):
  # App Config
  APP_ENV: str = Field("dev", alias="APP_ENVIRONMENT")
  
  # Database Config
  DB_HOST: str
  DB_PORT: int = 5432
  DB_NAME: str
  DB_USER: str
  DB_PASSWORD: str
  
  # Google AI Config (API Key)
  GEMINI_API_KEY: str
  EMBEDDING_MODEL: str = "models/gemini-embedding-001"
  EMBEDDING_DIMENSION: int = 768
  EMBEDDING_BATCH_SIZE: int = 100
  
  # Text Chunking Config
  CHUNK_SIZE: int = 1000
  CHUNK_OVERLAP: int = 200
  
  # Retrieval Config
  RETRIEVAL_TOP_K: int = 5
  RETRIEVAL_SIMILARITY_THRESHOLD: float = 0.5
  LLM_INFERENCE_MODEL: str = "gemini-2.5-pro"
  LLM_JUDGE_MODEL: str = "gemini-2.5-pro"
  LLM_TEACHER_MODEL: str = "gemini-3.1-pro-preview"

  # Proxy Config (Global toggle for LLM/Embeddings access from restricted regions)
  ENABLE_PROXY: bool = False
  HTTP_PROXY: str | None = None
  HTTPS_PROXY: str | None = None

  # Note on Pydantic V2 Architecture:
  # 'model_config' is a reserved class variable used by Pydantic's internal Rust engine.
  # It acts as a configuration blueprint. When 'Settings()' is instantiated, the engine implicitly
  # reads this dictionary to determine how to parse the '.env' file, cast types, and handle extra fields.
  # This eliminates the need for manual 'load_dotenv()' boilerplate.
  model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore"
  )

  @classmethod
  def settings_customise_sources(
    cls,
    settings_cls: type[BaseSettings],
    init_settings: PydanticBaseSettingsSource,
    env_settings: PydanticBaseSettingsSource,
    dotenv_settings: PydanticBaseSettingsSource,
    file_secret_settings: PydanticBaseSettingsSource,
  ) -> tuple[PydanticBaseSettingsSource, ...]:
    # Priority override: make .env file values override OS environment variables
    return (init_settings, dotenv_settings, env_settings, file_secret_settings)


settings = Settings()

# Initialize global logging configuration immediately after settings are loaded
# This ensures all modules importing 'settings' automatically get the correct log format and level
setup_logging(settings.APP_ENV)


# Apply Global Proxy if ENABLE_PROXY is true
if settings.ENABLE_PROXY:
  logger.info("Global proxy enabled. Setting HTTP_PROXY, HTTPS_PROXY and GRPC_PROXY environment variables.")
  logger.info(f"HTTP_PROXY: {settings.HTTP_PROXY}")
  logger.info(f"HTTPS_PROXY: {settings.HTTPS_PROXY}")
  if settings.HTTP_PROXY:
    os.environ["http_proxy"] = settings.HTTP_PROXY
    os.environ["HTTP_PROXY"] = settings.HTTP_PROXY
    os.environ["grpc_proxy"] = settings.HTTP_PROXY
    os.environ["GRPC_PROXY"] = settings.HTTP_PROXY
  if settings.HTTPS_PROXY:
    os.environ["https_proxy"] = settings.HTTPS_PROXY
    os.environ["HTTPS_PROXY"] = settings.HTTPS_PROXY



