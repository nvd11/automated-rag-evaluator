import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

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
    LLM_JUDGE_MODEL: str = "gemini-1.5-pro"
    
    # Proxy Config (Global toggle for LLM/Embeddings access from restricted regions)
    ENABLE_PROXY: bool = False
    HTTP_PROXY: str | None = None
    HTTPS_PROXY: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Apply Global Proxy if ENABLE_PROXY is true
if settings.ENABLE_PROXY:
    if settings.HTTP_PROXY:
        os.environ["http_proxy"] = settings.HTTP_PROXY
        os.environ["HTTP_PROXY"] = settings.HTTP_PROXY
    if settings.HTTPS_PROXY:
        os.environ["https_proxy"] = settings.HTTPS_PROXY
        os.environ["HTTPS_PROXY"] = settings.HTTPS_PROXY
