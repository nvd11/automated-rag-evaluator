from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger

from src.configs.settings import settings
from src.llm.llm_factory import ILLMFactory

class GeminiLLMFactory(ILLMFactory):
    """
    Concrete Factory implementation specifically for Google's Gemini models.
    Encapsulates vendor-specific logic such as proxy transport configurations
    and API key injection.
    """
    
    def create_llm(self, model_name: str, temperature: float = 0.0, **kwargs) -> BaseChatModel:
        logger.debug(f"[GeminiLLMFactory] Assembling LLM instance for model: {model_name} (Temp: {temperature})")
        
        # Base configuration required by the LangChain Google integration
        llm_kwargs = {
            "model": model_name,
            "google_api_key": settings.GEMINI_API_KEY,
            "temperature": temperature,
        }
        
        # Vendor-specific logic: Handle networking proxies if deployed in restricted regions
        if settings.ENABLE_PROXY:
            logger.debug(f"[GeminiLLMFactory] Applying REST transport for {model_name} due to proxy configuration.")
            llm_kwargs["transport"] = "rest"
            
        # Allow caller to override or append specific arguments
        llm_kwargs.update(kwargs)
        
        return ChatGoogleGenerativeAI(**llm_kwargs)
