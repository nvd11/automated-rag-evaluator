from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger

from src.configs.settings import settings

class LLMFactory:
    """
    Centralized factory for creating configured LLM instances.
    Provides Dependency Injection capabilities to decouple domain logic from specific model providers.
    """
    
    @staticmethod
    def create_llm(model_name: str, temperature: float = 0.0, **kwargs) -> BaseChatModel:
        """
        Creates and returns a LangChain BaseChatModel instance.
        
        Args:
            model_name: The identifier of the model (e.g., 'gemini-2.5-pro', 'deepseek-chat').
            temperature: Sampling temperature for generation.
            **kwargs: Additional provider-specific configurations.
        """
        logger.debug(f"Factory assembling LLM instance for model: {model_name} (Temp: {temperature})")
        
        # ---------------------------------------------------------------------
        # Provider: Google Gemini
        # ---------------------------------------------------------------------
        if "gemini" in model_name.lower():
            llm_kwargs = {
                "model": model_name,
                "google_api_key": settings.GEMINI_API_KEY,
                "temperature": temperature,
            }
            
            # Apply dynamic transport logic for proxy compatibility
            if settings.ENABLE_PROXY:
                logger.debug(f"Applying REST transport for {model_name} due to proxy configuration.")
                llm_kwargs["transport"] = "rest"
                
            # Merge any additional kwargs
            llm_kwargs.update(kwargs)
            
            return ChatGoogleGenerativeAI(**llm_kwargs)
            
        # ---------------------------------------------------------------------
        # Future-proofing: Example stub for DeepSeek or OpenAI
        # ---------------------------------------------------------------------
        # elif "deepseek" in model_name.lower():
        #     from langchain_openai import ChatOpenAI
        #     return ChatOpenAI(
        #         model=model_name,
        #         api_key=settings.DEEPSEEK_API_KEY,
        #         base_url="https://api.deepseek.com/v1",
        #         temperature=temperature,
        #         **kwargs
        #     )
            
        else:
            raise ValueError(f"Unsupported model provider for model_name: {model_name}")

