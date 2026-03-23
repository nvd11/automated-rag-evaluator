from abc import ABC, abstractmethod
from langchain_core.language_models.chat_models import BaseChatModel

class ILLMFactory(ABC):
    """
    Abstract Factory Interface for creating LLM instances.
    Enforces the Dependency Inversion Principle, ensuring that the RAG pipeline
    is decoupled from any specific model provider's instantiation logic.
    """
    
    @abstractmethod
    def create_llm(self, model_name: str, temperature: float = 0.0, **kwargs) -> BaseChatModel:
        """
        Creates and returns a LangChain BaseChatModel instance.
        
        Args:
            model_name: The identifier of the model (e.g., 'gemini-2.5-pro', 'gpt-4o').
            temperature: Sampling temperature for generation.
            **kwargs: Additional provider-specific configurations to override defaults.
        """
        pass
