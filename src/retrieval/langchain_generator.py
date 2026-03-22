from typing import Dict, Any, Optional, AsyncIterator
from loguru import logger

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from src.interfaces.retriever_interfaces import ILLMGenerator
from src.domain.models import RetrievedContext
from src.configs.settings import settings

class LangchainGeminiGenerator(ILLMGenerator):
    """
    Implements ILLMGenerator using LangChain's Expression Language (LCEL).
    Constructs an LLM chain consisting of a PromptTemplate, a Gemini model, and a String parser.
    """
    def __init__(self):
        # 1. Define the Prompt Template
        # We explicitly separate Context and Question to guide the LLM's attention.
        template = """You are a highly capable AI assistant specializing in corporate and financial compliance data.
        
Use the following retrieved context chunks to answer the user's question. 
If the answer is not explicitly contained in the context, do not guess; simply state that the information is not available in the provided documents.

Context:
{context}

Question:
{question}

Answer:"""
        self.prompt = ChatPromptTemplate.from_template(template)
        
        # 2. Instantiate the LLM
        # Using the specified Gemini model for the generator (LLM-as-a-Judge model can also be used here)
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_JUDGE_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.0  # Zero temperature for deterministic, factual extraction
        )
        
        # 3. Define the Output Parser
        self.output_parser = StrOutputParser()
        
        # 4. Construct the LCEL Chain
        self.chain = self.prompt | self.llm | self.output_parser
        
    def _format_context(self, contexts: list) -> str:
        """Helper to flatten RetrievedContext objects into a single readable string."""
        formatted = []
        for i, ctx in enumerate(contexts):
            # Assumes ctx is RetrievedContext
            meta_str = ", ".join([f"{k}={v}" for k, v in ctx.metadata.items()])
            formatted.append(f"[Source {i+1} | Score: {ctx.similarity_score:.4f} | {meta_str}]\n{ctx.text}")
        return "\n\n".join(formatted)

    async def ainvoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None, **kwargs) -> str:
        """
        Expects input dict containing:
        - 'context': List[RetrievedContext]
        - 'question': str
        """
        logger.debug(f"LLM Generator ainvoke triggered for question: {input.get('question')}")
        
        raw_contexts = input.get("context", [])
        formatted_context = self._format_context(raw_contexts)
        
        chain_input = {
            "context": formatted_context,
            "question": input.get("question")
        }
        
        logger.info(f"Generating answer using {settings.LLM_JUDGE_MODEL} via LCEL chain...")
        result = await self.chain.ainvoke(chain_input, config=config, **kwargs)
        return result
        
    async def astream(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None, **kwargs) -> AsyncIterator[str]:
        raw_contexts = input.get("context", [])
        formatted_context = self._format_context(raw_contexts)
        
        chain_input = {
            "context": formatted_context,
            "question": input.get("question")
        }
        
        logger.info(f"Streaming answer using {settings.LLM_JUDGE_MODEL} via LCEL chain...")
        async for chunk in self.chain.astream(chain_input, config=config, **kwargs):
            yield chunk

    def invoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None, **kwargs) -> str:
        raise NotImplementedError("LangchainGeminiGenerator is fully async. Use ainvoke instead.")
