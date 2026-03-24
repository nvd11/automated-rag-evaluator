import uuid
import asyncio
from loguru import logger

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import ValidationError

from src.interfaces.evaluator_interfaces import IDatasetGenerator
from src.domain.models import Chunk, QA_Pair, GoldenRecord
from src.configs.settings import settings

class LangchainDatasetGenerator(IDatasetGenerator):
    """
    Synthesizes High-Quality Ground Truth (Golden Records) for RAG Evaluation.
    Employs an advanced LLM ("The Teacher") via Dependency Injection and uses
    Structured Output (Function Calling) to enforce a reliable JSON schema.
    """
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        
        # 1. Define the rigorous Persona and Task for the LLM
        template = """You are an elite Financial Compliance Auditor and SME (Subject Matter Expert) creating a benchmark dataset to evaluate an enterprise AI assistant.
        
Your task is to analyze the following raw text snippet from a corporate report and synthesize ONE highly professional, specific, and unambiguous question that can ONLY be answered using the facts presented in this exact snippet.

Then, provide the exact, factual, and concise answer to your question based purely on the text. DO NOT introduce outside knowledge.

If the text snippet is too short, contains only boilerplate (e.g., page numbers, disclaimers), or lacks meaningful factual content to form a good question, you must still output a valid response, but set the question to "INVALID_CHUNK" and the answer to "REJECTED".

Source Text Snippet:
{text}
"""
        self.prompt = ChatPromptTemplate.from_template(template)
        
        # 2. Bind the LLM to the Pydantic schema to force Structured JSON Output.
        # This completely eliminates parsing errors common with StrOutputParser.
        self.structured_llm = self.llm.with_structured_output(QA_Pair)
        
        # 3. Assemble the LCEL Chain
        self.chain = self.prompt | self.structured_llm

    async def agenerate_qa_from_chunk(self, chunk: Chunk, batch_name: str) -> GoldenRecord | None:
        """
        Transforms a raw database Chunk into a structured GoldenRecord by orchestrating the LLM.
        Returns None if the LLM determines the chunk is unsuitable for question generation.
        """
        logger.debug(f"Generator evaluating Chunk ID: {getattr(chunk, 'chunk_index', 'Unknown')} (Page {chunk.page_number})...")
        
        try:
            # 1. Execute the LCEL Chain
            # The result here is guaranteed by LangChain to be a validated QA_Pair object
            if settings.ENABLE_PROXY:
                logger.debug(f"Executing LCEL Chain for Page {chunk.page_number} (Sync in Thread for Proxy Compatibility)...")
                # Using asyncio.to_thread forces the REST transport to use the synchronous requests library,
                # which correctly respects HTTP proxies and avoids gRPC timeouts behind GFW.
                qa_result: QA_Pair = await asyncio.to_thread(self.chain.invoke, {"text": chunk.text})
            else:
                logger.debug(f"Executing LCEL Chain for Page {chunk.page_number} (Native Async)...")
                qa_result: QA_Pair = await self.chain.ainvoke({"text": chunk.text})
            
            # 2. Handle LLM rejection of poor-quality chunks
            if "INVALID_CHUNK" in qa_result.question or qa_result.answer == "REJECTED":
                logger.warning(f"LLM rejected Chunk on Page {chunk.page_number} as unsuitable for QA generation. Skipping.")
                return None
                
            # 3. Map the LLM's QA_Pair back into our Enterprise Domain Model (GoldenRecord)
            # We inject the orchestrator's metadata (batch_name, ID, topics) here.
            
            # Note: For this simplified implementation, we aren't extracting specific expected_topics 
            # via a separate LLM call, so we leave it empty or default it. In a full production system, 
            # we might prompt the LLM to also classify the topic (e.g., 'Risk', 'Finance').
            record = GoldenRecord(
                id=str(uuid.uuid4()),
                batch_name=batch_name,
                question=qa_result.question,
                ground_truth=qa_result.answer,
                expected_topics=[],  # Can be enhanced later
                complexity=qa_result.complexity
            )
            
            logger.info(f"Successfully generated a '{record.complexity}' level question for Page {chunk.page_number}.")
            return record
            
        except ValidationError as e:
            logger.error(f"Structured Output Validation Failed for chunk on Page {chunk.page_number}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during QA generation for chunk on Page {chunk.page_number}: {e}")
            return None
