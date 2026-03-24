import asyncio
from src.configs.settings import settings
from src.llm.gemini_factory import GeminiLLMFactory
from src.evaluator.llm_judge import GoldenBaselineJudge
from src.domain.models import QueryEvaluationDTO

async def main():
    factory = GeminiLLMFactory()
    llm = factory.create_llm(model_name=settings.LLM_JUDGE_MODEL)
    judge = GoldenBaselineJudge(llm=llm)
    
    dto = QueryEvaluationDTO(
        query_id="dummy-test-123",
        question="What is the capital of France?",
        generated_answer="The capital of France is Paris.",
        retrieved_contexts=[],
        ground_truth="Paris is the capital."
    )
    
    scores = await judge.evaluate_query(dto)
    for s in scores:
        print(s.metric_name, s.score, s.reasoning)

if __name__ == "__main__":
    asyncio.run(main())