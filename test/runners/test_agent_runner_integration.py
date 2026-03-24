import pytest
import asyncio
from src.runners.agent_runner import main as runner_main

@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_runner_integration_end_to_end():
  """
  Executes the actual agent_runner.py main function without any mocking.
  This connects to the real GCP Cloud SQL database and real Gemini 2.5 Pro API.
  Used to verify the end-to-end RAG system functionality in a production-like environment.
  """
  try:
    # Run the actual runner which initializes the DB pool, creates components,
    # performs retrieval, calls the LLM, and logs the output.
    await runner_main()
    # If it completes without raising an exception, the integration is successful.
    assert True
  except Exception as e:
    pytest.fail(f"Agent runner failed with exception: {e}")
