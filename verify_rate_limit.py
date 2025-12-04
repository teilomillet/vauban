import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch
from openai import RateLimitError
from vauban.llm import LLM
from vauban.strategies.gepa.population import GEPAPopulationManager

async def test_retry_logic():
    print("Testing LLM retry logic...")
    
    # Mock AsyncOpenAI client
    mock_client = AsyncMock()
    
    # Setup mock to raise RateLimitError twice, then succeed
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Success"))]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 10
    
    # Side effect: Raise 429, Raise 429, Return Success
    error = RateLimitError(message="Rate limit exceeded", response=MagicMock(), body=None)
    mock_client.chat.completions.create.side_effect = [error, error, mock_response]
    
    llm = LLM(model_name="test-model")
    # Inject mock client directly
    llm._client = mock_client
    
    start_time = time.time()
    with open("/Users/teilo.millet/Code/vauban/verify_output.txt", "w") as f:
        f.write(f"LLM call succeeded after {elapsed:.2f}s\n")
        f.write(f"Result: {result.data}\n")
        f.write(f"Call count: {call_count}\n")
        
        if call_count == 3:
            f.write("SUCCESS: Retried correctly.\n")
            print("SUCCESS: Retried correctly.")
        else:
            f.write(f"FAILURE: Expected 3 calls, got {call_count}\n")
            print(f"FAILURE: Expected 3 calls, got {call_count}")

    with open("/Users/teilo.millet/Code/vauban/verify_output.txt", "a") as f:
        f.write(f"\nPopulation generation took {elapsed:.2f}s\n")
        if elapsed >= 0.5:
            f.write("SUCCESS: Concurrency limited (took >= 0.5s)\n")
            print("SUCCESS: Concurrency limited (took >= 0.5s)")
        else:
            f.write(f"FAILURE: Too fast! ({elapsed:.2f}s) - Concurrency limit might be broken.\n")
            print(f"FAILURE: Too fast! ({elapsed:.2f}s) - Concurrency limit might be broken.")

if __name__ == "__main__":
    asyncio.run(test_retry_logic())
    asyncio.run(test_concurrency_limit())
