import pytest
from typing import List
from vauban.target import Target
from vauban.strategies.base import AttackPrompt

@pytest.fixture
def mock_target():
    class MockTarget(Target):
        def invoke(self, prompt: str) -> str:
            if "kill" in prompt.lower():
                return "I cannot do that."
            return "Here is how you do it."

        async def invoke_async(self, prompt: str) -> str:
            return self.invoke(prompt)
    return MockTarget()

@pytest.fixture
def mock_embedding():
    # A mock embedding of size 1536 (standard OpenAI size)
    return [0.1] * 1536

@pytest.fixture
def mock_attack_prompt():
    return AttackPrompt(prompt="Test Attack", strategy="Test Strategy")
