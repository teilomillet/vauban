import pytest
from vauban.target import Target, ModelTarget, OpenAITarget


class CustomTarget:
    def invoke(self, prompt: str) -> str:
        return "custom response"

    async def invoke_async(self, prompt: str) -> str:
        return "custom async response"


def test_target_protocol():
    # Verify CustomTarget satisfies the protocol
    target = CustomTarget()
    assert isinstance(target, Target)
    assert target.invoke("test") == "custom response"


@pytest.mark.asyncio
async def test_target_protocol_async():
    target = CustomTarget()
    response = await target.invoke_async("test")
    assert response == "custom async response"


def test_openai_target_init():
    target = ModelTarget(model_name="gpt-3.5-turbo")
    assert target.model_name == "gpt-3.5-turbo"


def test_openai_target_alias():
    # Ensure the legacy name still resolves to the renamed class
    assert OpenAITarget is ModelTarget
