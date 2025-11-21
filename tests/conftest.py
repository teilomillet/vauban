import warnings
from sentry_sdk.hub import SentryHubDeprecationWarning
import pytest
from vauban.target import Target
from vauban.strategies.base import AttackPrompt
import asyncio
import inspect

# Ignore upstream Sentry deprecation noise from weave -> sentry-sdk.
warnings.filterwarnings("ignore", category=SentryHubDeprecationWarning)


# Minimal async support when pytest-asyncio is absent
def pytest_pyfunc_call(pyfuncitem):
    pm = pyfuncitem.config.pluginmanager
    if pm.hasplugin("pytest_asyncio"):
        return None
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        argnames = getattr(pyfuncitem._fixtureinfo, "argnames", ())
        kwargs = {name: pyfuncitem.funcargs[name] for name in argnames}
        asyncio.run(pyfuncitem.obj(**kwargs))
        return True
    return None


def pytest_configure(config):
    # Register asyncio marker to avoid warnings when plugin is absent
    config.addinivalue_line("markers", "asyncio: mark test as asynchronous")
    # Silence SentryHubDeprecationWarning emitted inside weave's Sentry shim (external dependency)
    # This warning comes from sentry-sdk using Hub; not actionable in our code path.
    config.addinivalue_line(
        "filterwarnings", "ignore::sentry_sdk.hub.SentryHubDeprecationWarning"
    )


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
