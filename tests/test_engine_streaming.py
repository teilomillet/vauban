import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from vauban.interfaces import AttackPrompt, AttackResultEvent, Strategy, Judge, Target
from vauban.intel import IntelSystem
from vauban.engine import SiegeEngine

@pytest.mark.asyncio
async def test_engine_streaming():
    """Verify that the engine yields results as they complete, not all at once."""
    
    # Mocks that satisfy Pydantic instance checks
    class MockStrategy(Strategy):
        def generator(self): pass
        async def update(self, results): pass
        
    class MockJudge(Judge):
        async def evaluate(self, *args, **kwargs): pass
        
    class MockTarget(Target):
        async def invoke_async(self, prompt): pass
        
    class MockIntel(IntelSystem):
        embedder = None

    strategy = MagicMock(spec=MockStrategy)
    # Create a generator that yields one squad of 3 attacks
    async def strategy_gen():
        yield [
            AttackPrompt(prompt="p1", strategy="s1"),
            AttackPrompt(prompt="p2", strategy="s2"),
            AttackPrompt(prompt="p3", strategy="s3"),
        ]
    strategy.generator.return_value = strategy_gen()
    strategy.update = AsyncMock()
    
    judge = MagicMock(spec=MockJudge)
    judge.evaluate.return_value = MagicMock(score=1.0, reasoning="test")
    
    intel = MagicMock(spec=MockIntel)
    intel.embedder.embed.return_value = [0.1] * 1536
    
    target = MagicMock(spec=MockTarget)
    
    # Simulate different delays for each attack
    async def delayed_invoke(prompt):
        delay = 0.1
        if "p1" in prompt: delay = 0.3
        if "p2" in prompt: delay = 0.1
        if "p3" in prompt: delay = 0.5
        await asyncio.sleep(delay)
        return "response"
        
    target.invoke_async.side_effect = delayed_invoke
    
    engine = SiegeEngine(
        strategy=strategy,
        judge=judge,
        intel_system=intel,
        target=target,
        goal="test",
        max_generations=1
    )
    
    # Collect events with timestamps
    events = []
    start_time = asyncio.get_running_loop().time()
    
    async for event in engine.run_stream(rate_limit_delay=0):
        if isinstance(event, AttackResultEvent):
            elapsed = asyncio.get_running_loop().time() - start_time
            events.append((event.attack.prompt, elapsed))
            
    # Verify order: p2 (0.1s) -> p1 (0.3s) -> p3 (0.5s)
    # If it was batch, they would all arrive around 0.5s
    
    assert len(events) == 3
    prompts = [e[0] for e in events]
    assert prompts == ["p2", "p1", "p3"]
    
    print("Streaming verified: Events arrived in order of completion duration.")
