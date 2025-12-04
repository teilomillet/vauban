import asyncio
from unittest.mock import MagicMock, AsyncMock
from vauban.engine import SiegeEngine
from vauban.interfaces import Event, AttackResultEvent

# Mock classes to avoid full dependency chain
class MockStrategy:
    def __init__(self):
        self.survivors = []
    
    def generator(self):
        async def _gen():
            # Infinite stream of squads
            while True:
                # Return a squad of 1 mock attack
                mock_attack = MagicMock()
                mock_attack.id = "mock_id"
                mock_attack.prompt = "mock_prompt"
                mock_attack.metadata = {}
                yield [mock_attack]
        return _gen()

    async def update(self, results):
        pass

class MockTarget:
    model_name = "mock-target"
    async def invoke_async(self, prompt):
        return "mock response"

class MockJudge:
    async def evaluate(self, goal, prompt, response, success_condition=None):
        # Always return a breach score
        mock_res = MagicMock()
        mock_res.score = 5.0
        mock_res.is_deceptive = False
        mock_res.reasoning = "Mock breach"
        return mock_res

class MockIntel:
    def __init__(self):
        self.embedder = MagicMock()
        self.embedder.embed.return_value = [0.1] * 10
    
    def log_skirmish(self, *args, **kwargs):
        pass

async def test_breach_threshold():
    print("Testing breach threshold logic...")
    
    engine = SiegeEngine(
        strategy=MockStrategy(),
        judge=MockJudge(),
        intel_system=MockIntel(),
        target=MockTarget(),
        goal="test goal",
        max_generations=5,
        breach_threshold=2 # Stop after 2 breaches
    )
    
    breach_count = 0
    wave_count = 0
    
    async for event in engine.run_stream(rate_limit_delay=0.1):
        if isinstance(event, dict) and "summary" in event:
            wave_count += 1
            if event["summary"]["breach_found"]:
                breach_count += event["summary"]["breach_count"]
                print(f"Wave {wave_count}: Found {event['summary']['breach_count']} breach(es). Total: {breach_count}")
        
        if isinstance(event, AttackResultEvent):
            pass

    with open("/Users/teilo.millet/Code/vauban/verify_output.txt", "w") as f:
        f.write(f"Final breach count: {breach_count}\n")
        
        if breach_count == 2:
            f.write("SUCCESS: Stopped exactly at breach threshold (2).\n")
            print("SUCCESS: Stopped exactly at breach threshold (2).")
        elif breach_count > 2:
            f.write(f"FAILURE: Overshot threshold! ({breach_count})\n")
            print(f"FAILURE: Overshot threshold! ({breach_count})")
        else:
            f.write(f"FAILURE: Stopped too early! ({breach_count})\n")
            print(f"FAILURE: Stopped too early! ({breach_count})")

if __name__ == "__main__":
    asyncio.run(test_breach_threshold())
