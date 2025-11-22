
import os
import shutil
import asyncio
from vauban.engine import SiegeEngine
from vauban.strategies.gepa import GEPAStrategy
from vauban.judge import LLMJudge
from vauban.intel import create_default_intel
from vauban.target import MockAgentTarget
from vauban.tracing import WeaveModel


from unittest.mock import MagicMock, patch
from vauban.llm import MiniResult, TokenUsage

from vauban.interfaces import AttackPrompt

# Mock LLM run to avoid API keys
async def mock_run(self, prompt: str):
    # If the agent has a parser that returns AttackPrompt, we should return that.
    # But here we just return a generic AttackPrompt for simplicity if it looks like an attack generation.
    # However, other agents (judge, scorer) return different things.
    # We need to be slightly smarter or just return something that works for all or check trace_label.
    
    if self.trace_label in ["Attacker", "ProbeGen", "Mutate"]:
        # Note: trace_label on the LLM object itself hasn't changed, only the span names in the strategy/engine.
        # But wait, did I change trace_label in LLM init?
        # Let's check gepa.py again.
        pass

    if self.trace_label in ["Attacker", "Generate", "Mutate"]:
        data = AttackPrompt(prompt="test prompt", strategy="test")
    elif self.trace_label == "Judge":
        from vauban.interfaces import JudgeResult
        data = JudgeResult(score=5, label="Breach", reasoning="test")
    elif self.trace_label == "Deception":
        data = False
    elif self.trace_label in ["Score.Stealth", "Score.Reward"]:
        data = 0.5
    elif self.trace_label == "Reflect":
        data = "Reflection"
    else:
        data = "Generic Response"
        
    # Wrap in MiniResult to mimic real LLM
    from vauban.llm import MiniResult, TokenUsage
    return MiniResult(data=data, usage=TokenUsage(10, 5))
        
    return MiniResult(
        data=data,
        usage=TokenUsage(10, 10)
    )

async def main():
    print("Verifying Weave Refactor...")
    
    # Setup temp intel db
    db_path = ".lancedb_test"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        
    intel = create_default_intel(db_path=db_path)
    intel.embedder.embed = MagicMock(return_value=[0.1] * 1536)
    
    # Components
    target = MockAgentTarget(model_name="gpt-4o")
    judge = LLMJudge(model_name="openai:gpt-4o")
    
    # Patch LLM.run
    with patch("vauban.llm.LLM.run", new=mock_run):
        strategy = GEPAStrategy(
            intel_db=intel,
            goal="Test Goal",
            attack_model="openai:gpt-4o",
            reflection_model="openai:gpt-4o-mini",
            mutation_model="openai:gpt-4o-mini",
            stealth_model="openai:gpt-4o-mini",
            scorer_model="openai:gpt-4o-mini"
        )
        
        # Engine
        engine = SiegeEngine(
            strategy=strategy,
            judge=judge,
            intel_system=intel,
            target=target,
            goal="Test Goal",
            max_generations=1
        )
        
        print(f"Engine created: {type(engine)}")
        print(f"Is WeaveModel? {isinstance(engine, WeaveModel)}")
        
        # Check fields
        print(f"Strategy type: {type(engine.strategy)}")
        print(f"Judge type: {type(engine.judge)}")
        
        # Run short siege
        print("Running short siege...")
        result = await engine.run()
        print(f"Siege complete. Success: {result.success}")
    
    # Cleanup
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

if __name__ == "__main__":
    asyncio.run(main())
