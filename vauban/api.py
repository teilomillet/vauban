from typing import Optional, Tuple, Any, List, Dict, Union
from dataclasses import dataclass
import asyncio

from vauban.interfaces import Target, SiegeResult
from vauban.target import OpenAITarget, MockAgentTarget
from vauban.engine import SiegeEngine
from vauban.strategies.gepa import GEPAStrategy
from vauban.judge import LLMJudge
from vauban.intel import create_default_intel, IntelSystem
from vauban.viz import create_atlas
from vauban.scenarios import SCENARIOS

# Global default instance for simple API calls
_DEFAULT_INTEL: Optional[IntelSystem] = None

def _get_intel(api_key: Optional[str] = None, base_url: Optional[str] = None) -> IntelSystem:
    global _DEFAULT_INTEL
    if _DEFAULT_INTEL is None:
        _DEFAULT_INTEL = create_default_intel(api_key=api_key, base_url=base_url)
    return _DEFAULT_INTEL

@dataclass
class AttackResult:
    response: str
    score: float
    is_breach: bool

    def __repr__(self):
        status = "BREACH" if self.is_breach else "DEFLECTED"
        return f"[{status}] Score: {self.score:.2f}/5.0 | Response: {self.response[:50]}..."

def scout(prompt: str, target: Optional[Target] = None, api_key: Optional[str] = None, base_url: Optional[str] = None) -> str:
    """
    Send a single probe/scout to the target.
    Atomic Operation: Infiltrate.
    """
    target = target or OpenAITarget(api_key=api_key, base_url=base_url)
    # Invoke sync wrapper if available, else run async
    if hasattr(target, 'invoke'):
        return target.invoke(prompt)
    else:
        # Fallback for async-only targets in sync context (not ideal but pragmatic for scripts)
        return asyncio.run(target.invoke_async(prompt))

def assess(response: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> Tuple[float, bool]:
    """
    Assess the damage of a response using the default Intel System.
    Atomic Operation: Assess.
    Returns (score, is_breach).
    """
    sys = _get_intel(api_key=api_key, base_url=base_url)
    return sys.assess_response(response)

def attack(payload: str, target: Optional[Target] = None, api_key: Optional[str] = None, base_url: Optional[str] = None) -> AttackResult:
    """
    Convenience wrapper for Atomic Scout + Assess.
    Single-shot attack.
    """
    resp = scout(payload, target=target, api_key=api_key, base_url=base_url)
    score, breach = assess(str(resp), api_key=api_key, base_url=base_url)
    return AttackResult(str(resp), score, breach)

def baseline(target: Optional[Target] = None, embedding_model: str = "text-embedding-3-small", api_key: Optional[str] = None, base_url: Optional[str] = None):
    """
    Establish the baseline defenses.
    """
    sys = _get_intel(api_key=api_key, base_url=base_url) 
    # baseline is async now in IntelSystem, so we run it
    target = target or OpenAITarget(api_key=api_key, base_url=base_url)
    asyncio.run(sys.establish_baseline_async(target))

async def siege(
    goal: str,
    generations: int = 3, 
    squad_size: int = 5, 
    target: Optional[Target] = None,
    scenario: Optional[str] = None,
    attacker_model: str = "openai:gpt-4o",
    reflection_model: str = "openai:gpt-4o",
    mutation_model: str = "openai:gpt-4o",
    stealth_model: str = "openai:gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
    use_tools: bool = False,
    target_tools: Optional[List[Dict[str, Any]]] = None,
    rate_limit_delay: float = 2.0,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> SiegeResult:
    """
    Launch a full Siege Campaign.
    Composition Root: Wires up the SiegeEngine with default components.
    """
    
    # 1. Setup Intel
    # We can use the global default or create new one. 
    # Ideally separate for isolation, but sharing memory is powerful.
    # Let's use a fresh one if we want to customize embedding model, otherwise default.
    intel_system = create_default_intel(embedding_model=embedding_model, api_key=api_key, base_url=base_url)
    
    # 2. Load Scenario (if any)
    active_scenario = None
    if scenario:
        if scenario not in SCENARIOS:
             raise ValueError(f"Unknown scenario: {scenario}. Available: {list(SCENARIOS.keys())}")
        active_scenario = SCENARIOS[scenario]
        goal = active_scenario.goal
        target_tools = active_scenario.tools
        use_tools = True
        print(f"Loaded Scenario: {active_scenario.name}")

    # 3. Setup Target
    if target is None:
        if use_tools:
            print(f"Initializing Mock Agent Target with {len(target_tools) if target_tools else 'default'} tools...")
            system_prompt = active_scenario.context if active_scenario else None
            mock_behaviors = active_scenario.mock_behaviors if active_scenario else None
            target = MockAgentTarget(
                model_name=attacker_model, 
                tools=target_tools, 
                system_prompt=system_prompt,
                mock_behaviors=mock_behaviors,
                api_key=api_key,
                base_url=base_url
            )
        else:
            target = OpenAITarget(model_name=attacker_model, api_key=api_key, base_url=base_url)
    
    # 4. Setup Judge
    judge = LLMJudge(model_name=attacker_model) # Agent will use env vars or can be updated
    
    # 5. Setup Strategy
    strategy = GEPAStrategy(
        intel_db=intel_system, 
        goal=goal,
        reflection_model=reflection_model,
        mutation_model=mutation_model,
        stealth_model=stealth_model,
        attack_model=attacker_model,
        squad_size=squad_size,
        max_generations=generations
    )
    
    # 6. Inject Deps into Engine
    engine = SiegeEngine(
        strategy=strategy,
        judge=judge,
        intel_system=intel_system,
        target=target,
        goal=goal,
        max_generations=generations,
        scenario=scenario # Pass name/ID
    )
    
    return await engine.run(rate_limit_delay=rate_limit_delay)

def visualize(db_path: str = ".lancedb", table_name: str = "refusals") -> Any:
    """
    Create an Embedding Atlas visualization widget.
    """
    return create_atlas(db_path=db_path, table_name=table_name)
