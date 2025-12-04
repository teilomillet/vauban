from typing import Optional, Tuple, Any, List, Dict
from dataclasses import dataclass
import asyncio

from vauban.tracing import trace
from vauban.interfaces import Target, SiegeResult
from vauban.target import ModelTarget, MockAgentTarget
from vauban.engine import SiegeEngine
from vauban.strategies.gepa import GEPAStrategy
from vauban.judge import LLMJudge
from vauban.intel import create_default_intel, IntelSystem
from vauban.viz import create_atlas
from vauban.scenarios import SCENARIOS

# Global default instance for simple API calls
_DEFAULT_INTEL: Optional[IntelSystem] = None


def _get_intel(
    api_key: Optional[str] = None, base_url: Optional[str] = None
) -> IntelSystem:
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


@trace
def scout(
    prompt: str,
    target: Optional[Target] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: int = 1024,
) -> str:
    """
    Send a single scout to the target.
    Atomic Operation: Infiltrate.
    """
    target = target or ModelTarget(api_key=api_key, base_url=base_url, max_tokens=max_tokens)
    # Invoke sync wrapper if available, else run async
    if hasattr(target, "invoke"):
        return target.invoke(prompt)
    else:
        # Fallback for async-only targets in sync context (not ideal but pragmatic for scripts)
        return asyncio.run(target.invoke_async(prompt))


@trace
def assess(
    response: str, api_key: Optional[str] = None, base_url: Optional[str] = None
) -> Tuple[float, bool]:
    """
    Assess the damage of a response using the default Intel System.
    Atomic Operation: Assess.
    Returns (score, is_breach).
    """
    sys = _get_intel(api_key=api_key, base_url=base_url)
    return sys.assess_response(response)


@trace
def attack(
    payload: str,
    target: Optional[Target] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: int = 1024,
) -> AttackResult:
    """
    Convenience wrapper for Atomic Scout + Assess.
    Single-shot attack.
    """
    resp = scout(payload, target=target, api_key=api_key, base_url=base_url, max_tokens=max_tokens)
    score, breach = assess(str(resp), api_key=api_key, base_url=base_url)
    return AttackResult(str(resp), score, breach)


def baseline(
    target: Optional[Target] = None,
    embedding_model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: int = 1024,
):
    """
    Establish the baseline defenses (sync helper).
    - If no event loop is running, executes baseline async via asyncio.run.
    - If already inside a loop (e.g., Jupyter), this schedules the task and returns it; prefer await baseline_async instead.
    """
    sys = _get_intel(api_key=api_key, base_url=base_url)
    target = target or ModelTarget(api_key=api_key, base_url=base_url, max_tokens=max_tokens)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(sys.establish_baseline_async(target))
        return None
    else:
        return loop.create_task(sys.establish_baseline_async(target))


async def baseline_async(
    target: Optional[Target] = None,
    embedding_model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: int = 1024,
):
    """
    Notebook-friendly baseline; await this inside async contexts.
    """
    sys = _get_intel(api_key=api_key, base_url=base_url)
    target = target or ModelTarget(api_key=api_key, base_url=base_url, max_tokens=max_tokens)
    await sys.establish_baseline_async(target)

# ...

from vauban.config import (
    DEFAULT_ATTACKER_MODEL,
    DEFAULT_STRATEGY_MODEL,
    DEFAULT_STEALTH_MODEL,
    DEFAULT_EMBEDDING_MODEL,
)

# ...

def prepare_siege(
    goal: str,
    generations: int = 3,
    squad_size: int = 5,
    target: Optional[Target] = None,
    scenario: Optional[str] = None,
    attacker_model: Optional[str] = None,
    reflection_model: Optional[str] = None,
    mutation_model: Optional[str] = None,
    stealth_model: Optional[str] = None,
    scorer_model: Optional[str] = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    use_tools: bool = False,
    target_tools: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
    max_tokens: int = 1024,
    selection_method: str = "map_elites",
    breach_threshold: int = 2,
) -> SiegeEngine:
    """
    Prepare a SiegeEngine without running it.
    scorer_model mirrors the FerRet reward scorer; when omitted we reuse stealth_model
    for backward compatibility with older CLI/API callers.
    """
    # Apply defaults
    attacker_model = attacker_model or DEFAULT_ATTACKER_MODEL
    reflection_model = reflection_model or DEFAULT_STRATEGY_MODEL
    mutation_model = mutation_model or DEFAULT_STRATEGY_MODEL
    stealth_model = stealth_model or DEFAULT_STEALTH_MODEL

    # 1. Setup Intel
    intel_system = create_default_intel(
        embedding_model=embedding_model, 
        api_key=embedding_api_key, 
        base_url=embedding_base_url
    )

    # 2. Load Scenario (if any)
    active_scenario = None
    if scenario:
        if scenario not in SCENARIOS:
            raise ValueError(
                f"Unknown scenario: {scenario}. Available: {list(SCENARIOS.keys())}"
            )
        active_scenario = SCENARIOS[scenario]
        goal = active_scenario.goal
        target_tools = active_scenario.tools
        use_tools = True
        print(f"Loaded Scenario: {active_scenario.name}")

    # 3. Setup Target
    if target is None:
        if use_tools:
            print(
                f"Initializing Mock Agent Target with {len(target_tools) if target_tools else 'default'} tools..."
            )
            system_prompt = active_scenario.context if active_scenario else None
            mock_behaviors = active_scenario.mock_behaviors if active_scenario else None
            target = MockAgentTarget(
                model_name=attacker_model,
                tools=target_tools,
                system_prompt=system_prompt,
                mock_behaviors=mock_behaviors,
                api_key=api_key,
                base_url=base_url,
                max_tokens=max_tokens,
            )
        else:
            target = ModelTarget(
                model_name=attacker_model, api_key=api_key, base_url=base_url, max_tokens=max_tokens
            )

    # 4. Setup Judge
    judge = LLMJudge(
        model_name=attacker_model
    )  # Agent will use env vars or can be updated

    # 4b. Resolve scorer model (FerRet reward). Reuse stealth model if caller
    #     did not supply a dedicated scorer to keep older CLI/API usage working.
    if scorer_model is None:
        scorer_model = stealth_model

    # 5. Setup Strategy
    strategy = GEPAStrategy(
        intel_db=intel_system,
        goal=goal,
        reflection_model=reflection_model,
        mutation_model=mutation_model,
        stealth_model=stealth_model,
        scorer_model=scorer_model,
        attack_model=attacker_model,
        squad_size=squad_size,
        max_generations=generations,
        selection_method=selection_method,
    )

    # 6. Inject Deps into Engine
    engine = SiegeEngine(
        strategy=strategy,
        judge=judge,
        intel_system=intel_system,
        target=target,
        goal=goal,
        max_generations=generations,
        scenario=scenario,  # Name for reporting
        active_scenario=active_scenario,  # Full scenario so judge can apply success_condition/tool logic
        breach_threshold=breach_threshold,
    )
    return engine


@trace
async def siege(
    goal: str,
    generations: int = 3,
    squad_size: int = 5,
    target: Optional[Target] = None,
    scenario: Optional[str] = None,
    attacker_model: Optional[str] = None,
    reflection_model: Optional[str] = None,
    mutation_model: Optional[str] = None,
    stealth_model: Optional[str] = None,
    scorer_model: Optional[str] = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    use_tools: bool = False,
    target_tools: Optional[List[Dict[str, Any]]] = None,
    rate_limit_delay: float = 2.0,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
    max_tokens: int = 1024,
    selection_method: str = "map_elites",
    breach_threshold: int = 2,
) -> SiegeResult:
    """
    Launch a full Siege Campaign.
    Composition Root: Wires up the SiegeEngine with default components.
    """
    engine = prepare_siege(
        goal=goal,
        generations=generations,
        squad_size=squad_size,
        target=target,
        scenario=scenario,
        attacker_model=attacker_model,
        reflection_model=reflection_model,
        mutation_model=mutation_model,
        stealth_model=stealth_model,
        scorer_model=scorer_model,
        embedding_model=embedding_model,
        use_tools=use_tools,
        target_tools=target_tools,
        api_key=api_key,
        base_url=base_url,
        embedding_base_url=embedding_base_url,
        embedding_api_key=embedding_api_key,
        max_tokens=max_tokens,
        selection_method=selection_method,
        breach_threshold=breach_threshold,
    )

    return await engine.run(rate_limit_delay=rate_limit_delay)


def visualize(db_path: str = ".lancedb", table_name: str = "refusals") -> Any:
    """
    Create an Embedding Atlas visualization widget.
    """
    return create_atlas(db_path=db_path, table_name=table_name)
