import asyncio
import random
import numpy as np
from typing import List, Any, Optional, AsyncGenerator, Dict
from pydantic import Field, PrivateAttr

try:
    import weave
except ImportError:
    weave = None

from vauban.interfaces import Strategy, AttackPrompt
from vauban.intel import IntelSystem
from vauban.tracing import trace, WeaveModel
from vauban.llm import MiniResult, TokenUsage

from .agents import GEPAAgents
from .population import GEPAPopulationManager
from .selection import pareto_selection
from .map_elites import MAPElitesArchive

class GEPAStrategy(WeaveModel, Strategy):
    """
    Upgraded to TAP (Tree of Attacks with Pruning) logic + Many-Shot capability.
    Enhanced with FerRet (Reward-Based Scoring + Explicit Styles).
    """
    intel_db: IntelSystem
    goal: str
    reflection_model: Optional[str] = None  # The "Brain": Analyzes failures (needs high reasoning, e.g. GPT-4o)
    mutation_model: Optional[str] = None    # The "Brain": Generates new prompts (needs high reasoning, e.g. GPT-4o)
    stealth_model: Optional[str] = None     # The "Judge": Rates stealthiness (can be smaller/faster)
    scorer_model: Optional[str] = None      # FerRet Reward Model Proxy
    attack_model: Optional[str] = None      # The "Victim": The target model we are testing/attacking
    squad_size: int = 5
    max_generations: int = 3
    branching_factor: int = 3
    width_limit: int = 5
    seed: Optional[int] = None
    selection_method: str = "map_elites"    # "nsga2" or "map_elites"
    
    # State
    current_generation: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    
    history: List[Any] = Field(default_factory=list, exclude=True)
    survivors: List[Dict[str, Any]] = Field(default_factory=list, exclude=True)
    current_batch_map: Dict[str, Any] = Field(default_factory=dict, exclude=True)

    # Components (Private)
    _agents: Optional[GEPAAgents] = PrivateAttr(default=None)
    _population_manager: Optional[GEPAPopulationManager] = PrivateAttr(default=None)
    _map_elites_archive: Optional[MAPElitesArchive] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # Validate required models
        if not self.reflection_model:
            raise ValueError("reflection_model is required")
        if not self.mutation_model:
            raise ValueError("mutation_model is required")
        if not self.attack_model:
            raise ValueError("attack_model is required")
        if not self.stealth_model:
            raise ValueError("stealth_model is required")
        if not self.scorer_model:
            raise ValueError("scorer_model is required")

        self._agents = GEPAAgents(
            goal=self.goal,
            reflection_model=self.reflection_model,
            mutation_model=self.mutation_model,
            stealth_model=self.stealth_model,
            scorer_model=self.scorer_model,
        )
        self._agents.initialize()
        
        self._population_manager = GEPAPopulationManager(
            agents=self._agents,
            squad_size=self.squad_size,
            branching_factor=self.branching_factor,
            goal=self.goal,
            usage_callback=self._track_usage,
            intel_system=self.intel_db
        )

        if self.selection_method == "map_elites":
            self._map_elites_archive = MAPElitesArchive()

    def _track_usage(self, result):
        """Track token usage from agent results."""
        if hasattr(result, "usage"):
            usage = result.usage()
            self.total_input_tokens += getattr(usage, "request_tokens", 0) or 0
            self.total_output_tokens += getattr(usage, "response_tokens", 0) or 0

    def get_cost_estimate(self) -> float:
        """
        Estimate cost based on configured pricing.
        """
        from vauban.config import PRICING_INPUT_PER_1M, PRICING_OUTPUT_PER_1M
        
        input_cost = (self.total_input_tokens / 1_000_000) * PRICING_INPUT_PER_1M
        output_cost = (self.total_output_tokens / 1_000_000) * PRICING_OUTPUT_PER_1M
        return input_cost + output_cost

    async def _calculate_stealth_result(self, prompt: str) -> Any:
        try:
            result = await self._agents.stealth_agent.run(f"Rate this prompt: {prompt}")
            self._track_usage(result)
            return result
        except Exception:
            # Return a dummy result with default data
            return MiniResult(data=0.5, usage=TokenUsage(0, 0))

    async def _score_mutation_result(self, prompt: str) -> Any:
        try:
            result = await self._agents.scorer_agent.run(f"Attack Prompt: {prompt}")
            self._track_usage(result)
            return result
        except Exception:
            return MiniResult(data=5.0, usage=TokenUsage(0, 0))

    async def generator(self) -> AsyncGenerator[List[AttackPrompt], None]:
        """
        Pipeline generator: Yields batches of attacks.
        Waits for feedback via `update()` before yielding the next batch.
        """
        # Gen 1: Initial Population (Random Personas)
        initial_population = await self._population_manager.generate_initial_population(self.attack_model)
        self._register_batch(initial_population)
        yield initial_population

        for _ in range(self.max_generations - 1):
            self.current_generation += 1

            # Wait for update() to populate self.survivors (which are the PRUNED results of previous batch)
            # Note: self.survivors is populated by update() which is called by engine

            # Expand (Branching)
            if not self.survivors:
                # Stop when no parent attacks survive pruning
                break

            next_batch = await self._population_manager.expand_population(self.survivors, self.current_generation)

            if not next_batch:
                break

            self._register_batch(next_batch)
            yield next_batch

    def _register_batch(self, attacks: List[AttackPrompt]):
        """Register current batch for tracking."""
        self.current_batch_map = {str(i): attack for i, attack in enumerate(attacks)}

    @trace
    async def update(self, results: List[Any]) -> None:
        """
        Consume results from the pipeline.
        results: List of (AttackPrompt, score, response, vector)
        PERFORMS PRUNING (Selection) here.
        """
        self.history.extend(results)

        # 1. Enrich results with stealth scores AND Reward Model scores
        # We score AFTER execution now to use it as a Pareto objective, not a filter.
        processed_results = []
        stealth_tasks = []
        reward_tasks = []

        for item in results:
            attack, score, response, vector = item
            @trace(name="Strategy.Score.Stealth")
            async def _stealth(prompt: str, wave: int, attack_id: str, model: str):
                res = await self._calculate_stealth_result(prompt)
                return res.data

            @trace(name="Strategy.Score.Reward")
            async def _reward(prompt: str, wave: int, attack_id: str, model: str):
                res = await self._score_mutation_result(prompt)
                return res.data

            stealth_tasks.append(_stealth(
                attack.prompt, 
                attack.metadata.get("wave"), 
                attack.id, 
                self._agents.stealth_agent.model_name
            ))
            reward_tasks.append(_reward(
                attack.prompt, 
                attack.metadata.get("wave"), 
                attack.id, 
                self._agents.scorer_agent.model_name
            ))

        stealth_scores = await asyncio.gather(*stealth_tasks)
        reward_scores = await asyncio.gather(*reward_tasks)

        for i, item in enumerate(results):
            attack, score, response, vector = item

            # Store metrics in metadata so they persist for logging
            attack.metadata["stealth"] = stealth_scores[i]
            attack.metadata["reward"] = reward_scores[i]

            processed_results.append(
                {
                    "attack": attack,
                    "score": score,  # Objective 1: Judge Score
                    "stealth": stealth_scores[i],  # Objective 2: Stealth
                    "reward": reward_scores[i],  # Objective 3: Proxy Reward
                    "response": response,
                    "vector": vector,
                }
            )

        # 2. Pruning (Selection)
        if self.selection_method == "map_elites":
            # MAP-Elites Strategy (Illumination)
            # Add all processed results to the archive
            added_count = 0
            for res in processed_results:
                if self._map_elites_archive.add(res):
                    added_count += 1
            
            # Select survivors from the archive
            self.survivors = self._map_elites_archive.select_survivors(self.width_limit)
            
            print(
                f"  >> MAP-Elites Update: {added_count} new elites added. Archive size: {len(self._map_elites_archive.get_all())}"
            )
            print(
                f"  >> Survivors Selected: {len(self.survivors)} (from {len(self._map_elites_archive.get_all())} elites)"
            )

        else:
            # NSGA-II Strategy (Pareto)
            # We combine previous survivors + new results to maintain the archive
            all_candidates = self.survivors + processed_results

            # Ensure we don't keep duplicates (by ID)
            unique_candidates = {c["attack"].id: c for c in all_candidates}.values()
            unique_list = list(unique_candidates)

            # Select Top K using Pareto
            self.survivors = pareto_selection(unique_list, self.width_limit)

            print(
                f"  >> GEPA Pareto Pruning: {len(unique_list)} candidates -> {len(self.survivors)} survivors"
            )

        if self.survivors:
            best = max(self.survivors, key=lambda x: x["score"])
            print(
                f"  >> Best Survivor: Score={best['score']:.1f} | Stealth={best['stealth']:.2f} | Reward={best['reward']:.1f}"
            )
