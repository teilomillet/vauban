"""Rollout scoring: run top-N candidates through the agent loop."""

from vauban.environment._loop import run_agent_loop
from vauban.types import (
    CausalLM,
    EnvironmentConfig,
    EnvironmentResult,
    Tokenizer,
)


def score_candidates_via_rollout(
    model: CausalLM,
    tokenizer: Tokenizer,
    env_config: EnvironmentConfig,
    candidate_texts: list[str],
    candidate_losses: list[float],
) -> tuple[list[float], list[EnvironmentResult]]:
    """Re-rank candidates by environment rollout reward.

    Runs each candidate text through the agent loop and returns
    combined scores (CE loss - reward) for re-ranking.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        env_config: Environment configuration.
        candidate_texts: Decoded candidate suffix texts.
        candidate_losses: Original CE losses for each candidate.

    Returns:
        Tuple of (adjusted_scores, environment_results).
        Lower score is better (same convention as CE loss).
    """
    adjusted: list[float] = []
    results: list[EnvironmentResult] = []

    for text, loss in zip(candidate_texts, candidate_losses, strict=True):
        env_result = run_agent_loop(model, tokenizer, env_config, text)
        results.append(env_result)
        # Reward is 0.0-1.0; invert and scale to combine with CE loss
        # A reward of 1.0 should strongly prefer this candidate
        adjusted.append(loss - env_result.reward * 10.0)

    return adjusted, results
