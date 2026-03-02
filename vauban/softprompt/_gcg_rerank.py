"""Transfer and rollout reranking helpers for GCG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vauban.softprompt._runtime import _score_transfer_loss

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM, EnvironmentConfig, Tokenizer


@dataclass(frozen=True, slots=True)
class GCGRerankContext:
    """Context required for post-score reranking."""

    model: CausalLM
    tokenizer: Tokenizer
    transfer_data: list[tuple[CausalLM, Tokenizer, list[Array], Array]]
    transfer_loss_weight: float
    transfer_rerank_count: int
    environment_config: EnvironmentConfig | None


def _apply_transfer_reranking(
    context: GCGRerankContext,
    candidates: list[list[int]],
    candidate_losses: list[float],
) -> None:
    """Re-rank the top candidates using transfer-model scores."""
    if not context.transfer_data:
        return
    n_rerank = min(context.transfer_rerank_count, len(candidates))
    ranked = sorted(
        range(len(candidate_losses)),
        key=lambda index: candidate_losses[index],
    )[:n_rerank]
    for idx in ranked:
        transfer_loss = _score_transfer_loss(
            context.tokenizer.decode(candidates[idx]),
            context.transfer_data,
        )
        candidate_losses[idx] += context.transfer_loss_weight * transfer_loss


def _apply_rollout_reranking(
    context: GCGRerankContext,
    candidates: list[list[int]],
    candidate_losses: list[float],
    step: int,
) -> None:
    """Re-rank top candidates using environment rollouts when enabled."""
    if context.environment_config is None:
        return
    stride = context.environment_config.rollout_every_n
    if stride > 1 and step % stride != 0:
        return
    from vauban.environment import score_candidates_via_rollout

    top_n = min(context.environment_config.rollout_top_n, len(candidates))
    ranked = sorted(
        range(len(candidate_losses)),
        key=lambda index: candidate_losses[index],
    )[:top_n]
    texts = [context.tokenizer.decode(candidates[idx]) for idx in ranked]
    losses = [candidate_losses[idx] for idx in ranked]
    adjusted, _env_results = score_candidates_via_rollout(
        context.model,
        context.tokenizer,
        context.environment_config,
        texts,
        losses,
    )
    for idx, adjusted_loss in zip(ranked, adjusted, strict=True):
        candidate_losses[idx] = adjusted_loss
