# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Search and selection helpers for soft prompt attacks."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from vauban import _ops as ops
from vauban._forward import force_eval
from vauban.softprompt._gcg_objective import (
    GCGDefenseConfig,
    GCGSharedState,
    _compute_prompt_objective_loss,
)

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM


def _select_prompt_ids(
    all_ids: list[Array],
    step: int,
    strategy: str,
) -> list[Array]:
    """Select prompt ID arrays for this optimization step."""
    if strategy == "first":
        return [all_ids[0]]
    if strategy == "cycle":
        idx = step % len(all_ids)
        return [all_ids[idx]]
    return all_ids


def _sample_prompt_ids(
    all_ids: list[Array],
    k: int,
) -> list[Array]:
    """Randomly sample k prompts per step."""
    if k >= len(all_ids):
        return all_ids
    return random.sample(all_ids, k)


def _compute_per_prompt_losses(
    model: CausalLM,
    soft_embeds: Array,
    all_ids: list[Array],
    target_ids: Array,
    n_tokens: int,
    direction: Array | None,
    direction_weight: float,
    direction_mode: str = "last",
    direction_layers: set[int] | None = None,
    eos_token_id: int | None = None,
    eos_loss_mode: str = "none",
    eos_loss_weight: float = 0.0,
    ref_model: CausalLM | None = None,
    kl_ref_weight: float = 0.0,
    loss_mode: str = "targeted",
    refusal_ids: Array | None = None,
    defense_aware_weight: float = 0.0,
    sic_layer: int | None = None,
    sic_threshold: float = 0.0,
    cast_layers: list[int] | None = None,
    cast_threshold: float = 0.0,
    perplexity_weight: float = 0.0,
    token_position: str = "prefix",
    suffix_token_ids: Array | None = None,
    infix_split: int | None = None,
) -> list[float]:
    """Compute final loss for each prompt individually."""
    resolved_infix_map = (
        {id(prompt_ids): infix_split for prompt_ids in all_ids}
        if infix_split is not None
        else None
    )
    state = GCGSharedState(
        model=model,
        target_ids=target_ids,
        n_tokens=n_tokens,
        loss_mode=loss_mode,
        direction=direction,
        direction_weight=direction_weight,
        direction_mode=direction_mode,
        direction_layers=direction_layers,
        eos_token_id=eos_token_id,
        eos_loss_mode=eos_loss_mode,
        eos_loss_weight=eos_loss_weight,
        ref_model=ref_model,
        kl_ref_weight=kl_ref_weight,
        refusal_ids=refusal_ids,
        defense=GCGDefenseConfig(
            weight=defense_aware_weight,
            sic_layer=sic_layer,
            sic_threshold=sic_threshold,
            cast_layers=cast_layers,
            cast_threshold=cast_threshold,
        ),
        perplexity_weight=perplexity_weight,
        token_position=token_position,
        infix_map=resolved_infix_map,
    )
    losses: list[float] = []
    for prompt_ids in all_ids:
        loss = _compute_prompt_objective_loss(
            state,
            soft_embeds,
            prompt_ids,
            suffix_token_ids=suffix_token_ids,
        )
        force_eval(loss)
        losses.append(float(loss.item()))
    return losses


def _select_worst_k_prompt_ids(
    model: CausalLM,
    soft_embeds: Array,
    all_ids: list[Array],
    target_ids: Array,
    n_tokens: int,
    k: int,
    direction: Array | None,
    direction_weight: float,
    direction_mode: str,
    direction_layers: set[int] | None,
    eos_token_id: int | None,
    eos_loss_mode: str,
    eos_loss_weight: float,
    ref_model: CausalLM | None,
    kl_ref_weight: float,
    loss_mode: str,
    refusal_ids: Array | None,
    defense_aware_weight: float = 0.0,
    sic_layer: int | None = None,
    sic_threshold: float = 0.0,
    cast_layers: list[int] | None = None,
    cast_threshold: float = 0.0,
    perplexity_weight: float = 0.0,
    token_position: str = "prefix",
    suffix_token_ids: Array | None = None,
    infix_split: int | None = None,
) -> list[Array]:
    """Select the top-k hardest prompts by loss."""
    stopped_embeds = ops.stop_gradient(soft_embeds)
    losses = _compute_per_prompt_losses(
        model,
        stopped_embeds,
        all_ids,
        target_ids,
        n_tokens,
        direction,
        direction_weight,
        direction_mode,
        direction_layers,
        eos_token_id,
        eos_loss_mode,
        eos_loss_weight,
        ref_model,
        kl_ref_weight,
        loss_mode=loss_mode,
        refusal_ids=refusal_ids,
        defense_aware_weight=defense_aware_weight,
        sic_layer=sic_layer,
        sic_threshold=sic_threshold,
        cast_layers=cast_layers,
        cast_threshold=cast_threshold,
        perplexity_weight=perplexity_weight,
        token_position=token_position,
        suffix_token_ids=suffix_token_ids,
        infix_split=infix_split,
    )
    effective_k = min(k, len(all_ids))
    indexed = sorted(enumerate(losses), key=lambda item: item[1], reverse=True)
    return [all_ids[i] for i, _ in indexed[:effective_k]]


def _split_into_batches(
    items: list[Array],
    n_batches: int,
) -> list[list[Array]]:
    """Split items into approximately equal batches."""
    if n_batches <= 1 or len(items) <= 1:
        return [items]

    effective_n = min(n_batches, len(items))
    batch_size = len(items) // effective_n
    remainder = len(items) % effective_n

    batches: list[list[Array]] = []
    start = 0
    for i in range(effective_n):
        end = start + batch_size + (1 if i < remainder else 0)
        batches.append(items[start:end])
        start = end
    return batches
