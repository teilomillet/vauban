"""Targeted soft prompt objective."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vauban import _nn
from vauban.softprompt._loss_common import (
    LossAuxConfig,
    LossPlacementConfig,
    _apply_shared_aux_terms,
    _assemble_targeted_sequence,
    _run_transformer_with_penalties,
)

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM


def _compute_loss(
    model: CausalLM,
    soft_embeds: Array,
    prompt_token_ids: Array,
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
    defense_aware_weight: float = 0.0,
    sic_layer: int | None = None,
    sic_threshold: float = 0.0,
    cast_layers: list[int] | None = None,
    cast_threshold: float = 0.0,
    perplexity_weight: float = 0.0,
    suffix_token_ids: Array | None = None,
    token_position: str = "prefix",
    infix_split: int | None = None,
) -> Array:
    """Compute the targeted teacher-forced loss."""
    placement = LossPlacementConfig(
        n_tokens=n_tokens,
        token_position=token_position,
        infix_split=infix_split,
    )
    aux_config = LossAuxConfig(
        direction=direction,
        direction_weight=direction_weight,
        direction_mode=direction_mode,
        direction_layers=direction_layers,
        eos_token_id=eos_token_id,
        eos_loss_mode=eos_loss_mode,
        eos_loss_weight=eos_loss_weight,
        ref_model=ref_model,
        kl_ref_weight=kl_ref_weight,
        defense_aware_weight=defense_aware_weight,
        sic_layer=sic_layer,
        sic_threshold=sic_threshold,
        cast_layers=cast_layers,
        cast_threshold=cast_threshold,
        perplexity_weight=perplexity_weight,
        suffix_token_ids=suffix_token_ids,
    )
    hidden_states, mask, n_prompt = _assemble_targeted_sequence(
        model.model,
        soft_embeds,
        prompt_token_ids,
        target_ids,
        placement,
    )
    trace = _run_transformer_with_penalties(
        model,
        hidden_states,
        mask,
        n_prompt,
        placement,
        aux_config,
    )
    n_target = target_ids.shape[0]
    target_logits = trace.logits[
        :,
        trace.prompt_last_pos : trace.prompt_last_pos + n_target,
        :,
    ]
    loss = _nn.cross_entropy(
        target_logits.reshape(-1, target_logits.shape[-1]),
        target_ids,
        reduction="mean",
    )
    return _apply_shared_aux_terms(
        loss,
        trace,
        model,
        soft_embeds,
        prompt_token_ids,
        placement,
        aux_config,
    )
