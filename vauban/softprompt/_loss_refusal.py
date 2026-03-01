"""Untargeted and defensive refusal objectives."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vauban import _ops as ops
from vauban.softprompt._loss_common import (
    LossAuxConfig,
    LossPlacementConfig,
    _apply_shared_aux_terms,
    _assemble_prefix_only_sequence,
    _run_transformer_with_penalties,
)

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM


def _shared_refusal_loss(
    model: CausalLM,
    soft_embeds: Array,
    prompt_token_ids: Array,
    n_tokens: int,
    refusal_ids: Array,
    direction: Array | None,
    direction_weight: float,
    direction_mode: str,
    direction_layers: set[int] | None,
    eos_token_id: int | None,
    eos_loss_mode: str,
    eos_loss_weight: float,
    ref_model: CausalLM | None,
    kl_ref_weight: float,
    defense_aware_weight: float,
    sic_layer: int | None,
    sic_threshold: float,
    cast_layers: list[int] | None,
    cast_threshold: float,
    perplexity_weight: float,
    suffix_token_ids: Array | None,
    token_position: str,
    infix_split: int | None,
    *,
    maximize_refusal: bool,
) -> Array:
    """Compute the shared refusal-probability objective."""
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
    hidden_states, mask, n_prompt = _assemble_prefix_only_sequence(
        model.model,
        soft_embeds,
        prompt_token_ids,
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
    last_logits = trace.logits[:, -1, :]
    probs = ops.softmax(last_logits, axis=-1)
    refusal_sum = ops.sum(probs[0, refusal_ids])
    eps = 1e-8
    if maximize_refusal:
        loss = -ops.log(refusal_sum + eps)
        direction_sign = -1.0
    else:
        loss = -ops.log(1.0 - refusal_sum + eps)
        direction_sign = 1.0
    return _apply_shared_aux_terms(
        loss,
        trace,
        model,
        soft_embeds,
        prompt_token_ids,
        placement,
        aux_config,
        direction_sign=direction_sign,
    )


def _compute_defensive_loss(
    model: CausalLM,
    soft_embeds: Array,
    prompt_token_ids: Array,
    n_tokens: int,
    refusal_ids: Array,
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
    """Compute the defensive objective that maximizes refusal mass."""
    return _shared_refusal_loss(
        model,
        soft_embeds,
        prompt_token_ids,
        n_tokens,
        refusal_ids,
        direction,
        direction_weight,
        direction_mode,
        direction_layers,
        eos_token_id,
        eos_loss_mode,
        eos_loss_weight,
        ref_model,
        kl_ref_weight,
        defense_aware_weight,
        sic_layer,
        sic_threshold,
        cast_layers,
        cast_threshold,
        perplexity_weight,
        suffix_token_ids,
        token_position,
        infix_split,
        maximize_refusal=True,
    )


def _compute_untargeted_loss(
    model: CausalLM,
    soft_embeds: Array,
    prompt_token_ids: Array,
    n_tokens: int,
    refusal_ids: Array,
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
    """Compute the untargeted objective that minimizes refusal mass."""
    return _shared_refusal_loss(
        model,
        soft_embeds,
        prompt_token_ids,
        n_tokens,
        refusal_ids,
        direction,
        direction_weight,
        direction_mode,
        direction_layers,
        eos_token_id,
        eos_loss_mode,
        eos_loss_weight,
        ref_model,
        kl_ref_weight,
        defense_aware_weight,
        sic_layer,
        sic_threshold,
        cast_layers,
        cast_threshold,
        perplexity_weight,
        suffix_token_ids,
        token_position,
        infix_split,
        maximize_refusal=False,
    )
