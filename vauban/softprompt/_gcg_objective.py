# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Shared objective dispatch for GCG and related soft prompt optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vauban._forward import force_eval, get_transformer
from vauban.softprompt._loss import (
    _compute_defensive_loss,
    _compute_externality_loss,
    _compute_loss,
    _compute_untargeted_loss,
)
from vauban.softprompt._runtime import _encode_refusal_tokens

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM, SoftPromptConfig, Tokenizer


@dataclass(frozen=True, slots=True)
class GCGDefenseConfig:
    """Shared defense-aware loss settings."""

    weight: float
    sic_layer: int | None
    sic_threshold: float
    cast_layers: list[int] | None
    cast_threshold: float


@dataclass(frozen=True, slots=True)
class GCGSharedState:
    """Resolved objective inputs shared across optimization steps."""

    model: CausalLM
    target_ids: Array
    n_tokens: int
    loss_mode: str
    direction: Array | None
    direction_weight: float
    direction_mode: str
    direction_layers: set[int] | None
    eos_token_id: int | None
    eos_loss_mode: str
    eos_loss_weight: float
    ref_model: CausalLM | None
    kl_ref_weight: float
    refusal_ids: Array | None
    defense: GCGDefenseConfig
    perplexity_weight: float
    token_position: str
    infix_map: dict[int, int] | None
    svf_boundary: object | None = None  # SVFBoundary for context-dependent directions


def _build_gcg_shared_state(
    model: CausalLM,
    tokenizer: Tokenizer,
    config: SoftPromptConfig,
    target_ids: Array,
    direction: Array | None,
    ref_model: CausalLM | None = None,
    infix_map: dict[int, int] | None = None,
    perplexity_weight_override: float | None = None,
    svf_boundary: object | None = None,
) -> GCGSharedState:
    """Resolve reusable objective inputs once per optimizer run."""
    direction_layers = (
        set(config.direction_layers)
        if config.direction_layers is not None
        else None
    )
    refusal_ids: Array | None = None
    if config.loss_mode in {"untargeted", "defensive"}:
        refusal_ids = _encode_refusal_tokens(tokenizer)
        force_eval(refusal_ids)
    sic_threshold = (
        config.defense_eval_sic_threshold
        if config.defense_eval_sic_threshold is not None
        else config.defense_eval_threshold
    )
    return GCGSharedState(
        model=model,
        target_ids=target_ids,
        n_tokens=config.n_tokens,
        loss_mode=config.loss_mode,
        direction=direction,
        direction_weight=config.direction_weight,
        direction_mode=config.direction_mode,
        direction_layers=direction_layers,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        eos_loss_mode=config.eos_loss_mode,
        eos_loss_weight=config.eos_loss_weight,
        ref_model=ref_model,
        kl_ref_weight=config.kl_ref_weight,
        refusal_ids=refusal_ids,
        defense=GCGDefenseConfig(
            weight=config.defense_aware_weight,
            sic_layer=config.defense_eval_layer,
            sic_threshold=sic_threshold,
            cast_layers=config.defense_eval_cast_layers,
            cast_threshold=config.defense_eval_threshold,
        ),
        perplexity_weight=(
            config.perplexity_weight
            if perplexity_weight_override is None
            else perplexity_weight_override
        ),
        token_position=config.token_position,
        infix_map=infix_map,
        svf_boundary=svf_boundary,
    )


def _prompt_infix_split(
    state: GCGSharedState,
    prompt_ids: Array,
) -> int | None:
    """Look up the infix insertion point for a prompt if one was precomputed."""
    if state.infix_map is None:
        return None
    return state.infix_map.get(id(prompt_ids))


def _compute_prompt_objective_loss(
    state: GCGSharedState,
    soft_embeds: Array,
    prompt_ids: Array,
    suffix_token_ids: Array | None = None,
) -> Array:
    """Compute the configured objective for a single prompt."""
    infix_split = _prompt_infix_split(state, prompt_ids)
    if state.loss_mode == "defensive" and state.refusal_ids is not None:
        return _compute_defensive_loss(
            state.model,
            soft_embeds,
            prompt_ids,
            state.n_tokens,
            state.refusal_ids,
            state.direction,
            state.direction_weight,
            state.direction_mode,
            state.direction_layers,
            state.eos_token_id,
            state.eos_loss_mode,
            state.eos_loss_weight,
            state.ref_model,
            state.kl_ref_weight,
            state.defense.weight,
            state.defense.sic_layer,
            state.defense.sic_threshold,
            state.defense.cast_layers,
            state.defense.cast_threshold,
            perplexity_weight=state.perplexity_weight,
            suffix_token_ids=suffix_token_ids,
            token_position=state.token_position,
            infix_split=infix_split,
            svf_boundary=state.svf_boundary,
        )
    if state.loss_mode == "untargeted" and state.refusal_ids is not None:
        return _compute_untargeted_loss(
            state.model,
            soft_embeds,
            prompt_ids,
            state.n_tokens,
            state.refusal_ids,
            state.direction,
            state.direction_weight,
            state.direction_mode,
            state.direction_layers,
            state.eos_token_id,
            state.eos_loss_mode,
            state.eos_loss_weight,
            state.ref_model,
            state.kl_ref_weight,
            state.defense.weight,
            state.defense.sic_layer,
            state.defense.sic_threshold,
            state.defense.cast_layers,
            state.defense.cast_threshold,
            perplexity_weight=state.perplexity_weight,
            suffix_token_ids=suffix_token_ids,
            token_position=state.token_position,
            infix_split=infix_split,
            svf_boundary=state.svf_boundary,
        )
    if state.loss_mode == "externality":
        return _compute_externality_loss(
            state.model,
            soft_embeds,
            prompt_ids,
            state.n_tokens,
            state.direction,
            state.direction_weight,
            state.perplexity_weight,
            suffix_token_ids=suffix_token_ids,
            token_position=state.token_position,
            infix_split=infix_split,
        )
    return _compute_loss(
        state.model,
        soft_embeds,
        prompt_ids,
        state.target_ids,
        state.n_tokens,
        state.direction,
        state.direction_weight,
        state.direction_mode,
        state.direction_layers,
        state.eos_token_id,
        state.eos_loss_mode,
        state.eos_loss_weight,
        state.ref_model,
        state.kl_ref_weight,
        state.defense.weight,
        state.defense.sic_layer,
        state.defense.sic_threshold,
        state.defense.cast_layers,
        state.defense.cast_threshold,
        perplexity_weight=state.perplexity_weight,
        suffix_token_ids=suffix_token_ids,
        token_position=state.token_position,
        infix_split=infix_split,
        svf_boundary=state.svf_boundary,
    )


def _compute_average_objective_loss(
    state: GCGSharedState,
    soft_embeds: Array,
    prompt_ids_list: list[Array],
    suffix_token_ids: Array | None = None,
) -> Array:
    """Compute the average objective over a list of prompts."""
    total = _compute_prompt_objective_loss(
        state,
        soft_embeds,
        prompt_ids_list[0],
        suffix_token_ids=suffix_token_ids,
    )
    for prompt_ids in prompt_ids_list[1:]:
        total = total + _compute_prompt_objective_loss(
            state,
            soft_embeds,
            prompt_ids,
            suffix_token_ids=suffix_token_ids,
        )
    return total / len(prompt_ids_list)


def _evaluate_candidate_loss(
    state: GCGSharedState,
    candidate_ids: list[int],
    selected_prompt_ids: list[Array],
) -> float:
    """Evaluate a discrete candidate suffix against the selected prompts."""
    from vauban import _ops as ops

    candidate_array = ops.array(candidate_ids)[None, :]
    candidate_embeds = get_transformer(state.model).embed_tokens(candidate_array)
    candidate_loss = _compute_average_objective_loss(
        state,
        candidate_embeds,
        selected_prompt_ids,
        suffix_token_ids=candidate_array,
    )
    force_eval(candidate_loss)
    return float(candidate_loss.item())
