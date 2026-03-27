# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Externality objective for soft prompt optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vauban import _ops as ops
from vauban._forward import get_transformer
from vauban.softprompt._loss_common import (
    LossAuxConfig,
    LossPlacementConfig,
    _assemble_prefix_only_sequence,
    _compute_perplexity_loss,
    _run_transformer_with_penalties,
    _soft_token_offset,
)

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM


def _compute_externality_loss(
    model: CausalLM,
    soft_embeds: Array,
    prompt_token_ids: Array,
    n_tokens: int,
    target_direction: Array | None,
    direction_weight: float,
    perplexity_weight: float = 0.0,
    suffix_token_ids: Array | None = None,
    token_position: str = "prefix",
    infix_split: int | None = None,
) -> Array:
    """Compute the steering externality objective."""
    placement = LossPlacementConfig(
        n_tokens=n_tokens,
        token_position=token_position,
        infix_split=infix_split,
    )
    hidden_states, mask, n_prompt = _assemble_prefix_only_sequence(
        get_transformer(model),
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
        LossAuxConfig(
            direction=None,
            direction_weight=0.0,
            direction_mode="last",
            direction_layers=None,
            eos_token_id=None,
            eos_loss_mode="none",
            eos_loss_weight=0.0,
            ref_model=None,
            kl_ref_weight=0.0,
            defense_aware_weight=0.0,
            sic_layer=None,
            sic_threshold=0.0,
            cast_layers=None,
            cast_threshold=0.0,
            perplexity_weight=perplexity_weight,
            suffix_token_ids=suffix_token_ids,
        ),
    )

    ppl_loss = (
        _compute_perplexity_loss(
            trace.logits,
            suffix_token_ids,
            n_tokens,
            _soft_token_offset(token_position, n_prompt, infix_split),
        )
        if suffix_token_ids is not None and perplexity_weight > 0.0
        else ops.array(0.0)
    )

    erosion_loss = ops.array(0.0)
    if target_direction is not None and direction_weight > 0.0:
        last_hidden = trace.hidden_states[:, trace.prompt_last_pos, :]
        proj = ops.sum(last_hidden * target_direction)
        erosion_loss = -direction_weight * proj

    return perplexity_weight * ppl_loss + erosion_loss
