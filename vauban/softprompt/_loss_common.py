"""Shared loss helpers for soft prompt objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vauban import _nn
from vauban import _ops as ops
from vauban._forward import embed_and_mask_with_prefix, lm_head_forward

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM, TransformerModel


@dataclass(frozen=True, slots=True)
class LossAuxConfig:
    """Shared auxiliary loss configuration."""

    direction: Array | None
    direction_weight: float
    direction_mode: str
    direction_layers: set[int] | None
    eos_token_id: int | None
    eos_loss_mode: str
    eos_loss_weight: float
    ref_model: CausalLM | None
    kl_ref_weight: float
    defense_aware_weight: float
    sic_layer: int | None
    sic_threshold: float
    cast_layers: list[int] | None
    cast_threshold: float
    perplexity_weight: float
    suffix_token_ids: Array | None


@dataclass(frozen=True, slots=True)
class LossPlacementConfig:
    """Placement configuration for soft tokens in the sequence."""

    n_tokens: int
    token_position: str = "prefix"
    infix_split: int | None = None


@dataclass(slots=True)
class LayerPenaltyAccumulator:
    """Mutable accumulator for per-layer auxiliary penalties."""

    direction_penalty: Array
    defense_penalty: Array
    n_penalty_layers: int


@dataclass(frozen=True, slots=True)
class ForwardTrace:
    """Outputs from the shared forward pass used by multiple objectives."""

    hidden_states: Array
    logits: Array
    n_prompt: int
    prompt_last_pos: int
    direction_penalty: Array
    defense_penalty: Array
    n_penalty_layers: int


def _compute_perplexity_loss(
    logits: Array,
    suffix_token_ids: Array,
    n_tokens: int,
    soft_token_offset: int,
) -> Array:
    """Compute perplexity loss over the soft-token span."""
    if n_tokens < 2:
        return ops.array(0.0)
    pred_logits = logits[:, soft_token_offset : soft_token_offset + n_tokens - 1, :]
    target = suffix_token_ids[0, 1:n_tokens]
    return _nn.cross_entropy(
        pred_logits.reshape(-1, pred_logits.shape[-1]),
        target,
        reduction="mean",
    )


def _add_perplexity_term(
    loss: Array,
    logits: Array,
    perplexity_weight: float,
    suffix_token_ids: Array | None,
    n_tokens: int,
    n_prompt: int,
    token_position: str,
    infix_split: int | None,
) -> Array:
    """Add perplexity regularization when suffix token IDs are available."""
    if perplexity_weight <= 0.0 or suffix_token_ids is None:
        return loss
    return loss + perplexity_weight * _compute_perplexity_loss(
        logits,
        suffix_token_ids,
        n_tokens,
        _soft_token_offset(token_position, n_prompt, infix_split),
    )


def _compute_eos_loss(
    logits: Array,
    position: int,
    eos_token_id: int,
    mode: str,
) -> Array:
    """Compute EOS control loss at a given position."""
    eps = 1e-8
    pos_logits = logits[:, position, :]
    probs = ops.softmax(pos_logits, axis=-1)
    p_eos = probs[0, eos_token_id]
    if mode == "force":
        return -ops.log(p_eos + eps)
    return -ops.log(1.0 - p_eos + eps)


def _compute_kl_collision_loss(
    model: CausalLM,
    ref_model: CausalLM,
    soft_embeds: Array,
    prompt_token_ids: Array,
    n_tokens: int,
) -> Array:
    """Compute KL divergence between the attacked and reference model."""
    transformer = model.model
    h, mask = embed_and_mask_with_prefix(transformer, soft_embeds, prompt_token_ids)
    for layer in transformer.layers:
        h = layer(h, mask)
    h = transformer.norm(h)
    attack_logits = lm_head_forward(model, h)

    ref_transformer = ref_model.model
    h_ref, mask_ref = embed_and_mask_with_prefix(
        ref_transformer,
        ops.stop_gradient(soft_embeds),
        prompt_token_ids,
    )
    for layer in ref_transformer.layers:
        h_ref = layer(h_ref, mask_ref)
    h_ref = ref_transformer.norm(h_ref)
    ref_logits = ops.stop_gradient(lm_head_forward(ref_model, h_ref))

    eps = 1e-8
    last_pos = n_tokens + prompt_token_ids.shape[1] - 1
    p_ref = ops.softmax(ref_logits[:, last_pos, :], axis=-1)
    q_attack = ops.softmax(attack_logits[:, last_pos, :], axis=-1)
    log_p_ref = ops.log(p_ref + eps)
    log_q_attack = ops.log(q_attack + eps)
    return ops.sum(p_ref * (log_p_ref - log_q_attack))


def _compute_defense_aware_penalty(
    h: Array,
    layer_idx: int,
    direction: Array | None,
    sic_layer: int | None,
    sic_threshold: float,
    cast_layers: list[int] | None,
    cast_threshold: float,
    prompt_last_pos: int = -1,
) -> Array:
    """Compute a penalty for being detectable by SIC or CAST defenses."""
    if direction is None:
        return ops.array(0.0)
    penalty = ops.array(0.0)
    if sic_layer is not None and layer_idx == sic_layer:
        proj = ops.sum(h[:, prompt_last_pos, :] * direction)
        penalty = penalty + ops.maximum(ops.array(0.0), sic_threshold - proj)
    if cast_layers is not None and layer_idx in cast_layers:
        proj = ops.sum(h[:, -1, :] * direction)
        penalty = penalty + ops.maximum(ops.array(0.0), proj - cast_threshold)
    return penalty


def _soft_token_offset(
    token_position: str,
    n_prompt: int,
    infix_split: int | None,
) -> int:
    """Compute the sequence offset of the soft-token span."""
    if token_position == "suffix":
        return n_prompt
    if token_position == "infix" and infix_split is not None:
        return infix_split
    return 0


def _assemble_targeted_sequence(
    transformer: TransformerModel,
    soft_embeds: Array,
    prompt_token_ids: Array,
    target_ids: Array,
    placement: LossPlacementConfig,
) -> tuple[Array, Array, int]:
    """Assemble a teacher-forced sequence containing prompt and targets."""
    prompt_embeds = transformer.embed_tokens(prompt_token_ids)
    target_embeds = transformer.embed_tokens(target_ids[None, :])
    if placement.token_position == "suffix":
        hidden_states = ops.concatenate(
            [prompt_embeds, soft_embeds, target_embeds],
            axis=1,
        )
    elif placement.token_position == "infix" and placement.infix_split is not None:
        part1 = prompt_embeds[:, : placement.infix_split, :]
        part2 = prompt_embeds[:, placement.infix_split :, :]
        hidden_states = ops.concatenate(
            [part1, soft_embeds, part2, target_embeds],
            axis=1,
        )
    else:
        hidden_states = ops.concatenate(
            [soft_embeds, prompt_embeds, target_embeds],
            axis=1,
        )
    mask = _nn.create_additive_causal_mask(hidden_states.shape[1]).astype(
        hidden_states.dtype,
    )
    return hidden_states, mask, prompt_token_ids.shape[1]


def _assemble_prefix_only_sequence(
    transformer: TransformerModel,
    soft_embeds: Array,
    prompt_token_ids: Array,
    placement: LossPlacementConfig,
) -> tuple[Array, Array, int]:
    """Assemble the prompt-only sequence used by refusal objectives."""
    hidden_states, mask = embed_and_mask_with_prefix(
        transformer,
        soft_embeds,
        prompt_token_ids,
        token_position=placement.token_position,
        infix_split=placement.infix_split,
    )
    return hidden_states, mask, prompt_token_ids.shape[1]


def _run_transformer_with_penalties(
    model: CausalLM,
    hidden_states: Array,
    mask: Array,
    n_prompt: int,
    placement: LossPlacementConfig,
    aux_config: LossAuxConfig,
) -> ForwardTrace:
    """Run the transformer while collecting shared layer penalties."""
    transformer = model.model
    prompt_last_pos = placement.n_tokens + n_prompt - 1
    penalties = LayerPenaltyAccumulator(
        direction_penalty=ops.array(0.0),
        defense_penalty=ops.array(0.0),
        n_penalty_layers=0,
    )
    h = hidden_states

    for layer_idx, layer in enumerate(transformer.layers):
        h = layer(h, mask)
        if (
            aux_config.direction is not None
            and aux_config.direction_weight > 0.0
            and aux_config.direction_mode != "last"
            and (
                aux_config.direction_layers is None
                or layer_idx in aux_config.direction_layers
            )
        ):
            if aux_config.direction_mode == "raid":
                proj = ops.sum(h[:, prompt_last_pos, :] * aux_config.direction)
            else:
                proj = ops.mean(ops.sum(h * aux_config.direction, axis=-1))
            penalties.direction_penalty = penalties.direction_penalty + proj
            penalties.n_penalty_layers += 1
        if aux_config.defense_aware_weight > 0.0:
            penalties.defense_penalty = penalties.defense_penalty + (
                _compute_defense_aware_penalty(
                    h,
                    layer_idx,
                    aux_config.direction,
                    aux_config.sic_layer,
                    aux_config.sic_threshold,
                    aux_config.cast_layers,
                    aux_config.cast_threshold,
                    prompt_last_pos=prompt_last_pos,
                )
            )

    h = transformer.norm(h)
    logits = lm_head_forward(model, h)
    return ForwardTrace(
        hidden_states=h,
        logits=logits,
        n_prompt=n_prompt,
        prompt_last_pos=prompt_last_pos,
        direction_penalty=penalties.direction_penalty,
        defense_penalty=penalties.defense_penalty,
        n_penalty_layers=penalties.n_penalty_layers,
    )


def _apply_shared_aux_terms(
    loss: Array,
    trace: ForwardTrace,
    model: CausalLM,
    soft_embeds: Array,
    prompt_token_ids: Array,
    placement: LossPlacementConfig,
    aux_config: LossAuxConfig,
    *,
    direction_sign: float = 1.0,
) -> Array:
    """Apply shared direction, defense, EOS, KL, and perplexity terms."""
    if aux_config.direction is not None and aux_config.direction_weight > 0.0:
        if aux_config.direction_mode == "last":
            last_hidden = trace.hidden_states[:, trace.prompt_last_pos, :]
            proj = ops.sum(last_hidden * aux_config.direction)
            loss = loss + direction_sign * aux_config.direction_weight * proj
        elif trace.n_penalty_layers > 0:
            loss = loss + direction_sign * aux_config.direction_weight * (
                trace.direction_penalty / trace.n_penalty_layers
            )

    if aux_config.defense_aware_weight > 0.0:
        loss = loss + aux_config.defense_aware_weight * trace.defense_penalty

    if (
        aux_config.eos_loss_weight > 0.0
        and aux_config.eos_loss_mode != "none"
        and aux_config.eos_token_id is not None
    ):
        loss = loss + aux_config.eos_loss_weight * _compute_eos_loss(
            trace.logits,
            trace.prompt_last_pos,
            aux_config.eos_token_id,
            aux_config.eos_loss_mode,
        )

    if aux_config.kl_ref_weight > 0.0 and aux_config.ref_model is not None:
        loss = loss + aux_config.kl_ref_weight * _compute_kl_collision_loss(
            model,
            aux_config.ref_model,
            soft_embeds,
            prompt_token_ids,
            placement.n_tokens,
        )

    return _add_perplexity_term(
        loss,
        trace.logits,
        aux_config.perplexity_weight,
        aux_config.suffix_token_ids,
        placement.n_tokens,
        trace.n_prompt,
        placement.token_position,
        placement.infix_split,
    )
