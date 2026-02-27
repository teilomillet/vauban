"""Loss computation for soft prompt attacks."""

from vauban import _nn
from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import embed_and_mask_with_prefix, lm_head_forward
from vauban.types import CausalLM


def _compute_eos_loss(
    logits: Array,
    position: int,
    eos_token_id: int,
    mode: str,
) -> Array:
    """Compute EOS control loss at a given position.

    Args:
        logits: Full logits tensor, shape (1, seq_len, vocab_size).
        position: Sequence position to evaluate.
        eos_token_id: Token ID for the EOS token.
        mode: "force" (push P(EOS) toward 1) or "suppress" (toward 0).

    Returns:
        Scalar loss value.
    """
    eps = 1e-8
    pos_logits = logits[:, position, :]  # (1, vocab_size)
    probs = ops.softmax(pos_logits, axis=-1)  # (1, vocab_size)
    p_eos = probs[0, eos_token_id]

    if mode == "force":
        return -ops.log(p_eos + eps)
    # "suppress"
    return -ops.log(1.0 - p_eos + eps)


def _compute_kl_collision_loss(
    model: CausalLM,
    ref_model: CausalLM,
    soft_embeds: Array,
    prompt_token_ids: Array,
    n_tokens: int,
) -> Array:
    """Compute KL divergence collision loss between attacked and reference model.

    Forward passes both models with the same [soft_prefix | prompt] input.
    Computes KL(P_ref || Q_attack) at the last position so gradients flow
    through the attacked model's branch w.r.t. soft_embeds.

    Args:
        model: The attacked causal language model.
        ref_model: The reference causal language model.
        soft_embeds: Learnable prefix, shape (1, n_tokens, d_model).
        prompt_token_ids: Prompt token IDs, shape (1, seq_len).
        n_tokens: Number of soft prompt tokens.

    Returns:
        Scalar KL divergence loss.
    """
    # Forward through attacked model
    transformer = model.model
    h, mask = embed_and_mask_with_prefix(transformer, soft_embeds, prompt_token_ids)
    for layer in transformer.layers:
        h = layer(h, mask)
    h = transformer.norm(h)
    attack_logits = lm_head_forward(model, h)  # (1, seq_len, vocab_size)

    # Forward through reference model (stop gradient)
    ref_transformer = ref_model.model
    h_ref, mask_ref = embed_and_mask_with_prefix(
        ref_transformer, ops.stop_gradient(soft_embeds), prompt_token_ids,
    )
    for layer in ref_transformer.layers:
        h_ref = layer(h_ref, mask_ref)
    h_ref = ref_transformer.norm(h_ref)
    ref_logits = lm_head_forward(ref_model, h_ref)
    ref_logits = ops.stop_gradient(ref_logits)

    # KL(P_ref || Q_attack) at last position
    eps = 1e-8
    last_pos = n_tokens + prompt_token_ids.shape[1] - 1
    p_ref = ops.softmax(ref_logits[:, last_pos, :], axis=-1)
    q_attack = ops.softmax(attack_logits[:, last_pos, :], axis=-1)
    log_p_ref = ops.log(p_ref + eps)
    log_q_attack = ops.log(q_attack + eps)

    kl = ops.sum(p_ref * (log_p_ref - log_q_attack))
    return kl


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
) -> Array:
    """Compute cross-entropy loss with teacher forcing.

    Feeds [soft_prefix | prompt | target] through the model and takes
    the cross-entropy loss at positions where the model should predict
    each target token.

    Supports RAID multi-layer direction penalty (Schwinn et al., 2024),
    all-positions direction penalty modes, EOS control losses, and
    KL collision loss (Geiping et al., 2024).

    Args:
        model: The causal language model.
        soft_embeds: Learnable prefix, shape (1, n_tokens, d_model).
        prompt_token_ids: Prompt token IDs, shape (1, seq_len).
        target_ids: Target token IDs to predict, shape (n_target,).
        n_tokens: Number of soft prompt tokens.
        direction: Optional refusal direction vector.
        direction_weight: Weight for direction auxiliary loss.
        direction_mode: "last" (original), "raid" (per-layer at last prompt
            position), or "all_positions" (mean over all positions per layer).
        direction_layers: Set of layer indices for RAID/all_positions penalty.
            None means all layers.
        eos_token_id: EOS token ID for EOS loss.
        eos_loss_mode: "none", "force", or "suppress".
        eos_loss_weight: Weight for EOS auxiliary loss.
        ref_model: Reference model for KL collision loss.
        kl_ref_weight: Weight for KL collision loss.

    Returns:
        Scalar loss value.
    """
    transformer = model.model
    prompt_embeds = transformer.embed_tokens(prompt_token_ids)
    target_embeds = transformer.embed_tokens(target_ids[None, :])

    # Teacher forcing: model sees prefix + prompt + target tokens
    h = ops.concatenate([soft_embeds, prompt_embeds, target_embeds], axis=1)

    mask = _nn.create_additive_causal_mask(h.shape[1])
    mask = mask.astype(h.dtype)

    n_prompt = prompt_token_ids.shape[1]

    # Per-layer direction penalty accumulation (RAID / all_positions)
    direction_penalty = ops.array(0.0)
    n_penalty_layers = 0

    for i, layer in enumerate(transformer.layers):
        h = layer(h, mask)
        if (
            direction is not None
            and direction_weight > 0.0
            and direction_mode != "last"
            and (direction_layers is None or i in direction_layers)
        ):
            if direction_mode == "raid":
                # Project at last prompt position (before targets)
                last_pos = n_tokens + n_prompt - 1
                proj = ops.sum(h[:, last_pos, :] * direction)
            else:  # "all_positions"
                proj = ops.mean(ops.sum(h * direction, axis=-1))
            direction_penalty = direction_penalty + proj
            n_penalty_layers += 1

    h = transformer.norm(h)
    logits = lm_head_forward(model, h)

    # Logits at position i predict token i+1.
    n_target = target_ids.shape[0]
    start = n_tokens + n_prompt - 1
    target_logits = logits[:, start : start + n_target, :]

    ce_loss = _nn.cross_entropy(
        target_logits.reshape(-1, target_logits.shape[-1]),
        target_ids,
        reduction="mean",
    )

    if direction is not None and direction_weight > 0.0:
        if direction_mode == "last":
            # Original single-position behavior
            last_prompt_pos = n_tokens + n_prompt - 1
            last_hidden = h[:, last_prompt_pos, :]
            proj = ops.sum(last_hidden * direction)
            ce_loss = ce_loss + direction_weight * proj
        elif n_penalty_layers > 0:
            ce_loss = ce_loss + direction_weight * (
                direction_penalty / n_penalty_layers
            )

    # EOS control loss (Geiping et al.)
    if (
        eos_loss_weight > 0.0
        and eos_loss_mode != "none"
        and eos_token_id is not None
    ):
        eos_pos = n_tokens + n_prompt - 1
        ce_loss = ce_loss + eos_loss_weight * _compute_eos_loss(
            logits, eos_pos, eos_token_id, eos_loss_mode,
        )

    # KL collision loss (Geiping et al.)
    if kl_ref_weight > 0.0 and ref_model is not None:
        ce_loss = ce_loss + kl_ref_weight * _compute_kl_collision_loss(
            model, ref_model, soft_embeds, prompt_token_ids, n_tokens,
        )

    return ce_loss


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
) -> Array:
    """Compute defensive loss: maximize refusal probability.

    Mirror of ``_compute_untargeted_loss`` — instead of pushing P(refusal)
    toward 0, pushes it toward 1. The direction penalty is flipped so it
    reinforces the refusal direction instead of suppressing it.

    loss = -log(P(refusal) + eps)

    Args:
        model: The causal language model.
        soft_embeds: Learnable prefix, shape (1, n_tokens, d_model).
        prompt_token_ids: Prompt token IDs, shape (1, seq_len).
        n_tokens: Number of soft prompt tokens.
        refusal_ids: Token IDs of refusal tokens, shape (n_refusal,).
        direction: Optional refusal direction vector.
        direction_weight: Weight for direction auxiliary loss.
        direction_mode: Direction penalty mode.
        direction_layers: Layer indices for direction penalty.
        eos_token_id: EOS token ID for EOS loss.
        eos_loss_mode: "none", "force", or "suppress".
        eos_loss_weight: Weight for EOS auxiliary loss.
        ref_model: Reference model for KL collision loss.
        kl_ref_weight: Weight for KL collision loss.

    Returns:
        Scalar loss value.
    """
    transformer = model.model
    h, mask = embed_and_mask_with_prefix(transformer, soft_embeds, prompt_token_ids)

    n_prompt = prompt_token_ids.shape[1]

    # Per-layer direction penalty accumulation
    direction_penalty = ops.array(0.0)
    n_penalty_layers = 0

    for i, layer in enumerate(transformer.layers):
        h = layer(h, mask)
        if (
            direction is not None
            and direction_weight > 0.0
            and direction_mode != "last"
            and (direction_layers is None or i in direction_layers)
        ):
            if direction_mode == "raid":
                last_pos = n_tokens + n_prompt - 1
                proj = ops.sum(h[:, last_pos, :] * direction)
            else:  # "all_positions"
                proj = ops.mean(ops.sum(h * direction, axis=-1))
            direction_penalty = direction_penalty + proj
            n_penalty_layers += 1

    h = transformer.norm(h)
    logits = lm_head_forward(model, h)

    # Logits at the last position predict the first generated token
    last_logits = logits[:, -1, :]  # (1, vocab_size)
    probs = ops.softmax(last_logits, axis=-1)  # (1, vocab_size)

    # Sum probability mass on refusal tokens
    refusal_probs = probs[0, refusal_ids]  # (n_refusal,)
    refusal_sum = ops.sum(refusal_probs)

    # Maximize P(refusal): minimize -log(P(refusal) + eps)
    eps = 1e-8
    loss = -ops.log(refusal_sum + eps)

    # Direction penalty — FLIPPED: push toward refusal direction
    # (subtract projection so higher projection = lower loss)
    if direction is not None and direction_weight > 0.0:
        if direction_mode == "last":
            last_prompt_pos = n_tokens + n_prompt - 1
            last_hidden = h[:, last_prompt_pos, :]
            proj = ops.sum(last_hidden * direction)
            loss = loss - direction_weight * proj
        elif n_penalty_layers > 0:
            loss = loss - direction_weight * (
                direction_penalty / n_penalty_layers
            )

    # EOS control loss
    if (
        eos_loss_weight > 0.0
        and eos_loss_mode != "none"
        and eos_token_id is not None
    ):
        eos_pos = n_tokens + n_prompt - 1
        loss = loss + eos_loss_weight * _compute_eos_loss(
            logits, eos_pos, eos_token_id, eos_loss_mode,
        )

    # KL collision loss
    if kl_ref_weight > 0.0 and ref_model is not None:
        loss = loss + kl_ref_weight * _compute_kl_collision_loss(
            model, ref_model, soft_embeds, prompt_token_ids, n_tokens,
        )

    return loss


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
) -> Array:
    """Compute untargeted jailbreak loss (UJA, Deng et al. 2024).

    Forward pass with [soft_prefix | prompt] (no target tokens). Gets logits
    at last position, computes softmax, penalizes sum of refusal token
    probabilities: loss = -log(1 - sum(softmax(logits)[refusal_ids]) + eps).

    Args:
        model: The causal language model.
        soft_embeds: Learnable prefix, shape (1, n_tokens, d_model).
        prompt_token_ids: Prompt token IDs, shape (1, seq_len).
        n_tokens: Number of soft prompt tokens.
        refusal_ids: Token IDs of refusal tokens, shape (n_refusal,).
        direction: Optional refusal direction vector.
        direction_weight: Weight for direction auxiliary loss.
        direction_mode: Direction penalty mode.
        direction_layers: Layer indices for direction penalty.
        eos_token_id: EOS token ID for EOS loss.
        eos_loss_mode: "none", "force", or "suppress".
        eos_loss_weight: Weight for EOS auxiliary loss.
        ref_model: Reference model for KL collision loss.
        kl_ref_weight: Weight for KL collision loss.

    Returns:
        Scalar loss value.
    """
    transformer = model.model
    h, mask = embed_and_mask_with_prefix(transformer, soft_embeds, prompt_token_ids)

    n_prompt = prompt_token_ids.shape[1]

    # Per-layer direction penalty accumulation
    direction_penalty = ops.array(0.0)
    n_penalty_layers = 0

    for i, layer in enumerate(transformer.layers):
        h = layer(h, mask)
        if (
            direction is not None
            and direction_weight > 0.0
            and direction_mode != "last"
            and (direction_layers is None or i in direction_layers)
        ):
            if direction_mode == "raid":
                last_pos = n_tokens + n_prompt - 1
                proj = ops.sum(h[:, last_pos, :] * direction)
            else:  # "all_positions"
                proj = ops.mean(ops.sum(h * direction, axis=-1))
            direction_penalty = direction_penalty + proj
            n_penalty_layers += 1

    h = transformer.norm(h)
    logits = lm_head_forward(model, h)

    # Logits at the last position predict the first generated token
    last_logits = logits[:, -1, :]  # (1, vocab_size)
    probs = ops.softmax(last_logits, axis=-1)  # (1, vocab_size)

    # Sum probability mass on refusal tokens
    refusal_probs = probs[0, refusal_ids]  # (n_refusal,)
    refusal_sum = ops.sum(refusal_probs)

    # Minimize: -log(1 - P(refusal) + eps)
    eps = 1e-8
    loss = -ops.log(1.0 - refusal_sum + eps)

    # Direction penalty
    if direction is not None and direction_weight > 0.0:
        if direction_mode == "last":
            last_prompt_pos = n_tokens + n_prompt - 1
            last_hidden = h[:, last_prompt_pos, :]
            proj = ops.sum(last_hidden * direction)
            loss = loss + direction_weight * proj
        elif n_penalty_layers > 0:
            loss = loss + direction_weight * (
                direction_penalty / n_penalty_layers
            )

    # EOS control loss (Geiping et al.)
    if (
        eos_loss_weight > 0.0
        and eos_loss_mode != "none"
        and eos_token_id is not None
    ):
        eos_pos = n_tokens + n_prompt - 1
        loss = loss + eos_loss_weight * _compute_eos_loss(
            logits, eos_pos, eos_token_id, eos_loss_mode,
        )

    # KL collision loss (Geiping et al.)
    if kl_ref_weight > 0.0 and ref_model is not None:
        loss = loss + kl_ref_weight * _compute_kl_collision_loss(
            model, ref_model, soft_embeds, prompt_token_ids, n_tokens,
        )

    return loss
