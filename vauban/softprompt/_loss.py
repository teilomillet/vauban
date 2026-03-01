"""Loss computation for soft prompt attacks."""

from vauban import _nn
from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import embed_and_mask_with_prefix, lm_head_forward
from vauban.types import CausalLM


def _compute_perplexity_loss(
    logits: Array,
    suffix_token_ids: Array,
    n_tokens: int,
    soft_token_offset: int,
) -> Array:
    """Compute perplexity loss to push suffixes toward fluent text.

    Extracts logits at soft token positions and computes cross-entropy
    against actual suffix token IDs (next-token prediction within the
    suffix). Lower loss = more fluent, natural-sounding suffix.

    Args:
        logits: Full logits tensor, shape (1, seq_len, vocab_size).
        suffix_token_ids: Actual suffix token IDs, shape (1, n_tokens).
        n_tokens: Number of soft prompt tokens.
        soft_token_offset: Position of soft tokens in the sequence
            (0 for prefix, n_prompt for suffix, infix_split for infix).

    Returns:
        Scalar mean cross-entropy loss over suffix positions.
    """
    if n_tokens < 2:
        return ops.array(0.0)
    # Logits at positions offset..offset+n_tokens-2 predict
    # suffix tokens at positions 1..n_tokens-1
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
    """Add perplexity regularization term to a loss value.

    Computes the correct soft-token offset based on token position
    and adds the weighted perplexity loss. Returns the original loss
    unchanged when perplexity is disabled or suffix IDs are absent.

    Args:
        loss: Current loss value to augment.
        logits: Full logits tensor, shape (1, seq_len, vocab_size).
        perplexity_weight: Weight for perplexity regularization.
        suffix_token_ids: Suffix token IDs, shape (1, n_tokens), or None.
        n_tokens: Number of soft prompt tokens.
        n_prompt: Number of prompt tokens (for suffix offset).
        token_position: "prefix", "suffix", or "infix".
        infix_split: Token index in prompt for infix insertion point.

    Returns:
        Loss with perplexity term added (or unchanged if disabled).
    """
    if perplexity_weight <= 0.0 or suffix_token_ids is None:
        return loss
    if token_position == "suffix":
        ppl_offset = n_prompt
    elif token_position == "infix" and infix_split is not None:
        ppl_offset = infix_split
    else:
        ppl_offset = 0
    return loss + perplexity_weight * _compute_perplexity_loss(
        logits, suffix_token_ids, n_tokens, soft_token_offset=ppl_offset,
    )


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
    """Penalty for being detectable by SIC or CAST defenses.

    Models the specific detection signals used by SIC and CAST so the
    optimizer learns to evade both during training.

    - SIC evasion: projection should stay ABOVE threshold (benign signal).
      SIC detects adversarial content when projection < threshold
      (low refusal = suspicious). Penalty: max(0, threshold - proj).
      Uses the last prompt position (before targets) to match SIC's
      runtime behavior of checking the prompt-only sequence.
    - CAST evasion: projection should stay BELOW threshold (avoid steering).
      CAST intervenes when projection > threshold (high refusal = steer).
      Penalty: max(0, proj - threshold).
      Uses the last token position because CAST fires during generation,
      where each generated token is the "last" in its context.

    When both are active on the same layer with the same threshold, the
    penalties push projection in opposite directions, creating a narrow
    evasion band. For a viable evasion zone, use a SIC threshold lower
    than the CAST threshold (e.g., SIC at the defense_eval_threshold,
    CAST layers at different layers or with a higher threshold).

    Args:
        h: Hidden state at current layer, shape (1, seq_len, d_model).
        layer_idx: Index of the current transformer layer.
        direction: Refusal direction vector.
        sic_layer: Layer index where SIC measures projection.
        sic_threshold: SIC detection threshold.
        cast_layers: Layer indices where CAST intervenes.
        cast_threshold: CAST intervention threshold.
        prompt_last_pos: Index of the last prompt token (before targets).
            SIC uses this position to match runtime behavior. Defaults
            to -1 (last token) for backward compatibility.

    Returns:
        Scalar penalty value.
    """
    if direction is None:
        return ops.array(0.0)
    penalty = ops.array(0.0)
    # SIC evasion: SIC detects adversarial when proj < threshold
    # Use prompt-end position (matches SIC runtime: prompt-only check)
    if sic_layer is not None and layer_idx == sic_layer:
        proj = ops.sum(h[:, prompt_last_pos, :] * direction)
        penalty = penalty + ops.maximum(ops.array(0.0), sic_threshold - proj)
    # CAST evasion: CAST intervenes when proj > threshold
    # Use last token (matches CAST runtime: per-generation-step)
    if cast_layers is not None and layer_idx in cast_layers:
        proj = ops.sum(h[:, -1, :] * direction)
        penalty = penalty + ops.maximum(ops.array(0.0), proj - cast_threshold)
    return penalty


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
        defense_aware_weight: Weight for defense-aware evasion penalty.
            0 disables the penalty.
        sic_layer: Layer index where SIC measures projection.
        sic_threshold: SIC detection threshold.
        cast_layers: Layer indices where CAST intervenes.
        cast_threshold: CAST intervention threshold.
        perplexity_weight: Weight for perplexity regularization loss.
        suffix_token_ids: Suffix token IDs for perplexity loss, shape (1, n_tokens).
        token_position: Soft token placement: "prefix", "suffix", or "infix".
        infix_split: Token index in prompt where soft tokens are inserted (infix only).

    Returns:
        Scalar loss value.
    """
    transformer = model.model
    prompt_embeds = transformer.embed_tokens(prompt_token_ids)
    target_embeds = transformer.embed_tokens(target_ids[None, :])

    # Teacher forcing: assemble [soft | prompt | target] based on position
    if token_position == "suffix":
        h = ops.concatenate(
            [prompt_embeds, soft_embeds, target_embeds], axis=1,
        )
    elif token_position == "infix" and infix_split is not None:
        part1 = prompt_embeds[:, :infix_split, :]
        part2 = prompt_embeds[:, infix_split:, :]
        h = ops.concatenate(
            [part1, soft_embeds, part2, target_embeds], axis=1,
        )
    else:  # prefix (default)
        h = ops.concatenate(
            [soft_embeds, prompt_embeds, target_embeds], axis=1,
        )

    mask = _nn.create_additive_causal_mask(h.shape[1])
    mask = mask.astype(h.dtype)

    n_prompt = prompt_token_ids.shape[1]

    # Per-layer direction penalty accumulation (RAID / all_positions)
    direction_penalty = ops.array(0.0)
    defense_penalty = ops.array(0.0)
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
        if defense_aware_weight > 0.0:
            defense_penalty = defense_penalty + _compute_defense_aware_penalty(
                h, i, direction, sic_layer, sic_threshold,
                cast_layers, cast_threshold,
                prompt_last_pos=n_tokens + n_prompt - 1,
            )

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

    # Defense-aware evasion penalty
    if defense_aware_weight > 0.0:
        ce_loss = ce_loss + defense_aware_weight * defense_penalty

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

    # Perplexity regularization — push suffix toward fluent text
    ce_loss = _add_perplexity_term(
        ce_loss, logits, perplexity_weight, suffix_token_ids,
        n_tokens, n_prompt, token_position, infix_split,
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
        defense_aware_weight: Weight for defense-aware evasion penalty.
            0 disables the penalty.
        sic_layer: Layer index where SIC measures projection.
        sic_threshold: SIC detection threshold.
        cast_layers: Layer indices where CAST intervenes.
        cast_threshold: CAST intervention threshold.
        perplexity_weight: Weight for perplexity regularization loss.
        suffix_token_ids: Suffix token IDs for perplexity loss, shape (1, n_tokens).
        token_position: Soft token placement: "prefix", "suffix", or "infix".
        infix_split: Token index in prompt where soft tokens are inserted (infix only).

    Returns:
        Scalar loss value.
    """
    transformer = model.model
    h, mask = embed_and_mask_with_prefix(
        transformer, soft_embeds, prompt_token_ids,
        token_position=token_position, infix_split=infix_split,
    )

    n_prompt = prompt_token_ids.shape[1]

    # Per-layer direction penalty accumulation
    direction_penalty = ops.array(0.0)
    defense_penalty = ops.array(0.0)
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
        if defense_aware_weight > 0.0:
            defense_penalty = defense_penalty + _compute_defense_aware_penalty(
                h, i, direction, sic_layer, sic_threshold,
                cast_layers, cast_threshold,
            )

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

    # Defense-aware evasion penalty — note this pushes toward evasion
    # even in defensive mode, which is intentional: a defensive prompt
    # that is also undetectable tests defense robustness
    if defense_aware_weight > 0.0:
        loss = loss + defense_aware_weight * defense_penalty

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

    # Perplexity regularization
    loss = _add_perplexity_term(
        loss, logits, perplexity_weight, suffix_token_ids,
        n_tokens, n_prompt, token_position, infix_split,
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
        defense_aware_weight: Weight for defense-aware evasion penalty.
            0 disables the penalty.
        sic_layer: Layer index where SIC measures projection.
        sic_threshold: SIC detection threshold.
        cast_layers: Layer indices where CAST intervenes.
        cast_threshold: CAST intervention threshold.
        perplexity_weight: Weight for perplexity regularization loss.
        suffix_token_ids: Suffix token IDs for perplexity loss, shape (1, n_tokens).
        token_position: Soft token placement: "prefix", "suffix", or "infix".
        infix_split: Token index in prompt where soft tokens are inserted (infix only).

    Returns:
        Scalar loss value.
    """
    transformer = model.model
    h, mask = embed_and_mask_with_prefix(
        transformer, soft_embeds, prompt_token_ids,
        token_position=token_position, infix_split=infix_split,
    )

    n_prompt = prompt_token_ids.shape[1]

    # Per-layer direction penalty accumulation
    direction_penalty = ops.array(0.0)
    defense_penalty = ops.array(0.0)
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
        if defense_aware_weight > 0.0:
            defense_penalty = defense_penalty + _compute_defense_aware_penalty(
                h, i, direction, sic_layer, sic_threshold,
                cast_layers, cast_threshold,
            )

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

    # Defense-aware evasion penalty
    if defense_aware_weight > 0.0:
        loss = loss + defense_aware_weight * defense_penalty

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

    # Perplexity regularization
    loss = _add_perplexity_term(
        loss, logits, perplexity_weight, suffix_token_ids,
        n_tokens, n_prompt, token_position, infix_split,
    )

    return loss


# ---------------------------------------------------------------------------
# Externality loss (Steering Externalities)
# ---------------------------------------------------------------------------


def _soft_token_offset(
    token_position: str,
    n_prompt: int,
    infix_split: int | None,
) -> int:
    """Compute the sequence offset of soft tokens based on placement mode."""
    if token_position == "suffix":
        return n_prompt
    if token_position == "infix" and infix_split is not None:
        return infix_split
    return 0  # prefix


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
    """Compute externality loss: minimize perplexity + maximize safety erosion.

    Dual objective from Steering Externalities (Xiong et al., 2026):
    - Minimize perplexity: push the suffix toward fluent, benign-looking text
    - Maximize projection onto target direction: erode safety margin

    The suffix should look innocuous while consuming the model's safety margin.

    Args:
        model: The causal language model.
        soft_embeds: Learnable prefix, shape (1, n_tokens, d_model).
        prompt_token_ids: Prompt token IDs, shape (1, seq_len).
        n_tokens: Number of soft prompt tokens.
        target_direction: Direction to maximize projection onto (erosion target).
        direction_weight: Weight for erosion term.
        perplexity_weight: Weight for perplexity term (fluency).
        suffix_token_ids: Suffix token IDs for perplexity loss.
        token_position: "prefix", "suffix", or "infix".
        infix_split: Infix split position.
    """
    transformer = model.model
    prompt_embeds = transformer.embed_tokens(prompt_token_ids)

    # Assemble embeddings based on token position
    if token_position == "suffix":
        h = ops.concatenate([prompt_embeds, soft_embeds], axis=1)
    elif token_position == "infix" and infix_split is not None:
        part1 = prompt_embeds[:, :infix_split, :]
        part2 = prompt_embeds[:, infix_split:, :]
        h = ops.concatenate([part1, soft_embeds, part2], axis=1)
    else:
        h = ops.concatenate([soft_embeds, prompt_embeds], axis=1)

    mask = _nn.create_additive_causal_mask(h.shape[1])
    mask = mask.astype(h.dtype)

    n_prompt = prompt_token_ids.shape[1]

    for layer in transformer.layers:
        h = layer(h, mask)

    h = transformer.norm(h)
    logits = lm_head_forward(model, h)

    # Perplexity loss (benign appearance)
    soft_offset = _soft_token_offset(token_position, n_prompt, infix_split)
    ppl_loss = (
        _compute_perplexity_loss(
            logits, suffix_token_ids, n_tokens, soft_offset,
        )
        if suffix_token_ids is not None and perplexity_weight > 0
        else ops.array(0.0)
    )

    # Safety erosion: maximize projection onto target direction
    erosion_loss = ops.array(0.0)
    if target_direction is not None and direction_weight > 0.0:
        last_pos = n_tokens + n_prompt - 1
        last_hidden = h[:, last_pos, :]
        proj = ops.sum(last_hidden * target_direction)
        # Negative = maximize erosion (push along target direction)
        erosion_loss = -direction_weight * proj

    return perplexity_weight * ppl_loss + erosion_loss
