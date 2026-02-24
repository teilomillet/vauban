"""Preprocessing and utility helpers for soft prompt attacks."""

import math

import mlx.core as mx
import mlx.nn as nn

from vauban.evaluate import DEFAULT_REFUSAL_PHRASES
from vauban.softprompt._loss import (
    _compute_defensive_loss,
    _compute_loss,
    _compute_untargeted_loss,
)
from vauban.types import CausalLM, Tokenizer


def _compute_learning_rate(
    base_lr: float,
    step: int,
    n_steps: int,
    schedule: str,
) -> float:
    """Compute learning rate for the given step.

    Args:
        base_lr: Base learning rate.
        step: Current step (0-indexed).
        n_steps: Total number of steps.
        schedule: "constant" or "cosine".

    Returns:
        Learning rate for this step.
    """
    if schedule == "cosine" and n_steps > 1:
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * step / (n_steps - 1)))
    return base_lr


def _build_vocab_mask(
    tokenizer: Tokenizer,
    vocab_size: int,
    constraint: str | None,
) -> mx.array | None:
    """Build a boolean mask of allowed token IDs for constrained search.

    Args:
        tokenizer: Tokenizer with decode support.
        vocab_size: Size of the vocabulary.
        constraint: "ascii", "alpha", "alphanumeric", or None.

    Returns:
        Boolean mask of shape (vocab_size,), or None if constraint is None.
    """
    if constraint is None:
        return None

    allowed = mx.zeros((vocab_size,), dtype=mx.bool_)
    for tid in range(vocab_size):
        text = tokenizer.decode([tid])
        if constraint == "ascii":
            ok = all(32 <= ord(c) < 127 for c in text) and len(text) > 0
        elif constraint == "alpha":
            ok = text.isalpha() and len(text) > 0
        elif constraint == "alphanumeric":
            ok = all(c.isalnum() or c == " " for c in text) and len(text) > 0
        else:
            msg = f"Unknown token constraint: {constraint!r}"
            raise ValueError(msg)
        if ok:
            allowed[tid] = True
    mx.eval(allowed)
    return allowed


def _compute_embed_regularization(
    soft_embeds: mx.array,
    embed_matrix: mx.array,
    weight: float,
) -> mx.array:
    """L2 penalty on embedding norm deviation (Huang et al.).

    Penalizes the soft prompt embeddings when their mean norm
    differs from the mean norm of real token embeddings.

    Args:
        soft_embeds: Soft prompt embeddings, shape (1, n_tokens, d_model).
        embed_matrix: Token embedding matrix, shape (vocab_size, d_model).
        weight: Regularization weight.

    Returns:
        Scalar regularization loss.
    """
    mean_soft_norm = mx.mean(mx.linalg.norm(soft_embeds[0], axis=-1))
    mean_real_norm = mx.mean(mx.linalg.norm(embed_matrix, axis=-1))
    return weight * (mean_soft_norm - mean_real_norm) ** 2


def _pre_encode_prompts(
    tokenizer: Tokenizer,
    prompts: list[str],
) -> list[mx.array]:
    """Pre-tokenize all prompts into token ID arrays.

    Args:
        tokenizer: Tokenizer with encode and chat template support.
        prompts: List of prompt strings.

    Returns:
        List of token ID arrays, each shape (1, seq_len).
    """
    encoded: list[mx.array] = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        if not isinstance(text, str):
            msg = "apply_chat_template must return str when tokenize=False"
            raise TypeError(msg)
        ids = mx.array(tokenizer.encode(text))[None, :]
        encoded.append(ids)
    return encoded


def _select_prompt_ids(
    all_ids: list[mx.array],
    step: int,
    strategy: str,
) -> list[mx.array]:
    """Select prompt ID arrays for this optimization step.

    Args:
        all_ids: All pre-encoded prompt ID arrays.
        step: Current optimization step (0-indexed).
        strategy: "all", "cycle", or "first".

    Returns:
        Subset of prompt ID arrays to use this step.
    """
    if strategy == "first":
        return [all_ids[0]]
    if strategy == "cycle":
        idx = step % len(all_ids)
        return [all_ids[idx]]
    # "all"
    return all_ids


def _compute_accessibility_score(final_loss: float) -> float:
    """Compute accessibility score from final loss (Nordby metric).

    Args:
        final_loss: Final optimization loss value.

    Returns:
        Accessibility score in (0, 1].
    """
    return math.exp(-final_loss)


def _compute_per_prompt_losses(
    model: CausalLM,
    soft_embeds: mx.array,
    all_ids: list[mx.array],
    target_ids: mx.array,
    n_tokens: int,
    direction: mx.array | None,
    direction_weight: float,
    direction_mode: str = "last",
    direction_layers: set[int] | None = None,
    eos_token_id: int | None = None,
    eos_loss_mode: str = "none",
    eos_loss_weight: float = 0.0,
    ref_model: CausalLM | None = None,
    kl_ref_weight: float = 0.0,
    loss_mode: str = "targeted",
    refusal_ids: mx.array | None = None,
) -> list[float]:
    """Compute final loss for each prompt individually.

    Args:
        model: The causal language model.
        soft_embeds: Optimized soft prompt embeddings.
        all_ids: Pre-encoded prompt token ID arrays.
        target_ids: Target token IDs.
        n_tokens: Number of soft prompt tokens.
        direction: Optional refusal direction vector.
        direction_weight: Weight for direction auxiliary loss.
        direction_mode: Direction penalty mode ("last", "raid", "all_positions").
        direction_layers: Layer indices for direction penalty (None = all).
        eos_token_id: EOS token ID for EOS loss.
        eos_loss_mode: "none", "force", or "suppress".
        eos_loss_weight: Weight for EOS auxiliary loss.
        ref_model: Reference model for KL collision loss.
        kl_ref_weight: Weight for KL collision loss.
        loss_mode: "targeted", "untargeted", or "defensive".
        refusal_ids: Refusal token IDs (required for untargeted/defensive).

    Returns:
        List of loss values, one per prompt.
    """
    losses: list[float] = []
    for prompt_ids in all_ids:
        if loss_mode == "defensive" and refusal_ids is not None:
            loss = _compute_defensive_loss(
                model, soft_embeds, prompt_ids,
                n_tokens, refusal_ids,
                direction, direction_weight,
                direction_mode, direction_layers,
                eos_token_id, eos_loss_mode, eos_loss_weight,
                ref_model, kl_ref_weight,
            )
        elif loss_mode == "untargeted" and refusal_ids is not None:
            loss = _compute_untargeted_loss(
                model, soft_embeds, prompt_ids,
                n_tokens, refusal_ids,
                direction, direction_weight,
                direction_mode, direction_layers,
                eos_token_id, eos_loss_mode, eos_loss_weight,
                ref_model, kl_ref_weight,
            )
        else:
            loss = _compute_loss(
                model, soft_embeds, prompt_ids, target_ids,
                n_tokens, direction, direction_weight,
                direction_mode, direction_layers,
                eos_token_id, eos_loss_mode, eos_loss_weight,
                ref_model, kl_ref_weight,
            )
        mx.eval(loss)
        losses.append(float(loss.item()))
    return losses


def _encode_refusal_tokens(tokenizer: Tokenizer) -> mx.array:
    """Encode refusal phrases into a deduplicated set of first-token IDs.

    Each refusal phrase is tokenized and only the first token is kept,
    as the first token is the strongest signal for refusal detection.

    Args:
        tokenizer: Tokenizer with encode support.

    Returns:
        1-D array of deduplicated refusal token IDs.
    """
    seen: set[int] = set()
    ids: list[int] = []
    for phrase in DEFAULT_REFUSAL_PHRASES:
        tokens = tokenizer.encode(phrase)
        if tokens and tokens[0] not in seen:
            seen.add(tokens[0])
            ids.append(tokens[0])
    return mx.array(ids)


def _forward_with_prefix(
    model: CausalLM,
    soft_embeds: mx.array,
    prompt_token_ids: mx.array,
) -> mx.array:
    """Forward pass with soft prefix prepended to prompt embeddings.

    Args:
        model: The causal language model.
        soft_embeds: Learnable prefix embeddings, shape (1, n_tokens, d_model).
        prompt_token_ids: Token IDs for the prompt, shape (1, seq_len).

    Returns:
        Logits tensor, shape (1, n_tokens + seq_len, vocab_size).
    """
    transformer = model.model
    prompt_embeds = transformer.embed_tokens(prompt_token_ids)
    h = mx.concatenate([soft_embeds, prompt_embeds], axis=1)

    mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
    mask = mask.astype(h.dtype)

    for layer in transformer.layers:
        h = layer(h, mask)

    h = transformer.norm(h)

    if hasattr(model, "lm_head"):
        lm_head: nn.Module = model.lm_head  # type: ignore[attr-defined]
        logits: mx.array = lm_head(h)
    else:
        logits = transformer.embed_tokens.as_linear(h)

    return logits


def _encode_targets(
    tokenizer: Tokenizer,
    target_prefixes: list[str],
) -> mx.array:
    """Encode target prefix strings into a flat token ID array.

    Args:
        tokenizer: Tokenizer with encode support.
        target_prefixes: Target strings to encode.

    Returns:
        1-D array of token IDs.
    """
    all_ids: list[int] = []
    for prefix in target_prefixes:
        ids = tokenizer.encode(prefix)
        all_ids.extend(ids)
    return mx.array(all_ids)
