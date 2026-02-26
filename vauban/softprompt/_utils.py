"""Preprocessing and utility helpers for soft prompt attacks."""

import math

import mlx.core as mx

from vauban._array import Array
from vauban._forward import embed_and_mask_with_prefix, force_eval, lm_head_forward
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
) -> Array | None:
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
    force_eval(allowed)
    return allowed


def _compute_embed_regularization(
    soft_embeds: Array,
    embed_matrix: Array,
    weight: float,
) -> Array:
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
) -> list[Array]:
    """Pre-tokenize all prompts into token ID arrays.

    Args:
        tokenizer: Tokenizer with encode and chat template support.
        prompts: List of prompt strings.

    Returns:
        List of token ID arrays, each shape (1, seq_len).
    """
    encoded: list[Array] = []
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
    all_ids: list[Array],
    step: int,
    strategy: str,
) -> list[Array]:
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
) -> list[Array]:
    """Select the top-k hardest prompts by loss (worst-k strategy).

    Computes per-prompt losses with stopped gradients, sorts descending,
    and returns the top-k prompt ID arrays.

    Args:
        model: The causal language model.
        soft_embeds: Current soft prompt embeddings (gradients stopped).
        all_ids: Pre-encoded prompt token ID arrays.
        target_ids: Target token IDs.
        n_tokens: Number of soft prompt tokens.
        k: Number of prompts to select.
        direction: Optional refusal direction vector.
        direction_weight: Weight for direction auxiliary loss.
        direction_mode: Direction penalty mode.
        direction_layers: Layer indices for direction penalty.
        eos_token_id: EOS token ID for EOS loss.
        eos_loss_mode: EOS loss mode.
        eos_loss_weight: Weight for EOS auxiliary loss.
        ref_model: Reference model for KL collision loss.
        kl_ref_weight: Weight for KL collision loss.
        loss_mode: Loss mode ("targeted", "untargeted", or "defensive").
        refusal_ids: Refusal token IDs.

    Returns:
        Top-k prompt ID arrays sorted by descending loss.
    """
    stopped_embeds = mx.stop_gradient(soft_embeds)
    losses = _compute_per_prompt_losses(
        model, stopped_embeds, all_ids, target_ids,
        n_tokens, direction, direction_weight,
        direction_mode, direction_layers,
        eos_token_id, eos_loss_mode, eos_loss_weight,
        ref_model, kl_ref_weight,
        loss_mode=loss_mode, refusal_ids=refusal_ids,
    )
    # Sort by loss descending, take top k
    effective_k = min(k, len(all_ids))
    indexed = sorted(enumerate(losses), key=lambda x: x[1], reverse=True)
    return [all_ids[i] for i, _ in indexed[:effective_k]]


def _split_into_batches(
    items: list[Array],
    n_batches: int,
) -> list[list[Array]]:
    """Split items into approximately equal batches.

    Args:
        items: List of items to split.
        n_batches: Target number of batches.

    Returns:
        List of batches. Returns [items] when n_batches <= 1.
    """
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


def _project_to_tokens(
    soft_embeds: Array,
    embed_matrix: Array,
) -> list[int]:
    """Project continuous embeddings to nearest discrete tokens.

    Args:
        soft_embeds: Continuous embeddings, shape (1, n_tokens, d_model).
        embed_matrix: Token embedding matrix, shape (vocab_size, d_model).

    Returns:
        List of token IDs (one per soft prompt position).
    """
    # (n_tokens, d_model) @ (d_model, vocab_size) -> (n_tokens, vocab_size)
    scores = soft_embeds[0] @ embed_matrix.T
    token_ids_array = mx.argmax(scores, axis=-1)
    force_eval(token_ids_array)
    raw = token_ids_array.tolist()
    if not isinstance(raw, list):
        return [int(raw)]
    return [int(t) for t in raw]


def _encode_refusal_tokens(tokenizer: Tokenizer) -> Array:
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
    soft_embeds: Array,
    prompt_token_ids: Array,
) -> Array:
    """Forward pass with soft prefix prepended to prompt embeddings.

    Args:
        model: The causal language model.
        soft_embeds: Learnable prefix embeddings, shape (1, n_tokens, d_model).
        prompt_token_ids: Token IDs for the prompt, shape (1, seq_len).

    Returns:
        Logits tensor, shape (1, n_tokens + seq_len, vocab_size).
    """
    transformer = model.model
    h, mask = embed_and_mask_with_prefix(transformer, soft_embeds, prompt_token_ids)

    for layer in transformer.layers:
        h = layer(h, mask)

    h = transformer.norm(h)
    return lm_head_forward(model, h)


def _encode_targets(
    tokenizer: Tokenizer,
    target_prefixes: list[str],
) -> Array:
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
