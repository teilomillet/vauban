"""Preprocessing and utility helpers for soft prompt attacks."""

import math
import random
import unicodedata

from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import embed_and_mask_with_prefix, force_eval, lm_head_forward
from vauban.evaluate import DEFAULT_REFUSAL_PHRASES
from vauban.softprompt._loss import (
    _compute_defensive_loss,
    _compute_loss,
    _compute_untargeted_loss,
)
from vauban.types import CausalLM, SoftPromptConfig, Tokenizer


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


def _is_invisible_char(c: str) -> bool:
    """Check if a character is invisible (zero-width, format, non-printable whitespace).

    Matches Unicode categories Cf (format), Zs/Zl/Zp (whitespace),
    Cc (control) minus printable ASCII (0x20-0x7E).
    """
    cat = unicodedata.category(c)
    if cat in {"Cf", "Zl", "Zp"}:
        return True
    if cat == "Zs" and c != " ":
        return True
    return bool(cat == "Cc" and ord(c) > 127)


def _is_emoji_char(c: str) -> bool:
    """Check if a character is an emoji or miscellaneous symbol.

    Matches Unicode category So (Other Symbol) plus common emoji ranges.
    """
    cat = unicodedata.category(c)
    if cat == "So":
        return True
    cp = ord(c)
    # Supplemental emoji ranges
    if 0x1F600 <= cp <= 0x1F64F:  # Emoticons
        return True
    if 0x1F300 <= cp <= 0x1F5FF:  # Misc Symbols and Pictographs
        return True
    if 0x1F680 <= cp <= 0x1F6FF:  # Transport and Map
        return True
    if 0x1F900 <= cp <= 0x1F9FF:  # Supplemental Symbols
        return True
    if 0x2600 <= cp <= 0x26FF:  # Misc Symbols
        return True
    return bool(0x2700 <= cp <= 0x27BF)  # Dingbats


def _matches_constraint(text: str, constraint: str) -> bool:
    """Check if decoded token text matches a single constraint.

    Args:
        text: Decoded token string.
        constraint: Constraint name.

    Returns:
        True if the token satisfies the constraint.

    Raises:
        ValueError: If constraint name is unknown.
    """
    if not text:
        return False
    if constraint == "ascii":
        return all(32 <= ord(c) < 127 for c in text)
    if constraint == "alpha":
        return text.isalpha()
    if constraint == "alphanumeric":
        return all(c.isalnum() or c == " " for c in text)
    if constraint == "non_latin":
        return all(ord(c) > 127 for c in text)
    if constraint == "chinese":
        return all(0x4E00 <= ord(c) <= 0x9FFF for c in text)
    if constraint == "non_alphabetic":
        return all(not c.isalpha() for c in text)
    if constraint == "invisible":
        return all(_is_invisible_char(c) for c in text)
    if constraint == "zalgo":
        return (
            all(
                0x0300 <= ord(c) <= 0x036F or c.isalpha()
                for c in text
            )
            and any(0x0300 <= ord(c) <= 0x036F for c in text)
        )
    if constraint == "emoji":
        return all(_is_emoji_char(c) for c in text)
    msg = f"Unknown token constraint: {constraint!r}"
    raise ValueError(msg)


def _build_vocab_mask(
    tokenizer: Tokenizer,
    vocab_size: int,
    constraint: str | list[str] | None,
) -> Array | None:
    """Build a boolean mask of allowed token IDs for constrained search.

    When multiple constraints are given, a token must satisfy ALL of them
    (intersection). This enables combined stealth attacks like
    ``["non_latin", "invisible"]``.

    Args:
        tokenizer: Tokenizer with decode support.
        vocab_size: Size of the vocabulary.
        constraint: Constraint name, list of names, or None.

    Returns:
        Boolean mask of shape (vocab_size,), or None if constraint is None.
    """
    if constraint is None:
        return None

    constraints = (
        [constraint] if isinstance(constraint, str) else constraint
    )

    allowed = ops.zeros((vocab_size,), dtype=ops.bool_)
    for tid in range(vocab_size):
        text = tokenizer.decode([tid])
        if all(_matches_constraint(text, c) for c in constraints):
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
    mean_soft_norm = ops.mean(ops.linalg.norm(soft_embeds[0], axis=-1))
    mean_real_norm = ops.mean(ops.linalg.norm(embed_matrix, axis=-1))
    return weight * (mean_soft_norm - mean_real_norm) ** 2


def _pre_encode_prompts(
    tokenizer: Tokenizer,
    prompts: list[str],
    system_prompt: str | None = None,
) -> list[Array]:
    """Pre-tokenize all prompts into token ID arrays.

    Args:
        tokenizer: Tokenizer with encode and chat template support.
        prompts: List of prompt strings.
        system_prompt: Optional system prompt to prepend to messages.

    Returns:
        List of token ID arrays, each shape (1, seq_len).
    """
    encoded: list[Array] = []
    for prompt in prompts:
        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        if not isinstance(text, str):
            msg = "apply_chat_template must return str when tokenize=False"
            raise TypeError(msg)
        ids = ops.array(tokenizer.encode(text))[None, :]
        encoded.append(ids)
    return encoded


def _pre_encode_prompts_with_history(
    tokenizer: Tokenizer,
    prompts: list[str],
    history: list[dict[str, str]],
    system_prompt: str | None = None,
) -> list[Array]:
    """Pre-tokenize prompts with conversation history prepended.

    Constructs ``messages = [system_prompt] + history + [user: prompt]``
    for each prompt and tokenizes the full sequence. History turns are
    baked in as hard token IDs so the attack optimizes against the
    full multi-turn context.

    Args:
        tokenizer: Tokenizer with encode and chat template support.
        prompts: List of prompt strings for the current turn.
        history: Previous conversation turns as role/content dicts.
        system_prompt: Optional system prompt to prepend.

    Returns:
        List of token ID arrays, each shape (1, seq_len).
    """
    encoded: list[Array] = []
    for prompt in prompts:
        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        if not isinstance(text, str):
            msg = "apply_chat_template must return str when tokenize=False"
            raise TypeError(msg)
        ids = ops.array(tokenizer.encode(text))[None, :]
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


def _sample_prompt_ids(
    all_ids: list[Array],
    k: int,
) -> list[Array]:
    """Randomly sample k prompts per step (mini-batch SGD).

    Unlike ``_select_worst_k_prompt_ids``, no extra forward pass is
    needed — prompts are drawn uniformly at random.

    Args:
        all_ids: All pre-encoded prompt ID arrays.
        k: Number of prompts to sample.

    Returns:
        Random subset of prompt ID arrays.
    """
    if k >= len(all_ids):
        return all_ids
    return random.sample(all_ids, k)


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
    defense_aware_weight: float = 0.0,
    sic_layer: int | None = None,
    sic_threshold: float = 0.0,
    cast_layers: list[int] | None = None,
    cast_threshold: float = 0.0,
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
                defense_aware_weight, sic_layer, sic_threshold,
                cast_layers, cast_threshold,
            )
        elif loss_mode == "untargeted" and refusal_ids is not None:
            loss = _compute_untargeted_loss(
                model, soft_embeds, prompt_ids,
                n_tokens, refusal_ids,
                direction, direction_weight,
                direction_mode, direction_layers,
                eos_token_id, eos_loss_mode, eos_loss_weight,
                ref_model, kl_ref_weight,
                defense_aware_weight, sic_layer, sic_threshold,
                cast_layers, cast_threshold,
            )
        else:
            loss = _compute_loss(
                model, soft_embeds, prompt_ids, target_ids,
                n_tokens, direction, direction_weight,
                direction_mode, direction_layers,
                eos_token_id, eos_loss_mode, eos_loss_weight,
                ref_model, kl_ref_weight,
                defense_aware_weight, sic_layer, sic_threshold,
                cast_layers, cast_threshold,
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
    stopped_embeds = ops.stop_gradient(soft_embeds)
    losses = _compute_per_prompt_losses(
        model, stopped_embeds, all_ids, target_ids,
        n_tokens, direction, direction_weight,
        direction_mode, direction_layers,
        eos_token_id, eos_loss_mode, eos_loss_weight,
        ref_model, kl_ref_weight,
        loss_mode=loss_mode, refusal_ids=refusal_ids,
        defense_aware_weight=defense_aware_weight,
        sic_layer=sic_layer, sic_threshold=sic_threshold,
        cast_layers=cast_layers, cast_threshold=cast_threshold,
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
    token_ids_array = ops.argmax(scores, axis=-1)
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
    return ops.array(ids)


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
    repeat_count: int = 0,
) -> Array:
    """Encode target prefix strings into a flat token ID array.

    Args:
        tokenizer: Tokenizer with encode support.
        target_prefixes: Target strings to encode.
        repeat_count: If > 0, repeat the base token sequence this many times.

    Returns:
        1-D array of token IDs.
    """
    all_ids: list[int] = []
    for prefix in target_prefixes:
        ids = tokenizer.encode(prefix)
        all_ids.extend(ids)
    if repeat_count > 0:
        all_ids = all_ids * repeat_count
    return ops.array(all_ids)


def _prepare_transfer_data(
    transfer_models: list[tuple[str, CausalLM, Tokenizer]] | None,
    config: SoftPromptConfig,
    prompts: list[str],
) -> list[tuple[CausalLM, Tokenizer, list[Array], Array]]:
    """Pre-encode prompts and targets for each transfer model.

    Returns an empty list when transfer scoring is disabled
    (no models supplied or ``transfer_loss_weight == 0``).

    Args:
        transfer_models: Named transfer model triples.
        config: Soft prompt configuration (reads target_prefixes,
            target_repeat_count, system_prompt, transfer_loss_weight).
        prompts: Attack prompts to encode.

    Returns:
        List of ``(model, tokenizer, prompt_ids, target_ids)`` tuples
        ready for loss evaluation.
    """
    if not transfer_models or config.transfer_loss_weight <= 0.0:
        return []

    effective_prompts = prompts if prompts else ["Hello"]
    result: list[tuple[CausalLM, Tokenizer, list[Array], Array]] = []
    for _name, t_model, t_tok in transfer_models:
        t_prompt_ids = _pre_encode_prompts(
            t_tok, effective_prompts, config.system_prompt,
        )
        t_target_ids = _encode_targets(
            t_tok, config.target_prefixes,
            config.target_repeat_count,
        )
        force_eval(t_target_ids)
        result.append((t_model, t_tok, t_prompt_ids, t_target_ids))
    return result


def _score_transfer_loss(
    token_text: str,
    transfer_data: list[tuple[CausalLM, Tokenizer, list[Array], Array]],
) -> float:
    """Compute mean loss across transfer models for a candidate suffix.

    Re-encodes ``token_text`` with each transfer model's tokenizer,
    computes the target-prefix loss averaged over all pre-encoded
    prompts, then returns the mean across models.

    Args:
        token_text: Decoded candidate suffix text (from the primary
            tokenizer).
        transfer_data: Pre-encoded transfer model tuples from
            :func:`_prepare_transfer_data`.

    Returns:
        Mean loss averaged over all transfer models and prompts.
    """
    total = 0.0
    for t_model, t_tok, t_prompts, t_targets in transfer_data:
        t_tokens = t_tok.encode(token_text)
        t_n = len(t_tokens)
        t_embeds = t_model.model.embed_tokens(
            ops.array(t_tokens)[None, :],
        )
        t_loss = ops.array(0.0)
        for t_pid in t_prompts:
            t_loss = t_loss + _compute_loss(
                t_model, t_embeds, t_pid, t_targets,
                t_n, None, 0.0,
            )
        t_avg = t_loss / len(t_prompts)
        force_eval(t_avg)
        total += float(t_avg.item())
    return total / len(transfer_data)
