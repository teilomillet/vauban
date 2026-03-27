# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Direction computation and instruction boundary detection."""

from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import encode_user_prompt, force_eval
from vauban.measure._activations import _clip_activation, _forward_collect
from vauban.types import CausalLM, Tokenizer


def _best_direction(
    harmful_acts: list[Array],
    harmless_acts: list[Array],
) -> tuple[Array, int, list[float]]:
    """Find the best refusal direction across layers.

    Computes difference-in-means at each layer, normalizes, and selects
    the layer with highest cosine separation.

    Args:
        harmful_acts: Per-layer mean activations for harmful prompts.
        harmless_acts: Per-layer mean activations for harmless prompts.

    Returns:
        Tuple of (unit direction, best layer index, per-layer cosine scores).
    """
    num_layers = len(harmful_acts)

    best_layer = 0
    best_score = -1.0
    cosine_scores: list[float] = []

    for i in range(num_layers):
        diff = harmful_acts[i] - harmless_acts[i]
        direction = diff / (ops.linalg.norm(diff) + 1e-8)
        score = _cosine_separation(
            harmful_acts[i], harmless_acts[i], direction,
        )
        cosine_scores.append(float(score.item()))
        if cosine_scores[-1] > best_score:
            best_score = cosine_scores[-1]
            best_layer = i

    # Recompute best direction
    best_diff = harmful_acts[best_layer] - harmless_acts[best_layer]
    best_dir = best_diff / (ops.linalg.norm(best_diff) + 1e-8)
    force_eval(best_dir)

    return best_dir, best_layer, cosine_scores


def find_instruction_boundary(tokenizer: Tokenizer, prompt: str) -> int:
    """Find the instruction-final token position in a chat-templated sequence.

    Tokenizes the chat template with empty content vs. actual content.
    Matches suffix tokens from the end to find the generation prompt
    boundary. The instruction-final token is ``len(full) - suffix_len - 1``.

    Falls back to ``len(full) - 1`` if no suffix is detected.

    Args:
        tokenizer: Tokenizer with chat template support.
        prompt: The user prompt text.

    Returns:
        Token index of the instruction-final position.
    """
    messages_full = [{"role": "user", "content": prompt}]
    messages_empty = [{"role": "user", "content": ""}]

    full_result = tokenizer.apply_chat_template(messages_full, tokenize=True)
    empty_result = tokenizer.apply_chat_template(messages_empty, tokenize=True)

    if isinstance(full_result, str):
        msg = "apply_chat_template must return list[int] when tokenize=True"
        raise TypeError(msg)
    if isinstance(empty_result, str):
        msg = "apply_chat_template must return list[int] when tokenize=True"
        raise TypeError(msg)

    full_ids: list[int] = full_result
    empty_ids: list[int] = empty_result

    suffix_len = _match_suffix(full_ids, empty_ids)
    if suffix_len == 0:
        return len(full_ids) - 1

    return len(full_ids) - suffix_len - 1


def _match_suffix(full: list[int], empty: list[int]) -> int:
    """Count matching tokens from the end of both sequences.

    Args:
        full: Token IDs of the full (content-bearing) template.
        empty: Token IDs of the empty-content template.

    Returns:
        Number of matching suffix tokens.
    """
    count = 0
    i_full = len(full) - 1
    i_empty = len(empty) - 1
    while i_full >= 0 and i_empty >= 0:
        if full[i_full] != empty[i_empty]:
            break
        count += 1
        i_full -= 1
        i_empty -= 1
    return count


def _collect_activations_at_instruction_end(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    clip_quantile: float = 0.0,
) -> list[Array]:
    """Collect per-layer mean activations at the instruction-final token.

    Like ``_collect_activations`` but computes ``find_instruction_boundary``
    per-prompt and passes it to ``_forward_collect``. Needed because the
    instruction boundary varies per prompt.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        prompts: Prompts to collect activations for.
        clip_quantile: If > 0, clip per-prompt activations by this quantile.

    Returns a list of length num_layers, each element shape (d_model,).
    """
    means: list[Array] | None = None

    for count, prompt in enumerate(prompts, start=1):
        token_ids = encode_user_prompt(tokenizer, prompt)
        boundary = find_instruction_boundary(tokenizer, prompt)
        residuals = _forward_collect(model, token_ids, boundary)

        if clip_quantile > 0.0:
            residuals = [_clip_activation(r, clip_quantile) for r in residuals]

        if means is None:
            means = [r.astype(ops.float32) for r in residuals]
        else:
            for i, r in enumerate(residuals):
                delta = r.astype(ops.float32) - means[i]
                means[i] = means[i] + delta / count

        if count % 16 == 0 and means is not None:
            force_eval(*means)

    if means is None:
        msg = "No prompts provided for activation collection"
        raise ValueError(msg)

    force_eval(*means)
    return means


def _cosine_separation(
    harmful_mean: Array,
    harmless_mean: Array,
    direction: Array,
) -> Array:
    """Cosine separation: projection difference onto direction."""
    proj_harmful = ops.sum(harmful_mean * direction)
    proj_harmless = ops.sum(harmless_mean * direction)
    return proj_harmful - proj_harmless
