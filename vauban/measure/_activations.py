# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Activation collection helpers for the measure pipeline."""

from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import (
    embed_and_mask,
    encode_user_prompt,
    force_eval,
    get_transformer,
    make_ssm_mask,
    select_mask,
)
from vauban.types import CausalLM, Tokenizer


def _collect_activations(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    clip_quantile: float = 0.0,
    token_position: int = -1,
) -> list[Array]:
    """Collect per-layer mean activations across prompts.

    Uses Welford's online algorithm for numerically stable streaming
    mean computation — O(d_model) memory per layer instead of
    O(num_prompts * d_model).

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        prompts: Prompts to collect activations for.
        clip_quantile: If > 0, clip per-prompt activations by this
            quantile before accumulating into the running mean.
        token_position: Token index to extract activations from.
            Defaults to -1 (last token).

    Returns a list of length num_layers, each element shape (d_model,).
    """
    means: list[Array] | None = None

    for count, prompt in enumerate(prompts, start=1):
        token_ids = encode_user_prompt(tokenizer, prompt)
        residuals = _forward_collect(model, token_ids, token_position)

        if clip_quantile > 0.0:
            residuals = [_clip_activation(r, clip_quantile) for r in residuals]

        if means is None:
            means = [r.astype(ops.float32) for r in residuals]
        else:
            # Welford online mean: mean += (x - mean) / n
            for i, r in enumerate(residuals):
                delta = r.astype(ops.float32) - means[i]
                means[i] = means[i] + delta / count

        # Evaluate periodically to avoid graph buildup
        if count % 16 == 0 and means is not None:
            force_eval(*means)

    if means is None:
        msg = "No prompts provided for activation collection"
        raise ValueError(msg)

    force_eval(*means)
    return means


def _collect_per_prompt_activations(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    clip_quantile: float = 0.0,
    token_position: int = -1,
) -> list[Array]:
    """Collect per-prompt activations at each layer (no averaging).

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        prompts: Prompts to collect activations for.
        clip_quantile: If > 0, clip per-prompt activations by this quantile.
        token_position: Token index to extract activations from.
            Defaults to -1 (last token).

    Returns a list of length num_layers, each element shape (num_prompts, d_model).
    """
    if not prompts:
        msg = "prompts must be non-empty"
        raise ValueError(msg)

    all_residuals: list[list[Array]] = []

    for prompt in prompts:
        token_ids = encode_user_prompt(tokenizer, prompt)
        residuals = _forward_collect(model, token_ids, token_position)

        if clip_quantile > 0.0:
            residuals = [_clip_activation(r, clip_quantile) for r in residuals]

        all_residuals.append(residuals)

    # Stack per-prompt activations for each layer
    num_layers = len(all_residuals[0])
    per_layer: list[Array] = []
    for layer_idx in range(num_layers):
        stacked = ops.stack([r[layer_idx] for r in all_residuals])
        force_eval(stacked)
        per_layer.append(stacked)

    return per_layer


def _forward_collect(
    model: CausalLM,
    token_ids: Array,
    token_position: int = -1,
) -> list[Array]:
    """Manual layer-by-layer forward pass, capturing residual stream.

    Returns per-layer activations at the given token position.
    Each element has shape (d_model,).

    Handles hybrid architectures (e.g. Qwen3.5) where some layers use
    standard causal attention and others use SSM/linear attention, each
    requiring a different mask format.

    Args:
        model: The causal language model.
        token_ids: Input token IDs of shape (1, seq_len).
        token_position: Token index to extract activations from.
            Defaults to -1 (last token).
    """
    transformer = get_transformer(model)
    h, attn_mask = embed_and_mask(transformer, token_ids)

    # Build SSM mask for hybrid architectures (Qwen3.5 GatedDeltaNet)
    ssm_mask = make_ssm_mask(transformer, h)

    residuals: list[Array] = []
    for layer in transformer.layers:
        h = layer(h, select_mask(layer, attn_mask, ssm_mask))
        # Upcast to float32 for numerical stability (like Heretic)
        activation = ops.stop_gradient(h[0, token_position, :]).astype(ops.float32)
        residuals.append(activation)

    return residuals


def _clip_activation(activation: Array, quantile: float) -> Array:
    """Winsorize an activation vector by clamping extreme values.

    Clips each dimension to the ``[quantile, 1-quantile]`` range of
    its absolute value distribution. This tames "massive activations"
    that can distort the difference-in-means computation.

    Args:
        activation: Activation vector of shape (d_model,).
        quantile: Fraction of extremes to clip (e.g. 0.01 = clip top/bottom 1%).
    """
    abs_vals = ops.abs(activation)
    sorted_vals = ops.sort(abs_vals)
    n = sorted_vals.shape[0]
    high_idx = min(n - 1, int(n * (1.0 - quantile)))
    threshold = sorted_vals[high_idx]
    force_eval(threshold)
    return ops.clip(activation, -threshold, threshold)
