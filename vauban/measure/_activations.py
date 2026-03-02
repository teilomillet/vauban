"""Activation collection helpers for the measure pipeline."""

from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import embed_and_mask, force_eval
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
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
        )
        if not isinstance(text, str):
            msg = "apply_chat_template must return str when tokenize=False"
            raise TypeError(msg)
        token_ids = ops.array(tokenizer.encode(text))[None, :]
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
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
        )
        if not isinstance(text, str):
            msg = "apply_chat_template must return str when tokenize=False"
            raise TypeError(msg)
        token_ids = ops.array(tokenizer.encode(text))[None, :]
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

    Args:
        model: The causal language model.
        token_ids: Input token IDs of shape (1, seq_len).
        token_position: Token index to extract activations from.
            Defaults to -1 (last token).
    """
    transformer = model.model
    h, mask = embed_and_mask(transformer, token_ids)

    residuals: list[Array] = []
    for layer in transformer.layers:
        h = layer(h, mask)
        # Upcast to float32 for numerical stability (like Heretic)
        activation = h[0, token_position, :].astype(ops.float32)
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
