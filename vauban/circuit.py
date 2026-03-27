# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Circuit tracing via activation patching.

Implements causal tracing to identify which model components (layers,
attention, MLP) are causally responsible for behavioral differences
between clean and corrupt inputs.

Granularity:
    - ``"layer"``: Patch entire residual stream at each layer.
    - ``"component"``: Decompose into attention + MLP, patch each
      independently.

Metrics:
    - ``"kl"``: KL divergence between patched and original logits.
    - ``"logit_diff"``: Difference in logit values for target tokens.
"""

from vauban import _ops as ops
from vauban._arch import detect_layer_components
from vauban._array import Array
from vauban._forward import (
    embed_and_mask,
    force_eval,
    get_transformer,
    lm_head_forward,
    make_ssm_mask,
    select_mask,
)
from vauban.types import CausalLM, CircuitResult, ComponentEffect, Tokenizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_causal_mask(h: Array) -> Array:
    """Create a causal attention mask matching h's sequence length."""
    from vauban import _nn

    mask = _nn.create_additive_causal_mask(h.shape[1])
    return mask.astype(h.dtype)


def _match_seq_len(tensor: Array, target_len: int) -> Array:
    """Truncate or pad tensor along sequence dimension to target length.

    Used when clean and corrupt prompts have different tokenization
    lengths during activation patching.

    Args:
        tensor: Shape (batch, seq_len, d_model).
        target_len: Desired sequence length.

    Returns:
        Tensor with sequence dimension matched to target_len.
    """
    src_len = tensor.shape[1]
    if src_len == target_len:
        return tensor
    if src_len > target_len:
        # Truncate: keep the last target_len tokens
        return tensor[:, src_len - target_len :, :]
    # Pad: zero-pad on the left (causal: early positions are least important)
    pad_size = target_len - src_len
    padding = ops.zeros((tensor.shape[0], pad_size, tensor.shape[2]))
    return ops.concatenate([padding, tensor], axis=1)


# ---------------------------------------------------------------------------
# Forward passes with residual caching
# ---------------------------------------------------------------------------


def _forward_cache_residuals(
    model: CausalLM,
    token_ids: Array,
) -> tuple[Array, list[Array]]:
    """Run forward pass and cache residual stream at each layer.

    Args:
        model: The causal language model.
        token_ids: Input token IDs, shape (1, seq_len).

    Returns:
        Tuple of (logits, cached_residuals) where cached_residuals[i]
        is the residual stream after layer i, shape (1, seq_len, d_model).
    """
    transformer = get_transformer(model)
    h, mask = embed_and_mask(transformer, token_ids)

    cached: list[Array] = []
    ssm_mask = make_ssm_mask(transformer, h)
    for layer in transformer.layers:
        h = layer(h, select_mask(layer, mask, ssm_mask))
        cached.append(h)

    h = transformer.norm(h)
    logits = lm_head_forward(model, h)
    force_eval(logits)
    return logits, cached


def _forward_cache_components(
    model: CausalLM,
    token_ids: Array,
) -> tuple[Array, list[tuple[Array, Array]]]:
    """Run forward pass and cache attention + MLP outputs per layer.

    Args:
        model: The causal language model.
        token_ids: Input token IDs, shape (1, seq_len).

    Returns:
        Tuple of (logits, component_outputs) where component_outputs[i]
        is (attn_output, mlp_output) for layer i.
    """
    transformer = get_transformer(model)
    h, mask = embed_and_mask(transformer, token_ids)

    component_outputs: list[tuple[Array, Array]] = []
    ssm_mask = make_ssm_mask(transformer, h)
    for layer in transformer.layers:
        if getattr(layer, "is_linear", False):
            # SSM/linear layers can't be decomposed into attn + mlp;
            # record the full layer delta as (attn=delta, mlp=zeros).
            h_before = h
            h = layer(h, select_mask(layer, mask, ssm_mask))
            delta = h - h_before
            component_outputs.append((delta, ops.zeros_like(delta)))
        else:
            components = detect_layer_components(layer)
            normed = components.input_norm(h)  # type: ignore[operator]
            attn_out = components.self_attn(normed, mask)  # type: ignore[operator]
            h_mid = h + attn_out
            normed_mid = components.post_attn_norm(h_mid)  # type: ignore[operator]
            mlp_out = components.mlp(normed_mid)  # type: ignore[operator]
            h = h_mid + mlp_out
            component_outputs.append((attn_out, mlp_out))

    h = transformer.norm(h)
    logits = lm_head_forward(model, h)
    force_eval(logits)
    return logits, component_outputs


# ---------------------------------------------------------------------------
# Patched forward passes
# ---------------------------------------------------------------------------


def _patched_forward_layer(
    model: CausalLM,
    token_ids: Array,
    patch_layer: int,
    patch_residual: Array,
) -> Array:
    """Forward pass with one layer's residual stream replaced.

    Args:
        model: The causal language model.
        token_ids: Input token IDs, shape (1, seq_len).
        patch_layer: Index of layer to patch.
        patch_residual: Replacement residual stream from clean pass.

    Returns:
        Logits from the patched forward pass.
    """
    transformer = get_transformer(model)
    h, mask = embed_and_mask(transformer, token_ids)

    ssm_mask = make_ssm_mask(transformer, h)
    for i, layer in enumerate(transformer.layers):
        if i == patch_layer:
            h = patch_residual
            mask = _create_causal_mask(h)
            ssm_mask = make_ssm_mask(transformer, h)
        else:
            h = layer(h, select_mask(layer, mask, ssm_mask))

    h = transformer.norm(h)
    logits = lm_head_forward(model, h)
    force_eval(logits)
    return logits


def _patched_forward_component(
    model: CausalLM,
    token_ids: Array,
    patch_layer: int,
    component_name: str,
    clean_component_output: Array,
) -> Array:
    """Forward pass with one component's output replaced at one layer.

    Args:
        model: The causal language model.
        token_ids: Input token IDs, shape (1, seq_len).
        patch_layer: Layer index where patching occurs.
        component_name: ``"attn"`` or ``"mlp"``.
        clean_component_output: Replacement output from clean pass.

    Returns:
        Logits from the patched forward pass.
    """
    transformer = get_transformer(model)
    h, mask = embed_and_mask(transformer, token_ids)

    ssm_mask = make_ssm_mask(transformer, h)
    for i, layer in enumerate(transformer.layers):
        if i == patch_layer:
            if getattr(layer, "is_linear", False):
                # SSM layer — can't decompose; patch full delta if "attn"
                h_before = h
                h = layer(h, select_mask(layer, mask, ssm_mask))
                if component_name == "attn":
                    seq_len = h.shape[1]
                    patched = _match_seq_len(clean_component_output, seq_len)
                    h = h_before + patched
            else:
                seq_len = h.shape[1]
                components = detect_layer_components(layer)
                normed = components.input_norm(h)  # type: ignore[operator]
                attn_out = components.self_attn(normed, mask)  # type: ignore[operator]
                if component_name == "attn":
                    attn_out = _match_seq_len(clean_component_output, seq_len)
                h_mid = h + attn_out
                normed_mid = components.post_attn_norm(h_mid)  # type: ignore[operator]
                mlp_out = components.mlp(normed_mid)  # type: ignore[operator]
                if component_name == "mlp":
                    mlp_out = _match_seq_len(clean_component_output, seq_len)
                h = h_mid + mlp_out
        else:
            h = layer(h, select_mask(layer, mask, ssm_mask))

    h = transformer.norm(h)
    logits = lm_head_forward(model, h)
    force_eval(logits)
    return logits


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _kl_divergence(
    logits_p: Array,
    logits_q: Array,
    token_position: int = -1,
) -> float:
    """Compute KL divergence between two logit distributions at a token position.

    KL(P || Q) where P = softmax(logits_p), Q = softmax(logits_q).

    Args:
        logits_p: Reference logits, shape (1, seq_len, vocab_size).
        logits_q: Comparison logits, shape (1, seq_len, vocab_size).
        token_position: Token index to compare at.

    Returns:
        Scalar KL divergence (non-negative).
    """
    p = ops.softmax(logits_p[0, token_position, :])
    q = ops.softmax(logits_q[0, token_position, :])
    eps = 1e-10
    kl = ops.sum(p * (ops.log(p + eps) - ops.log(q + eps)))
    force_eval(kl)
    return max(0.0, float(kl.item()))


def _logit_diff(
    logits: Array,
    target_tokens: list[int],
    token_position: int = -1,
) -> float:
    """Compute mean logit value for target tokens at a position.

    Args:
        logits: Model logits, shape (1, seq_len, vocab_size).
        target_tokens: Token IDs to measure.
        token_position: Token index to measure at.

    Returns:
        Mean logit value for target tokens.
    """
    pos_logits = logits[0, token_position, :]
    indices = ops.array(target_tokens)
    total = ops.sum(pos_logits[indices])
    force_eval(total)
    return float(total.item()) / len(target_tokens)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def trace_circuit(
    model: CausalLM,
    tokenizer: Tokenizer,
    clean_prompts: list[str],
    corrupt_prompts: list[str],
    *,
    metric: str = "kl",
    granularity: str = "layer",
    layers: list[int] | None = None,
    token_position: int = -1,
    direction: Array | None = None,
    attribute_direction: bool = False,
    logit_diff_tokens: list[int] | None = None,
) -> CircuitResult:
    """Trace causal circuits via activation patching.

    For each layer (or component), patches clean activations into a corrupt
    forward pass and measures the effect on output.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        clean_prompts: Harmless prompts (clean run).
        corrupt_prompts: Harmful prompts (corrupt run).
        metric: Effect metric — ``"kl"`` or ``"logit_diff"``.
        granularity: ``"layer"`` or ``"component"``.
        layers: Specific layers to trace (None = all).
        token_position: Token position for metric computation.
        direction: Refusal direction for attribution (optional).
        attribute_direction: Whether to compute direction attribution.
        logit_diff_tokens: Target token IDs for logit_diff metric.

    Returns:
        CircuitResult with per-component causal effects.
    """
    if metric == "logit_diff" and not logit_diff_tokens:
        msg = "logit_diff metric requires logit_diff_tokens"
        raise ValueError(msg)

    transformer = get_transformer(model)
    n_layers = len(transformer.layers)
    trace_layers = layers if layers is not None else list(range(n_layers))

    # Aggregate effects across prompt pairs
    all_effects: dict[tuple[int, str], list[float]] = {}
    all_attributions: dict[tuple[int, str], list[float]] = {}

    for clean_prompt, corrupt_prompt in zip(
        clean_prompts, corrupt_prompts, strict=True,
    ):
        clean_ids = _tokenize_prompt(tokenizer, clean_prompt)
        corrupt_ids = _tokenize_prompt(tokenizer, corrupt_prompt)

        if granularity == "component":
            effects, attributions = _trace_components(
                model, clean_ids, corrupt_ids, trace_layers,
                metric, token_position, direction,
                attribute_direction, logit_diff_tokens,
            )
        else:
            effects, attributions = _trace_layers(
                model, clean_ids, corrupt_ids, trace_layers,
                metric, token_position, direction,
                attribute_direction, logit_diff_tokens,
            )

        for key, val in effects.items():
            all_effects.setdefault(key, []).append(val)
        for key, val in attributions.items():
            all_attributions.setdefault(key, []).append(val)

    # Average effects across prompt pairs
    result_effects: list[ComponentEffect] = []
    for key in sorted(all_effects):
        layer_idx, comp_name = key
        mean_effect = sum(all_effects[key]) / len(all_effects[key])
        mean_attr: float | None = None
        if key in all_attributions:
            mean_attr = sum(all_attributions[key]) / len(all_attributions[key])
        result_effects.append(ComponentEffect(
            layer=layer_idx,
            component=comp_name,
            effect=mean_effect,
            direction_attribution=mean_attr,
        ))

    return CircuitResult(
        effects=result_effects,
        metric=metric,
        granularity=granularity,
        n_layers=n_layers,
        clean_prompts=clean_prompts,
        corrupt_prompts=corrupt_prompts,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tokenize_prompt(tokenizer: Tokenizer, prompt: str) -> Array:
    """Tokenize a prompt through the chat template."""
    from vauban._forward import encode_user_prompt

    return encode_user_prompt(tokenizer, prompt)


def _trace_layers(
    model: CausalLM,
    clean_ids: Array,
    corrupt_ids: Array,
    trace_layers: list[int],
    metric: str,
    token_position: int,
    direction: Array | None,
    attribute_direction: bool,
    logit_diff_tokens: list[int] | None,
) -> tuple[dict[tuple[int, str], float], dict[tuple[int, str], float]]:
    """Trace at layer granularity."""
    corrupt_logits, _ = _forward_cache_residuals(model, corrupt_ids)
    _, clean_residuals = _forward_cache_residuals(model, clean_ids)

    effects: dict[tuple[int, str], float] = {}
    attributions: dict[tuple[int, str], float] = {}

    for layer_idx in trace_layers:
        patched_logits = _patched_forward_layer(
            model, corrupt_ids, layer_idx, clean_residuals[layer_idx],
        )
        effect = _compute_effect(
            corrupt_logits, patched_logits, metric,
            token_position, logit_diff_tokens,
        )
        effects[(layer_idx, "full")] = effect

        if attribute_direction and direction is not None:
            residual = clean_residuals[layer_idx]
            last_token = residual[0, token_position, :]
            proj = ops.sum(last_token * direction)
            force_eval(proj)
            attributions[(layer_idx, "full")] = float(proj.item())

    return effects, attributions


def _trace_components(
    model: CausalLM,
    clean_ids: Array,
    corrupt_ids: Array,
    trace_layers: list[int],
    metric: str,
    token_position: int,
    direction: Array | None,
    attribute_direction: bool,
    logit_diff_tokens: list[int] | None,
) -> tuple[dict[tuple[int, str], float], dict[tuple[int, str], float]]:
    """Trace at component granularity (attn + mlp)."""
    corrupt_logits, _ = _forward_cache_components(model, corrupt_ids)
    _, clean_components = _forward_cache_components(model, clean_ids)

    effects: dict[tuple[int, str], float] = {}
    attributions: dict[tuple[int, str], float] = {}

    for layer_idx in trace_layers:
        clean_attn, clean_mlp = clean_components[layer_idx]

        for comp_name, clean_out in [("attn", clean_attn), ("mlp", clean_mlp)]:
            patched_logits = _patched_forward_component(
                model, corrupt_ids, layer_idx, comp_name, clean_out,
            )
            effect = _compute_effect(
                corrupt_logits, patched_logits, metric,
                token_position, logit_diff_tokens,
            )
            effects[(layer_idx, comp_name)] = effect

            if attribute_direction and direction is not None:
                proj = ops.sum(clean_out[0, token_position, :] * direction)
                force_eval(proj)
                attributions[(layer_idx, comp_name)] = float(proj.item())

    return effects, attributions


def _compute_effect(
    original_logits: Array,
    patched_logits: Array,
    metric: str,
    token_position: int,
    logit_diff_tokens: list[int] | None,
) -> float:
    """Compute the effect of patching using the chosen metric."""
    if metric == "logit_diff":
        if logit_diff_tokens is None:
            msg = "logit_diff metric requires logit_diff_tokens"
            raise ValueError(msg)
        orig_diff = _logit_diff(original_logits, logit_diff_tokens, token_position)
        patched_diff = _logit_diff(patched_logits, logit_diff_tokens, token_position)
        return abs(patched_diff - orig_diff)
    return _kl_divergence(original_logits, patched_logits, token_position)
