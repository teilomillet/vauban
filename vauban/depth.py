"""Deep-thinking token analysis: DTR metric and depth direction extraction.

Based on Chen et al. 2026 ("Think Deep, Not Just Long", arxiv 2602.13517).
Measures per-token settling depth via JSD between intermediate and final layer
logit distributions. Classifies tokens as "deep-thinking" vs "shallow".
"""

import math

import mlx.core as mx
import mlx.nn as nn

from vauban.probe import _make_cache
from vauban.types import (
    CausalLM,
    DepthConfig,
    DepthDirectionResult,
    DepthResult,
    DirectionResult,
    LayerCache,
    TokenDepth,
    Tokenizer,
)


def depth_profile(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    config: DepthConfig,
) -> DepthResult:
    """Static depth analysis (prompt-only, no generation).

    Single forward pass, capture hidden states at each layer for all token
    positions, compute JSD profile per token.
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not isinstance(text, str):
        msg = "apply_chat_template must return str when tokenize=False"
        raise TypeError(msg)
    token_ids_list = tokenizer.encode(text)
    token_ids = mx.array(token_ids_list)[None, :]

    transformer = model.model
    h = transformer.embed_tokens(token_ids)
    mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
    mask = mask.astype(h.dtype)

    # Collect hidden states at each layer
    hidden_states: list[mx.array] = []
    for layer in transformer.layers:
        h = layer(h, mask)
        hidden_states.append(h)
        mx.eval(h)

    num_layers = len(hidden_states)
    seq_len = hidden_states[0].shape[1]

    # Get final-layer logits top-k indices for efficiency
    final_h = hidden_states[-1]
    final_normed = transformer.norm(final_h)
    final_logits_full = _lm_head_forward(model, final_normed)
    mx.eval(final_logits_full)

    # Build per-token results
    tokens: list[TokenDepth] = []
    deep_threshold_layer = math.ceil((1 - config.deep_fraction) * num_layers)

    for t in range(seq_len):
        # Final layer distribution (top-k)
        final_logits_t = final_logits_full[0, t, :]
        if config.top_k_logits < final_logits_t.shape[0]:
            top_k_indices = mx.argpartition(
                final_logits_t, kth=final_logits_t.shape[0] - config.top_k_logits,
            )[final_logits_t.shape[0] - config.top_k_logits :]
        else:
            top_k_indices = mx.arange(final_logits_t.shape[0])
        mx.eval(top_k_indices)

        final_probs = mx.softmax(final_logits_t[top_k_indices])
        mx.eval(final_probs)

        jsd_profile: list[float] = []
        for layer_idx in range(num_layers):
            layer_h = hidden_states[layer_idx][0, t, :]
            layer_normed = transformer.norm(layer_h[None, None, :])
            layer_logits = _lm_head_forward(model, layer_normed)
            layer_logits_t = layer_logits[0, 0, :]
            layer_probs = mx.softmax(layer_logits_t[top_k_indices])
            mx.eval(layer_probs)

            jsd_val = _jsd(final_probs, layer_probs)
            jsd_profile.append(jsd_val)

        settling = _settling_depth(jsd_profile, config.settling_threshold)
        is_deep = settling >= deep_threshold_layer

        token_id = token_ids_list[t]
        token_str = tokenizer.decode([token_id])

        tokens.append(TokenDepth(
            token_id=token_id,
            token_str=token_str,
            settling_depth=settling,
            is_deep_thinking=is_deep,
            jsd_profile=jsd_profile,
        ))

    return _build_depth_result(tokens, num_layers, config, prompt)


def depth_generate(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    config: DepthConfig,
) -> DepthResult:
    """Generation-mode depth analysis.

    Token-by-token generation with KV cache, compute JSD per generated token.
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not isinstance(text, str):
        msg = "apply_chat_template must return str when tokenize=False"
        raise TypeError(msg)
    input_ids = mx.array(tokenizer.encode(text))[None, :]

    transformer = model.model
    num_layers = len(transformer.layers)
    cache = _make_cache(model)
    deep_threshold_layer = math.ceil((1 - config.deep_fraction) * num_layers)

    # Prefill: run through prompt to populate cache
    _prefill_cache(model, input_ids, cache)

    # Generate tokens
    tokens: list[TokenDepth] = []
    # Get first token by running the full prompt
    logits = _cached_forward_all_hidden(model, input_ids, cache=None)
    next_token_logits = logits[:, -1:, :]
    next_token = mx.argmax(next_token_logits[0, 0, :])
    mx.eval(next_token)

    # Re-create cache for generation with hidden state capture
    cache = _make_cache(model)
    _prefill_cache(model, input_ids, cache)

    current_token = next_token[None, None]

    for _ in range(config.max_tokens):
        token_id = int(current_token.item())

        # Forward pass capturing all hidden states
        hidden_states = _forward_collect_hidden(model, current_token, cache)

        # Final layer logits
        final_h = hidden_states[-1]
        final_normed = transformer.norm(final_h)
        final_logits = _lm_head_forward(model, final_normed)
        mx.eval(final_logits)

        final_logits_t = final_logits[0, 0, :]
        if config.top_k_logits < final_logits_t.shape[0]:
            top_k_indices = mx.argpartition(
                final_logits_t,
                kth=final_logits_t.shape[0] - config.top_k_logits,
            )[final_logits_t.shape[0] - config.top_k_logits :]
        else:
            top_k_indices = mx.arange(final_logits_t.shape[0])
        mx.eval(top_k_indices)

        final_probs = mx.softmax(final_logits_t[top_k_indices])
        mx.eval(final_probs)

        jsd_profile: list[float] = []
        for layer_idx in range(num_layers):
            layer_h = hidden_states[layer_idx]
            layer_normed = transformer.norm(layer_h)
            layer_logits = _lm_head_forward(model, layer_normed)
            layer_probs = mx.softmax(layer_logits[0, 0, :][top_k_indices])
            mx.eval(layer_probs)
            jsd_profile.append(_jsd(final_probs, layer_probs))

        settling = _settling_depth(jsd_profile, config.settling_threshold)
        is_deep = settling >= deep_threshold_layer
        token_str = tokenizer.decode([token_id])

        tokens.append(TokenDepth(
            token_id=token_id,
            token_str=token_str,
            settling_depth=settling,
            is_deep_thinking=is_deep,
            jsd_profile=jsd_profile,
        ))

        # Next token
        next_token = mx.argmax(final_logits[0, 0, :])
        mx.eval(next_token)
        current_token = next_token[None, None]

    return _build_depth_result(tokens, num_layers, config, prompt)


def depth_direction(
    model: CausalLM,
    tokenizer: Tokenizer,
    depth_results: list[DepthResult],
    refusal_direction: DirectionResult | None = None,
    clip_quantile: float = 0.0,
) -> DepthDirectionResult:
    """Extract a depth direction from DTR classification.

    Splits prompts by median DTR into deep/shallow groups, then computes
    difference-in-means direction using the existing measure() infrastructure.
    """
    from vauban.measure import measure

    # Compute per-prompt DTR and sort
    dtr_scores = [(r.deep_thinking_ratio, r.prompt) for r in depth_results]
    dtr_scores.sort(key=lambda x: x[0])

    if len(dtr_scores) < 2:
        msg = "Need at least 2 prompts for depth direction extraction"
        raise ValueError(msg)

    # Median split
    median_idx = len(dtr_scores) // 2
    median_dtr = dtr_scores[median_idx][0]

    shallow_prompts = [p for dtr, p in dtr_scores[:median_idx]]
    deep_prompts = [p for dtr, p in dtr_scores[median_idx:]]

    # Ensure both groups are non-empty
    if not shallow_prompts or not deep_prompts:
        msg = "DTR median split produced an empty group"
        raise ValueError(msg)

    # Use measure() with deep=harmful slot, shallow=harmless slot
    direction_result = measure(
        model, tokenizer,
        deep_prompts, shallow_prompts,
        clip_quantile,
    )

    # Compute cosine with refusal direction if available
    refusal_cosine: float | None = None
    if refusal_direction is not None:
        cos = mx.sum(direction_result.direction * refusal_direction.direction)
        norm_d = mx.linalg.norm(direction_result.direction)
        norm_r = mx.linalg.norm(refusal_direction.direction)
        denom = norm_d * norm_r
        if float(denom.item()) > 0:
            cos_val = cos / denom
            mx.eval(cos_val)
            refusal_cosine = float(cos_val.item())
        else:
            refusal_cosine = 0.0

    return DepthDirectionResult(
        direction=direction_result.direction,
        layer_index=direction_result.layer_index,
        cosine_scores=direction_result.cosine_scores,
        d_model=direction_result.d_model,
        refusal_cosine=refusal_cosine,
        deep_prompts=deep_prompts,
        shallow_prompts=shallow_prompts,
        median_dtr=median_dtr,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _jsd(p: mx.array, q: mx.array) -> float:
    """Jensen-Shannon divergence between two probability distributions.

    Both p and q must be valid probability distributions (non-negative, sum to 1).
    Returns a value in [0, ln(2)] ≈ [0, 0.693].
    """
    eps = 1e-10
    m = 0.5 * (p + q)
    # KL(p || m)
    kl_pm = mx.sum(p * mx.log(p / (m + eps) + eps))
    # KL(q || m)
    kl_qm = mx.sum(q * mx.log(q / (m + eps) + eps))
    jsd = 0.5 * (kl_pm + kl_qm)
    mx.eval(jsd)
    return max(0.0, float(jsd.item()))


def _settling_depth(jsd_profile: list[float], threshold: float) -> int:
    """First layer l where JSD[l] drops below threshold.

    Returns the layer index. If JSD never drops below threshold,
    returns len(jsd_profile) - 1 (last layer).
    """
    for i, jsd_val in enumerate(jsd_profile):
        if jsd_val <= threshold:
            return i
    return len(jsd_profile) - 1


def _lm_head_forward(model: CausalLM, h: mx.array) -> mx.array:
    """Apply the language model head to get logits."""
    if hasattr(model, "lm_head"):
        lm_head: nn.Module = model.lm_head  # type: ignore[attr-defined]
        return lm_head(h)
    return model.model.embed_tokens.as_linear(h)


def _prefill_cache(
    model: CausalLM,
    token_ids: mx.array,
    cache: list[LayerCache],
) -> mx.array:
    """Run input tokens through the model to populate the KV cache."""
    transformer = model.model
    h = transformer.embed_tokens(token_ids)
    mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
    mask = mask.astype(h.dtype)

    for i, layer in enumerate(transformer.layers):
        h = layer(h, mask, cache=cache[i])
    mx.eval(h)
    return h


def _forward_collect_hidden(
    model: CausalLM,
    token_ids: mx.array,
    cache: list[LayerCache],
) -> list[mx.array]:
    """Forward pass collecting hidden states at each layer."""
    transformer = model.model
    h = transformer.embed_tokens(token_ids)
    mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
    mask = mask.astype(h.dtype)

    hidden_states: list[mx.array] = []
    for i, layer in enumerate(transformer.layers):
        h = layer(h, mask, cache=cache[i])
        hidden_states.append(h)
        mx.eval(h)

    return hidden_states


def _cached_forward_all_hidden(
    model: CausalLM,
    token_ids: mx.array,
    cache: list[LayerCache] | None = None,
) -> mx.array:
    """Full forward pass returning logits (for initial token selection)."""
    transformer = model.model
    h = transformer.embed_tokens(token_ids)
    mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
    mask = mask.astype(h.dtype)

    if cache is not None:
        for i, layer in enumerate(transformer.layers):
            h = layer(h, mask, cache=cache[i])
    else:
        for layer in transformer.layers:
            h = layer(h, mask)

    h = transformer.norm(h)
    logits = _lm_head_forward(model, h)
    mx.eval(logits)
    return logits


def _build_depth_result(
    tokens: list[TokenDepth],
    num_layers: int,
    config: DepthConfig,
    prompt: str,
) -> DepthResult:
    """Aggregate per-token results into a DepthResult."""
    deep_count = sum(1 for t in tokens if t.is_deep_thinking)
    total = len(tokens)
    dtr = deep_count / total if total > 0 else 0.0
    mean_depth = (
        sum(t.settling_depth for t in tokens) / total if total > 0 else 0.0
    )

    return DepthResult(
        tokens=tokens,
        deep_thinking_ratio=dtr,
        deep_thinking_count=deep_count,
        mean_settling_depth=mean_depth,
        layer_count=num_layers,
        settling_threshold=config.settling_threshold,
        deep_fraction=config.deep_fraction,
        prompt=prompt,
    )
