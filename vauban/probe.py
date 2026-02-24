"""Probe activations and steer generation at runtime."""

import mlx.core as mx
import mlx.nn as nn

from vauban.types import CausalLM, LayerCache, ProbeResult, SteerResult, Tokenizer


def probe(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    direction: mx.array,
) -> ProbeResult:
    """Measure per-layer projection of activations onto a direction.

    Runs a forward pass and returns how strongly each layer's residual
    stream aligns with the given direction.
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not isinstance(text, str):
        msg = "apply_chat_template must return str when tokenize=False"
        raise TypeError(msg)
    token_ids = mx.array(tokenizer.encode(text))[None, :]

    transformer = model.model
    h = transformer.embed_tokens(token_ids)
    mask = nn.MultiHeadAttention.create_additive_causal_mask(
        h.shape[1],
    )
    mask = mask.astype(h.dtype)

    projections: list[float] = []
    for layer in transformer.layers:
        h = layer(h, mask)
        last_token = h[0, -1, :]
        proj = mx.sum(last_token * direction)
        mx.eval(proj)
        projections.append(float(proj.item()))

    return ProbeResult(
        projections=projections,
        layer_count=len(projections),
        prompt=prompt,
    )


def multi_probe(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    directions: dict[str, mx.array],
) -> dict[str, ProbeResult]:
    """Probe activations against multiple named directions."""
    return {
        name: probe(model, tokenizer, prompt, direction)
        for name, direction in directions.items()
    }


def steer(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    direction: mx.array,
    layers: list[int],
    alpha: float = 1.0,
    max_tokens: int = 100,
) -> SteerResult:
    """Generate text while removing a direction at specified layers.

    Manual token-by-token loop with KV cache, intervening between layers.
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not isinstance(text, str):
        msg = "apply_chat_template must return str when tokenize=False"
        raise TypeError(msg)
    token_ids = mx.array(tokenizer.encode(text))[None, :]

    generated: list[int] = []
    cache = _make_cache(model)
    proj_before_all: list[float] = []
    proj_after_all: list[float] = []

    for _ in range(max_tokens):
        logits, p_before, p_after = _steered_forward(
            model, token_ids, direction, layers, alpha, cache,
        )
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        token_id = int(next_token.item())
        generated.append(token_id)
        token_ids = next_token[:, None]

        # Record mean projections across steer layers
        if p_before:
            proj_before_all.append(
                sum(p_before) / len(p_before),
            )
        if p_after:
            proj_after_all.append(
                sum(p_after) / len(p_after),
            )

    return SteerResult(
        text=tokenizer.decode(generated),
        projections_before=proj_before_all,
        projections_after=proj_after_all,
    )


def _make_cache(model: CausalLM) -> list[LayerCache]:
    """Create a KV cache for the model.

    Uses model.make_cache() if available (real mlx-lm and mock),
    otherwise falls back to importing from mlx_lm.
    """
    if hasattr(model, "make_cache"):
        return model.make_cache()  # type: ignore[no-any-return]
    from mlx_lm.models.cache import make_prompt_cache

    return make_prompt_cache(model)  # type: ignore[no-any-return]


def _steered_forward(
    model: CausalLM,
    token_ids: mx.array,
    direction: mx.array,
    steer_layers: list[int],
    alpha: float,
    cache: list[LayerCache],
) -> tuple[mx.array, list[float], list[float]]:
    """Forward pass with mid-layer steering.

    Returns (logits, projections_before, projections_after).
    Cache is mutated in-place.
    """
    transformer = model.model
    h = transformer.embed_tokens(token_ids)

    mask = nn.MultiHeadAttention.create_additive_causal_mask(
        h.shape[1],
    )
    mask = mask.astype(h.dtype)

    proj_before: list[float] = []
    proj_after: list[float] = []
    steer_set = set(steer_layers)

    for i, layer in enumerate(transformer.layers):
        h = layer(h, mask, cache=cache[i])

        if i in steer_set:
            last_token = h[0, -1, :]
            proj = mx.sum(last_token * direction)
            mx.eval(proj)
            proj_before.append(float(proj.item()))

            # Steer: remove direction from last token activations
            if float(proj.item()) > 0:
                correction = alpha * proj * direction
                h_list = [h[0, j, :] for j in range(h.shape[1])]
                h_list[-1] = h_list[-1] - correction
                h = mx.stack(h_list)[None, :, :]

            last_after = h[0, -1, :]
            proj_a = mx.sum(last_after * direction)
            mx.eval(proj_a)
            proj_after.append(float(proj_a.item()))

    h = transformer.norm(h)
    # Compute logits via the lm_head or tied embeddings
    if hasattr(model, "lm_head"):
        lm_head: nn.Module = model.lm_head  # type: ignore[attr-defined]
        logits: mx.array = lm_head(h)
    else:
        logits = transformer.embed_tokens.as_linear(h)

    mx.eval(logits)
    return logits, proj_before, proj_after
