# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Probe activations and steer generation at runtime."""

from typing import TYPE_CHECKING

from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import (
    embed_and_mask,
    encode_user_prompt,
    force_eval,
    get_transformer,
    lm_head_forward,
    make_cache,
    make_ssm_mask,
    select_mask,
)
from vauban.types import CausalLM, LayerCache, ProbeResult, SteerResult, Tokenizer

if TYPE_CHECKING:
    from vauban.svf import SVFBoundary


def probe(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    direction: Array,
) -> ProbeResult:
    """Measure per-layer projection of activations onto a direction.

    Runs a forward pass and returns how strongly each layer's residual
    stream aligns with the given direction.
    """
    token_ids = encode_user_prompt(tokenizer, prompt)

    transformer = get_transformer(model)
    h, mask = embed_and_mask(transformer, token_ids)
    direction = ops.to_device_like(direction, h)

    projections: list[float] = []
    ssm_mask = make_ssm_mask(transformer, h)
    for layer in transformer.layers:
        h = layer(h, select_mask(layer, mask, ssm_mask))
        last_token = h[0, -1, :]
        proj = ops.sum(last_token * direction)
        force_eval(proj)
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
    directions: dict[str, Array],
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
    direction: Array,
    layers: list[int],
    alpha: float = 1.0,
    max_tokens: int = 100,
) -> SteerResult:
    """Generate text while removing a direction at specified layers.

    Manual token-by-token loop with KV cache, intervening between layers.
    """
    token_ids = encode_user_prompt(tokenizer, prompt)

    generated: list[int] = []
    cache = make_cache(model)
    proj_before_all: list[float] = []
    proj_after_all: list[float] = []

    for _ in range(max_tokens):
        logits, p_before, p_after = _steered_forward(
            model, token_ids, direction, layers, alpha, cache,
        )
        next_token = ops.argmax(logits[:, -1, :], axis=-1)
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


def _steered_forward(
    model: CausalLM,
    token_ids: Array,
    direction: Array,
    steer_layers: list[int],
    alpha: float,
    cache: list[LayerCache],
) -> tuple[Array, list[float], list[float]]:
    """Forward pass with mid-layer steering.

    Returns (logits, projections_before, projections_after).
    Cache is mutated in-place.
    """
    transformer = get_transformer(model)
    h, mask = embed_and_mask(transformer, token_ids)
    direction = ops.to_device_like(direction, h)

    proj_before: list[float] = []
    proj_after: list[float] = []
    steer_set = set(steer_layers)
    ssm_mask = make_ssm_mask(transformer, h)

    for i, layer in enumerate(transformer.layers):
        h = layer(h, select_mask(layer, mask, ssm_mask), cache=cache[i])

        if i in steer_set:
            last_token = h[0, -1, :]
            proj = ops.sum(last_token * direction)
            force_eval(proj)
            proj_before.append(float(proj.item()))

            # Steer: remove direction from last token activations
            if float(proj.item()) > 0:
                correction = alpha * proj * direction
                h_list = [h[0, j, :] for j in range(h.shape[1])]
                h_list[-1] = h_list[-1] - correction
                h = ops.stack(h_list)[None, :, :]

            last_after = h[0, -1, :]
            proj_a = ops.sum(last_after * direction)
            force_eval(proj_a)
            proj_after.append(float(proj_a.item()))

    h = transformer.norm(h)
    logits = lm_head_forward(model, h)
    force_eval(logits)
    return logits, proj_before, proj_after


def steer_svf(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    boundary: "SVFBoundary",
    layers: list[int],
    alpha: float = 1.0,
    max_tokens: int = 100,
) -> SteerResult:
    """Generate text with SVF boundary-gradient steering.

    Uses the SVF boundary's gradient as a context-dependent direction at each
    layer, instead of a fixed linear direction.

    Reference: Li, Li & Huang (2026) — arxiv.org/abs/2602.01654
    """
    token_ids = encode_user_prompt(tokenizer, prompt)

    generated: list[int] = []
    cache = make_cache(model)
    scores_before: list[float] = []
    scores_after: list[float] = []

    for _ in range(max_tokens):
        logits, s_before, s_after = _svf_steered_forward(
            model, token_ids, boundary, layers, alpha, cache,
        )
        next_token = ops.argmax(logits[:, -1, :], axis=-1)
        token_id = int(next_token.item())
        generated.append(token_id)
        token_ids = next_token[:, None]

        if s_before:
            scores_before.append(sum(s_before) / len(s_before))
        if s_after:
            scores_after.append(sum(s_after) / len(s_after))

    return SteerResult(
        text=tokenizer.decode(generated),
        projections_before=scores_before,
        projections_after=scores_after,
    )


def _svf_steered_forward(
    model: CausalLM,
    token_ids: Array,
    boundary: "SVFBoundary",
    steer_layers: list[int],
    alpha: float,
    cache: list[LayerCache],
) -> tuple[Array, list[float], list[float]]:
    """Forward pass with SVF boundary-gradient steering.

    At each steer layer, computes the boundary score and gradient. If the
    score is positive (harmful side), steers by removing the gradient
    direction scaled by alpha.
    """
    from vauban.svf import svf_gradient

    transformer = get_transformer(model)
    h, mask = embed_and_mask(transformer, token_ids)

    scores_before: list[float] = []
    scores_after: list[float] = []
    steer_set = set(steer_layers)
    ssm_mask = make_ssm_mask(transformer, h)

    for i, layer in enumerate(transformer.layers):
        h = layer(h, select_mask(layer, mask, ssm_mask), cache=cache[i])

        if i in steer_set:
            last_token = h[0, -1, :]
            score, grad = svf_gradient(boundary, last_token, i)
            scores_before.append(score)

            if score > 0:
                correction = alpha * score * grad
                h_list = [h[0, j, :] for j in range(h.shape[1])]
                h_list[-1] = h_list[-1] - correction
                h = ops.stack(h_list)[None, :, :]

            # Score after steering
            last_after = h[0, -1, :]
            score_after, _ = svf_gradient(boundary, last_after, i)
            scores_after.append(score_after)

    h = transformer.norm(h)
    logits = lm_head_forward(model, h)
    force_eval(logits)
    return logits, scores_before, scores_after
