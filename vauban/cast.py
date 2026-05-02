# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Conditional Activation Steering (CAST) runtime generation."""

from typing import TYPE_CHECKING

from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import (
    embed_and_mask,
    encode_chat_prompt,
    force_eval,
    get_transformer,
    lm_head_forward,
    make_cache,
    make_ssm_mask,
    select_mask,
)
from vauban.types import AlphaTier, CastResult, CausalLM, LayerCache, Tokenizer

if TYPE_CHECKING:
    from vauban.svf import SVFBoundary


def _resolve_alpha(
    projection_value: float,
    base_alpha: float,
    alpha_tiers: list[AlphaTier] | None,
) -> float:
    """Resolve the effective alpha from tiered thresholds.

    Walks sorted tiers (ascending threshold) and returns the alpha of
    the highest tier where ``projection_value >= tier.threshold``.
    Returns ``base_alpha`` if no tiers are configured or no tier matches.
    """
    if alpha_tiers is None or len(alpha_tiers) == 0:
        return base_alpha

    resolved = base_alpha
    for tier in alpha_tiers:
        if projection_value >= tier.threshold:
            resolved = tier.alpha
        else:
            break
    return resolved


def cast_generate(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    direction: Array,
    layers: list[int],
    alpha: float = 1.0,
    threshold: float = 0.0,
    max_tokens: int = 100,
    condition_direction: Array | None = None,
    alpha_tiers: list[AlphaTier] | None = None,
    baseline_activations: dict[int, Array] | None = None,
    displacement_threshold: float = 0.0,
) -> CastResult:
    """Generate text with conditional activation steering.

    Steering is applied only when the per-layer projection of the current
    last-token activation on the detection direction is greater than
    ``threshold``.

    When ``condition_direction`` is provided, it is used for the gating
    check (detect), while ``direction`` is always used for the actual
    steering correction (steer). This implements the dual-direction
    pattern from AdaSteer.

    When ``alpha_tiers`` is provided, the effective alpha is resolved
    from the projection magnitude via ``_resolve_alpha()``.
    """
    messages = [{"role": "user", "content": prompt}]
    return _cast_generate_from_messages(
        model, tokenizer, messages, prompt, direction, layers,
        alpha, threshold, max_tokens, condition_direction, alpha_tiers,
        baseline_activations, displacement_threshold,
    )


def cast_generate_with_messages(
    model: CausalLM,
    tokenizer: Tokenizer,
    messages: list[dict[str, str]],
    direction: Array,
    layers: list[int],
    alpha: float = 1.0,
    threshold: float = 0.0,
    max_tokens: int = 100,
    condition_direction: Array | None = None,
    alpha_tiers: list[AlphaTier] | None = None,
    baseline_activations: dict[int, Array] | None = None,
    displacement_threshold: float = 0.0,
) -> CastResult:
    """Generate text with CAST steering over an arbitrary message list.

    Same as ``cast_generate`` but takes a full message list instead of
    a single prompt string, enabling multi-turn context for steering
    decisions.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        messages: Full conversation as role/content dicts.
        direction: Refusal direction for steering correction.
        layers: Layer indices to apply CAST on.
        alpha: Base steering strength.
        threshold: Minimum projection to trigger steering.
        max_tokens: Maximum tokens to generate.
        condition_direction: Separate detection direction (AdaSteer).
        alpha_tiers: Adaptive alpha tiers (TRYLOCK/AlphaSteer).
        baseline_activations: Per-layer baseline for externality monitoring.
        displacement_threshold: L2 displacement threshold for intervention.

    Returns:
        CastResult with generation and intervention stats.
    """
    prompt_str = ""
    for m in reversed(messages):
        if m["role"] == "user":
            prompt_str = m["content"]
            break
    return _cast_generate_from_messages(
        model, tokenizer, messages, prompt_str, direction, layers,
        alpha, threshold, max_tokens, condition_direction, alpha_tiers,
        baseline_activations, displacement_threshold,
    )


def _cast_generate_from_messages(
    model: CausalLM,
    tokenizer: Tokenizer,
    messages: list[dict[str, str]],
    prompt_label: str,
    direction: Array,
    layers: list[int],
    alpha: float,
    threshold: float,
    max_tokens: int,
    condition_direction: Array | None,
    alpha_tiers: list[AlphaTier] | None,
    baseline_activations: dict[int, Array] | None = None,
    displacement_threshold: float = 0.0,
) -> CastResult:
    """Shared CAST decode loop for single-prompt and multi-turn entry points.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        messages: Full conversation as role/content dicts.
        prompt_label: String stored in ``CastResult.prompt``.
        direction: Refusal direction for steering correction.
        layers: Layer indices to apply CAST on.
        alpha: Base steering strength.
        threshold: Minimum projection to trigger steering.
        max_tokens: Maximum tokens to generate.
        condition_direction: Separate detection direction (AdaSteer).
        alpha_tiers: Adaptive alpha tiers (TRYLOCK/AlphaSteer).
        baseline_activations: Per-layer baseline for externality monitoring.
        displacement_threshold: L2 displacement threshold for intervention.

    Returns:
        CastResult with generation and intervention stats.
    """
    token_ids = encode_chat_prompt(tokenizer, messages)
    generated: list[int] = []
    cache = make_cache(model)
    projections_before_all: list[float] = []
    projections_after_all: list[float] = []
    interventions = 0
    considered = 0
    disp_interventions_total = 0
    max_disp_total = 0.0

    for _ in range(max_tokens):
        (
            logits,
            projections_before_step,
            projections_after_step,
            interventions_step,
            considered_step,
            disp_interventions_step,
            max_disp_step,
        ) = _cast_forward(
            model,
            token_ids,
            direction,
            layers,
            alpha,
            threshold,
            cache,
            condition_direction=condition_direction,
            alpha_tiers=alpha_tiers,
            baseline_activations=baseline_activations,
            displacement_threshold=displacement_threshold,
        )
        next_token = ops.argmax(logits[:, -1, :], axis=-1)
        token_id = int(next_token.item())
        generated.append(token_id)
        token_ids = next_token[:, None]

        if projections_before_step:
            projections_before_all.append(
                sum(projections_before_step) / len(projections_before_step),
            )
        if projections_after_step:
            projections_after_all.append(
                sum(projections_after_step) / len(projections_after_step),
            )
        interventions += interventions_step
        considered += considered_step
        disp_interventions_total += disp_interventions_step
        if max_disp_step > max_disp_total:
            max_disp_total = max_disp_step

    return CastResult(
        prompt=prompt_label,
        text=tokenizer.decode(generated),
        projections_before=projections_before_all,
        projections_after=projections_after_all,
        interventions=interventions,
        considered=considered,
        displacement_interventions=disp_interventions_total,
        max_displacement=max_disp_total,
    )


def _cast_forward(
    model: CausalLM,
    token_ids: Array,
    direction: Array,
    cast_layers: list[int],
    alpha: float,
    threshold: float,
    cache: list[LayerCache],
    *,
    condition_direction: Array | None = None,
    alpha_tiers: list[AlphaTier] | None = None,
    baseline_activations: dict[int, Array] | None = None,
    displacement_threshold: float = 0.0,
) -> tuple[Array, list[float], list[float], int, int, int, float]:
    """Run one forward step with conditional steering.

    Returns:
        ``(logits, projections_before, projections_after, interventions,
        considered, displacement_interventions, max_displacement)``
        where ``considered`` is the number of cast layers visited in this step.
    """
    transformer = get_transformer(model)
    h, mask = embed_and_mask(transformer, token_ids)

    # Use condition_direction for gating if provided, else primary direction
    direction = ops.to_device_like(direction, h)
    detect_dir = condition_direction if condition_direction is not None else direction
    detect_dir = ops.to_device_like(detect_dir, h)

    projections_before: list[float] = []
    projections_after: list[float] = []
    cast_layer_set = set(cast_layers)
    interventions = 0
    considered = 0
    displacement_interventions = 0
    max_displacement = 0.0
    ssm_mask = make_ssm_mask(transformer, h)

    for i, layer in enumerate(transformer.layers):
        h = layer(h, select_mask(layer, mask, ssm_mask), cache=cache[i])

        if i not in cast_layer_set:
            continue

        last_token = h[0, -1, :]

        # Detect: project onto condition direction for gating
        detect_projection = ops.sum(last_token * detect_dir)
        force_eval(detect_projection)
        detect_value = float(detect_projection.item())

        # Report the steer-direction projection as "before"
        steer_projection = ops.sum(last_token * direction)
        force_eval(steer_projection)
        projection_value = float(steer_projection.item())
        projections_before.append(projection_value)
        considered += 1

        # Externality monitoring: check activation displacement
        displacement_triggered = False
        if (
            baseline_activations is not None
            and displacement_threshold > 0.0
            and i in baseline_activations
        ):
            baseline = ops.to_device_like(baseline_activations[i], last_token)
            diff = last_token - baseline
            disp = float(
                ops.sqrt(ops.sum(diff * diff)).item(),
            )
            if disp > max_displacement:
                max_displacement = disp
            if disp > displacement_threshold:
                displacement_triggered = True
                displacement_interventions += 1

        if detect_value > threshold or displacement_triggered:
            effective_alpha = _resolve_alpha(
                detect_value, alpha, alpha_tiers,
            )
            correction = effective_alpha * steer_projection * direction
            h_list = [h[0, j, :] for j in range(h.shape[1])]
            h_list[-1] = h_list[-1] - correction
            h = ops.stack(h_list)[None, :, :]
            interventions += 1

        last_after = h[0, -1, :]
        projection_after = ops.sum(last_after * direction)
        force_eval(projection_after)
        projections_after.append(float(projection_after.item()))

    h = transformer.norm(h)
    logits = lm_head_forward(model, h)
    force_eval(logits)
    return (
        logits, projections_before, projections_after,
        interventions, considered,
        displacement_interventions, max_displacement,
    )


# ---------------------------------------------------------------------------
# SVF-aware CAST
# ---------------------------------------------------------------------------


def cast_generate_svf(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    boundary: "SVFBoundary",
    layers: list[int],
    alpha: float = 1.0,
    max_tokens: int = 100,
) -> CastResult:
    """Generate text with SVF boundary-gradient CAST.

    Uses the SVF boundary score as the gate (positive = harmful) and the
    boundary gradient as the steering direction. Context-dependent per-layer.

    Reference: Li, Li & Huang (2026) — arxiv.org/abs/2602.01654
    """
    messages = [{"role": "user", "content": prompt}]
    token_ids = encode_chat_prompt(tokenizer, messages)
    generated: list[int] = []
    cache = make_cache(model)
    scores_before_all: list[float] = []
    scores_after_all: list[float] = []
    interventions = 0
    considered = 0

    for _ in range(max_tokens):
        (
            logits,
            scores_before_step,
            scores_after_step,
            interventions_step,
            considered_step,
        ) = _cast_forward_svf(
            model, token_ids, boundary, layers, alpha, cache,
        )
        next_token = ops.argmax(logits[:, -1, :], axis=-1)
        token_id = int(next_token.item())
        generated.append(token_id)
        token_ids = next_token[:, None]

        if scores_before_step:
            scores_before_all.append(
                sum(scores_before_step) / len(scores_before_step),
            )
        if scores_after_step:
            scores_after_all.append(
                sum(scores_after_step) / len(scores_after_step),
            )
        interventions += interventions_step
        considered += considered_step

    return CastResult(
        prompt=prompt,
        text=tokenizer.decode(generated),
        projections_before=scores_before_all,
        projections_after=scores_after_all,
        interventions=interventions,
        considered=considered,
    )


def _cast_forward_svf(
    model: CausalLM,
    token_ids: Array,
    boundary: "SVFBoundary",
    cast_layers: list[int],
    alpha: float,
    cache: list[LayerCache],
) -> tuple[Array, list[float], list[float], int, int]:
    """One forward step with SVF boundary-gradient conditional steering.

    Gate: boundary score > 0 (harmful side).
    Direction: normalized gradient of boundary at current activation.
    """
    from vauban.svf import svf_gradient

    transformer = get_transformer(model)
    h, mask = embed_and_mask(transformer, token_ids)

    scores_before: list[float] = []
    scores_after: list[float] = []
    cast_set = set(cast_layers)
    interventions = 0
    considered = 0
    ssm_mask = make_ssm_mask(transformer, h)

    for i, layer in enumerate(transformer.layers):
        h = layer(h, select_mask(layer, mask, ssm_mask), cache=cache[i])

        if i not in cast_set:
            continue

        last_token = h[0, -1, :]
        score, grad = svf_gradient(boundary, last_token, i)
        scores_before.append(score)
        considered += 1

        # Gate: steer only when boundary score is positive
        if score > 0:
            correction = alpha * score * grad
            steered_last = h[0, -1, :] - correction
            h = ops.concatenate(
                [h[:, :-1, :], steered_last[None, None, :]], axis=1,
            )
            interventions += 1

        last_after = h[0, -1, :]
        score_after, _ = svf_gradient(boundary, last_after, i)
        scores_after.append(score_after)

    h = transformer.norm(h)
    logits = lm_head_forward(model, h)
    force_eval(logits)
    return logits, scores_before, scores_after, interventions, considered
