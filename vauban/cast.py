"""Conditional Activation Steering (CAST) runtime generation."""

from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import embed_and_mask, force_eval, lm_head_forward, make_cache
from vauban.types import AlphaTier, CastResult, CausalLM, LayerCache, Tokenizer


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
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not isinstance(text, str):
        msg = "apply_chat_template must return str when tokenize=False"
        raise TypeError(msg)

    token_ids = ops.array(tokenizer.encode(text))[None, :]
    generated: list[int] = []
    cache = make_cache(model)
    projections_before_all: list[float] = []
    projections_after_all: list[float] = []
    interventions = 0
    considered = 0

    for _ in range(max_tokens):
        (
            logits,
            projections_before_step,
            projections_after_step,
            interventions_step,
            considered_step,
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

    return CastResult(
        prompt=prompt,
        text=tokenizer.decode(generated),
        projections_before=projections_before_all,
        projections_after=projections_after_all,
        interventions=interventions,
        considered=considered,
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
) -> tuple[Array, list[float], list[float], int, int]:
    """Run one forward step with conditional steering.

    Returns:
        ``(logits, projections_before, projections_after, interventions, considered)``
        where ``considered`` is the number of cast layers visited in this step.
    """
    transformer = model.model
    h, mask = embed_and_mask(transformer, token_ids)

    # Use condition_direction for gating if provided, else primary direction
    detect_dir = condition_direction if condition_direction is not None else direction

    projections_before: list[float] = []
    projections_after: list[float] = []
    cast_layer_set = set(cast_layers)
    interventions = 0
    considered = 0

    for i, layer in enumerate(transformer.layers):
        h = layer(h, mask, cache=cache[i])

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

        if detect_value > threshold:
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
    return logits, projections_before, projections_after, interventions, considered
