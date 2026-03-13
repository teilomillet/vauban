"""Sensitivity-Scaled Steering (SSS) generation runtime.

Adaptive activation-space attack using Jacobian sensitivity analysis.
Seeds a perturbation at the BOS token in compression-valley layers,
then applies per-token micro-injections scaled by directional gain
and correlation with the dominant amplification direction.

Reference: "Steering in the Shadows" (arxiv 2511.17194).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vauban import _ops as ops
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
from vauban.sensitivity import SensitivityProfile, compute_sensitivity_profile

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM, SSSConfig, SSSResult, Tokenizer


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _sss_calibrate(
    model: CausalLM,
    tokenizer: Tokenizer,
    calibration_prompt: str,
    direction: Array,
    n_power_iterations: int = 5,
    fd_epsilon: float = 1e-4,
    valley_window: int = 3,
    top_k_valleys: int = 3,
) -> SensitivityProfile:
    """Run a calibration forward pass to build the sensitivity profile.

    Args:
        model: The language model.
        tokenizer: Tokenizer for encoding the calibration prompt.
        calibration_prompt: Short prompt for calibration.
        direction: Steering direction, shape ``(D,)``.
        n_power_iterations: Power iterations for dominant vector.
        fd_epsilon: Finite-difference step size.
        valley_window: Window for compression valley detection.
        top_k_valleys: Max number of valleys.

    Returns:
        Sensitivity profile for all layers.
    """
    token_ids = encode_user_prompt(tokenizer, calibration_prompt)
    transformer = get_transformer(model)
    h, mask = embed_and_mask(transformer, token_ids)
    force_eval(h, mask)

    return compute_sensitivity_profile(
        model, h, mask, direction,
        n_power_iterations=n_power_iterations,
        fd_epsilon=fd_epsilon,
        valley_window=valley_window,
        top_k_valleys=top_k_valleys,
    )


# ---------------------------------------------------------------------------
# Forward passes
# ---------------------------------------------------------------------------

type LayerCache = object


def _sss_seed_forward(
    model: CausalLM,
    token_ids: Array,
    direction: Array,
    profile: SensitivityProfile,
    alpha: float,
    seed_floor: float,
    cache: list[LayerCache],
) -> tuple[Array, float]:
    """Prompt-processing forward pass with BOS seeding.

    Seeds the perturbation at ``h[0, 0, :]`` (BOS position) in
    compression-valley layers.  Injection strength is scaled by
    ``gain * |correlation|`` per layer.

    Args:
        model: The language model.
        token_ids: Encoded prompt tokens, shape ``(1, T)``.
        direction: Steering direction, shape ``(D,)``.
        profile: Pre-computed sensitivity profile.
        alpha: Global steering strength multiplier.
        seed_floor: Minimum injection strength.
        cache: KV cache list (mutated in-place).

    Returns:
        ``(logits, total_seed_strength)`` where logits has shape
        ``(1, T, V)`` and total_seed_strength sums all injections.
    """
    transformer = get_transformer(model)
    h, mask = embed_and_mask(transformer, token_ids)
    ssm_mask = make_ssm_mask(transformer, h)

    seed_set = set(profile.valley_layers)
    gain_map = {ls.layer_index: ls for ls in profile.layers}
    total_seed = 0.0

    for i, layer in enumerate(transformer.layers):
        layer_mask = select_mask(layer, mask, ssm_mask)
        h = layer(h, layer_mask, cache=cache[i])

        if i in seed_set:
            ls = gain_map[i]
            strength = alpha * ls.directional_gain * abs(ls.correlation)
            strength = max(strength, seed_floor)
            total_seed += strength

            # Inject at BOS position (index 0)
            bos_token = h[0, 0, :]
            correction = strength * direction
            new_bos = bos_token + correction

            # Reconstruct h with modified BOS
            h_list = [h[0, j, :] for j in range(h.shape[1])]
            h_list[0] = new_bos
            h = ops.stack(h_list)[None, :, :]

    h = transformer.norm(h)
    logits = lm_head_forward(model, h)
    force_eval(logits)
    return logits, total_seed


def _sss_reinforcement_forward(
    model: CausalLM,
    token_ids: Array,
    direction: Array,
    profile: SensitivityProfile,
    alpha: float,
    steer_layers: set[int],
    cache: list[LayerCache],
) -> tuple[Array, float, list[float], list[float]]:
    """Decode-step forward pass with per-token micro-injections.

    Injects at ``h[0, -1, :]`` (last token) in the specified steer
    layers, scaled by directional gain and correlation.

    Args:
        model: The language model.
        token_ids: Single-token input, shape ``(1, 1)``.
        direction: Steering direction, shape ``(D,)``.
        profile: Pre-computed sensitivity profile.
        alpha: Global steering strength multiplier.
        steer_layers: Layer indices to steer.
        cache: KV cache list (mutated in-place).

    Returns:
        ``(logits, gain_this_step, proj_before, proj_after)``
    """
    transformer = get_transformer(model)
    h, mask = embed_and_mask(transformer, token_ids)
    ssm_mask = make_ssm_mask(transformer, h)

    gain_map = {ls.layer_index: ls for ls in profile.layers}
    proj_before: list[float] = []
    proj_after: list[float] = []
    total_gain = 0.0

    for i, layer in enumerate(transformer.layers):
        layer_mask = select_mask(layer, mask, ssm_mask)
        h = layer(h, layer_mask, cache=cache[i])

        if i in steer_layers:
            ls = gain_map[i]
            strength = alpha * ls.directional_gain * abs(ls.correlation)
            total_gain += strength

            # Measure projection before
            last_token = h[0, -1, :]
            proj = ops.sum(last_token * direction)
            force_eval(proj)
            proj_before.append(float(proj.item()))

            # Inject: ADD direction (attack, not removal)
            correction = strength * direction
            new_last = last_token + correction
            h_list = [h[0, j, :] for j in range(h.shape[1])]
            h_list[-1] = new_last
            h = ops.stack(h_list)[None, :, :]

            # Measure projection after
            last_after = h[0, -1, :]
            proj_a = ops.sum(last_after * direction)
            force_eval(proj_a)
            proj_after.append(float(proj_a.item()))

    h = transformer.norm(h)
    logits = lm_head_forward(model, h)
    force_eval(logits)
    return logits, total_gain, proj_before, proj_after


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sss_generate(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    direction: Array,
    config: SSSConfig,
    profile: SensitivityProfile | None = None,
) -> SSSResult:
    """Generate text using Sensitivity-Scaled Steering.

    1. Calibrate (or reuse) a sensitivity profile.
    2. Seed the perturbation at BOS in compression-valley layers.
    3. Reinforce with per-token micro-injections during decode.

    Args:
        model: The language model.
        tokenizer: Tokenizer.
        prompt: User prompt to generate from.
        direction: Steering direction, shape ``(D,)``.
        config: SSS configuration.
        profile: Optional pre-computed profile (skips calibration).

    Returns:
        SSSResult with generated text and per-token diagnostics.
    """
    from vauban.types import SSSResult

    # 1. Calibrate if needed
    if profile is None:
        profile = _sss_calibrate(
            model, tokenizer,
            config.calibration_prompt,
            direction,
            n_power_iterations=config.n_power_iterations,
            fd_epsilon=config.fd_epsilon,
            valley_window=config.valley_window,
            top_k_valleys=config.top_k_valleys,
        )

    # Determine steer layers
    if config.layers is not None:
        steer_layer_set = set(config.layers)
    else:
        # Use all layers for reinforcement (valleys for seeding only)
        n_layers = len(get_transformer(model).layers)
        steer_layer_set = set(range(n_layers))

    # 2. Seed forward pass (prompt processing)
    token_ids = encode_user_prompt(tokenizer, prompt)
    cache = make_cache(model)
    logits, seed_strength = _sss_seed_forward(
        model, token_ids, direction, profile,
        config.alpha, config.seed_floor, cache,
    )

    # 3. Decode loop with reinforcement
    generated: list[int] = []
    all_gains: list[float] = []
    all_proj_before: list[float] = []
    all_proj_after: list[float] = []

    eos_id: int | None = getattr(tokenizer, "eos_token_id", None)

    # First token from seed pass
    next_token = ops.argmax(logits[:, -1, :], axis=-1)
    token_id = int(next_token.item())
    generated.append(token_id)

    for _ in range(config.max_tokens - 1):
        token_input = next_token[:, None]
        logits, gain, p_before, p_after = _sss_reinforcement_forward(
            model, token_input, direction, profile,
            config.alpha, steer_layer_set, cache,
        )
        all_gains.append(gain)

        if p_before:
            all_proj_before.append(sum(p_before) / len(p_before))
        if p_after:
            all_proj_after.append(sum(p_after) / len(p_after))

        next_token = ops.argmax(logits[:, -1, :], axis=-1)
        token_id = int(next_token.item())
        generated.append(token_id)

        if eos_id is not None and token_id == eos_id:
            break

    return SSSResult(
        text=tokenizer.decode(generated),
        prompt=prompt,
        seed_layers=profile.valley_layers,
        seed_strength=seed_strength,
        per_token_gains=all_gains,
        projections_before=all_proj_before,
        projections_after=all_proj_after,
    )
