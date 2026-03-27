# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Steering awareness detection via Jacobian sensitivity comparison.

Paper 2 in the sensitivity analysis series.  SSS (paper 1) uses Jacobian
sensitivity to *attack* — awareness uses the same primitives to *detect*
whether activations are being steered at inference time.

Algorithm:
  1. Calibrate: compute a SensitivityProfile on a clean prompt → baseline.
  2. For each test prompt: compute a profile, then per-layer compare
     gain ratio, rank ratio, and correlation delta against the baseline.
  3. Confidence: mean of top-half per-layer anomaly scores.
  4. Verdict: steered if enough anomalous layers *and* confidence above
     threshold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from vauban._forward import (
    LayerModule,
    embed_and_mask,
    encode_user_prompt,
    get_transformer,
    make_ssm_mask,
    select_mask,
)
from vauban.sensitivity import (
    SensitivityProfile,
    compute_sensitivity_profile,
    directional_gain,
)
from vauban.types import AwarenessLayerResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from vauban._array import Array
    from vauban.sensitivity import LayerSensitivity
    from vauban.types import (
        AwarenessConfig,
        AwarenessResult,
        CausalLM,
        Tokenizer,
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def awareness_calibrate(
    model: CausalLM,
    tokenizer: Tokenizer,
    calibration_prompt: str,
    direction: Array,
    config: AwarenessConfig,
) -> SensitivityProfile:
    """Compute a baseline sensitivity profile on a clean calibration prompt.

    Args:
        model: The language model.
        tokenizer: Tokenizer for encoding the calibration prompt.
        calibration_prompt: Short benign prompt for calibration.
        direction: Steering direction, shape ``(D,)``.
        config: Awareness configuration.

    Returns:
        Baseline sensitivity profile for all layers.
    """
    ids = encode_user_prompt(tokenizer, calibration_prompt)
    transformer = get_transformer(model)
    h, mask = embed_and_mask(transformer, ids)

    if config.mode == "fast":
        return _fast_gain_profile(model, h, mask, direction, config)

    return compute_sensitivity_profile(
        model, h, mask, direction,
        n_power_iterations=config.n_power_iterations,
        fd_epsilon=config.fd_epsilon,
        valley_window=config.valley_window,
        top_k_valleys=config.top_k_valleys,
    )


# ---------------------------------------------------------------------------
# Fast mode: gain-only profile
# ---------------------------------------------------------------------------

def _fast_gain_profile(
    model: CausalLM,
    h: Array,
    mask: Array,
    direction: Array,
    config: AwarenessConfig,
) -> SensitivityProfile:
    """Compute a gain-only sensitivity profile (fast mode).

    Only computes directional gain per layer — skips SVD.  Correlation and
    effective rank are set to sentinel values (0.0 and 1.0 respectively).

    Args:
        model: The language model.
        h: Embedded hidden states, shape ``(B, T, D)``.
        mask: Attention mask.
        direction: Steering direction, shape ``(D,)``.
        config: Awareness configuration.

    Returns:
        Sensitivity profile with gain populated, rank/correlation as sentinels.
    """
    from vauban.sensitivity import LayerSensitivity

    transformer = get_transformer(model)
    ssm_mask = make_ssm_mask(transformer, h)
    layer_results: list[LayerSensitivity] = []

    def _make_layer_fn(
        _layer: LayerModule, _mask: Array | None,
    ) -> Callable[[Array], Array]:
        def fn(x: Array) -> Array:
            return _layer(x, _mask)
        return fn

    h_cur = h
    for i, layer in enumerate(transformer.layers):
        typed_layer = cast("LayerModule", layer)
        layer_mask = select_mask(layer, mask, ssm_mask)
        layer_fn = _make_layer_fn(typed_layer, layer_mask)

        gain = directional_gain(layer_fn, h_cur, direction, config.fd_epsilon)

        layer_results.append(LayerSensitivity(
            layer_index=i,
            directional_gain=gain,
            correlation=0.0,  # sentinel — not computed in fast mode
            effective_rank=1.0,  # sentinel — not computed in fast mode
        ))

        h_cur = typed_layer(h_cur, layer_mask)

    return SensitivityProfile(layers=layer_results, valley_layers=[])


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _layer_anomaly_score(
    baseline: LayerSensitivity,
    test: LayerSensitivity,
    config: AwarenessConfig,
) -> tuple[AwarenessLayerResult, float]:
    """Compare one layer and return the result plus its raw anomaly score."""
    # Gain ratio
    baseline_gain = baseline.directional_gain
    test_gain = test.directional_gain
    if baseline_gain > 1e-10:
        gain_ratio = test_gain / baseline_gain
    elif test_gain > 1e-10:
        gain_ratio = test_gain / 1e-10
    else:
        gain_ratio = 1.0  # both near-zero → no anomaly

    # Rank ratio
    baseline_rank = baseline.effective_rank
    test_rank = test.effective_rank
    rank_ratio = test_rank / baseline_rank if baseline_rank > 1e-10 else 1.0

    # Correlation delta
    correlation_delta = abs(test.correlation - baseline.correlation)

    # Anomaly detection
    gain_anomalous = gain_ratio >= config.gain_ratio_threshold
    rank_anomalous = rank_ratio <= config.rank_ratio_threshold
    corr_anomalous = correlation_delta >= config.correlation_delta_threshold

    # In fast mode, only gain triggers anomaly
    is_fast = config.mode == "fast"
    if is_fast:
        anomalous = gain_anomalous
    else:
        anomalous = gain_anomalous or rank_anomalous or corr_anomalous

    # Raw anomaly score: max of normalized signals
    signals: list[float] = [gain_ratio / config.gain_ratio_threshold]
    if not is_fast:
        # Invert rank_ratio so lower = higher anomaly score
        if config.rank_ratio_threshold > 0:
            signals.append(
                config.rank_ratio_threshold / max(rank_ratio, 1e-10),
            )
        signals.append(correlation_delta / config.correlation_delta_threshold)

    raw_score = max(signals) if signals else 0.0

    layer_result = AwarenessLayerResult(
        layer_index=test.layer_index,
        baseline_gain=baseline_gain,
        test_gain=test_gain,
        gain_ratio=gain_ratio,
        baseline_rank=baseline_rank,
        test_rank=test_rank,
        rank_ratio=rank_ratio,
        baseline_correlation=baseline.correlation,
        test_correlation=test.correlation,
        correlation_delta=correlation_delta,
        anomalous=anomalous,
    )
    return layer_result, raw_score


def awareness_detect(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    direction: Array,
    config: AwarenessConfig,
    baseline: SensitivityProfile,
) -> AwarenessResult:
    """Detect whether a prompt is being steered by comparing against baseline.

    Args:
        model: The language model.
        tokenizer: Tokenizer for encoding the prompt.
        prompt: Test prompt to analyze.
        direction: Steering direction, shape ``(D,)``.
        config: Awareness configuration.
        baseline: Baseline sensitivity profile from calibration.

    Returns:
        Detection result with per-layer metrics and verdict.
    """
    from vauban.types import AwarenessResult

    # Compute test profile
    ids = encode_user_prompt(tokenizer, prompt)
    transformer = get_transformer(model)
    h, mask = embed_and_mask(transformer, ids)

    if config.mode == "fast":
        test_profile = _fast_gain_profile(model, h, mask, direction, config)
    else:
        test_profile = compute_sensitivity_profile(
            model, h, mask, direction,
            n_power_iterations=config.n_power_iterations,
            fd_epsilon=config.fd_epsilon,
            valley_window=config.valley_window,
            top_k_valleys=config.top_k_valleys,
        )

    # Compare per-layer
    layer_results: list[AwarenessLayerResult] = []
    raw_scores: list[float] = []
    anomalous_layers: list[int] = []
    evidence: list[str] = []

    n_layers = min(len(baseline.layers), len(test_profile.layers))
    for i in range(n_layers):
        layer_result, raw_score = _layer_anomaly_score(
            baseline.layers[i], test_profile.layers[i], config,
        )
        layer_results.append(layer_result)
        raw_scores.append(raw_score)
        if layer_result.anomalous:
            anomalous_layers.append(layer_result.layer_index)
            reasons: list[str] = []
            if layer_result.gain_ratio >= config.gain_ratio_threshold:
                reasons.append(
                    f"gain_ratio={layer_result.gain_ratio:.2f}",
                )
            if config.mode != "fast":
                if layer_result.rank_ratio <= config.rank_ratio_threshold:
                    reasons.append(
                        f"rank_ratio={layer_result.rank_ratio:.2f}",
                    )
                if layer_result.correlation_delta >= config.correlation_delta_threshold:
                    reasons.append(
                        f"corr_delta={layer_result.correlation_delta:.2f}",
                    )
            evidence.append(
                f"layer {layer_result.layer_index}: {', '.join(reasons)}",
            )

    # Confidence: mean of top-half scores (steering affects a subset)
    if raw_scores:
        sorted_scores = sorted(raw_scores, reverse=True)
        top_half = sorted_scores[: max(1, len(sorted_scores) // 2)]
        confidence = min(1.0, sum(top_half) / len(top_half))
    else:
        confidence = 0.0

    # Verdict
    steered = (
        len(anomalous_layers) >= config.min_anomalous_layers
        and confidence >= config.confidence_threshold
    )

    return AwarenessResult(
        prompt=prompt,
        steered=steered,
        confidence=confidence,
        anomalous_layers=anomalous_layers,
        layers=layer_results,
        evidence=evidence,
    )
