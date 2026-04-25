# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Jacobian sensitivity analysis for transformer layers.

Provides finite-difference JVP, directional gain, dominant singular
vector extraction, and compression-valley detection.  Reusable across
SSS (paper 1), steering-awareness detection (paper 2), and
component-level probing (paper 3).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from vauban import _ops as ops
from vauban._forward import (
    LayerModule,
    LayerRuntime,
    advance_layer,
    force_eval,
    get_transformer,
    make_ssm_mask,
    select_mask,
    svd_stable,
)
from vauban.subspace import effective_rank

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LayerSensitivity:
    """Sensitivity metrics for a single transformer layer."""

    layer_index: int
    directional_gain: float  # ||J_l * v|| / ||v||
    correlation: float  # cos(dominant_direction, steering_direction)
    effective_rank: float


@dataclass(frozen=True, slots=True)
class SensitivityProfile:
    """Full sensitivity profile across all layers."""

    layers: list[LayerSensitivity]
    valley_layers: list[int]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

type LayerFn = Callable[[Array], Array]


def jvp_finite_diff(
    layer_fn: LayerFn,
    h: Array,
    v: Array,
    epsilon: float = 1e-4,
) -> Array:
    """Compute Jacobian-vector product via forward finite differences.

    ``(layer_fn(h + eps*v) - layer_fn(h)) / eps``

    Works with any callable including SSM layers (no VJP needed).

    Args:
        layer_fn: Callable mapping hidden state to hidden state.
        h: Hidden-state tensor, shape ``(B, T, D)``.
        v: Perturbation direction, shape ``(D,)``.
        epsilon: Finite-difference step size.

    Returns:
        JVP result, same shape as ``h``.
    """
    v_expanded = v[None, None, :]  # (1, 1, D)
    h_plus = layer_fn(h + epsilon * v_expanded)
    h_base = layer_fn(h)
    return (h_plus - h_base) * (1.0 / epsilon)


def directional_gain(
    layer_fn: LayerFn,
    h: Array,
    direction: Array,
    epsilon: float = 1e-4,
) -> float:
    """Compute directional gain: ||J * v|| / ||v|| for a given layer.

    Args:
        layer_fn: Layer forward callable.
        h: Hidden-state tensor, shape ``(B, T, D)``.
        direction: Unit direction vector, shape ``(D,)``.
        epsilon: Finite-difference step size.

    Returns:
        Directional gain (scalar).
    """
    jv = jvp_finite_diff(layer_fn, h, direction, epsilon)
    # Use last token of first batch element (consistent with dominant_singular_vector)
    jv_last = jv[0, -1, :]
    jv_norm = float(ops.sqrt(ops.sum(jv_last * jv_last)).item())
    v_norm = float(ops.sqrt(ops.sum(direction * direction)).item())
    if v_norm < 1e-10:
        return 0.0
    return jv_norm / v_norm


def dominant_singular_vector(
    layer_fn: LayerFn,
    h: Array,
    n_iter: int = 5,
    epsilon: float = 1e-4,
) -> tuple[Array, list[float]]:
    """Extract the dominant singular vector of the layer Jacobian.

    Uses randomized power iteration: generate random probes, compute
    JVPs, form a matrix, and take the top-1 left singular vector.

    Args:
        layer_fn: Layer forward callable.
        h: Hidden-state tensor, shape ``(B, T, D)``.
        n_iter: Number of random probe vectors.
        epsilon: Finite-difference step size.

    Returns:
        ``(dominant, singular_values)`` where *dominant* has shape
        ``(D,)`` (unit-normalized) and *singular_values* is a list of
        floats from the SVD (reusable for effective-rank computation).
    """
    d = h.shape[-1]
    # Generate random probe vectors
    probes = ops.random.normal((n_iter, d))
    # Compute JVP for each probe, extract last-token response
    responses: list[Array] = []
    for i in range(n_iter):
        probe_i = probes[i]
        jv = jvp_finite_diff(layer_fn, h, probe_i, epsilon)
        # Use last token of first batch element
        responses.append(jv[0, -1, :])

    # Stack into (n_iter, D) matrix and SVD
    mat = ops.stack(responses)  # (n_iter, D)
    force_eval(mat)
    _u, s, _vt = svd_stable(mat)
    sv_list = [float(s[k].item()) for k in range(s.shape[0])]
    # First right singular vector = dominant direction
    dominant = _vt[0]  # (D,)
    # Normalize
    norm = ops.sqrt(ops.sum(dominant * dominant))
    return dominant * (1.0 / float(norm.item())), sv_list


def find_compression_valleys(
    effective_ranks: list[float],
    window: int = 3,
    top_k: int = 3,
) -> list[int]:
    """Identify compression-valley layers (local minima of effective rank).

    A valley is a layer whose effective rank is lower than all layers
    in the surrounding window.

    Args:
        effective_ranks: Per-layer effective rank values.
        window: Half-window size for local minimum detection.
        top_k: Maximum number of valleys to return.

    Returns:
        Layer indices of compression valleys, sorted by effective rank
        (lowest first).
    """
    n = len(effective_ranks)
    if n == 0:
        return []

    valleys: list[tuple[float, int]] = []
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        neighbors = [effective_ranks[j] for j in range(lo, hi) if j != i]
        if neighbors and effective_ranks[i] <= min(neighbors):
            valleys.append((effective_ranks[i], i))

    valleys.sort()  # sort by effective rank (lowest first)
    return [idx for _, idx in valleys[:top_k]]


def compute_sensitivity_profile(
    model: CausalLM,
    h: Array,
    mask: Array | None,
    direction: Array,
    n_power_iterations: int = 5,
    fd_epsilon: float = 1e-4,
    valley_window: int = 3,
    top_k_valleys: int = 3,
    runtime: LayerRuntime | None = None,
) -> SensitivityProfile:
    """Compute full sensitivity profile for all transformer layers.

    Runs a single forward pass to collect per-layer hidden states, then
    computes directional gain, dominant direction correlation, and
    effective rank for each layer.

    Args:
        model: The CausalLM model.
        h: Embedded input hidden states, shape ``(B, T, D)``.
        mask: Attention mask.
        direction: Steering direction, shape ``(D,)``.
        n_power_iterations: Power iterations for dominant vector.
        fd_epsilon: Finite-difference epsilon.
        valley_window: Window for valley detection.
        top_k_valleys: Max valleys to return.
        runtime: Optional prepared layer runtime for architectures that
            need per-layer masks or auxiliary inputs during manual layer
            stepping.

    Returns:
        Full sensitivity profile with per-layer metrics and valley layers.
    """
    transformer = get_transformer(model)
    layer_sensitivities: list[LayerSensitivity] = []
    ranks: list[float] = []

    ssm_mask = make_ssm_mask(transformer, h) if runtime is None else None
    h_cur = runtime.hidden if runtime is not None else h
    for i, layer in enumerate(transformer.layers):
        if runtime is None:
            layer_mask = select_mask(layer, mask, ssm_mask)
            typed_layer = cast("LayerModule", layer)

            # Build a closure that captures the current hidden state context.
            def fn(
                x: Array,
                *,
                _layer: LayerModule = typed_layer,
                _mask: Array | None = layer_mask,
            ) -> Array:
                return _layer(x, _mask)

            layer_fn: LayerFn = fn
        else:
            # Reuse the same per-layer runtime configuration while cloning the
            # mutable hidden/intermediate state for each finite-difference call.
            def fn(x: Array, *, _layer_index: int = i) -> Array:
                local_runtime = LayerRuntime(
                    hidden=x,
                    masks=runtime.masks,
                    cache=runtime.cache,
                    per_layer_inputs=runtime.per_layer_inputs,
                    previous_kvs=runtime.previous_kvs,
                    intermediates=(
                        None if runtime.intermediates is None
                        else list(runtime.intermediates)
                    ),
                )
                return advance_layer(transformer, local_runtime, _layer_index)

            layer_fn = fn

        # Directional gain
        gain = directional_gain(layer_fn, h_cur, direction, fd_epsilon)

        # Dominant singular vector + correlation + effective rank
        # (single SVD pass — singular values reused for rank estimate)
        dominant, sv_list = dominant_singular_vector(
            layer_fn, h_cur, n_power_iterations, fd_epsilon,
        )
        force_eval(dominant)
        cos_sim = float(ops.sum(dominant * direction).item())
        correlation = max(-1.0, min(1.0, cos_sim))
        eff_rank = effective_rank(sv_list)
        ranks.append(eff_rank)

        layer_sensitivities.append(LayerSensitivity(
            layer_index=i,
            directional_gain=gain,
            correlation=correlation,
            effective_rank=eff_rank,
        ))

        # Advance hidden state through this layer
        if runtime is None:
            h_cur = typed_layer(h_cur, layer_mask)
        else:
            h_cur = advance_layer(transformer, runtime, i)

    valley_layers = find_compression_valleys(ranks, valley_window, top_k_valleys)

    return SensitivityProfile(
        layers=layer_sensitivities,
        valley_layers=valley_layers,
    )
