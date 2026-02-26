"""Cut (abliterate) a refusal direction from model weights."""

from pathlib import Path

import mlx.core as mx

from vauban._array import Array
from vauban._forward import force_eval


def cut(
    weights: dict[str, Array],
    direction: Array,
    target_layers: list[int],
    alpha: float = 1.0,
    norm_preserve: bool = False,
    layer_weights: list[float] | None = None,
) -> dict[str, Array]:
    """Remove a direction from model weights via rank-1 projection.

    Modifies o_proj.weight and down_proj.weight for the specified layers.
    Returns a new dict with modified weights (unmodified keys are shared).

    Args:
        weights: Flat dict of model weights (key -> mx.array).
        direction: Unit refusal direction of shape (d_model,).
        target_layers: Layer indices to modify.
        alpha: Base scaling factor for the projection removal.
        norm_preserve: If True, rescale rows to preserve original norms.
        layer_weights: Per-layer alpha multipliers, one per target layer.
            Final alpha for layer i is ``alpha * layer_weights[i]``.
            If None, uniform weighting (all 1.0).
    """
    layer_alpha = _resolve_layer_alpha(alpha, target_layers, layer_weights)
    result = dict(weights)

    for layer_idx, a in zip(target_layers, layer_alpha, strict=True):
        keys = _keys_for_layer(list(weights.keys()), layer_idx)
        for key in keys:
            w = weights[key]
            if norm_preserve:
                result[key] = _orthogonalize_norm_preserve(w, direction, a)
            else:
                result[key] = _orthogonalize_matrix(w, direction, a)
            force_eval(result[key])

    return result


def cut_biprojected(
    weights: dict[str, Array],
    refusal_direction: Array,
    harmless_direction: Array,
    target_layers: list[int],
    alpha: float = 1.0,
    norm_preserve: bool = False,
    layer_weights: list[float] | None = None,
) -> dict[str, Array]:
    """Remove a biprojected direction from model weights.

    First orthogonalizes the refusal direction against the harmless
    direction via Gram-Schmidt, then applies the standard cut.
    """
    direction = _biprojected_direction(refusal_direction, harmless_direction)
    return cut(
        weights, direction, target_layers, alpha, norm_preserve, layer_weights,
    )


def cut_false_refusal_ortho(
    weights: dict[str, Array],
    refusal_direction: Array,
    false_refusal_direction: Array,
    target_layers: list[int],
    alpha: float = 1.0,
    norm_preserve: bool = False,
    layer_weights: list[float] | None = None,
) -> dict[str, Array]:
    """Remove a refusal direction orthogonalized against false refusal.

    First orthogonalizes the refusal direction against the false refusal
    direction via Gram-Schmidt (same math as biprojected), then applies
    the standard cut. This reduces over-refusal on borderline-safe queries.

    Args:
        weights: Flat dict of model weights (key -> mx.array).
        refusal_direction: Unit refusal direction of shape (d_model,).
        false_refusal_direction: Unit false-refusal direction from
            borderline-safe prompts.
        target_layers: Layer indices to modify.
        alpha: Base scaling factor for the projection removal.
        norm_preserve: If True, rescale rows to preserve original norms.
        layer_weights: Per-layer alpha multipliers.
    """
    direction = _biprojected_direction(refusal_direction, false_refusal_direction)
    return cut(
        weights, direction, target_layers, alpha, norm_preserve, layer_weights,
    )


def _orthogonalize_matrix(
    w: Array,
    direction: Array,
    alpha: float,
) -> Array:
    """Remove a direction from the output space of a weight matrix.

    For 2D weights (d_model, in_features):
        W' = W - alpha * d (d^T W) = W - alpha * outer(d, d @ W)

    For 3D weights (num_experts, d_model, in_features) — MoE experts:
        Applies the same projection removal to each expert independently.
    """
    if w.ndim == 3:
        # Batched experts: (N, d_model, F)
        # proj[n] = direction @ w[n] -> (N, F)
        proj = mx.sum(
            w * direction[None, :, None], axis=1,
        )  # (N, F)
        # update[n] = outer(direction, proj[n]) -> (N, d_model, F)
        update = direction[None, :, None] * proj[:, None, :]
        return w - alpha * update

    proj = direction @ w  # (in_features,)
    return w - alpha * mx.outer(direction, proj)


def _orthogonalize_norm_preserve(
    w: Array,
    direction: Array,
    alpha: float,
) -> Array:
    """Remove a direction and rescale rows to preserve original norms.

    Uses last axis for norm computation, works for both 2D and 3D.
    """
    original_norms = mx.linalg.norm(w, axis=-1, keepdims=True)
    w_new = _orthogonalize_matrix(w, direction, alpha)
    new_norms = mx.linalg.norm(w_new, axis=-1, keepdims=True)
    return w_new * (original_norms / (new_norms + 1e-8))


def _biprojected_direction(
    refusal_dir: Array,
    harmless_dir: Array,
) -> Array:
    """Gram-Schmidt: orthogonalize refusal direction against harmless direction."""
    proj = mx.sum(refusal_dir * harmless_dir) * harmless_dir
    orthogonal = refusal_dir - proj
    return orthogonal / (mx.linalg.norm(orthogonal) + 1e-8)


def target_weight_keys(
    all_keys: list[str],
    target_layers: list[int],
) -> list[str]:
    """Select o_proj and down_proj weight keys for target layers.

    Matches standard dense layers, MoE shared experts, and MoE
    per-expert (batched) weight matrices.
    """
    suffixes = (
        # Standard dense
        "self_attn.o_proj.weight",
        "mlp.down_proj.weight",
        # MoE shared expert
        "mlp.shared_experts.down_proj.weight",
        # MoE batched experts (3D tensor)
        "mlp.experts.down_proj.weight",
    )
    targets: list[str] = []
    for layer_idx in target_layers:
        for suffix in suffixes:
            key = f"model.layers.{layer_idx}.{suffix}"
            if key in all_keys:
                targets.append(key)
    return targets


def cut_subspace(
    weights: dict[str, Array],
    basis: Array,
    target_layers: list[int],
    alpha: float = 1.0,
    norm_preserve: bool = False,
    layer_weights: list[float] | None = None,
) -> dict[str, Array]:
    """Remove a multi-dimensional subspace from model weights.

    Iterates over basis vectors and applies rank-1 projection removal
    for each direction. If norm_preserve is True, rows are rescaled
    after all projections have been removed.

    Args:
        weights: Flat dict of model weights (key -> mx.array).
        basis: Orthonormal basis of shape (k, d_model).
        target_layers: Layer indices to modify.
        alpha: Base scaling factor for the projection removal.
        norm_preserve: If True, rescale rows to preserve original norms.
        layer_weights: Per-layer alpha multipliers, one per target layer.
    """
    layer_alpha = _resolve_layer_alpha(alpha, target_layers, layer_weights)
    result = dict(weights)

    for layer_idx, a in zip(target_layers, layer_alpha, strict=True):
        keys = _keys_for_layer(list(weights.keys()), layer_idx)
        for key in keys:
            w = weights[key]
            if norm_preserve:
                original_norms = mx.linalg.norm(w, axis=-1, keepdims=True)

            for i in range(basis.shape[0]):
                d = basis[i]
                w = _orthogonalize_matrix(w, d, a)

            if norm_preserve:
                new_norms = mx.linalg.norm(w, axis=-1, keepdims=True)
                w = w * (original_norms / (new_norms + 1e-8))

            force_eval(w)
            result[key] = w

    return result


def sparsify_direction(direction: Array, sparsity: float) -> Array:
    """Zero out low-magnitude components of a direction vector.

    Keeps only the top ``(1 - sparsity)`` fraction of components by
    absolute value, zeroing the rest. This reduces collateral damage
    to non-refusal behavior by focusing the cut on the strongest
    signal dimensions.

    Args:
        direction: Direction vector of shape (d_model,).
        sparsity: Fraction of components to zero out (0.0 = keep all,
            0.9 = keep top 10%).

    Returns:
        Sparsified direction vector (not re-normalized).
    """
    if sparsity <= 0.0:
        return direction
    if sparsity >= 1.0:
        return mx.zeros_like(direction)

    abs_vals = mx.abs(direction)
    # Number of components to keep
    k = max(1, int(direction.shape[0] * (1.0 - sparsity)))
    # Find the k-th largest value as threshold
    sorted_vals = mx.sort(abs_vals)
    threshold = sorted_vals[-k]
    force_eval(threshold)
    mask = abs_vals >= threshold
    result = direction * mask
    force_eval(result)
    return result


def _resolve_layer_alpha(
    alpha: float,
    target_layers: list[int],
    layer_weights: list[float] | None,
) -> list[float]:
    """Compute per-layer alpha from base alpha and optional layer weights."""
    if layer_weights is None:
        return [alpha] * len(target_layers)
    if len(layer_weights) != len(target_layers):
        msg = (
            f"layer_weights length ({len(layer_weights)}) must match "
            f"target_layers length ({len(target_layers)})"
        )
        raise ValueError(msg)
    return [alpha * w for w in layer_weights]


def _keys_for_layer(all_keys: list[str], layer_idx: int) -> list[str]:
    """Get target weight keys for a single layer."""
    suffixes = (
        "self_attn.o_proj.weight",
        "mlp.down_proj.weight",
        "mlp.shared_experts.down_proj.weight",
        "mlp.experts.down_proj.weight",
    )
    keys: list[str] = []
    for suffix in suffixes:
        key = f"model.layers.{layer_idx}.{suffix}"
        if key in all_keys:
            keys.append(key)
    return keys


def save_weights(
    weights: dict[str, Array],
    output_path: str | Path,
) -> Path:
    """Save weights to a safetensors file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output_path), weights)
    return output_path
