"""LoRA adapter construction, loading, and analysis.

Converts the rank-1 weight update ``W' = W - alpha * d (d^T W)`` into
LoRA matrices ``delta_W = B @ A`` where ``B = -d`` and ``A = d^T W``.

Also provides adapter loading (single or multi-adapter merge) and
SVD-based adapter decomposition for structural analysis.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, cast

from vauban import _ops as ops
from vauban.cut import _keys_for_layer
from vauban.types import LoraMatrices

if TYPE_CHECKING:
    from pathlib import Path

    from vauban._array import Array
    from vauban.types import LoraAnalysisResult


# ---------------------------------------------------------------------------
# Core conversions
# ---------------------------------------------------------------------------


def direction_to_lora(
    direction: Array,
    weight: Array,
    polarity: str = "remove",
) -> tuple[Array, Array]:
    """Convert a rank-1 direction into LoRA A/B matrices.

    The cut operation is ``W' = W - alpha * outer(d, d @ W)``, so
    ``delta_W = -alpha * outer(d, d @ W)``.

    LoRA parameterizes ``delta_W = B @ A * (lora_alpha / rank)``.
    We set ``rank = 1``, ``lora_alpha = alpha`` (applied externally),
    so ``B @ A = outer(d, d @ W)`` with sign from polarity.

    Args:
        direction: Unit direction of shape ``(d_model,)``.
        weight: Weight matrix of shape ``(d_model, in_features)``.
        polarity: ``"remove"`` subtracts the direction (abliteration),
            ``"add"`` adds it (trait amplification).

    Returns:
        ``(lora_a, lora_b)`` where ``lora_a`` has shape ``(1, in_features)``
        and ``lora_b`` has shape ``(d_model, 1)``.
    """
    # A = d^T @ W -> (in_features,) -> (1, in_features)
    a_vec = direction @ weight  # (in_features,)
    lora_a = ops.expand_dims(a_vec, axis=0)  # (1, in_features)

    # B = d -> (d_model,) -> (d_model, 1)
    lora_b = ops.expand_dims(direction, axis=1)  # (d_model, 1)

    # Polarity: remove = negative delta (subtract direction),
    # add = positive delta (amplify direction)
    if polarity == "remove":
        lora_b = -lora_b
    elif polarity != "add":
        msg = f"polarity must be 'remove' or 'add', got {polarity!r}"
        raise ValueError(msg)

    return lora_a, lora_b


def subspace_to_lora(
    basis: Array,
    weight: Array,
    polarity: str = "remove",
) -> tuple[Array, Array]:
    """Convert a rank-k subspace basis into LoRA A/B matrices.

    For a basis of shape ``(k, d_model)`` the combined update is the
    sum of rank-1 projections: ``delta_W = sum_i outer(d_i, d_i @ W)``.

    This factors as ``B @ A`` where ``B = basis^T`` (d_model, k) and
    ``A = basis @ W`` (k, in_features).

    Args:
        basis: Orthonormal basis of shape ``(k, d_model)``.
        weight: Weight matrix of shape ``(d_model, in_features)``.
        polarity: ``"remove"`` or ``"add"``.

    Returns:
        ``(lora_a, lora_b)`` where ``lora_a`` has shape ``(k, in_features)``
        and ``lora_b`` has shape ``(d_model, k)``.
    """
    k = basis.shape[0]

    # A = basis @ W -> (k, in_features)
    lora_a = ops.matmul(basis, weight)

    # B = basis^T -> (d_model, k)
    lora_b = ops.stack([basis[i] for i in range(k)], axis=1)  # (d_model, k)

    if polarity == "remove":
        lora_b = -lora_b
    elif polarity != "add":
        msg = f"polarity must be 'remove' or 'add', got {polarity!r}"
        raise ValueError(msg)

    return lora_a, lora_b


# ---------------------------------------------------------------------------
# Build adapter weight dict
# ---------------------------------------------------------------------------


def build_lora_weights(
    direction: Array,
    flat_weights: dict[str, Array],
    target_layers: list[int],
    alpha: float = 1.0,
    polarity: str = "remove",
    layer_weights: list[float] | None = None,
) -> list[LoraMatrices]:
    """Build LoRA matrices for all target weight keys.

    Iterates over target layers/keys and produces ``(lora_a, lora_b)``
    pairs. Skips 3D weights (MoE experts) since LoRA has no standard
    3D convention.

    The per-key alpha is baked into lora_b so the final LoRA scaling
    ``(lora_alpha / rank)`` is set to 1.0 at save time.

    Args:
        direction: Unit direction of shape ``(d_model,)`` or basis
            of shape ``(k, d_model)``.
        flat_weights: Full flat weight dict from the model.
        target_layers: Layer indices to create adapters for.
        alpha: Base scaling factor (from ``[cut].alpha``).
        polarity: ``"remove"`` or ``"add"``.
        layer_weights: Per-layer alpha multipliers. If None, uniform.

    Returns:
        List of ``LoraMatrices`` for each target weight key.
    """
    all_keys = list(flat_weights.keys())
    is_subspace = direction.ndim == 2

    if layer_weights is None:
        per_layer_alpha = [alpha] * len(target_layers)
    else:
        if len(layer_weights) != len(target_layers):
            msg = (
                f"layer_weights length ({len(layer_weights)}) must match "
                f"target_layers length ({len(target_layers)})"
            )
            raise ValueError(msg)
        per_layer_alpha = [alpha * w for w in layer_weights]

    matrices: list[LoraMatrices] = []

    for layer_idx, a in zip(target_layers, per_layer_alpha, strict=True):
        keys = _keys_for_layer(all_keys, layer_idx)
        for key in keys:
            w = flat_weights[key]
            # Skip 3D weights (MoE experts)
            if w.ndim != 2:
                continue

            if is_subspace:
                lora_a, lora_b = subspace_to_lora(direction, w, polarity)
            else:
                lora_a, lora_b = direction_to_lora(direction, w, polarity)

            # Bake per-key alpha into lora_b
            lora_b = lora_b * a

            matrices.append(LoraMatrices(key=key, lora_a=lora_a, lora_b=lora_b))

    return matrices


# ---------------------------------------------------------------------------
# Save adapters
# ---------------------------------------------------------------------------


def _mlx_lora_key(weight_key: str) -> tuple[str, str]:
    """Convert a flat weight key to mlx-lm LoRA key pair.

    mlx-lm format: strip ``model.`` prefix and ``.weight`` suffix,
    append ``.lora_a`` / ``.lora_b``.
    """
    k = weight_key
    if k.startswith("model."):
        k = k[len("model."):]
    if k.endswith(".weight"):
        k = k[: -len(".weight")]
    return f"{k}.lora_a", f"{k}.lora_b"


def _peft_lora_key(weight_key: str) -> tuple[str, str]:
    """Convert a flat weight key to PEFT LoRA key pair.

    PEFT format: prepend ``base_model.model.``, strip ``.weight``,
    append ``.lora_A.weight`` / ``.lora_B.weight``.
    """
    k = weight_key
    if not k.startswith("base_model.model."):
        k = f"base_model.model.{k}"
    if k.endswith(".weight"):
        k = k[: -len(".weight")]
    return f"{k}.lora_A.weight", f"{k}.lora_B.weight"


def save_adapter_mlx(
    matrices: list[LoraMatrices],
    output_dir: Path,
    rank: int,
    model_path: str,
) -> Path:
    """Write LoRA adapter in mlx-lm format.

    Creates ``adapters.safetensors`` and ``adapter_config.json``.
    Alpha is already baked into the matrices by ``build_lora_weights``,
    so ``lora_alpha`` is set equal to ``rank`` (scaling = 1.0).

    Args:
        matrices: LoRA matrices from ``build_lora_weights``.
        output_dir: Directory to write to.
        rank: LoRA rank.
        model_path: Base model path (for config metadata).

    Returns:
        Path to the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, Array] = {}
    for m in matrices:
        a_key, b_key = _mlx_lora_key(m.key)
        tensors[a_key] = m.lora_a
        tensors[b_key] = m.lora_b

    ops.save_safetensors(str(output_dir / "adapters.safetensors"), tensors)

    # Alpha is already baked into lora_b by build_lora_weights,
    # so set lora_alpha = rank to make the runtime scaling
    # (lora_alpha / rank) = 1.0 — a no-op.
    config = {
        "lora_rank": rank,
        "lora_alpha": float(rank),
        "model_path": model_path,
        "num_lora_layers": len(matrices),
    }
    (output_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2),
    )

    return output_dir


def save_adapter_peft(
    matrices: list[LoraMatrices],
    output_dir: Path,
    rank: int,
    model_path: str,
) -> Path:
    """Write LoRA adapter in PEFT/HuggingFace format.

    Creates ``adapter_model.safetensors`` and ``adapter_config.json``.
    Alpha is already baked into the matrices by ``build_lora_weights``,
    so ``lora_alpha`` is set equal to ``rank`` (scaling = 1.0).

    Args:
        matrices: LoRA matrices from ``build_lora_weights``.
        output_dir: Directory to write to.
        rank: LoRA rank.
        model_path: Base model path (for config metadata).

    Returns:
        Path to the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, Array] = {}
    for m in matrices:
        a_key, b_key = _peft_lora_key(m.key)
        tensors[a_key] = m.lora_a
        tensors[b_key] = m.lora_b

    ops.save_safetensors(
        str(output_dir / "adapter_model.safetensors"), tensors,
    )

    # Collect target module names for PEFT config
    target_modules: list[str] = []
    for m in matrices:
        k = m.key
        if k.endswith(".weight"):
            k = k[: -len(".weight")]
        # Extract the last component (e.g. "o_proj", "down_proj")
        parts = k.rsplit(".", maxsplit=1)
        module_name = parts[-1] if len(parts) > 1 else k
        if module_name not in target_modules:
            target_modules.append(module_name)

    # Alpha is already baked into lora_b by build_lora_weights,
    # so set lora_alpha = rank to make the runtime scaling
    # (lora_alpha / rank) = 1.0 — a no-op.
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": model_path,
        "r": rank,
        "lora_alpha": float(rank),
        "target_modules": target_modules,
        "bias": "none",
        "fan_in_fan_out": False,
        "task_type": "CAUSAL_LM",
    }
    (output_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2),
    )

    return output_dir


# ---------------------------------------------------------------------------
# Adapter merging (task arithmetic)
# ---------------------------------------------------------------------------


def merge_adapters(
    adapter_paths: list[Path],
    weights: list[float],
) -> dict[str, Array]:
    """Merge multiple LoRA adapters via weighted sum (task arithmetic).

    Loads each adapter's safetensors file and computes the weighted
    sum of matching keys. Keys present in only some adapters are
    weighted by their respective scalar only.

    Args:
        adapter_paths: Paths to adapter directories. Each must contain
            either ``adapters.safetensors`` (mlx) or
            ``adapter_model.safetensors`` (PEFT).
        weights: Scalar weight for each adapter.

    Returns:
        Merged weight dict ready for ``save_adapter_*``.
    """
    if len(adapter_paths) != len(weights):
        msg = (
            f"adapter_paths length ({len(adapter_paths)}) must match "
            f"weights length ({len(weights)})"
        )
        raise ValueError(msg)

    merged: dict[str, Array] = {}

    for path, w in zip(adapter_paths, weights, strict=True):
        st_file = _find_safetensors(path)
        tensors = cast("dict[str, Array]", ops.load(str(st_file)))
        for key, tensor in tensors.items():
            if key in merged:
                merged[key] = merged[key] + w * tensor
            else:
                merged[key] = w * tensor

    return merged


def _find_safetensors(adapter_dir: Path) -> Path:
    """Locate the safetensors file in an adapter directory."""
    for name in ("adapters.safetensors", "adapter_model.safetensors"):
        candidate = adapter_dir / name
        if candidate.exists():
            return candidate
    msg = f"No safetensors file found in {adapter_dir}"
    raise FileNotFoundError(msg)


# ---------------------------------------------------------------------------
# Key normalization
# ---------------------------------------------------------------------------

# Suffixes in order of specificity (PEFT first, then mlx).
_LORA_SUFFIXES: tuple[str, ...] = (
    ".lora_A.weight",
    ".lora_B.weight",
    ".lora_a",
    ".lora_b",
)

# PEFT adapters may carry a ``base_model.model.`` prefix that doesn't
# appear in the mlx-lm weight namespace.
_PEFT_PREFIX = "base_model.model."


def _strip_lora_suffix(key: str) -> str | None:
    """Return the base key with the LoRA suffix and PEFT prefix removed.

    Returns ``None`` if the key doesn't end with a known LoRA suffix.
    """
    for suffix in _LORA_SUFFIXES:
        if key.endswith(suffix):
            base = key[: -len(suffix)]
            if base.startswith(_PEFT_PREFIX):
                base = base[len(_PEFT_PREFIX):]
            return base
    return None


# ---------------------------------------------------------------------------
# Adapter loading
# ---------------------------------------------------------------------------


def load_and_apply_adapter(model: object, adapter_path: str) -> None:
    """Load a LoRA adapter and fuse into model weights.

    Uses mlx-lm's ``load_adapters`` to inject LoRA layers, then fuses
    each ``LoRALinear`` back into its parent as a plain ``Linear``.
    Mutates the model in-place.

    Args:
        model: An mlx-lm model instance.
        adapter_path: Path to adapter directory containing
            ``adapters.safetensors`` and ``adapter_config.json``.
    """
    from mlx_lm.tuner.utils import apply_lora_layers  # type: ignore[import-untyped]

    apply_lora_layers(model, adapter_path)
    _fuse_lora_layers(model)


def _fuse_lora_layers(model: object) -> None:
    """Walk the model tree and fuse all LoRALinear into plain Linear."""
    for _name, module in model.named_modules():  # type: ignore[attr-defined]
        # Check children for LoRALinear instances
        for child_name in list(vars(module)):
            child = getattr(module, child_name, None)
            if child is None:
                continue
            if hasattr(child, "to_linear"):
                fused = child.to_linear()
                setattr(module, child_name, fused)


def load_and_merge_adapters(
    model: object,
    adapter_paths: list[str],
    weights: list[float] | None = None,
) -> None:
    """Merge multiple adapters via task arithmetic and apply to model.

    Groups lora_a/lora_b pairs from the merged weight dict, reconstructs
    the full-rank delta ``B @ A``, and adds it to the corresponding model
    weight. Mutates the model in-place.

    Args:
        model: An mlx-lm model instance.
        adapter_paths: Paths to adapter directories.
        weights: Scalar weight for each adapter. Defaults to uniform 1.0.
    """
    from pathlib import Path

    from vauban._forward import force_eval

    path_objs = [Path(p) for p in adapter_paths]
    if weights is None:
        weights = [1.0] * len(path_objs)

    merged = merge_adapters(path_objs, weights)

    # Group lora_a / lora_b pairs, normalizing both mlx and PEFT key formats
    pairs: dict[str, dict[str, Array]] = {}
    for key, tensor in merged.items():
        base = _strip_lora_suffix(key)
        if base is None:
            continue
        if key.endswith((".lora_a", ".lora_A.weight")):
            pairs.setdefault(base, {})["lora_a"] = tensor
        elif key.endswith((".lora_b", ".lora_B.weight")):
            pairs.setdefault(base, {})["lora_b"] = tensor

    # Apply deltas to model weights
    flat_weights = cast(
        "dict[str, Array]",
        dict(ops.tree_flatten(model.parameters())),  # type: ignore[attr-defined]
    )

    updates: list[tuple[str, Array]] = []
    for base_key, ab in pairs.items():
        lora_a = ab["lora_a"]
        lora_b = ab["lora_b"]
        delta = ops.matmul(lora_b, lora_a)

        # Reconstruct the full weight key from the normalized base
        weight_key = f"model.{base_key}.weight"
        if weight_key not in flat_weights:
            weight_key = f"{base_key}.weight"
        if weight_key in flat_weights:
            new_w = flat_weights[weight_key] + delta
            force_eval(new_w)
            updates.append((weight_key, new_w))

    # Apply updates via tree_unflatten pattern
    if updates:
        model.load_weights(updates)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Adapter analysis
# ---------------------------------------------------------------------------


def analyze_adapter(
    adapter_path: str,
    variance_threshold: float = 0.99,
    direction: Array | None = None,
) -> LoraAnalysisResult:
    """Decompose a LoRA adapter via SVD for structural analysis.

    Per weight pair (lora_a, lora_b):
    1. Reconstruct ``delta = B @ A``.
    2. SVD decomposition.
    3. Compute Frobenius norm, effective rank, variance cutoff.
    4. Optionally compute alignment with a measured direction.

    Args:
        adapter_path: Path to adapter directory.
        variance_threshold: Cumulative variance threshold for rank cutoff.
        direction: Optional direction vector for alignment computation.

    Returns:
        Full analysis result with per-layer and aggregate metrics.
    """
    from pathlib import Path

    from vauban._forward import force_eval, svd_stable
    from vauban.subspace import effective_rank, explained_variance_ratio
    from vauban.types import LoraAnalysisResult as _Result
    from vauban.types import LoraLayerAnalysis as _LayerResult

    adapter_dir = Path(adapter_path)
    st_file = _find_safetensors(adapter_dir)
    tensors = cast("dict[str, Array]", ops.load(str(st_file)))

    # Group lora_a / lora_b pairs, normalizing both mlx and PEFT key formats
    pairs: dict[str, dict[str, Array]] = {}
    for key, tensor in tensors.items():
        base = _strip_lora_suffix(key)
        if base is None:
            continue
        if key.endswith((".lora_a", ".lora_A.weight")):
            pairs.setdefault(base, {})["lora_a"] = tensor
        elif key.endswith((".lora_b", ".lora_B.weight")):
            pairs.setdefault(base, {})["lora_b"] = tensor

    layers: list[_LayerResult] = []
    total_params = 0
    norm_profile: list[float] = []

    for base_key in sorted(pairs):
        ab = pairs[base_key]
        if "lora_a" not in ab or "lora_b" not in ab:
            continue

        lora_a = ab["lora_a"]
        lora_b = ab["lora_b"]
        total_params += lora_a.shape[0] * lora_a.shape[1]
        total_params += lora_b.shape[0] * lora_b.shape[1]

        # Reconstruct delta
        delta = ops.matmul(lora_b, lora_a)
        force_eval(delta)

        # SVD
        u, s, _vt = svd_stable(delta)
        force_eval(u, s)

        sv_raw: list[object] = s.tolist()  # type: ignore[assignment]
        sv_list = [float(v) for v in sv_raw]  # type: ignore[arg-type]

        # Frobenius norm from singular values
        frob = math.sqrt(sum(v * v for v in sv_list))

        # Effective rank
        eff_rank = effective_rank(sv_list) if sv_list else 1.0

        # Variance cutoff
        ratios = explained_variance_ratio(sv_list)
        cumsum = 0.0
        cutoff = len(sv_list)
        for i, r in enumerate(ratios):
            cumsum += r
            if cumsum >= variance_threshold:
                cutoff = i + 1
                break

        # Direction alignment
        alignment: float | None = None
        if direction is not None and u.shape[0] == direction.shape[0]:
            # cos(U[:,0], direction)
            u0 = u[:, 0]
            dot = float(ops.sum(u0 * direction))
            alignment = abs(dot)

        layers.append(_LayerResult(
            key=base_key,
            frobenius_norm=frob,
            singular_values=sv_list,
            effective_rank=eff_rank,
            variance_cutoff=cutoff,
            direction_alignment=alignment,
        ))
        norm_profile.append(frob)

    mean_eff_rank = (
        sum(layer.effective_rank for layer in layers) / len(layers)
        if layers
        else 0.0
    )

    return _Result(
        adapter_path=adapter_path,
        layers=layers,
        total_params=total_params,
        mean_effective_rank=mean_eff_rank,
        norm_profile=norm_profile,
    )
