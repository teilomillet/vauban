"""LoRA adapter construction from measured directions.

Converts the rank-1 weight update ``W' = W - alpha * d (d^T W)`` into
LoRA matrices ``delta_W = B @ A`` where ``B = -d`` and ``A = d^T W``.

Pure functions, no pipeline dependencies beyond ``cut._keys_for_layer``
and ``cut._OUTPUT_SUFFIXES``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from vauban import _ops as ops
from vauban.cut import _keys_for_layer
from vauban.types import LoraMatrices

if TYPE_CHECKING:
    from pathlib import Path

    from vauban._array import Array


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
    # Stack column vectors to avoid needing a transpose op
    cols = [ops.expand_dims(basis[i], axis=1) for i in range(k)]
    lora_b = ops.concatenate(cols, axis=1)  # (d_model, k)

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
    sum of matching keys. All adapters must have the same keys.

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
