"""Subspace bank composition for Steer2Adapt composed steering.

Loads named subspace bases from a safetensors bank file and computes
weighted linear combinations of first basis vectors.

Reference: Han et al. (2026) — arxiv.org/abs/2602.07276
"""

from pathlib import Path

from vauban import _ops as ops
from vauban._array import Array


def load_bank(path: str | Path) -> dict[str, Array]:
    """Load a subspace bank from a safetensors file.

    Returns a dict mapping subspace names to their basis arrays.
    """
    loaded = ops.load(str(path))
    if not isinstance(loaded, dict):
        msg = f"Expected dict from bank file, got {type(loaded).__name__}"
        raise TypeError(msg)
    return {str(k): v for k, v in loaded.items()}


def compose_direction(
    bank: dict[str, Array],
    composition: dict[str, float],
) -> Array:
    """Compose a direction from named bank entries with weights.

    Takes the first basis vector from each named entry, computes a weighted
    sum, and L2-normalizes the result.

    Args:
        bank: Named subspace bases (shape (k, d_model) each).
        composition: Mapping of subspace name to weight.

    Raises:
        KeyError: If a composition name is not found in the bank.
        ValueError: If the composed direction has zero norm.
    """
    result: Array | None = None
    for name, weight in composition.items():
        if name not in bank:
            msg = (
                f"Composition references unknown bank entry {name!r}."
                f" Available: {sorted(bank.keys())}"
            )
            raise KeyError(msg)
        basis = bank[name]
        # Take first basis vector (most important direction)
        first_vec = basis[0] if basis.ndim == 2 else basis
        scaled = weight * first_vec
        result = scaled if result is None else result + scaled

    if result is None:
        msg = "Composition is empty"
        raise ValueError(msg)

    # L2 normalize
    norm = ops.sqrt(ops.sum(result * result))
    norm_val = float(norm.item())
    if norm_val < 1e-10:
        msg = "Composed direction has near-zero norm"
        raise ValueError(msg)
    return result / norm
