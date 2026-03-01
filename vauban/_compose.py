"""Subspace bank composition for Steer2Adapt composed steering.

Loads named subspace bases from a safetensors bank file and computes
weighted linear combinations of first basis vectors.

Reference: Han et al. (2026) — arxiv.org/abs/2602.07276
"""

from pathlib import Path

from vauban import _ops as ops
from vauban._array import Array
from vauban.types import DirectionSpace


def load_bank(path: str | Path) -> dict[str, Array]:
    """Load a subspace bank from a safetensors file.

    Returns a dict mapping subspace names to their basis arrays.
    """
    raw = ops.load(str(path))
    if not isinstance(raw, dict):
        msg = f"Expected dict from bank file, got {type(raw).__name__}"
        raise TypeError(msg)
    # ops.load returns dict[str, Array]; narrow values through isinstance
    loaded: dict[str, Array] = {}
    for k, v in raw.items():
        if not isinstance(v, Array):
            msg = f"Bank value for {k!r} is not an Array"
            raise TypeError(msg)
        loaded[str(k)] = v
    return loaded


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


def compose_subspaces(
    bank: dict[str, Array],
    composition: dict[str, float],
    max_rank: int | None = None,
) -> DirectionSpace:
    """Compose DirectionSpaces from a bank with weights (full-subspace Steer2Adapt).

    Unlike compose_direction() which only uses the first basis vector,
    this uses all basis vectors from each entry, weighted by their
    singular values.

    Args:
        bank: Named subspace bases (shape (k, d_model) each).
        composition: Mapping of subspace name to weight.
        max_rank: Maximum rank of the output space.

    Raises:
        KeyError: If a composition name is not found in the bank.
        ValueError: If the composition is empty.
    """
    from vauban.algebra import compose, from_array

    if not composition:
        msg = "Composition is empty"
        raise ValueError(msg)

    spaces: list[DirectionSpace] = []
    weights: list[float] = []
    for name, weight in composition.items():
        if name not in bank:
            msg = (
                f"Composition references unknown bank entry {name!r}."
                f" Available: {sorted(bank.keys())}"
            )
            raise KeyError(msg)
        spaces.append(from_array(bank[name], label=name))
        weights.append(weight)

    return compose(spaces, weights, max_rank=max_rank)
