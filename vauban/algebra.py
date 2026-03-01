"""Direction algebra — closed operations on directional subspaces.

Pure functions operating on DirectionSpace. Every algebraic operation
returns a new DirectionSpace, ensuring closure. Math delegates to
existing subspace.py primitives.
"""

from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import force_eval, svd_stable
from vauban.subspace import orthonormalize, remove_subspace, subspace_overlap
from vauban.types import (
    DBDIResult,
    DiffResult,
    DirectionProvenance,
    DirectionResult,
    DirectionSpace,
    SubspaceResult,
)

# ---------------------------------------------------------------------------
# Converters — from existing result types to DirectionSpace
# ---------------------------------------------------------------------------


def from_direction_result(result: DirectionResult, label: str = "") -> DirectionSpace:
    """Wrap a single-vector DirectionResult as a rank-1 DirectionSpace."""
    direction = result.direction
    basis = ops.expand_dims(direction, axis=0) if direction.ndim == 1 else direction
    force_eval(basis)
    return DirectionSpace(
        basis=basis,
        d_model=result.d_model,
        rank=basis.shape[0],
        label=label or f"direction_L{result.layer_index}",
        layer_index=result.layer_index,
        provenance=DirectionProvenance("convert", (), {"source": "DirectionResult"}),
    )


def from_subspace_result(result: SubspaceResult, label: str = "") -> DirectionSpace:
    """Convert a SubspaceResult to a DirectionSpace."""
    rank = result.basis.shape[0]
    return DirectionSpace(
        basis=result.basis,
        d_model=result.d_model,
        rank=rank,
        label=label or f"subspace_L{result.layer_index}",
        layer_index=result.layer_index,
        singular_values=list(result.singular_values),
        provenance=DirectionProvenance("convert", (), {"source": "SubspaceResult"}),
    )


def from_diff_result(result: DiffResult, label: str = "") -> DirectionSpace:
    """Convert a DiffResult to a DirectionSpace."""
    rank = result.basis.shape[0]
    return DirectionSpace(
        basis=result.basis,
        d_model=result.d_model,
        rank=rank,
        label=label or f"diff_L{result.best_layer}",
        layer_index=result.best_layer,
        singular_values=list(result.singular_values),
        provenance=DirectionProvenance("convert", (), {"source": "DiffResult"}),
    )


def from_dbdi_result(result: DBDIResult, label: str = "") -> DirectionSpace:
    """Convert a DBDIResult to a rank-2 DirectionSpace (HDD + RED).

    Stacks the harm-detection and refusal-execution directions and
    orthonormalizes them.
    """
    hdd = result.hdd
    red = result.red
    if hdd.ndim == 1:
        hdd = ops.expand_dims(hdd, axis=0)
    if red.ndim == 1:
        red = ops.expand_dims(red, axis=0)
    stacked = ops.concatenate([hdd, red], axis=0)  # (2, d_model)
    basis = orthonormalize(stacked)
    force_eval(basis)
    rank = basis.shape[0]
    return DirectionSpace(
        basis=basis,
        d_model=result.d_model,
        rank=rank,
        label=label or "dbdi",
        provenance=DirectionProvenance("convert", (), {"source": "DBDIResult"}),
    )


def from_array(arr: Array, label: str = "") -> DirectionSpace:
    """Create a DirectionSpace from a raw array.

    1D array (d,) → rank-1 space. 2D array (k, d) → rank-k space.
    The basis is orthonormalized.
    """
    if arr.ndim == 1:
        norm = ops.sqrt(ops.sum(arr * arr))
        force_eval(norm)
        norm_val = float(norm.item())
        if norm_val < 1e-10:
            # Zero vector → rank-0 space
            d_model = arr.shape[0]
            basis = ops.zeros((0, d_model))
            force_eval(basis)
            return DirectionSpace(
                basis=basis,
                d_model=d_model,
                rank=0,
                label=label,
                provenance=DirectionProvenance("convert", (), {"source": "array"}),
            )
        normalized = ops.expand_dims(arr / norm, axis=0)
        force_eval(normalized)
        return DirectionSpace(
            basis=normalized,
            d_model=arr.shape[0],
            rank=1,
            label=label,
            provenance=DirectionProvenance("convert", (), {"source": "array"}),
        )

    if arr.ndim == 2:
        k, d_model = arr.shape
        if k == 0:
            basis = ops.zeros((0, d_model))
            force_eval(basis)
            return DirectionSpace(
                basis=basis,
                d_model=d_model,
                rank=0,
                label=label,
                provenance=DirectionProvenance("convert", (), {"source": "array"}),
            )
        basis = orthonormalize(arr)
        force_eval(basis)
        return DirectionSpace(
            basis=basis,
            d_model=d_model,
            rank=basis.shape[0],
            label=label,
            provenance=DirectionProvenance("convert", (), {"source": "array"}),
        )

    msg = f"Expected 1D or 2D array, got ndim={arr.ndim}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Algebra operations — all return DirectionSpace (closure)
# ---------------------------------------------------------------------------


def _validate_compatible(a: DirectionSpace, b: DirectionSpace) -> None:
    """Raise ValueError if two spaces have different d_model."""
    if a.d_model != b.d_model:
        msg = (
            f"Dimension mismatch: {a.label!r} has d_model={a.d_model}, "
            f"{b.label!r} has d_model={b.d_model}"
        )
        raise ValueError(msg)


def _add_provenance(
    a: DirectionSpace,
    b: DirectionSpace,
    max_rank: int | None,
    variance_threshold: float | None,
) -> DirectionProvenance:
    """Build provenance for an add operation."""
    params: dict[str, float | int | str] = {}
    if max_rank is not None:
        params["max_rank"] = max_rank
    if variance_threshold is not None:
        params["variance_threshold"] = variance_threshold
    return DirectionProvenance("add", (a.label, b.label), params)


def add(
    a: DirectionSpace,
    b: DirectionSpace,
    max_rank: int | None = None,
    variance_threshold: float | None = None,
) -> DirectionSpace:
    """Add two subspaces: union of their spans, truncated by SVD.

    Stacks bases, performs SVD, retains components above the variance
    threshold (default: 1/(k1+k2) of total variance) or up to max_rank.
    """
    _validate_compatible(a, b)
    d_model = a.d_model

    # Handle rank-0 cases
    if a.rank == 0:
        return DirectionSpace(
            basis=b.basis,
            d_model=d_model,
            rank=b.rank,
            label=f"({a.label} + {b.label})",
            singular_values=list(b.singular_values),
            provenance=_add_provenance(
                a, b, max_rank, variance_threshold,
            ),
        )
    if b.rank == 0:
        return DirectionSpace(
            basis=a.basis,
            d_model=d_model,
            rank=a.rank,
            label=f"({a.label} + {b.label})",
            singular_values=list(a.singular_values),
            provenance=_add_provenance(
                a, b, max_rank, variance_threshold,
            ),
        )

    stacked = ops.concatenate([a.basis, b.basis], axis=0)  # (k1+k2, d_model)
    u, s, vt = svd_stable(stacked)
    force_eval(u, s, vt)

    # Determine how many components to keep
    total_k = a.rank + b.rank
    threshold = 1.0 / total_k if variance_threshold is None else variance_threshold

    s_list = [float(s[i].item()) for i in range(min(s.shape[0], total_k))]
    total_var = sum(sv * sv for sv in s_list)
    if total_var < 1e-10:
        basis = ops.zeros((0, d_model))
        force_eval(basis)
        return DirectionSpace(
            basis=basis,
            d_model=d_model,
            rank=0,
            label=f"({a.label} + {b.label})",
            provenance=_add_provenance(a, b, max_rank, threshold),
        )

    keep = 0
    for sv in s_list:
        if (sv * sv) / total_var >= threshold:
            keep += 1

    if max_rank is not None:
        keep = min(keep, max_rank)
    keep = max(keep, 1)  # keep at least 1 if there's signal

    # Take top-k rows from Vt (right singular vectors = directions in d_model space)
    basis = vt[:keep]
    force_eval(basis)
    kept_svs = s_list[:keep]

    return DirectionSpace(
        basis=basis,
        d_model=d_model,
        rank=keep,
        label=f"({a.label} + {b.label})",
        singular_values=kept_svs,
        provenance=_add_provenance(a, b, max_rank, threshold),
    )


def subtract(a: DirectionSpace, b: DirectionSpace) -> DirectionSpace:
    """Subtract b from a: remove b's subspace from each of a's basis vectors.

    May produce lower rank or rank-0 if a is a subset of b.
    """
    _validate_compatible(a, b)
    d_model = a.d_model

    if a.rank == 0 or b.rank == 0:
        return DirectionSpace(
            basis=a.basis,
            d_model=d_model,
            rank=a.rank,
            label=f"({a.label} - {b.label})",
            singular_values=list(a.singular_values),
            provenance=DirectionProvenance("subtract", (a.label, b.label), {}),
        )

    # Project each of a's basis vectors out of b's subspace
    residuals: list[Array] = []
    for i in range(a.rank):
        v = a.basis[i]
        r = remove_subspace(v, b.basis)
        force_eval(r)
        norm = float(ops.sqrt(ops.sum(r * r)).item())
        if norm > 1e-6:
            residuals.append(ops.expand_dims(r, axis=0))

    if not residuals:
        basis = ops.zeros((0, d_model))
        force_eval(basis)
        return DirectionSpace(
            basis=basis,
            d_model=d_model,
            rank=0,
            label=f"({a.label} - {b.label})",
            provenance=DirectionProvenance("subtract", (a.label, b.label), {}),
        )

    stacked = ops.concatenate(residuals, axis=0)
    basis = orthonormalize(stacked)
    force_eval(basis)

    return DirectionSpace(
        basis=basis,
        d_model=d_model,
        rank=basis.shape[0],
        label=f"({a.label} - {b.label})",
        provenance=DirectionProvenance("subtract", (a.label, b.label), {}),
    )


def intersect(
    a: DirectionSpace,
    b: DirectionSpace,
    threshold: float = 0.3,
) -> DirectionSpace:
    """Intersect two subspaces: directions with small principal angles.

    Computes cross-matrix M = a.basis @ b.basis.T, takes SVD, and
    retains directions where cos(angle) > threshold.
    """
    _validate_compatible(a, b)
    d_model = a.d_model

    if a.rank == 0 or b.rank == 0:
        basis = ops.zeros((0, d_model))
        force_eval(basis)
        return DirectionSpace(
            basis=basis,
            d_model=d_model,
            rank=0,
            label=f"({a.label} \u2229 {b.label})",
            provenance=DirectionProvenance(
                "intersect", (a.label, b.label), {"threshold": threshold},
            ),
        )

    # Cross-matrix SVD
    m = a.basis @ b.basis.T  # (k1, k2)
    u, s, _vt = svd_stable(m)
    force_eval(u, s)

    # s contains cos(principal_angles) — keep those above threshold
    n_angles = min(a.rank, b.rank)
    keep_indices: list[int] = []
    kept_svs: list[float] = []
    for i in range(min(s.shape[0], n_angles)):
        cos_angle = float(s[i].item())
        if cos_angle > threshold:
            keep_indices.append(i)
            kept_svs.append(cos_angle)

    if not keep_indices:
        basis = ops.zeros((0, d_model))
        force_eval(basis)
        return DirectionSpace(
            basis=basis,
            d_model=d_model,
            rank=0,
            label=f"({a.label} \u2229 {b.label})",
            provenance=DirectionProvenance(
                "intersect", (a.label, b.label), {"threshold": threshold},
            ),
        )

    # Extract intersection directions: U columns in a's coordinate system
    # u[:, i] gives coefficients in a's basis → project back to d_model
    directions: list[Array] = []
    for i in keep_indices:
        coeffs = u[:, i]  # (k1,)
        # Linear combination of a's basis vectors
        direction = coeffs @ a.basis  # (d_model,)
        directions.append(ops.expand_dims(direction, axis=0))

    stacked = ops.concatenate(directions, axis=0)
    basis = orthonormalize(stacked)
    force_eval(basis)

    return DirectionSpace(
        basis=basis,
        d_model=d_model,
        rank=basis.shape[0],
        label=f"({a.label} \u2229 {b.label})",
        singular_values=kept_svs,
        provenance=DirectionProvenance(
            "intersect", (a.label, b.label), {"threshold": threshold},
        ),
    )


def negate(space: DirectionSpace) -> DirectionSpace:
    """Negate a subspace (flip all basis vectors)."""
    negated = -space.basis
    force_eval(negated)
    return DirectionSpace(
        basis=negated,
        d_model=space.d_model,
        rank=space.rank,
        label=f"(-{space.label})",
        layer_index=space.layer_index,
        singular_values=list(space.singular_values),
        provenance=DirectionProvenance("negate", (space.label,), {}),
    )


def compose(
    spaces: list[DirectionSpace],
    weights: list[float],
    max_rank: int | None = None,
) -> DirectionSpace:
    """Compose multiple spaces with weights (generalized Steer2Adapt).

    Scales all basis vectors by weight * singular_value (or weight alone
    if no singular values), stacks everything, and takes SVD to produce
    the combined space.
    """
    if len(spaces) != len(weights):
        msg = (
            f"spaces ({len(spaces)}) and weights ({len(weights)})"
            " must have same length"
        )
        raise ValueError(msg)
    if not spaces:
        msg = "Cannot compose empty list of spaces"
        raise ValueError(msg)

    d_model = spaces[0].d_model
    for s in spaces[1:]:
        if s.d_model != d_model:
            msg = f"Dimension mismatch: expected d_model={d_model}, got {s.d_model}"
            raise ValueError(msg)

    # Stack weighted basis vectors
    all_rows: list[Array] = []
    for space, w in zip(spaces, weights, strict=True):
        if space.rank == 0:
            continue
        for i in range(space.rank):
            sv = space.singular_values[i] if i < len(space.singular_values) else 1.0
            scaled = (w * sv) * space.basis[i]
            all_rows.append(ops.expand_dims(scaled, axis=0))

    labels = [
        f"{w:.1f}*{sp.label}"
        for sp, w in zip(spaces, weights, strict=True)
    ]
    prov = DirectionProvenance(
        "compose",
        tuple(sp.label for sp in spaces),
        {"weights": ",".join(f"{w:.2f}" for w in weights)},
    )

    if not all_rows:
        basis = ops.zeros((0, d_model))
        force_eval(basis)
        return DirectionSpace(
            basis=basis,
            d_model=d_model,
            rank=0,
            label=f"compose({', '.join(labels)})",
            provenance=prov,
        )

    stacked = ops.concatenate(all_rows, axis=0)
    u, s, vt = svd_stable(stacked)
    force_eval(u, s, vt)

    total_rows = stacked.shape[0]
    s_list = [float(s[i].item()) for i in range(min(s.shape[0], total_rows))]

    # Filter near-zero singular values
    s_list = [sv for sv in s_list if sv > 1e-6]

    if not s_list:
        basis = ops.zeros((0, d_model))
        force_eval(basis)
        return DirectionSpace(
            basis=basis,
            d_model=d_model,
            rank=0,
            label=f"compose({', '.join(labels)})",
            provenance=prov,
        )

    keep = len(s_list)
    if max_rank is not None:
        keep = min(keep, max_rank)
    keep = max(keep, 1)

    basis = vt[:keep]
    force_eval(basis)
    kept_svs = s_list[:keep]

    return DirectionSpace(
        basis=basis,
        d_model=d_model,
        rank=keep,
        label=f"compose({', '.join(labels)})",
        singular_values=kept_svs,
        provenance=prov,
    )


def similarity(a: DirectionSpace, b: DirectionSpace) -> float:
    """Compute subspace overlap between two DirectionSpaces.

    Returns a value in [0, 1]. 1.0 = identical subspaces, 0.0 = orthogonal.
    """
    _validate_compatible(a, b)
    if a.rank == 0 or b.rank == 0:
        return 0.0
    return subspace_overlap(a.basis, b.basis)


# ---------------------------------------------------------------------------
# Extractors — backward compatibility
# ---------------------------------------------------------------------------


def to_direction(space: DirectionSpace) -> Array:
    """Extract the first basis vector (rank-1 direction).

    Raises ValueError if the space is rank-0.
    """
    if space.rank == 0:
        msg = f"Cannot extract direction from rank-0 space {space.label!r}"
        raise ValueError(msg)
    return space.basis[0]


def to_basis(space: DirectionSpace) -> Array:
    """Extract the full basis matrix."""
    return space.basis
