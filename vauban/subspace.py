"""Subspace geometry tools for analyzing refusal subspaces.

Pure mx.array linear algebra — no model dependency. Implements the
analysis toolkit needed for GRP-Oblit Section 3.3.2 style subspace
comparison: principal angles, Grassmann distance, overlap metrics,
and projection/removal operations.
"""

import math

import mlx.core as mx


def principal_angles(u: mx.array, v: mx.array) -> mx.array:
    """Compute principal angles between two subspaces.

    Args:
        u: Orthonormal basis, shape (k1, d).
        v: Orthonormal basis, shape (k2, d).

    Returns:
        Array of min(k1, k2) principal angles in [0, pi/2].
    """
    # Cross-product matrix M = U @ V^T
    m = u @ v.T
    _, s, _ = mx.linalg.svd(m, stream=mx.cpu)  # type: ignore[arg-type]
    mx.eval(s)
    # Clamp to [0, 1] for numerical stability before arccos
    s = mx.clip(s, 0.0, 1.0)
    angles = mx.arccos(s)
    mx.eval(angles)
    return angles


def grassmann_distance(u: mx.array, v: mx.array) -> float:
    """Grassmann distance between two subspaces.

    Equal to the L2 norm of the vector of principal angles.

    Args:
        u: Orthonormal basis, shape (k1, d).
        v: Orthonormal basis, shape (k2, d).

    Returns:
        Non-negative distance. Zero iff subspaces are identical.
    """
    angles = principal_angles(u, v)
    dist = mx.sqrt(mx.sum(angles * angles))
    mx.eval(dist)
    return float(dist.item())


def subspace_overlap(u: mx.array, v: mx.array) -> float:
    """Mean squared cosine of principal angles (subspace overlap).

    Args:
        u: Orthonormal basis, shape (k1, d).
        v: Orthonormal basis, shape (k2, d).

    Returns:
        Value in [0, 1]. 1.0 = identical subspaces, 0.0 = orthogonal.
    """
    angles = principal_angles(u, v)
    cos_sq = mx.cos(angles) ** 2
    overlap = mx.mean(cos_sq)
    mx.eval(overlap)
    return float(overlap.item())


def project_subspace(x: mx.array, basis: mx.array) -> mx.array:
    """Project a vector onto a subspace spanned by an orthonormal basis.

    Args:
        x: Vector of shape (d,).
        basis: Orthonormal basis, shape (k, d).

    Returns:
        Projection of x onto the subspace, shape (d,).
    """
    # coeffs[i] = <x, basis[i]>
    coeffs = basis @ x  # (k,)
    return coeffs @ basis  # (d,)


def remove_subspace(x: mx.array, basis: mx.array) -> mx.array:
    """Remove the subspace component from a vector.

    Args:
        x: Vector of shape (d,).
        basis: Orthonormal basis, shape (k, d).

    Returns:
        x minus its projection onto the subspace, shape (d,).
    """
    return x - project_subspace(x, basis)


def orthonormalize(vectors: mx.array) -> mx.array:
    """Orthonormalize a set of vectors via QR decomposition.

    Args:
        vectors: Matrix of shape (k, d) where k <= d.

    Returns:
        Orthonormal matrix of shape (k, d).
    """
    # QR on the transpose: vectors^T = Q @ R
    # Then Q^T gives us the orthonormal rows
    q, _ = mx.linalg.qr(vectors.T, stream=mx.cpu)  # type: ignore[arg-type]
    mx.eval(q)
    k = vectors.shape[0]
    return q[:, :k].T


def explained_variance_ratio(singular_values: list[float]) -> list[float]:
    """Compute explained variance ratio from singular values.

    Args:
        singular_values: List of singular values (decreasing).

    Returns:
        List of variance ratios summing to 1.0.
    """
    total = sum(s * s for s in singular_values)
    if total < 1e-10:
        return [0.0] * len(singular_values)
    return [(s * s) / total for s in singular_values]


def effective_rank(singular_values: list[float]) -> float:
    """Compute the effective rank (Shannon entropy of normalized squared SVs).

    A measure of the dimensionality of the subspace.
    Returns 1.0 for a perfectly rank-1 matrix, higher for spread-out spectra.

    Args:
        singular_values: List of singular values.

    Returns:
        Effective rank (>= 1.0).
    """
    ratios = explained_variance_ratio(singular_values)
    entropy = 0.0
    for r in ratios:
        if r > 1e-10:
            entropy -= r * math.log(r)
    return math.exp(entropy)
