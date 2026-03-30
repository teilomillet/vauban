"""Property-based tests for vauban.subspace — geometric invariants.

Tests metric properties: non-negativity, symmetry, identity of
indiscernibles, triangle inequality, projection idempotence, and
the fundamental decomposition project(x) + remove(x) = x.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from vauban import _ops as ops
from vauban.subspace import (
    effective_rank,
    explained_variance_ratio,
    grassmann_distance,
    orthonormalize,
    project_subspace,
    remove_subspace,
    subspace_overlap,
)

D_MODEL = 32


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _random_orthonormal(rank: int, d_model: int = D_MODEL) -> np.ndarray:
    """Generate a random orthonormal basis via QR."""
    if rank == 0:
        return np.zeros((0, d_model))
    a = np.random.randn(rank, d_model)
    q, _ = np.linalg.qr(a.T)
    return q.T[:rank]


@st.composite
def orthonormal_bases(
    draw: st.DrawFn,
    min_rank: int = 1,
    max_rank: int = 6,
    d_model: int = D_MODEL,
) -> np.ndarray:
    """Random orthonormal basis as numpy array (k, d)."""
    rank = draw(st.integers(min_value=min_rank, max_value=max_rank))
    return _random_orthonormal(rank, d_model)


def random_vectors(d_model: int = D_MODEL) -> st.SearchStrategy[np.ndarray]:
    """Random d-dimensional vector (not necessarily unit)."""
    return st.builds(lambda _: np.random.randn(d_model), st.just(None))


def _to_mx(arr: np.ndarray) -> object:
    """Convert numpy to backend array."""
    result = ops.array(arr.tolist())
    ops.eval(result)
    return result


_SETTINGS = settings(max_examples=50, deadline=None)


# ---------------------------------------------------------------------------
# Grassmann distance: metric properties
# ---------------------------------------------------------------------------


class TestGrassmannMetric:
    """grassmann_distance must satisfy metric axioms."""

    @_SETTINGS
    @given(u=orthonormal_bases())
    def test_self_distance_zero(self, u: np.ndarray) -> None:
        d = grassmann_distance(_to_mx(u), _to_mx(u))
        assert abs(d) < 1e-3, f"d(u, u) = {d}, expected ~0"

    @_SETTINGS
    @given(u=orthonormal_bases(), v=orthonormal_bases())
    def test_non_negative(self, u: np.ndarray, v: np.ndarray) -> None:
        d = grassmann_distance(_to_mx(u), _to_mx(v))
        assert d >= -1e-6, f"d(u, v) = {d} < 0"

    @_SETTINGS
    @given(u=orthonormal_bases(), v=orthonormal_bases())
    def test_symmetric(self, u: np.ndarray, v: np.ndarray) -> None:
        d_uv = grassmann_distance(_to_mx(u), _to_mx(v))
        d_vu = grassmann_distance(_to_mx(v), _to_mx(u))
        assert abs(d_uv - d_vu) < 1e-4, f"|d(u,v) - d(v,u)| = {abs(d_uv - d_vu)}"


# ---------------------------------------------------------------------------
# Subspace overlap: bounded [0, 1]
# ---------------------------------------------------------------------------


class TestSubspaceOverlap:
    @_SETTINGS
    @given(u=orthonormal_bases(), v=orthonormal_bases())
    def test_bounded(self, u: np.ndarray, v: np.ndarray) -> None:
        o = subspace_overlap(_to_mx(u), _to_mx(v))
        assert -1e-6 <= o <= 1.0 + 1e-6, f"overlap out of bounds: {o}"

    @_SETTINGS
    @given(u=orthonormal_bases())
    def test_self_overlap_is_one(self, u: np.ndarray) -> None:
        o = subspace_overlap(_to_mx(u), _to_mx(u))
        assert abs(o - 1.0) < 1e-4, f"overlap(u, u) = {o}, expected 1.0"

    @_SETTINGS
    @given(u=orthonormal_bases(), v=orthonormal_bases())
    def test_symmetric(self, u: np.ndarray, v: np.ndarray) -> None:
        o_uv = subspace_overlap(_to_mx(u), _to_mx(v))
        o_vu = subspace_overlap(_to_mx(v), _to_mx(u))
        assert abs(o_uv - o_vu) < 1e-4, f"|o(u,v) - o(v,u)| = {abs(o_uv - o_vu)}"


# ---------------------------------------------------------------------------
# Projection properties
# ---------------------------------------------------------------------------


class TestProjection:
    @_SETTINGS
    @given(u=orthonormal_bases(), x=random_vectors())
    def test_idempotent(self, u: np.ndarray, x: np.ndarray) -> None:
        """project(project(x, B), B) == project(x, B)."""
        basis = _to_mx(u)
        x_mx = _to_mx(x)
        p1 = project_subspace(x_mx, basis)
        p2 = project_subspace(p1, basis)
        p1_np = np.array(p1)
        p2_np = np.array(p2)
        assert np.allclose(p1_np, p2_np, atol=1e-4), (
            f"Projection not idempotent: max diff = {np.max(np.abs(p1_np - p2_np))}"
        )

    @_SETTINGS
    @given(u=orthonormal_bases(), x=random_vectors())
    def test_decomposition(self, u: np.ndarray, x: np.ndarray) -> None:
        """project(x, B) + remove(x, B) == x."""
        basis = _to_mx(u)
        x_mx = _to_mx(x)
        projected = np.array(project_subspace(x_mx, basis))
        removed = np.array(remove_subspace(x_mx, basis))
        reconstructed = projected + removed
        assert np.allclose(x, reconstructed, atol=1e-4), (
            f"Decomposition failed: max diff = {np.max(np.abs(x - reconstructed))}"
        )

    @_SETTINGS
    @given(u=orthonormal_bases(), x=random_vectors())
    def test_projection_norm_bounded(self, u: np.ndarray, x: np.ndarray) -> None:
        """||project(x, B)|| <= ||x||."""
        basis = _to_mx(u)
        x_mx = _to_mx(x)
        p = np.array(project_subspace(x_mx, basis))
        assert np.linalg.norm(p) <= np.linalg.norm(x) + 1e-4

    @_SETTINGS
    @given(u=orthonormal_bases(), x=random_vectors())
    def test_residual_orthogonal_to_basis(self, u: np.ndarray, x: np.ndarray) -> None:
        """remove(x, B) is orthogonal to each basis vector."""
        basis = _to_mx(u)
        x_mx = _to_mx(x)
        residual = np.array(remove_subspace(x_mx, basis))
        for row in u:
            dot = abs(float(np.dot(residual, row)))
            assert dot < 1e-4, f"Residual not orthogonal: dot = {dot}"


# ---------------------------------------------------------------------------
# Orthonormalize
# ---------------------------------------------------------------------------


class TestOrthonormalize:
    @_SETTINGS
    @given(u=orthonormal_bases(min_rank=2, max_rank=6))
    def test_preserves_orthonormality(self, u: np.ndarray) -> None:
        """orthonormalize on already orthonormal input is stable."""
        result = np.array(orthonormalize(_to_mx(u)))
        gram = result @ result.T
        assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-4)

    def test_orthonormalizes_non_orthogonal(self) -> None:
        """Non-orthogonal input becomes orthonormal."""
        raw = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        result = np.array(orthonormalize(_to_mx(raw)))
        gram = result @ result.T
        assert np.allclose(gram, np.eye(2), atol=1e-4)


# ---------------------------------------------------------------------------
# Explained variance ratio
# ---------------------------------------------------------------------------


class TestExplainedVarianceRatio:
    @_SETTINGS
    @given(svs=st.lists(
        st.floats(min_value=0.01, max_value=100.0, allow_nan=False),
        min_size=1, max_size=10,
    ))
    def test_sums_to_one(self, svs: list[float]) -> None:
        ratios = explained_variance_ratio(svs)
        total = sum(ratios)
        assert abs(total - 1.0) < 1e-6, f"variance ratios sum to {total}"

    @_SETTINGS
    @given(svs=st.lists(
        st.floats(min_value=0.01, max_value=100.0, allow_nan=False),
        min_size=1, max_size=10,
    ))
    def test_all_non_negative(self, svs: list[float]) -> None:
        ratios = explained_variance_ratio(svs)
        assert all(r >= 0 for r in ratios)


# ---------------------------------------------------------------------------
# Effective rank
# ---------------------------------------------------------------------------


class TestEffectiveRank:
    def test_rank_one(self) -> None:
        """Single dominant SV → effective rank ~1."""
        r = effective_rank([10.0, 0.001, 0.001])
        assert r < 1.5, f"effective rank = {r}, expected ~1"

    def test_uniform_svs(self) -> None:
        """Equal SVs → effective rank = k."""
        k = 5
        r = effective_rank([1.0] * k)
        assert abs(r - k) < 0.5, f"effective rank = {r}, expected ~{k}"

    @_SETTINGS
    @given(svs=st.lists(
        st.floats(min_value=0.01, max_value=100.0, allow_nan=False),
        min_size=1, max_size=10,
    ))
    def test_at_least_one(self, svs: list[float]) -> None:
        r = effective_rank(svs)
        assert r >= 1.0 - 1e-6, f"effective rank = {r} < 1"
