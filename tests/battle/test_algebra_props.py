"""Property-based tests for vauban.algebra — direction space algebra.

Tests universal algebraic properties: closure, commutativity,
orthonormality preservation, rank bounds, similarity metrics.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vauban import _ops as ops
from vauban.algebra import (
    add,
    compose,
    from_array,
    intersect,
    negate,
    similarity,
    subtract,
    to_basis,
    to_direction,
)
from vauban.types import DirectionSpace

# ---------------------------------------------------------------------------
# Strategies: generate random DirectionSpaces
# ---------------------------------------------------------------------------

D_MODEL = 32

SpacePair = tuple[DirectionSpace, DirectionSpace]


def _make_space(rank: int, d_model: int = D_MODEL, label: str = "") -> DirectionSpace:
    """Build a random DirectionSpace with given rank."""
    if rank == 0:
        basis = ops.zeros((0, d_model))
        ops.eval(basis)
        return DirectionSpace(
            basis=basis, d_model=d_model, rank=0, label=label or "zero",
        )
    arr = ops.random.normal((rank, d_model))
    ops.eval(arr)
    return from_array(arr, label=label or f"r{rank}")


@st.composite
def direction_spaces(
    draw: st.DrawFn,
    min_rank: int = 1,
    max_rank: int = 6,
    d_model: int = D_MODEL,
) -> DirectionSpace:
    """Hypothesis strategy for random DirectionSpace objects."""
    rank = draw(st.integers(min_value=min_rank, max_value=max_rank))
    label = draw(st.text(min_size=1, max_size=5, alphabet="abcdefgh"))
    return _make_space(rank, d_model, label)


@st.composite
def direction_space_pairs(
    draw: st.DrawFn,
    min_rank: int = 1,
    max_rank: int = 6,
) -> tuple[DirectionSpace, DirectionSpace]:
    """Two independent DirectionSpaces."""
    a = draw(direction_spaces(min_rank=min_rank, max_rank=max_rank))
    b = draw(direction_spaces(min_rank=min_rank, max_rank=max_rank))
    return a, b


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _basis_np(space: DirectionSpace) -> np.ndarray:
    """Extract basis as numpy array."""
    return np.array(to_basis(space))


def _is_orthonormal(basis: np.ndarray, tol: float = 1e-4) -> bool:
    """Check if rows are orthonormal."""
    if basis.shape[0] == 0:
        return True
    gram = basis @ basis.T
    return bool(np.allclose(gram, np.eye(gram.shape[0]), atol=tol))


def _assert_valid_space(space: DirectionSpace) -> None:
    """Assert invariants of a valid DirectionSpace."""
    assert space.d_model == D_MODEL, f"d_model mismatch: {space.d_model}"
    assert space.rank >= 0, f"negative rank: {space.rank}"
    assert space.rank <= space.d_model, f"rank {space.rank} > d_model {space.d_model}"
    if space.rank > 0:
        basis = _basis_np(space)
        assert basis.shape == (space.rank, space.d_model), (
            f"shape mismatch: {basis.shape}"
        )
        assert _is_orthonormal(basis), "basis not orthonormal"


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

_SETTINGS = settings(max_examples=50, deadline=None)


class TestClosure:
    """Every algebra operation returns a valid DirectionSpace."""

    @_SETTINGS
    @given(data=direction_space_pairs())
    def test_add_closure(self, data: SpacePair) -> None:
        a, b = data
        result = add(a, b)
        _assert_valid_space(result)

    @_SETTINGS
    @given(data=direction_space_pairs())
    def test_subtract_closure(self, data: SpacePair) -> None:
        a, b = data
        result = subtract(a, b)
        _assert_valid_space(result)

    @_SETTINGS
    @given(data=direction_space_pairs())
    def test_intersect_closure(self, data: SpacePair) -> None:
        a, b = data
        result = intersect(a, b)
        _assert_valid_space(result)

    @_SETTINGS
    @given(space=direction_spaces())
    def test_negate_closure(self, space: DirectionSpace) -> None:
        result = negate(space)
        _assert_valid_space(result)

    @_SETTINGS
    @given(space=direction_spaces())
    def test_compose_single_closure(self, space: DirectionSpace) -> None:
        result = compose([space], [1.0])
        _assert_valid_space(result)


class TestSymmetricOps:
    """Operations that ARE commutative/symmetric."""

    @_SETTINGS
    @given(data=direction_space_pairs())
    def test_similarity_commutative(self, data: SpacePair) -> None:
        a, b = data
        assert abs(similarity(a, b) - similarity(b, a)) < 1e-6

    @_SETTINGS
    @given(space=direction_spaces())
    def test_add_self_preserves_space(self, space: DirectionSpace) -> None:
        """add(a, a) should recover a (same subspace)."""
        result = add(space, space)
        sim = similarity(space, result)
        assert sim > 0.99, f"add(a, a) similarity={sim:.4f}"


class TestSelfOperations:
    """Operations on a space with itself."""

    @_SETTINGS
    @given(space=direction_spaces())
    def test_subtract_self_rank_zero(self, space: DirectionSpace) -> None:
        result = subtract(space, space)
        assert result.rank == 0, f"subtract(a, a) has rank {result.rank}, expected 0"

    @_SETTINGS
    @given(space=direction_spaces())
    def test_similarity_self_is_one(self, space: DirectionSpace) -> None:
        sim = similarity(space, space)
        assert abs(sim - 1.0) < 1e-4, f"similarity(a, a) = {sim}, expected ~1.0"


class TestNegateInvolution:
    """negate(negate(x)) should recover the original space."""

    @_SETTINGS
    @given(space=direction_spaces())
    def test_double_negate_recovers(self, space: DirectionSpace) -> None:
        double = negate(negate(space))
        sim = similarity(space, double)
        assert sim > 0.99, f"negate involution broken: similarity={sim:.4f}"


class TestIntersectBounds:
    """Intersection rank is bounded by the smaller operand."""

    @_SETTINGS
    @given(data=direction_space_pairs())
    def test_intersect_rank_bounded(self, data: SpacePair) -> None:
        a, b = data
        result = intersect(a, b)
        assert result.rank <= min(a.rank, b.rank), (
            f"intersect rank {result.rank} > min({a.rank}, {b.rank})"
        )


class TestAddRankBounds:
    """Add rank is bounded by the sum of operand ranks."""

    @_SETTINGS
    @given(data=direction_space_pairs())
    def test_add_rank_bounded(self, data: SpacePair) -> None:
        a, b = data
        result = add(a, b)
        assert result.rank <= min(a.rank + b.rank, D_MODEL), (
            f"add rank {result.rank} > {a.rank} + {b.rank}"
        )


class TestSimilarityBounds:
    """Similarity is always in [0, 1]."""

    @_SETTINGS
    @given(data=direction_space_pairs())
    def test_similarity_bounded(self, data: SpacePair) -> None:
        sim = similarity(*data)
        assert -1e-6 <= sim <= 1.0 + 1e-6, f"similarity out of bounds: {sim}"


class TestZeroSpace:
    """Operations with rank-0 spaces."""

    @_SETTINGS
    @given(space=direction_spaces())
    def test_add_zero_is_identity(self, space: DirectionSpace) -> None:
        zero = _make_space(0)
        result = add(space, zero)
        sim = similarity(space, result)
        assert sim > 0.99, f"add(a, 0) not identity: similarity={sim:.4f}"

    @_SETTINGS
    @given(space=direction_spaces())
    def test_intersect_with_zero(self, space: DirectionSpace) -> None:
        zero = _make_space(0)
        result = intersect(space, zero)
        assert result.rank == 0


class TestExtractors:
    """to_direction and to_basis extractors."""

    @_SETTINGS
    @given(space=direction_spaces())
    def test_to_direction_is_unit(self, space: DirectionSpace) -> None:
        d = np.array(to_direction(space))
        norm = float(np.linalg.norm(d))
        assert abs(norm - 1.0) < 1e-4, f"to_direction norm={norm}"

    def test_to_direction_rank_zero_raises(self) -> None:
        zero = _make_space(0)
        with pytest.raises((ValueError, IndexError)):
            to_direction(zero)
