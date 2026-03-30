"""Metamorphic relation tests for vauban algebra.

Tests algebraic *relationships* rather than exact values:
- subtract(a, a) → rank 0
- negate(negate(x)) → identity
- add(a, zero) → a
- similarity(a, a) → 1.0
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from vauban import _ops as ops
from vauban.algebra import (
    add,
    from_array,
    negate,
    similarity,
    subtract,
)
from vauban.types import DirectionSpace

D_MODEL = 32


def _make_space(rank: int) -> DirectionSpace:
    """Build a random DirectionSpace."""
    if rank == 0:
        basis = ops.zeros((0, D_MODEL))
        ops.eval(basis)
        return DirectionSpace(basis=basis, d_model=D_MODEL, rank=0, label="zero")
    arr = ops.random.normal((rank, D_MODEL))
    ops.eval(arr)
    return from_array(arr, label=f"r{rank}")


@st.composite
def direction_spaces(draw: st.DrawFn) -> DirectionSpace:
    """Random DirectionSpace with rank 1-6."""
    rank = draw(st.integers(min_value=1, max_value=6))
    return _make_space(rank)


_SETTINGS = settings(max_examples=50, deadline=None)


# ---------------------------------------------------------------------------
# Metamorphic relations using ordeal framework
# ---------------------------------------------------------------------------


class TestSubtractSelfRelation:
    """subtract(a, a) always yields rank-0."""

    @_SETTINGS
    @given(space=direction_spaces())
    def test_self_subtraction_rank_zero(self, space: DirectionSpace) -> None:
        result = subtract(space, space)
        assert result.rank == 0, f"subtract(a, a).rank = {result.rank}"


class TestNegateInvolution:
    """negate(negate(x)) recovers x (measured by similarity)."""

    @_SETTINGS
    @given(space=direction_spaces())
    def test_double_negate(self, space: DirectionSpace) -> None:
        recovered = negate(negate(space))
        sim = similarity(space, recovered)
        assert sim > 0.99, f"negate involution: similarity={sim:.4f}"


class TestAdditiveIdentity:
    """add(a, zero) should recover a."""

    @_SETTINGS
    @given(space=direction_spaces())
    def test_add_zero_identity(self, space: DirectionSpace) -> None:
        zero = _make_space(0)
        result = add(space, zero)
        sim = similarity(space, result)
        assert sim > 0.99, f"add(a, 0) identity: similarity={sim:.4f}"


class TestSimilaritySelf:
    """similarity(a, a) = 1.0 exactly."""

    @_SETTINGS
    @given(space=direction_spaces())
    def test_self_similarity(self, space: DirectionSpace) -> None:
        sim = similarity(space, space)
        assert abs(sim - 1.0) < 1e-4, f"similarity(a, a) = {sim}"


class TestSimilaritySymmetry:
    """similarity(a, b) = similarity(b, a)."""

    @_SETTINGS
    @given(a=direction_spaces(), b=direction_spaces())
    def test_symmetric(self, a: DirectionSpace, b: DirectionSpace) -> None:
        sim_ab = similarity(a, b)
        sim_ba = similarity(b, a)
        assert abs(sim_ab - sim_ba) < 1e-6, (
            f"similarity not symmetric: {sim_ab:.6f} vs {sim_ba:.6f}"
        )


class TestComposeMetamorphic:
    """Composing a single space with weight 1.0 should approximate identity."""

    @_SETTINGS
    @given(space=direction_spaces())
    def test_compose_single_identity(self, space: DirectionSpace) -> None:
        from vauban.algebra import compose

        result = compose([space], [1.0])
        sim = similarity(space, result)
        assert sim > 0.99, f"compose([a], [1.0]) identity: similarity={sim:.4f}"


class TestSubtractOrthogonality:
    """subtract(a, b) should be approximately orthogonal to b."""

    @_SETTINGS
    @given(a=direction_spaces(), b=direction_spaces())
    def test_subtract_orthogonal(self, a: DirectionSpace, b: DirectionSpace) -> None:
        result = subtract(a, b)
        if result.rank == 0:
            return  # trivially orthogonal
        sim = similarity(result, b)
        assert sim < 0.15, (
            f"subtract(a, b) not orthogonal to b: similarity={sim:.4f}"
        )
