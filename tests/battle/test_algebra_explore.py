# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""ChaosTest state machine for vauban.algebra.

Explores the algebra API as a stateful system: a pool of
DirectionSpaces is built up via random operations, with
invariants checked after every step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import hypothesis.strategies as st
from hypothesis import settings
from hypothesis.stateful import Bundle, invariant, rule
from ordeal import ChaosTest
from ordeal.invariants import (
    finite,
    orthonormal,
    positive_semi_definite,
    rank_bounded,
    symmetric,
)

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
)

if TYPE_CHECKING:
    from vauban.types import DirectionSpace

D_MODEL = 16  # small for fast exploration


class AlgebraChaosTest(ChaosTest):
    """Stateful exploration of the direction space algebra."""

    faults: ClassVar[list[object]] = []

    spaces = Bundle("spaces")

    def __init__(self) -> None:
        super().__init__()
        self._all_spaces: list[DirectionSpace] = []

    @rule(target=spaces, rank=st.integers(min_value=1, max_value=4))
    def create_random_space(self, rank: int) -> DirectionSpace:
        """Generate a fresh random DirectionSpace."""
        arr = ops.random.normal((rank, D_MODEL))
        ops.eval(arr)
        space = from_array(arr, label=f"r{rank}_{len(self._all_spaces)}")
        self._all_spaces.append(space)
        return space

    @rule(target=spaces, a=spaces, b=spaces)
    def do_add(self, a: DirectionSpace, b: DirectionSpace) -> DirectionSpace:
        """add(a, b)."""
        result = add(a, b)
        self._all_spaces.append(result)
        return result

    @rule(target=spaces, a=spaces, b=spaces)
    def do_subtract(self, a: DirectionSpace, b: DirectionSpace) -> DirectionSpace:
        """subtract(a, b)."""
        result = subtract(a, b)
        self._all_spaces.append(result)
        return result

    @rule(target=spaces, a=spaces, b=spaces)
    def do_intersect(self, a: DirectionSpace, b: DirectionSpace) -> DirectionSpace:
        """intersect(a, b)."""
        result = intersect(a, b)
        self._all_spaces.append(result)
        return result

    @rule(target=spaces, space=spaces)
    def do_negate(self, space: DirectionSpace) -> DirectionSpace:
        """negate(space)."""
        result = negate(space)
        self._all_spaces.append(result)
        return result

    @rule(target=spaces, a=spaces, b=spaces)
    def do_compose(self, a: DirectionSpace, b: DirectionSpace) -> DirectionSpace:
        """compose([a, b], [1.0, 1.0])."""
        result = compose([a, b], [1.0, 1.0])
        self._all_spaces.append(result)
        return result

    # -- Invariants (checked after every step) --

    _check_orthonormal = orthonormal(tol=1e-4)
    _check_rank = rank_bounded(0, D_MODEL)
    _check_symmetric = symmetric(tol=1e-4)
    _check_psd = positive_semi_definite(tol=1e-4)

    @invariant()
    def all_spaces_valid(self) -> None:
        """Every DirectionSpace must satisfy structural invariants."""
        for space in self._all_spaces:
            assert space.d_model == D_MODEL, f"d_model={space.d_model}"
            self._check_rank(space.rank, name=space.label or "space")
            if space.rank > 0:
                basis = to_basis(space)
                assert basis.shape == (space.rank, space.d_model)
                self._check_orthonormal(basis, name=space.label or "basis")
                finite(basis, name=space.label or "basis")
                # Gram matrix B @ B.T must be symmetric PSD (identity-like
                # for orthonormal bases). Catches numerical drift after
                # chained compose/subtract operations.
                gram = basis @ basis.T
                label = space.label or "gram"
                self._check_symmetric(gram, name=label)
                self._check_psd(gram, name=label)

    @invariant()
    def similarity_self_is_one(self) -> None:
        """similarity(a, a) ≈ 1.0 for all non-zero spaces."""
        for space in self._all_spaces:
            if space.rank > 0:
                sim = similarity(space, space)
                assert abs(sim - 1.0) < 0.01, (
                    f"similarity(a, a) = {sim} for {space.label}"
                )


TestAlgebraChaos = AlgebraChaosTest.TestCase
TestAlgebraChaos.settings = settings(
    max_examples=30,
    stateful_step_count=15,
    deadline=None,
)
