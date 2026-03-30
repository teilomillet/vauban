"""ChaosTest state machine for vauban.algebra.

Explores the algebra API as a stateful system: a pool of
DirectionSpaces is built up via random operations, with
invariants checked after every step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from hypothesis import settings
from hypothesis.stateful import Bundle, invariant, rule
from ordeal import ChaosTest

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

    faults: ClassVar[list[object]] = []  # type: ignore[assignment]

    spaces = Bundle("spaces")

    def __init__(self) -> None:
        super().__init__()
        self._all_spaces: list[DirectionSpace] = []

    @rule(target=spaces)
    def create_random_space(self) -> DirectionSpace:
        """Generate a fresh random DirectionSpace."""
        rank = np.random.randint(1, 5)
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

    @invariant()
    def all_spaces_valid(self) -> None:
        """Every DirectionSpace must satisfy structural invariants."""
        for space in self._all_spaces:
            assert space.d_model == D_MODEL, f"d_model={space.d_model}"
            assert space.rank >= 0, f"rank={space.rank}"
            assert space.rank <= space.d_model, f"rank={space.rank} > d_model"
            if space.rank > 0:
                basis = np.array(to_basis(space))
                assert basis.shape == (space.rank, space.d_model)
                # Check orthonormality
                gram = basis @ basis.T
                identity = np.eye(gram.shape[0])
                max_err = float(np.max(np.abs(gram - identity)))
                assert max_err < 1e-3, (
                    f"Basis not orthonormal: max |G-I| = {max_err:.6f}"
                )

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
