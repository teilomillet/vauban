# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Mutation testing for vauban.algebra — validate test strength.

Uses ordeal's mutation testing to verify that the property tests in
test_algebra_props.py can detect injected bugs.  A surviving mutant
means the property tests have a gap.

Since ordeal 0.2.103, equivalence filtering is tunable via
``equivalence_samples`` — higher values give more confident
equivalent-mutant classification, allowing a tighter kill threshold.
"""

from __future__ import annotations

import pytest
from ordeal.mutations import mutate_function_and_test

# Target: the functions under test must be importable via dotted path
_TARGETS = [
    "vauban.algebra.similarity",
    "vauban.algebra.subtract",
    "vauban.algebra.negate",
]

# Operators to apply — use the lightweight ones to keep runtime manageable
_OPERATORS = ["arithmetic", "comparison", "negate", "return_none"]


def _run_property_tests() -> None:
    """Run a subset of algebra property tests as the mutation oracle."""
    from vauban import _ops as ops
    from vauban.algebra import (
        from_array,
        negate,
        similarity,
        subtract,
    )

    d = 16
    # Generate two random spaces
    ops.random.seed(42)
    arr_a = ops.random.normal((2, d))
    arr_b = ops.random.normal((3, d))
    ops.eval(arr_a)
    ops.eval(arr_b)
    a = from_array(arr_a, label="a")
    b = from_array(arr_b, label="b")

    # Property: similarity(a, a) ≈ 1.0
    sim = similarity(a, a)
    assert abs(sim - 1.0) < 0.01, f"self-similarity: {sim}"

    # Property: similarity bounded [0, 1]
    sim_ab = similarity(a, b)
    assert -0.01 <= sim_ab <= 1.01, f"similarity out of bounds: {sim_ab}"

    # Property: similarity symmetric
    sim_ba = similarity(b, a)
    assert abs(sim_ab - sim_ba) < 1e-4, "similarity not symmetric"

    # Property: subtract(a, a).rank == 0
    s = subtract(a, a)
    assert s.rank == 0, f"subtract(a,a).rank = {s.rank}"

    # Property: negate involution
    nn = negate(negate(a))
    sim_nn = similarity(a, nn)
    assert sim_nn > 0.99, f"negate involution: {sim_nn}"


class TestMutationKillRate:
    """Verify property tests detect injected bugs."""

    @pytest.mark.parametrize("target", _TARGETS)
    def test_mutation_score(self, target: str) -> None:
        result = mutate_function_and_test(
            target, _run_property_tests, operators=_OPERATORS,
            # Raise equivalence_samples from default 10 to 50 so that
            # equivalent mutants (e.g. x+0, reordered commutative ops)
            # are classified correctly instead of inflating the denominator.
            equivalence_samples=50,
        )
        if result.total == 0:
            pytest.skip(f"No mutants generated for {target}")

        surviving = [m for m in result.survived]
        # Report surviving mutants for debugging
        for m in surviving:
            print(f"  SURVIVED: {m.location} {m.description}")

        # With better equivalence filtering (50 samples), the kill rate
        # threshold can be tighter — surviving mutants are genuine gaps.
        assert result.score >= 0.6, (
            f"{target}: mutation score {result.score:.0%} "
            f"({result.killed}/{result.total}), "
            f"survived: {[m.description for m in surviving]}"
        )
