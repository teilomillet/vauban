# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Invariant checker framework and fault injection utilities.

Declare properties that must *always* hold,
then systematically search for violations.

Invariants are simple functions that raise ``AssertionError`` on
violation.  Fault injectors are context managers that corrupt
state to test defensive code paths.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


# ── Invariant registry ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Invariant:
    """A named property that must hold."""

    name: str
    description: str
    check: Callable[..., None]


_REGISTRY: list[Invariant] = []


def invariant(name: str, description: str) -> Callable[
    [Callable[..., None]], Callable[..., None]
]:
    """Decorator: register a function as a named invariant."""

    def _wrap(fn: Callable[..., None]) -> Callable[..., None]:
        _REGISTRY.append(Invariant(name=name, description=description, check=fn))
        return fn

    return _wrap


def all_invariants() -> list[Invariant]:
    """Return all registered invariants."""
    return list(_REGISTRY)


# ── Numeric guards ────────────────────────────────────────────────────


def assert_finite(value: float, label: str = "value") -> None:
    """Assert a float is finite (not NaN, not Inf)."""
    if math.isnan(value):
        msg = f"{label} is NaN"
        raise AssertionError(msg)
    if math.isinf(value):
        msg = f"{label} is Inf"
        raise AssertionError(msg)


def assert_bounded(
    value: float,
    lo: float,
    hi: float,
    label: str = "value",
) -> None:
    """Assert a float is in [lo, hi]."""
    assert_finite(value, label)
    if not (lo <= value <= hi):
        msg = f"{label} = {value} not in [{lo}, {hi}]"
        raise AssertionError(msg)


def assert_all_finite(values: list[float], label: str = "values") -> None:
    """Assert every element is finite."""
    for i, v in enumerate(values):
        assert_finite(v, f"{label}[{i}]")


def assert_all_bounded(
    values: list[float],
    lo: float,
    hi: float,
    label: str = "values",
) -> None:
    """Assert every element is in [lo, hi]."""
    for i, v in enumerate(values):
        assert_bounded(v, lo, hi, f"{label}[{i}]")


# ── Determinism guard ─────────────────────────────────────────────────


def assert_deterministic[T](
    fn: Callable[[], T],
    *,
    runs: int = 3,
    label: str = "function",
) -> T:
    """Call *fn* multiple times and assert all results are identical.

    Returns the first result on success.
    """
    results = [fn() for _ in range(runs)]
    first = results[0]
    for i, r in enumerate(results[1:], 1):
        if r != first:
            msg = f"{label}: run 0 != run {i}: {first!r} != {r!r}"
            raise AssertionError(msg)
    return first


# ── Fault injection ──────────────────────────────────────────────────


@contextmanager
def corrupted_floats(
    values: list[float],
    corruption: str = "nan",
) -> Iterator[list[float]]:
    """Yield a copy of *values* with injected corruption.

    Args:
        values: Original float list.
        corruption: One of "nan", "inf", "neg_inf", "zero", "huge".

    Yields:
        The corrupted copy (original is not modified).
    """
    corrupted = list(values)
    inject: float
    if corruption == "nan":
        inject = float("nan")
    elif corruption == "inf":
        inject = float("inf")
    elif corruption == "neg_inf":
        inject = float("-inf")
    elif corruption == "zero":
        inject = 0.0
    elif corruption == "huge":
        inject = 1e308
    else:
        msg = f"Unknown corruption: {corruption!r}"
        raise ValueError(msg)

    if corrupted:
        corrupted[0] = inject
        if len(corrupted) > 1:
            corrupted[len(corrupted) // 2] = inject
    yield corrupted


@contextmanager
def empty_input() -> Iterator[list[str]]:
    """Yield an empty list — simulates missing prompt data."""
    yield []


@contextmanager
def extreme_string(length: int = 100_000) -> Iterator[str]:
    """Yield an extremely long string for stress testing."""
    yield "x" * length


@contextmanager
def unicode_edge_cases() -> Iterator[list[str]]:
    """Yield strings that stress Unicode handling."""
    yield [
        "",                              # empty
        "\x00",                          # null byte
        "\n\n\n",                        # only newlines
        "\u200b\u200c\u200d",            # zero-width chars
        "\U0001f4a3" * 50,               # emoji (bomb) x50
        "a" * 50_000,                    # very long single word
        " " * 100,                       # only spaces
        "\t\r\n \t\r\n",                 # mixed whitespace
        "\u0430\u0441\u0435",            # Cyrillic lookalikes
        "SELECT * FROM users; DROP TABLE users;--",  # SQL injection
        "{{payload}} {payload} %s %d",   # format string chars
        "\x1b[31mred\x1b[0m",           # ANSI escape
    ]
