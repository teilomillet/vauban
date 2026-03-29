"""Battle test configuration — auto-mark and Hypothesis profiles."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, settings

# ── Auto-mark every test in this directory as "battle" ────────────────


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Add the ``battle`` marker to every test collected from this package."""
    battle_dir = str(__file__).rsplit("/", 1)[0]
    for item in items:
        if str(item.fspath).startswith(battle_dir):
            item.add_marker(pytest.mark.battle)


# ── Hypothesis profiles ──────────────────────────────────────────────

# Default: fast CI-friendly exploration.
settings.register_profile(
    "battle_ci",
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Deep: thorough state-space sweep (run manually / nightly).
settings.register_profile(
    "battle_deep",
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Use battle_ci by default; override with --hypothesis-profile=battle_deep
settings.load_profile("battle_ci")
