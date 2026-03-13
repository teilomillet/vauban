"""Convergence detection for the flywheel loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vauban.types import FlywheelCycleMetrics


def check_convergence(
    metrics: list[FlywheelCycleMetrics],
    window: int,
    threshold: float,
) -> bool:
    """Check whether the flywheel has converged.

    Converged when the evasion rate change over the last *window* cycles
    is below *threshold*.

    Args:
        metrics: Metrics from all completed cycles.
        window: Number of recent cycles to examine.
        threshold: Maximum change to consider converged.

    Returns:
        True if the flywheel has converged.
    """
    if len(metrics) < window:
        return False

    recent = metrics[-window:]
    rates = [m.evasion_rate for m in recent]

    # Max absolute change across the window
    max_change = max(rates) - min(rates)
    return max_change <= threshold
