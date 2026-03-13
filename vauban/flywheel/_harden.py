"""Defense parameter hardening for the flywheel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vauban.types import FlywheelDefenseParams

if TYPE_CHECKING:
    from vauban.types import DefendedTrace


def harden_defense(
    current: FlywheelDefenseParams,
    evaded: list[DefendedTrace],
    adaptation_rate: float,
    utility_score: float,
    utility_floor: float,
) -> FlywheelDefenseParams:
    """Adapt defense parameters based on evasion analysis.

    Increases CAST alpha and decreases SIC threshold proportionally to
    the evasion rate, clamped by the adaptation_rate. If utility_score
    drops below utility_floor, scales down changes to prevent
    over-hardening.

    Args:
        current: Current defense parameters.
        evaded: Traces that evaded defense this cycle.
        adaptation_rate: Maximum fractional change per cycle.
        utility_score: Current utility score (0.0-1.0).
        utility_floor: Minimum acceptable utility.

    Returns:
        Updated FlywheelDefenseParams.
    """
    if not evaded:
        return current

    # Scale factor based on how many traces evaded
    n_evaded = len(evaded)
    evasion_pressure = min(n_evaded / 10.0, 1.0)  # cap at 1.0

    # Base deltas proportional to evasion pressure and adaptation rate
    alpha_delta = (
        current.cast_alpha * adaptation_rate * evasion_pressure
    )
    threshold_delta = max(
        current.sic_threshold * adaptation_rate * evasion_pressure,
        0.01,
    )

    # Scale down if utility is suffering
    if utility_score < utility_floor:
        utility_ratio = (
            utility_score / utility_floor if utility_floor > 0 else 0.0
        )
        scale = max(utility_ratio, 0.1)  # never fully zero out
        alpha_delta *= scale
        threshold_delta *= scale

    # Apply deltas: increase CAST alpha, decrease SIC threshold
    new_alpha = current.cast_alpha + alpha_delta
    new_sic_threshold = max(current.sic_threshold - threshold_delta, 0.0)

    # Optionally increase SIC iterations if many evasions
    new_sic_iters = current.sic_iterations
    if n_evaded >= 5 and current.sic_iterations < 10:
        new_sic_iters = current.sic_iterations + 1

    return FlywheelDefenseParams(
        cast_alpha=new_alpha,
        cast_threshold=current.cast_threshold,
        sic_threshold=new_sic_threshold,
        sic_iterations=new_sic_iters,
        sic_mode=current.sic_mode,
        cast_layers=current.cast_layers,
    )
