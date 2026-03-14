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
    n_successful: int = 0,
    prev_evasion_rates: list[float] | None = None,
) -> FlywheelDefenseParams:
    """Adapt defense parameters based on evasion analysis.

    Uses complementarity-aware escalation: analyzes which defense
    (CAST vs SIC) is weaker on the evaded traces and prioritizes
    hardening that defense.  Implements TRYLOCK danger-zone detection:
    if evasion increased in 2+ of the last 3 cycles, backs off alpha
    to avoid oscillation in the non-monotonic region.

    Args:
        current: Current defense parameters.
        evaded: Traces that evaded defense this cycle.
        adaptation_rate: Maximum fractional change per cycle.
        utility_score: Current utility score (0.0-1.0).
        utility_floor: Minimum acceptable utility.
        n_successful: Total successful attacks this cycle (for rate).
        prev_evasion_rates: Recent evasion rates (oldest first).
            Used for danger-zone detection with a multi-cycle window.

    Returns:
        Updated FlywheelDefenseParams.
    """
    if not evaded:
        return current

    # -- Evasion pressure (rate-based) --
    n_evaded = len(evaded)
    if n_successful > 0:
        evasion_pressure = min(n_evaded / n_successful, 1.0)
    else:
        evasion_pressure = min(n_evaded / 10.0, 1.0)

    # -- TRYLOCK danger-zone detection --
    # Check whether evasion increased in 2+ of the last 3 cycles.
    # A single spike could be noise; repeated increases signal the
    # non-monotonic danger zone where more alpha makes things worse.
    in_danger_zone = _detect_danger_zone(
        prev_evasion_rates or [],
        n_evaded / n_successful if n_successful > 0 else 0.0,
    )

    # -- Complementarity analysis --
    # Determine which defense is weaker on evaded traces.
    # CAST-active means CAST tried but failed to prevent evasion.
    # SIC-inactive means SIC never ran or didn't block.
    cast_active_count = sum(
        1 for t in evaded if t.cast_interventions > 0
    )
    sic_inactive_count = sum(
        1 for t in evaded if not t.sic_blocked
    )

    # Fractions: how much of the evasion escapes each defense.
    cast_active_frac = cast_active_count / n_evaded
    sic_inactive_frac = sic_inactive_count / n_evaded

    # If CAST is already intervening on most evaders but they still
    # escape, CAST needs LESS escalation (it's already doing its
    # job — the attack bypasses it structurally).  Focus on SIC.
    # Conversely, if SIC is inactive on most evaders, SIC needs more.
    cast_weight = 1.0 - cast_active_frac  # high → CAST needs help
    sic_weight = sic_inactive_frac        # high → SIC needs help

    # -- Base deltas --
    alpha_delta = (
        current.cast_alpha
        * adaptation_rate
        * evasion_pressure
        * max(cast_weight, 0.1)  # always some CAST hardening
    )
    threshold_delta = max(
        current.sic_threshold
        * adaptation_rate
        * evasion_pressure
        * max(sic_weight, 0.1),  # always some SIC hardening
        0.01,
    )

    # -- Utility floor dampening --
    if utility_score < utility_floor:
        utility_ratio = (
            utility_score / utility_floor if utility_floor > 0 else 0.0
        )
        scale = max(utility_ratio, 0.1)
        alpha_delta *= scale
        threshold_delta *= scale

    # -- Apply deltas --
    if in_danger_zone:
        # TRYLOCK: reduce alpha (back off from danger zone)
        new_alpha = max(
            current.cast_alpha - alpha_delta * 0.5,
            1.0,  # never drop below baseline alpha=1.0
        )
    else:
        new_alpha = current.cast_alpha + alpha_delta

    new_sic_threshold = max(current.sic_threshold - threshold_delta, 0.0)

    # Increase SIC iterations if many evasions escape SIC
    new_sic_iters = current.sic_iterations
    if (
        n_evaded >= 5
        and current.sic_iterations < 10
        and sic_weight > 0.5
    ):
        new_sic_iters = current.sic_iterations + 1

    return FlywheelDefenseParams(
        cast_alpha=new_alpha,
        cast_threshold=current.cast_threshold,
        sic_threshold=new_sic_threshold,
        sic_iterations=new_sic_iters,
        sic_mode=current.sic_mode,
        cast_layers=current.cast_layers,
    )


def _detect_danger_zone(
    prev_rates: list[float],
    current_rate: float,
) -> bool:
    """Return whether the flywheel is in the TRYLOCK danger zone.

    True when evasion increased in at least 2 of the last 3 transitions.
    A single spike is treated as noise; repeated increases signal the
    non-monotonic region where more alpha makes things worse.
    """
    # Build the full rate sequence (recent history + current)
    rates = [*prev_rates[-3:], current_rate]
    if len(rates) < 2:
        return False

    # Count transitions where evasion increased
    from itertools import pairwise

    increases = sum(1 for a, b in pairwise(rates) if b > a)
    # Danger zone: 2+ increases in the window
    return increases >= 2
