# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Objective assessment helpers for deployment assurance loops."""

from __future__ import annotations

from typing import Literal

from vauban.types import (
    FlywheelCycleMetrics,
    ObjectiveAssessment,
    ObjectiveConfig,
    ObjectiveMetricAssessment,
    ObjectiveMetricSpec,
)

FLYWHEEL_OBJECTIVE_METRICS: frozenset[str] = frozenset({
    "attack_success_rate",
    "defense_block_rate",
    "evasion_rate",
    "utility_score",
    "cast_block_fraction",
    "sic_block_fraction",
    "n_new_payloads",
    "n_previous_blocked",
})

OBJECTIVE_COMPARISONS: tuple[str, ...] = ("at_least", "at_most")
OBJECTIVE_AGGREGATES: tuple[str, ...] = ("final", "mean", "min", "max")
OBJECTIVE_ACCESS_MODES: tuple[str, ...] = ("weights", "api", "hybrid", "system")


def assess_flywheel_objective(
    objective: ObjectiveConfig,
    cycles: list[FlywheelCycleMetrics],
) -> ObjectiveAssessment:
    """Assess flywheel metrics against a deployment objective contract."""
    checks: list[ObjectiveMetricAssessment] = []
    safety_checks = _assess_metric_group("safety", objective.safety, cycles)
    utility_checks = _assess_metric_group("utility", objective.utility, cycles)
    checks.extend(safety_checks)
    checks.extend(utility_checks)

    safety_passed = all(check.passed for check in safety_checks)
    utility_passed = all(check.passed for check in utility_checks)
    passed = safety_passed and utility_passed

    if passed:
        summary = "Objective met"
    elif not safety_passed and not utility_passed:
        summary = "Safety and utility thresholds failed"
    elif not safety_passed:
        summary = "Safety thresholds failed"
    else:
        summary = "Utility thresholds failed"

    return ObjectiveAssessment(
        objective_name=objective.name,
        deployment=objective.deployment,
        access=objective.access,
        passed=passed,
        safety_passed=safety_passed,
        utility_passed=utility_passed,
        summary=summary,
        checks=checks,
    )


def _assess_metric_group(
    kind: Literal["safety", "utility"],
    specs: list[ObjectiveMetricSpec],
    cycles: list[FlywheelCycleMetrics],
) -> list[ObjectiveMetricAssessment]:
    """Assess one objective metric group over flywheel cycle metrics."""
    assessments: list[ObjectiveMetricAssessment] = []
    for spec in specs:
        actual = _aggregate_metric(cycles, spec.metric, spec.aggregate)
        if spec.comparison == "at_least":
            passed = actual >= spec.threshold
        else:
            passed = actual <= spec.threshold
        assessments.append(ObjectiveMetricAssessment(
            kind=kind,
            metric=spec.metric,
            threshold=spec.threshold,
            actual=actual,
            comparison=spec.comparison,
            aggregate=spec.aggregate,
            passed=passed,
            label=spec.label,
            description=spec.description,
        ))
    return assessments


def _aggregate_metric(
    cycles: list[FlywheelCycleMetrics],
    metric: str,
    aggregate: str,
) -> float:
    """Aggregate one flywheel metric across cycle history."""
    if not cycles:
        msg = "Cannot assess an objective without flywheel cycle metrics"
        raise ValueError(msg)

    values = [_cycle_metric_value(cycle, metric) for cycle in cycles]
    if aggregate == "final":
        return values[-1]
    if aggregate == "mean":
        return sum(values) / float(len(values))
    if aggregate == "min":
        return min(values)
    if aggregate == "max":
        return max(values)

    msg = f"Unsupported objective aggregate {aggregate!r}"
    raise ValueError(msg)


def _cycle_metric_value(cycle: FlywheelCycleMetrics, metric: str) -> float:
    """Read one supported objective metric from a flywheel cycle."""
    if metric not in FLYWHEEL_OBJECTIVE_METRICS:
        msg = f"Unsupported flywheel objective metric {metric!r}"
        raise ValueError(msg)
    return float(getattr(cycle, metric))
