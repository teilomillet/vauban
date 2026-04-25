# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Behavior-diff regression threshold evaluation."""

from __future__ import annotations

from vauban.behavior._primitives import (
    BehaviorMetricDelta,
    BehaviorThresholdResult,
    BehaviorThresholdSpec,
)


def evaluate_behavior_thresholds(
    deltas: tuple[BehaviorMetricDelta, ...],
    thresholds: tuple[BehaviorThresholdSpec, ...],
) -> tuple[BehaviorThresholdResult, ...]:
    """Evaluate regression thresholds against behavior metric deltas."""
    return tuple(_evaluate_threshold(deltas, threshold) for threshold in thresholds)


def behavior_threshold_summary(
    results: tuple[BehaviorThresholdResult, ...],
) -> dict[str, int | bool]:
    """Summarize threshold results for machine-readable reports."""
    failed = sum(
        1 for result in results
        if not result.passed and result.severity == "fail"
    )
    warned = sum(
        1 for result in results
        if not result.passed and result.severity == "warn"
    )
    return {
        "passed": failed == 0,
        "n_thresholds": len(results),
        "n_failed": failed,
        "n_warned": warned,
    }


def _evaluate_threshold(
    deltas: tuple[BehaviorMetricDelta, ...],
    threshold: BehaviorThresholdSpec,
) -> BehaviorThresholdResult:
    """Evaluate one threshold."""
    delta = _find_delta(deltas, threshold)
    if delta is None:
        return BehaviorThresholdResult(
            metric=threshold.metric,
            category=threshold.category,
            severity=threshold.severity,
            passed=False,
            max_delta=threshold.max_delta,
            min_delta=threshold.min_delta,
            max_absolute_delta=threshold.max_absolute_delta,
            description=threshold.description,
            reason="matching metric delta was not available",
        )

    violations: list[str] = []
    if threshold.max_delta is not None and delta.delta > threshold.max_delta:
        violations.append(
            f"delta {delta.delta:.6g} > max_delta {threshold.max_delta:.6g}",
        )
    if threshold.min_delta is not None and delta.delta < threshold.min_delta:
        violations.append(
            f"delta {delta.delta:.6g} < min_delta {threshold.min_delta:.6g}",
        )
    if (
        threshold.max_absolute_delta is not None
        and abs(delta.delta) > threshold.max_absolute_delta
    ):
        violations.append(
            (
                f"abs(delta) {abs(delta.delta):.6g} >"
                f" max_absolute_delta {threshold.max_absolute_delta:.6g}"
            ),
        )

    return BehaviorThresholdResult(
        metric=threshold.metric,
        category=threshold.category,
        severity=threshold.severity,
        passed=not violations,
        delta=delta.delta,
        max_delta=threshold.max_delta,
        min_delta=threshold.min_delta,
        max_absolute_delta=threshold.max_absolute_delta,
        description=threshold.description,
        reason="; ".join(violations) if violations else "passed",
    )


def _find_delta(
    deltas: tuple[BehaviorMetricDelta, ...],
    threshold: BehaviorThresholdSpec,
) -> BehaviorMetricDelta | None:
    """Find the matching metric delta for a threshold."""
    threshold_category = threshold.category or "overall"
    for delta in deltas:
        delta_category = delta.category or "overall"
        if delta.name == threshold.metric and delta_category == threshold_category:
            return delta
    return None
