# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for flywheel convergence detection."""

from vauban.flywheel._convergence import check_convergence
from vauban.types import FlywheelCycleMetrics


def _make_metrics(evasion_rates: list[float]) -> list[FlywheelCycleMetrics]:
    """Create cycle metrics with given evasion rates."""
    return [
        FlywheelCycleMetrics(
            cycle=i,
            n_worlds=10,
            n_attacks=50,
            attack_success_rate=0.5,
            defense_block_rate=0.8,
            evasion_rate=rate,
            utility_score=0.95,
            cast_alpha=2.0,
            sic_threshold=0.5,
            n_new_payloads=0,
            n_previous_blocked=0,
        )
        for i, rate in enumerate(evasion_rates)
    ]


class TestConvergence:
    def test_not_enough_data(self) -> None:
        metrics = _make_metrics([0.5, 0.3])
        assert check_convergence(metrics, window=3, threshold=0.01) is False

    def test_converged(self) -> None:
        metrics = _make_metrics([0.5, 0.3, 0.1, 0.1, 0.1])
        assert check_convergence(metrics, window=3, threshold=0.01) is True

    def test_not_converged(self) -> None:
        metrics = _make_metrics([0.5, 0.3, 0.1, 0.05, 0.15])
        assert check_convergence(
            metrics, window=3, threshold=0.01,
        ) is False

    def test_edge_at_threshold(self) -> None:
        metrics = _make_metrics([0.1, 0.11, 0.1])
        assert check_convergence(metrics, window=3, threshold=0.01) is True

    def test_single_value_window(self) -> None:
        metrics = _make_metrics([0.5, 0.5])
        assert check_convergence(metrics, window=2, threshold=0.0) is True

    def test_decreasing_trend_not_converged(self) -> None:
        metrics = _make_metrics([0.5, 0.4, 0.3])
        assert check_convergence(
            metrics, window=3, threshold=0.05,
        ) is False

    def test_zero_attack_success_does_not_converge(self) -> None:
        """Evasion rate 0 from zero attack success is vacuous."""
        metrics = [
            FlywheelCycleMetrics(
                cycle=i,
                n_worlds=10,
                n_attacks=50,
                attack_success_rate=0.0,
                defense_block_rate=1.0,
                evasion_rate=0.0,
                utility_score=0.95,
                cast_alpha=2.0,
                sic_threshold=0.5,
                n_new_payloads=0,
                n_previous_blocked=0,
            )
            for i in range(5)
        ]
        assert check_convergence(
            metrics, window=3, threshold=0.01,
        ) is False
