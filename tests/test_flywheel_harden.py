"""Tests for flywheel defense hardening."""

from vauban.flywheel._harden import harden_defense
from vauban.types import DefendedTrace, FlywheelDefenseParams


def _make_evaded(n: int = 1) -> list[DefendedTrace]:
    """Create n synthetic evaded traces."""
    return [
        DefendedTrace(
            world_index=i,
            payload_index=0,
            payload_text=f"test_{i}",
            reward=0.9,
            target_called=True,
            turns_used=1,
            tool_calls_made=1,
        )
        for i in range(n)
    ]


class TestHardenDefense:
    def test_no_evasions_returns_unchanged(self) -> None:
        params = FlywheelDefenseParams(
            cast_alpha=2.0,
            cast_threshold=0.0,
            sic_threshold=0.5,
            sic_iterations=3,
            sic_mode="direction",
        )
        result = harden_defense(params, [], 0.1, 0.95, 0.90)
        assert result is params

    def test_evasions_increase_alpha(self) -> None:
        params = FlywheelDefenseParams(
            cast_alpha=2.0,
            cast_threshold=0.0,
            sic_threshold=0.5,
            sic_iterations=3,
            sic_mode="direction",
        )
        result = harden_defense(params, _make_evaded(1), 0.1, 0.95, 0.90)
        assert result.cast_alpha > params.cast_alpha

    def test_evasions_decrease_sic_threshold(self) -> None:
        params = FlywheelDefenseParams(
            cast_alpha=2.0,
            cast_threshold=0.0,
            sic_threshold=0.5,
            sic_iterations=3,
            sic_mode="direction",
        )
        result = harden_defense(params, _make_evaded(1), 0.1, 0.95, 0.90)
        assert result.sic_threshold < params.sic_threshold

    def test_utility_floor_dampens_changes(self) -> None:
        params = FlywheelDefenseParams(
            cast_alpha=2.0,
            cast_threshold=0.0,
            sic_threshold=0.5,
            sic_iterations=3,
            sic_mode="direction",
        )
        # High utility (above floor) — full adaptation
        high_util = harden_defense(
            params, _make_evaded(3), 0.1, 0.95, 0.90,
        )
        # Low utility (below floor) — dampened adaptation
        low_util = harden_defense(
            params, _make_evaded(3), 0.1, 0.5, 0.90,
        )
        # Low utility should have smaller alpha increase
        assert (
            high_util.cast_alpha - params.cast_alpha
            > low_util.cast_alpha - params.cast_alpha
        )

    def test_sic_threshold_floors_at_zero(self) -> None:
        params = FlywheelDefenseParams(
            cast_alpha=2.0,
            cast_threshold=0.0,
            sic_threshold=0.01,
            sic_iterations=3,
            sic_mode="direction",
        )
        result = harden_defense(params, _make_evaded(10), 1.0, 0.95, 0.90)
        assert result.sic_threshold >= 0.0

    def test_many_evasions_increase_sic_iterations(self) -> None:
        params = FlywheelDefenseParams(
            cast_alpha=2.0,
            cast_threshold=0.0,
            sic_threshold=0.5,
            sic_iterations=3,
            sic_mode="direction",
        )
        result = harden_defense(params, _make_evaded(5), 0.1, 0.95, 0.90)
        assert result.sic_iterations > params.sic_iterations

    def test_rate_based_pressure_with_n_successful(self) -> None:
        params = FlywheelDefenseParams(
            cast_alpha=2.0,
            cast_threshold=0.0,
            sic_threshold=0.5,
            sic_iterations=3,
            sic_mode="direction",
        )
        # 5 evasions out of 100 successful = 5% rate → low pressure
        low_rate = harden_defense(
            params, _make_evaded(5), 0.1, 0.95, 0.90,
            n_successful=100,
        )
        # 5 evasions out of 10 successful = 50% rate → high pressure
        high_rate = harden_defense(
            params, _make_evaded(5), 0.1, 0.95, 0.90,
            n_successful=10,
        )
        # Higher evasion rate → more alpha increase
        assert high_rate.cast_alpha > low_rate.cast_alpha

    def test_preserves_unchanged_fields(self) -> None:
        params = FlywheelDefenseParams(
            cast_alpha=2.0,
            cast_threshold=0.5,
            sic_threshold=0.5,
            sic_iterations=3,
            sic_mode="generation",
            cast_layers=[4, 8],
        )
        result = harden_defense(params, _make_evaded(1), 0.1, 0.95, 0.90)
        assert result.cast_threshold == params.cast_threshold
        assert result.sic_mode == params.sic_mode
        assert result.cast_layers == params.cast_layers
