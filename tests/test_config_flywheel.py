"""Tests for the [flywheel] config parser."""

import pytest

from vauban.config._parse_flywheel import _parse_flywheel


class TestParseFlywheel:
    def test_absent_returns_none(self) -> None:
        assert _parse_flywheel({}) is None

    def test_section_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_flywheel({"flywheel": "bad"})

    def test_minimal_valid(self) -> None:
        cfg = _parse_flywheel({"flywheel": {}})
        assert cfg is not None
        assert cfg.n_cycles == 10
        assert cfg.worlds_per_cycle == 50
        assert cfg.payloads_per_world == 5
        assert cfg.skeletons == ["email", "doc", "code", "calendar", "search"]
        assert cfg.model_expand is True
        assert cfg.difficulty_range == (1, 5)
        assert cfg.positions == ["infix"]
        assert cfg.warmstart_gcg is False
        assert cfg.cast_alpha == 2.0
        assert cfg.sic_mode == "direction"
        assert cfg.harden is True
        assert cfg.adaptation_rate == 0.1
        assert cfg.utility_floor == 0.90
        assert cfg.convergence_window == 3
        assert cfg.convergence_threshold == 0.01
        assert cfg.seed is None
        assert cfg.max_turns == 6

    def test_custom_values(self) -> None:
        cfg = _parse_flywheel({"flywheel": {
            "n_cycles": 5,
            "worlds_per_cycle": 20,
            "skeletons": ["email", "code"],
            "positions": ["prefix", "suffix"],
            "difficulty_range": [2, 4],
            "seed": 42,
            "sic_mode": "generation",
        }})
        assert cfg is not None
        assert cfg.n_cycles == 5
        assert cfg.worlds_per_cycle == 20
        assert cfg.skeletons == ["email", "code"]
        assert cfg.positions == ["prefix", "suffix"]
        assert cfg.difficulty_range == (2, 4)
        assert cfg.seed == 42
        assert cfg.sic_mode == "generation"

    def test_n_cycles_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="n_cycles"):
            _parse_flywheel({"flywheel": {"n_cycles": 0}})

    def test_worlds_per_cycle_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="worlds_per_cycle"):
            _parse_flywheel({"flywheel": {"worlds_per_cycle": 0}})

    def test_payloads_per_world_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="payloads_per_world"):
            _parse_flywheel({"flywheel": {"payloads_per_world": 0}})

    def test_skeletons_must_be_nonempty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _parse_flywheel({"flywheel": {"skeletons": []}})

    def test_positions_must_be_nonempty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _parse_flywheel({"flywheel": {"positions": []}})

    def test_positions_invalid_value(self) -> None:
        with pytest.raises(ValueError, match="invalid position"):
            _parse_flywheel({"flywheel": {"positions": ["middle"]}})

    def test_difficulty_range_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="exactly 2"):
            _parse_flywheel({"flywheel": {"difficulty_range": [1]}})

    def test_difficulty_range_min_gt_max(self) -> None:
        with pytest.raises(ValueError, match="<="):
            _parse_flywheel({"flywheel": {"difficulty_range": [4, 2]}})

    def test_difficulty_range_out_of_bounds(self) -> None:
        with pytest.raises(ValueError, match="\\[1, 5\\]"):
            _parse_flywheel({"flywheel": {"difficulty_range": [0, 3]}})

    def test_sic_iterations_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="sic_iterations"):
            _parse_flywheel({"flywheel": {"sic_iterations": 0}})

    def test_sic_mode_invalid(self) -> None:
        with pytest.raises(ValueError, match="sic_mode"):
            _parse_flywheel({"flywheel": {"sic_mode": "invalid"}})

    def test_adaptation_rate_zero(self) -> None:
        with pytest.raises(ValueError, match="adaptation_rate"):
            _parse_flywheel({"flywheel": {"adaptation_rate": 0.0}})

    def test_adaptation_rate_above_one(self) -> None:
        with pytest.raises(ValueError, match="adaptation_rate"):
            _parse_flywheel({"flywheel": {"adaptation_rate": 1.5}})

    def test_utility_floor_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="utility_floor"):
            _parse_flywheel({"flywheel": {"utility_floor": 1.5}})

    def test_convergence_window_too_small(self) -> None:
        with pytest.raises(ValueError, match="convergence_window"):
            _parse_flywheel({"flywheel": {"convergence_window": 1}})
