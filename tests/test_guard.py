# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for guard config parsing and calibration."""

import pytest

from vauban.config._parse_guard import _parse_guard
from vauban.types import _DEFAULT_GUARD_TIERS

# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParseGuard:
    """Tests for the [guard] TOML parser."""

    def test_absent_returns_none(self) -> None:
        assert _parse_guard({}) is None

    def test_minimal_config(self) -> None:
        raw = {"guard": {"prompts": ["test"]}}
        config = _parse_guard(raw)
        assert config is not None
        assert config.prompts == ["test"]
        assert config.max_tokens == 100
        assert config.max_rewinds == 3
        assert config.checkpoint_interval == 1
        assert len(config.tiers) == 4  # defaults

    def test_custom_tiers(self) -> None:
        raw = {
            "guard": {
                "prompts": ["test"],
                "tiers": [
                    {"threshold": 0.0, "zone": "green", "alpha": 0.0},
                    {"threshold": 0.5, "zone": "red", "alpha": 2.0},
                ],
            },
        }
        config = _parse_guard(raw)
        assert config is not None
        assert len(config.tiers) == 2
        assert config.tiers[1].zone == "red"
        assert config.tiers[1].alpha == 2.0

    def test_empty_prompts_raises(self) -> None:
        raw = {"guard": {"prompts": []}}
        with pytest.raises(ValueError, match="non-empty"):
            _parse_guard(raw)

    def test_invalid_zone_raises(self) -> None:
        raw = {
            "guard": {
                "prompts": ["test"],
                "tiers": [
                    {"threshold": 0.0, "zone": "invalid", "alpha": 0.0},
                ],
            },
        }
        with pytest.raises(ValueError, match="zone"):
            _parse_guard(raw)

    def test_unsorted_tiers_raises(self) -> None:
        raw = {
            "guard": {
                "prompts": ["test"],
                "tiers": [
                    {"threshold": 0.5, "zone": "yellow", "alpha": 0.5},
                    {"threshold": 0.1, "zone": "green", "alpha": 0.0},
                ],
            },
        }
        with pytest.raises(ValueError, match="ascending"):
            _parse_guard(raw)

    def test_empty_tiers_raises(self) -> None:
        raw = {
            "guard": {
                "prompts": ["test"],
                "tiers": [],
            },
        }
        with pytest.raises(ValueError, match="empty"):
            _parse_guard(raw)

    def test_missing_tier_keys_raises(self) -> None:
        raw = {
            "guard": {
                "prompts": ["test"],
                "tiers": [{"threshold": 0.0}],
            },
        }
        with pytest.raises(ValueError, match=r"zone.*alpha"):
            _parse_guard(raw)

    def test_max_tokens_validation(self) -> None:
        raw = {"guard": {"prompts": ["test"], "max_tokens": 0}}
        with pytest.raises(ValueError, match="max_tokens"):
            _parse_guard(raw)

    def test_max_rewinds_negative_raises(self) -> None:
        raw = {"guard": {"prompts": ["test"], "max_rewinds": -1}}
        with pytest.raises(ValueError, match="max_rewinds"):
            _parse_guard(raw)

    def test_checkpoint_interval_zero_raises(self) -> None:
        raw = {"guard": {"prompts": ["test"], "checkpoint_interval": 0}}
        with pytest.raises(ValueError, match="checkpoint_interval"):
            _parse_guard(raw)

    def test_all_options(self) -> None:
        raw = {
            "guard": {
                "prompts": ["test1", "test2"],
                "layers": [5, 10],
                "max_tokens": 200,
                "max_rewinds": 5,
                "checkpoint_interval": 3,
                "calibrate": True,
                "calibrate_prompts": "harmful",
                "defensive_prompt": "Be safe",
                "condition_direction": "cond.npy",
            },
        }
        config = _parse_guard(raw)
        assert config is not None
        assert config.prompts == ["test1", "test2"]
        assert config.layers == [5, 10]
        assert config.max_tokens == 200
        assert config.max_rewinds == 5
        assert config.checkpoint_interval == 3
        assert config.calibrate is True
        assert config.calibrate_prompts == "harmful"
        assert config.defensive_prompt == "Be safe"
        assert config.condition_direction_path == "cond.npy"


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------


class TestCalibration:
    """Tests for guard tier calibration."""

    def test_returns_four_tiers(
        self,
        mock_model: "MockCausalLM",  # noqa: F821
        mock_tokenizer: "MockTokenizer",  # noqa: F821
        direction: "Array",  # noqa: F821
    ) -> None:
        from vauban.guard import calibrate_guard_thresholds

        tiers = calibrate_guard_thresholds(
            mock_model, mock_tokenizer,
            ["test prompt", "another prompt"],
            direction, layers=[0],
        )
        assert len(tiers) == 4
        assert tiers[0].zone == "green"
        assert tiers[1].zone == "yellow"
        assert tiers[2].zone == "orange"
        assert tiers[3].zone == "red"
        # Thresholds must be ascending
        for i in range(1, len(tiers)):
            assert tiers[i].threshold >= tiers[i - 1].threshold

    def test_empty_prompts_returns_defaults(
        self,
        mock_model: "MockCausalLM",  # noqa: F821
        mock_tokenizer: "MockTokenizer",  # noqa: F821
        direction: "Array",  # noqa: F821
    ) -> None:
        from vauban.guard import calibrate_guard_thresholds

        tiers = calibrate_guard_thresholds(
            mock_model, mock_tokenizer,
            [], direction, layers=[0],
        )
        assert len(tiers) == 4
        # Should match default tiers
        for t, d in zip(tiers, _DEFAULT_GUARD_TIERS, strict=True):
            assert t.threshold == d.threshold
            assert t.zone == d.zone


# ---------------------------------------------------------------------------
# Serializer tests
# ---------------------------------------------------------------------------


class TestGuardSerializer:
    """Tests for GuardResult JSON serialization."""

    def test_round_trip(self) -> None:
        from vauban._serializers import _guard_to_dict
        from vauban.types import GuardEvent, GuardResult

        result = GuardResult(
            prompt="test",
            text="response",
            events=[
                GuardEvent(
                    token_index=0, token_id=42, token_str="A",
                    projection=0.1, zone="green", action="pass",
                    alpha_applied=0.0, rewind_count=0,
                    checkpoint_offset=0,
                ),
            ],
            total_rewinds=0,
            circuit_broken=False,
            tokens_generated=1,
            tokens_rewound=0,
            final_zone_counts={"green": 1, "yellow": 0, "orange": 0, "red": 0},
        )
        d = _guard_to_dict(result)
        assert d["prompt"] == "test"
        assert d["text"] == "response"
        assert d["circuit_broken"] is False
        assert len(d["events"]) == 1  # type: ignore[arg-type]
        assert d["events"][0]["zone"] == "green"  # type: ignore[index]
