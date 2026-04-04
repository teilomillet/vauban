# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Extra coverage tests for small config parsers."""

import pytest

from vauban.config._parse_cast import _parse_cast
from vauban.config._parse_circuit import _parse_circuit
from vauban.config._types import TomlDict


def _minimal_cast() -> TomlDict:
    return {
        "cast": {
            "prompts": ["hello"],
        },
    }


def _minimal_circuit() -> TomlDict:
    return {
        "circuit": {
            "clean_prompts": ["a"],
            "corrupt_prompts": ["b"],
        },
    }


class TestParseCastExtra:
    def test_parse_cast_supports_optional_monitoring_fields(self) -> None:
        raw: dict[str, object] = {
            "cast": {
                "prompts": ["hello"],
                "direction_source": "svf",
                "svf_boundary_path": "boundary.npz",
                "bank_path": "bank.json",
                "composition": {"guard": 0.75},
                "externality_monitor": True,
                "displacement_threshold": 0.4,
                "baseline_activations_path": "baseline.npz",
                "alpha_tiers": [
                    {"threshold": 0.2, "alpha": 0.5},
                    {"threshold": 0.8, "alpha": 1.0},
                ],
            },
        }

        config = _parse_cast(raw)

        assert config is not None
        assert config.direction_source == "svf"
        assert config.svf_boundary_path == "boundary.npz"
        assert config.bank_path == "bank.json"
        assert config.composition == {"guard": 0.75}
        assert config.externality_monitor is True
        assert config.displacement_threshold == pytest.approx(0.4)
        assert config.baseline_activations_path == "baseline.npz"
        assert config.alpha_tiers is not None
        assert len(config.alpha_tiers) == 2

    def test_parse_cast_requires_svf_boundary_path(self) -> None:
        raw: TomlDict = {
            "cast": {
                "prompts": ["hello"],
                "direction_source": "svf",
            },
        }

        with pytest.raises(ValueError, match="svf_boundary_path"):
            _parse_cast(raw)

    def test_parse_cast_rejects_non_list_alpha_tiers(self) -> None:
        raw: TomlDict = {
            "cast": {
                "prompts": ["hello"],
                "alpha_tiers": "bad",
            },
        }

        with pytest.raises(TypeError, match="list of tables"):
            _parse_cast(raw)

    def test_parse_cast_rejects_missing_alpha_tier_keys(self) -> None:
        raw: TomlDict = {
            "cast": {
                "prompts": ["hello"],
                "alpha_tiers": [{"threshold": 0.2}],
            },
        }

        with pytest.raises(ValueError, match="must have 'threshold' and 'alpha'"):
            _parse_cast(raw)

    def test_parse_cast_rejects_invalid_alpha_tier_types(self) -> None:
        raw_threshold: TomlDict = {
            "cast": {
                "prompts": ["hello"],
                "alpha_tiers": [{"threshold": "high", "alpha": 0.5}],
            },
        }

        with pytest.raises(TypeError, match="threshold must be a number"):
            _parse_cast(raw_threshold)

        raw_alpha: TomlDict = {
            "cast": {
                "prompts": ["hello"],
                "alpha_tiers": [{"threshold": 0.1, "alpha": "high"}],
            },
        }

        with pytest.raises(TypeError, match="alpha must be a number"):
            _parse_cast(raw_alpha)

    def test_parse_cast_rejects_unsorted_alpha_tiers(self) -> None:
        raw: TomlDict = {
            "cast": {
                "prompts": ["hello"],
                "alpha_tiers": [
                    {"threshold": 0.8, "alpha": 1.0},
                    {"threshold": 0.2, "alpha": 0.5},
                ],
            },
        }

        with pytest.raises(ValueError, match="sorted by ascending threshold"):
            _parse_cast(raw)


class TestParseCircuitExtra:
    def test_parse_circuit_rejects_empty_clean_prompts(self) -> None:
        raw: TomlDict = {
            "circuit": {
                "clean_prompts": [],
                "corrupt_prompts": ["b"],
            },
        }

        with pytest.raises(ValueError, match="clean_prompts must not be empty"):
            _parse_circuit(raw)

    def test_parse_circuit_rejects_empty_corrupt_prompts(self) -> None:
        raw: TomlDict = {
            "circuit": {
                "clean_prompts": ["a"],
                "corrupt_prompts": [],
            },
        }

        with pytest.raises(ValueError, match="corrupt_prompts must not be empty"):
            _parse_circuit(raw)

    def test_parse_circuit_rejects_mismatched_prompt_lengths(self) -> None:
        raw: TomlDict = {
            "circuit": {
                "clean_prompts": ["a", "b"],
                "corrupt_prompts": ["c"],
            },
        }

        with pytest.raises(ValueError, match="same length"):
            _parse_circuit(raw)

    def test_parse_circuit_rejects_empty_logit_diff_tokens(self) -> None:
        raw: TomlDict = {
            "circuit": {
                "clean_prompts": ["a"],
                "corrupt_prompts": ["b"],
                "logit_diff_tokens": [],
            },
        }

        with pytest.raises(ValueError, match="logit_diff_tokens must not be empty"):
            _parse_circuit(raw)
