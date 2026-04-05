# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for the [objective] config parser."""

import pytest

from vauban.config._parse_objective import _parse_objective


class TestParseObjective:
    def test_absent_returns_none(self) -> None:
        assert _parse_objective({}) is None

    def test_section_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_objective({"objective": "bad"})

    def test_minimal_valid_utility_objective(self) -> None:
        cfg = _parse_objective({
            "objective": {
                "name": "customer_support_gate",
                "utility": [{
                    "metric": "utility_score",
                    "threshold": 0.90,
                }],
            },
        })

        assert cfg is not None
        assert cfg.name == "customer_support_gate"
        assert cfg.access == "system"
        assert cfg.preserve == []
        assert cfg.prevent == []
        assert cfg.safety == []
        assert len(cfg.utility) == 1
        assert cfg.utility[0].metric == "utility_score"
        assert cfg.utility[0].comparison == "at_least"
        assert cfg.utility[0].aggregate == "final"

    def test_custom_values(self) -> None:
        cfg = _parse_objective({
            "objective": {
                "name": "refund_abuse_gate",
                "deployment": "customer_support",
                "summary": "Preserve support quality while resisting abuse.",
                "access": "api",
                "preserve": ["resolve billing issue", "explain refund policy"],
                "prevent": ["unauthorized refund", "PII disclosure"],
                "safety": [{
                    "metric": "evasion_rate",
                    "threshold": 0.05,
                    "comparison": "at_most",
                    "aggregate": "max",
                    "label": "peak adaptive evasion",
                }],
                "utility": [{
                    "metric": "utility_score",
                    "threshold": 0.92,
                    "comparison": "at_least",
                    "aggregate": "mean",
                    "description": "Average benign retention.",
                }],
            },
        })

        assert cfg is not None
        assert cfg.deployment == "customer_support"
        assert cfg.summary == "Preserve support quality while resisting abuse."
        assert cfg.access == "api"
        assert cfg.preserve == [
            "resolve billing issue",
            "explain refund policy",
        ]
        assert cfg.prevent == ["unauthorized refund", "PII disclosure"]
        assert cfg.safety[0].aggregate == "max"
        assert cfg.safety[0].label == "peak adaptive evasion"
        assert cfg.utility[0].aggregate == "mean"
        assert cfg.utility[0].description == "Average benign retention."

    def test_name_must_be_non_empty(self) -> None:
        with pytest.raises(ValueError, match="name must be non-empty"):
            _parse_objective({"objective": {
                "name": "",
                "utility": [{
                    "metric": "utility_score",
                    "threshold": 0.90,
                }],
            }})

    def test_requires_at_least_one_threshold(self) -> None:
        with pytest.raises(ValueError, match="must define at least one"):
            _parse_objective({"objective": {"name": "empty"}})

    def test_invalid_metric_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            _parse_objective({"objective": {
                "name": "bad_metric",
                "safety": [{
                    "metric": "made_up_metric",
                    "threshold": 0.1,
                }],
            }})

    def test_metric_group_must_be_list(self) -> None:
        with pytest.raises(TypeError, match="array of tables"):
            _parse_objective({"objective": {
                "name": "bad_group",
                "utility": "bad",
            }})

    def test_metric_entry_must_be_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_objective({"objective": {
                "name": "bad_entry",
                "utility": ["bad"],
            }})
