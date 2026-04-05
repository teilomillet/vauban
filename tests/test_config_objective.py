# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for the [objective] config parser."""

from pathlib import Path

import pytest

from vauban.config._parse_objective import _parse_objective
from vauban.types import ObjectiveConfig


class TestParseObjective:
    def _parse(
        self,
        raw: dict[str, object],
        *,
        base_dir: Path | None = None,
    ) -> ObjectiveConfig | None:
        return _parse_objective(base_dir or Path("/tmp"), raw)

    def test_absent_returns_none(self) -> None:
        assert self._parse({}) is None

    def test_section_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            self._parse({"objective": "bad"})

    def test_minimal_valid_utility_objective(self) -> None:
        cfg = self._parse({
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
        assert cfg.benign_inquiry_source == "generated"
        assert cfg.benign_inquiries_path is None
        assert cfg.preserve == []
        assert cfg.prevent == []
        assert cfg.safety == []
        assert len(cfg.utility) == 1
        assert cfg.utility[0].metric == "utility_score"
        assert cfg.utility[0].comparison == "at_least"
        assert cfg.utility[0].aggregate == "final"

    def test_custom_values(self) -> None:
        cfg = self._parse({
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

    def test_dataset_path_infers_dataset_source_and_resolves_relative_path(
        self,
        tmp_path: Path,
    ) -> None:
        cfg = self._parse({
            "objective": {
                "name": "support_dataset_gate",
                "benign_inquiries": "data/benign.jsonl",
                "utility": [{
                    "metric": "utility_score",
                    "threshold": 0.9,
                }],
            },
        }, base_dir=tmp_path)

        assert cfg is not None
        assert cfg.benign_inquiry_source == "dataset"
        assert cfg.benign_inquiries_path == tmp_path / "data" / "benign.jsonl"

    def test_dataset_source_requires_path(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"requires \[objective\]\.benign_inquiries",
        ):
            self._parse({"objective": {
                "name": "missing_dataset",
                "benign_inquiry_source": "dataset",
                "utility": [{
                    "metric": "utility_score",
                    "threshold": 0.90,
                }],
            }})

    def test_generated_source_rejects_explicit_dataset_path(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"requires.*benign_inquiry_source = \"dataset\"",
        ):
            self._parse({"objective": {
                "name": "mismatch",
                "benign_inquiry_source": "generated",
                "benign_inquiries": "data/benign.jsonl",
                "utility": [{
                    "metric": "utility_score",
                    "threshold": 0.90,
                }],
            }})

    def test_name_must_be_non_empty(self) -> None:
        with pytest.raises(ValueError, match="name must be non-empty"):
            self._parse({"objective": {
                "name": "",
                "utility": [{
                    "metric": "utility_score",
                    "threshold": 0.90,
                }],
            }})

    def test_requires_at_least_one_threshold(self) -> None:
        with pytest.raises(ValueError, match="must define at least one"):
            self._parse({"objective": {"name": "empty"}})

    def test_invalid_metric_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            self._parse({"objective": {
                "name": "bad_metric",
                "safety": [{
                    "metric": "made_up_metric",
                    "threshold": 0.1,
                }],
            }})

    def test_metric_group_must_be_list(self) -> None:
        with pytest.raises(TypeError, match="array of tables"):
            self._parse({"objective": {
                "name": "bad_group",
                "utility": "bad",
            }})

    def test_metric_entry_must_be_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            self._parse({"objective": {
                "name": "bad_entry",
                "utility": ["bad"],
            }})
