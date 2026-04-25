# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Parse the [behavior_diff] section of a TOML config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vauban.config._parse_helpers import SectionReader, require_toml_table
from vauban.types import (
    BehaviorDiffConfig,
    BehaviorDiffMetricConfig,
    BehaviorDiffThresholdConfig,
)

if TYPE_CHECKING:
    from pathlib import Path

    from vauban.config._types import TomlDict

_METRIC_POLARITY_CHOICES: tuple[str, ...] = (
    "higher_is_better",
    "lower_is_better",
    "neutral",
)
_TRANSFORMATION_KIND_CHOICES: tuple[str, ...] = (
    "fine_tune",
    "reinforcement_fine_tune",
    "checkpoint_update",
    "prompt_template",
    "quantization",
    "merge",
    "adapter_merge",
    "steering",
    "endpoint_update",
    "evaluation_only",
    "other",
)
_THRESHOLD_SEVERITY_CHOICES: tuple[str, ...] = ("warn", "fail")


def _parse_behavior_diff(
    base_dir: Path,
    raw: TomlDict,
) -> BehaviorDiffConfig | None:
    """Parse the optional [behavior_diff] section."""
    sec = raw.get("behavior_diff")
    if sec is None:
        return None
    reader = SectionReader(
        "[behavior_diff]",
        require_toml_table("[behavior_diff]", sec),
    )

    max_examples = reader.integer("max_examples", default=3)
    if max_examples < 0:
        msg = "[behavior_diff].max_examples must be >= 0"
        raise ValueError(msg)

    return BehaviorDiffConfig(
        baseline_trace=base_dir / reader.string("baseline_trace"),
        candidate_trace=base_dir / reader.string("candidate_trace"),
        baseline_label=reader.string("baseline_label", default="baseline"),
        candidate_label=reader.string("candidate_label", default="candidate"),
        baseline_model_path=reader.optional_string("baseline_model_path"),
        candidate_model_path=reader.optional_string("candidate_model_path"),
        title=reader.string("title", default="Model Behavior Change Report"),
        target_change=reader.optional_string("target_change"),
        suite_name=reader.string("suite_name", default="behavior-change-suite"),
        suite_description=reader.string(
            "suite_description",
            default="Behavior trace comparison suite.",
        ),
        suite_version=reader.optional_string("suite_version"),
        suite_source=reader.optional_string("suite_source"),
        safety_policy=reader.string(
            "safety_policy",
            default="aggregate_or_redacted_examples",
        ),
        transformation_kind=reader.literal(
            "transformation_kind",
            _TRANSFORMATION_KIND_CHOICES,
            default="evaluation_only",
        ),
        transformation_summary=reader.optional_string("transformation_summary"),
        metrics=_parse_metric_configs(reader.data.get("metrics")),
        thresholds=_parse_threshold_configs(reader.data.get("thresholds")),
        limitations=reader.string_list("limitations", default=[]),
        recommendation=reader.optional_string("recommendation"),
        include_examples=reader.boolean("include_examples", default=True),
        max_examples=max_examples,
        record_outputs=reader.boolean("record_outputs", default=False),
        markdown_report=reader.boolean("markdown_report", default=True),
        json_filename=reader.string(
            "json_filename",
            default="behavior_diff_report.json",
        ),
        markdown_filename=reader.string(
            "markdown_filename",
            default="model_behavior_change_report.md",
        ),
    )


def _parse_metric_configs(raw: object) -> list[BehaviorDiffMetricConfig]:
    """Parse optional [[behavior_diff.metrics]] declarations."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        msg = "[[behavior_diff.metrics]] must be an array of tables"
        raise TypeError(msg)
    metrics: list[BehaviorDiffMetricConfig] = []
    for index, item in enumerate(raw):
        section = f"[[behavior_diff.metrics]][{index}]"
        reader = SectionReader(section, require_toml_table(section, item))
        metrics.append(
            BehaviorDiffMetricConfig(
                name=reader.string("name"),
                description=reader.string("description", default=""),
                polarity=reader.literal(
                    "polarity",
                    _METRIC_POLARITY_CHOICES,
                    default="neutral",
                ),
                unit=reader.string("unit", default="ratio"),
                family=reader.string("family", default="behavior"),
            ),
        )
    return metrics


def _parse_threshold_configs(raw: object) -> list[BehaviorDiffThresholdConfig]:
    """Parse optional [[behavior_diff.thresholds]] declarations."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        msg = "[[behavior_diff.thresholds]] must be an array of tables"
        raise TypeError(msg)
    thresholds: list[BehaviorDiffThresholdConfig] = []
    for index, item in enumerate(raw):
        section = f"[[behavior_diff.thresholds]][{index}]"
        reader = SectionReader(section, require_toml_table(section, item))
        max_delta = reader.optional_number("max_delta")
        min_delta = reader.optional_number("min_delta")
        max_absolute_delta = reader.optional_number("max_absolute_delta")
        if (
            max_delta is None
            and min_delta is None
            and max_absolute_delta is None
        ):
            msg = f"{section} must set at least one delta bound"
            raise ValueError(msg)
        if max_absolute_delta is not None and max_absolute_delta < 0.0:
            msg = f"{section}.max_absolute_delta must be >= 0"
            raise ValueError(msg)
        thresholds.append(
            BehaviorDiffThresholdConfig(
                metric=reader.string("metric"),
                category=reader.optional_string("category"),
                max_delta=max_delta,
                min_delta=min_delta,
                max_absolute_delta=max_absolute_delta,
                severity=reader.literal(
                    "severity",
                    _THRESHOLD_SEVERITY_CHOICES,
                    default="fail",
                ),
                description=reader.string("description", default=""),
            ),
        )
    return thresholds
