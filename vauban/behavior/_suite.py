# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Reusable behavior-suite TOML parsing primitives."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast

from vauban.behavior._primitives import (
    BehaviorMetricSpec,
    BehaviorPrompt,
    BehaviorSuite,
    ExampleRedaction,
    ExpectedBehavior,
    MetricPolarity,
)

EXPECTED_BEHAVIOR_CHOICES: tuple[ExpectedBehavior, ...] = (
    "refuse",
    "comply",
    "express_uncertainty",
    "ask_clarifying_question",
    "defer",
    "unknown",
)
REDACTION_CHOICES: tuple[ExampleRedaction, ...] = (
    "safe",
    "redacted",
    "omitted",
)
METRIC_POLARITY_CHOICES: tuple[MetricPolarity, ...] = (
    "higher_is_better",
    "lower_is_better",
    "neutral",
)

DEFAULT_BEHAVIOR_METRIC_SPECS: tuple[BehaviorMetricSpec, ...] = (
    BehaviorMetricSpec(
        name="refusal_rate",
        description="Fraction of observations with refusal-style output.",
        polarity="neutral",
        unit="ratio",
        family="behavior",
    ),
    BehaviorMetricSpec(
        name="expected_behavior_match_rate",
        description="Fraction of observations matching expected behavior labels.",
        polarity="higher_is_better",
        unit="ratio",
        family="behavior",
    ),
    BehaviorMetricSpec(
        name="uncertainty_expression_rate",
        description="Fraction of observations expressing uncertainty.",
        polarity="neutral",
        unit="ratio",
        family="behavior",
    ),
    BehaviorMetricSpec(
        name="clarifying_question_rate",
        description="Fraction of observations asking a clarifying question.",
        polarity="neutral",
        unit="ratio",
        family="behavior",
    ),
    BehaviorMetricSpec(
        name="direct_answer_rate",
        description="Fraction of observations giving a direct non-refusal answer.",
        polarity="neutral",
        unit="ratio",
        family="behavior",
    ),
    BehaviorMetricSpec(
        name="assertive_language_rate",
        description="Fraction of observations using assertive language markers.",
        polarity="neutral",
        unit="ratio",
        family="behavior",
    ),
    BehaviorMetricSpec(
        name="output_length_chars",
        description="Generated output length in characters.",
        polarity="neutral",
        unit="count",
        family="behavior",
    ),
    BehaviorMetricSpec(
        name="output_word_count",
        description="Generated output length in whitespace-delimited words.",
        polarity="neutral",
        unit="count",
        family="behavior",
    ),
)


def load_behavior_suite_toml(path: str | Path) -> BehaviorSuite:
    """Load a reusable behavior suite from TOML."""
    path_obj = Path(path)
    if not path_obj.exists():
        msg = f"behavior suite TOML does not exist: {path_obj}"
        raise FileNotFoundError(msg)
    with path_obj.open("rb") as handle:
        raw_obj: object = tomllib.load(handle)
    raw = _require_mapping("suite TOML", raw_obj)
    section = _require_mapping("[behavior_suite]", raw.get("behavior_suite"))
    return parse_behavior_suite_table(
        section,
        source_default=str(path_obj),
    )


def parse_behavior_suite_table(
    section: dict[str, object],
    *,
    source_default: str | None = None,
) -> BehaviorSuite:
    """Parse a [behavior_suite] table into a validated BehaviorSuite."""
    prompts = parse_behavior_prompts(
        section.get("prompts"),
        "[behavior_suite].prompts",
    )
    metric_specs = parse_behavior_metric_specs(section.get("metrics"))
    return BehaviorSuite(
        name=_string(section, "name", default="behavior-change-suite"),
        description=_string(
            section,
            "description",
            default="Behavior trace collection suite.",
        ),
        version=_optional_string(section, "version"),
        source=_optional_string(section, "source") or source_default,
        safety_policy=_string(
            section,
            "safety_policy",
            default="safe_or_redacted_prompts",
        ),
        prompts=tuple(prompts),
        metric_specs=metric_specs or DEFAULT_BEHAVIOR_METRIC_SPECS,
    )


def parse_behavior_prompts(raw: object, section: str) -> tuple[BehaviorPrompt, ...]:
    """Parse behavior prompt tables."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        msg = f"{section} must be an array"
        raise TypeError(msg)
    prompts: list[BehaviorPrompt] = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            prompts.append(
                BehaviorPrompt(
                    prompt_id=f"prompt-{index + 1:03d}",
                    category="default",
                    prompt=item,
                ),
            )
            continue
        table = _require_mapping(f"{section}[{index}]", item)
        prompt_id = _optional_string(table, "id")
        if prompt_id is None:
            prompt_id = _optional_string(table, "prompt_id")
        if prompt_id is None:
            msg = f"{section}[{index}].id is required"
            raise ValueError(msg)
        prompt_text = _optional_string(table, "text")
        if prompt_text is None:
            prompt_text = _optional_string(table, "prompt")
        if prompt_text is None:
            msg = f"{section}[{index}].text is required"
            raise ValueError(msg)
        prompts.append(
            BehaviorPrompt(
                prompt_id=prompt_id,
                category=_string(table, "category", default="default"),
                prompt=prompt_text,
                expected_behavior=_literal(
                    table,
                    "expected_behavior",
                    EXPECTED_BEHAVIOR_CHOICES,
                    default="unknown",
                ),
                tags=tuple(_string_list(table, "tags", default=[])),
                redaction=_literal(
                    table,
                    "redaction",
                    REDACTION_CHOICES,
                    default="safe",
                ),
            ),
        )
    return tuple(prompts)


def parse_behavior_metric_specs(raw: object) -> tuple[BehaviorMetricSpec, ...]:
    """Parse optional behavior metric declarations."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        msg = "[[behavior_suite.metrics]] must be an array of tables"
        raise TypeError(msg)
    metrics: list[BehaviorMetricSpec] = []
    for index, item in enumerate(raw):
        table = _require_mapping(f"[[behavior_suite.metrics]][{index}]", item)
        name = _string(table, "name")
        metrics.append(
            BehaviorMetricSpec(
                name=name,
                description=_string(
                    table,
                    "description",
                    default=f"Behavior metric declared by suite: {name}.",
                ),
                polarity=_literal(
                    table,
                    "polarity",
                    METRIC_POLARITY_CHOICES,
                    default="neutral",
                ),
                unit=_string(table, "unit", default="ratio"),
                family=_string(table, "family", default="behavior"),
            ),
        )
    return tuple(metrics)


def _require_mapping(section: str, raw: object) -> dict[str, object]:
    """Validate a TOML table."""
    if not isinstance(raw, dict):
        msg = f"{section} must be a table, got {type(raw).__name__}"
        raise TypeError(msg)
    parsed: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            msg = f"{section} keys must be strings"
            raise TypeError(msg)
        parsed[key] = value
    return parsed


def _string(
    table: dict[str, object],
    key: str,
    *,
    default: str | None = None,
) -> str:
    """Read a required or defaulted string."""
    raw = table.get(key, default)
    if not isinstance(raw, str):
        msg = f"{key} must be a string"
        raise TypeError(msg)
    if not raw.strip():
        msg = f"{key} must not be empty"
        raise ValueError(msg)
    return raw


def _optional_string(table: dict[str, object], key: str) -> str | None:
    """Read an optional string."""
    raw = table.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str):
        msg = f"{key} must be a string"
        raise TypeError(msg)
    if not raw.strip():
        msg = f"{key} must not be empty"
        raise ValueError(msg)
    return raw


def _string_list(
    table: dict[str, object],
    key: str,
    *,
    default: list[str],
) -> list[str]:
    """Read a list of strings."""
    raw = table.get(key, default)
    if not isinstance(raw, list):
        msg = f"{key} must be a list of strings"
        raise TypeError(msg)
    values: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            msg = f"{key} elements must be strings"
            raise TypeError(msg)
        values.append(item)
    return values


def _literal[LiteralT: str](
    table: dict[str, object],
    key: str,
    choices: tuple[LiteralT, ...],
    *,
    default: LiteralT,
) -> LiteralT:
    """Read and validate a literal string."""
    raw = table.get(key, default)
    if not isinstance(raw, str):
        msg = f"{key} must be a string"
        raise TypeError(msg)
    if raw not in choices:
        msg = f"{key} must be one of {choices!r}, got {raw!r}"
        raise ValueError(msg)
    return cast("LiteralT", raw)
