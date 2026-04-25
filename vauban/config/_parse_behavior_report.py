# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Parse the [behavior_report] section of a TOML config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vauban.behavior import (
    ActivationFinding,
    BehaviorExample,
    BehaviorMetric,
    BehaviorReport,
    BehaviorSuiteRef,
    ExampleRedaction,
    FindingSeverity,
    MetricPolarity,
    ModelRole,
    ReportModelRef,
    ReproducibilityInfo,
    compare_behavior_metrics,
)
from vauban.config._parse_helpers import SectionReader, require_toml_table
from vauban.types import BehaviorReportConfig

if TYPE_CHECKING:
    from vauban.config._types import TomlDict

_POLARITY_CHOICES: tuple[MetricPolarity, ...] = (
    "higher_is_better",
    "lower_is_better",
    "neutral",
)
_MODEL_ROLE_CHOICES: tuple[ModelRole, ...] = (
    "baseline",
    "candidate",
    "reference",
    "intervention",
)
_SEVERITY_CHOICES: tuple[FindingSeverity, ...] = (
    "info",
    "low",
    "medium",
    "high",
    "critical",
)
_REDACTION_CHOICES: tuple[ExampleRedaction, ...] = (
    "safe",
    "redacted",
    "omitted",
)


def _parse_behavior_report(raw: TomlDict) -> BehaviorReportConfig | None:
    """Parse the optional [behavior_report] section."""
    sec = raw.get("behavior_report")
    if sec is None:
        return None
    reader = SectionReader(
        "[behavior_report]",
        require_toml_table("[behavior_report]", sec),
    )

    title = reader.string("title", default="Model Behavior Change Report")
    markdown_report = reader.boolean("markdown_report", default=True)
    json_filename = reader.string("json_filename", default="behavior_report.json")
    markdown_filename = reader.string(
        "markdown_filename",
        default="behavior_report.md",
    )

    baseline = _parse_model_ref(
        reader.data.get("baseline"),
        "[behavior_report.baseline]",
        default_role="baseline",
    )
    candidate = _parse_model_ref(
        reader.data.get("candidate"),
        "[behavior_report.candidate]",
        default_role="candidate",
    )
    suite = _parse_suite_ref(reader.data.get("suite"))
    metrics = tuple(_parse_metrics(reader.data.get("metrics")))
    metric_deltas = compare_behavior_metrics(
        tuple(metric for metric in metrics if metric.model_label == baseline.label),
        tuple(metric for metric in metrics if metric.model_label == candidate.label),
    )

    report = BehaviorReport(
        title=title,
        baseline=baseline,
        candidate=candidate,
        suite=suite,
        target_change=reader.optional_string("target_change"),
        findings=tuple(reader.string_list("findings", default=[])),
        metrics=metrics,
        metric_deltas=metric_deltas,
        activation_findings=tuple(
            _parse_activation_findings(reader.data.get("activation_findings")),
        ),
        examples=tuple(_parse_examples(reader.data.get("examples"))),
        limitations=tuple(reader.string_list("limitations", default=[])),
        recommendation=reader.optional_string("recommendation"),
        reproducibility=_parse_reproducibility(
            reader.data.get("reproducibility"),
        ),
    )
    return BehaviorReportConfig(
        report=report,
        markdown_report=markdown_report,
        json_filename=json_filename,
        markdown_filename=markdown_filename,
    )


def _parse_model_ref(
    raw: object,
    section: str,
    *,
    default_role: ModelRole,
) -> ReportModelRef:
    """Parse a report model reference table."""
    reader = SectionReader(section, require_toml_table(section, raw))
    return ReportModelRef(
        label=reader.string("label"),
        model_path=reader.string("model_path"),
        role=reader.literal("role", _MODEL_ROLE_CHOICES, default=default_role),
        checkpoint=reader.optional_string("checkpoint"),
        adapter_path=reader.optional_string("adapter_path"),
        prompt_template=reader.optional_string("prompt_template"),
        quantization=reader.optional_string("quantization"),
    )


def _parse_suite_ref(raw: object) -> BehaviorSuiteRef:
    """Parse the report suite metadata table."""
    section = "[behavior_report.suite]"
    reader = SectionReader(section, require_toml_table(section, raw))
    return BehaviorSuiteRef(
        name=reader.string("name"),
        description=reader.string("description"),
        categories=tuple(reader.string_list("categories")),
        metrics=tuple(reader.string_list("metrics")),
        version=reader.optional_string("version"),
        source=reader.optional_string("source"),
        safety_policy=reader.string(
            "safety_policy",
            default="aggregate_or_redacted_examples",
        ),
    )


def _parse_metrics(raw: object) -> list[BehaviorMetric]:
    """Parse [[behavior_report.metrics]] entries."""
    rows = _array_of_tables("[[behavior_report.metrics]]", raw, allow_empty=True)
    return [
        _parse_metric(row, f"[[behavior_report.metrics]][{index}]")
        for index, row in enumerate(rows)
    ]


def _parse_metric(raw: TomlDict, section: str) -> BehaviorMetric:
    """Parse one behavior metric table."""
    reader = SectionReader(section, raw)
    model_label = reader.optional_string("model_label")
    if model_label is None:
        model_label = reader.optional_string("model")
    if model_label is None:
        msg = f"{section}.model_label is required"
        raise ValueError(msg)
    sample_size = reader.optional_integer("sample_size")
    if sample_size is not None and sample_size < 0:
        msg = f"{section}.sample_size must be >= 0"
        raise ValueError(msg)
    return BehaviorMetric(
        name=reader.string("name"),
        value=reader.number("value"),
        model_label=model_label,
        category=reader.optional_string("category"),
        unit=reader.string("unit", default="ratio"),
        polarity=reader.literal("polarity", _POLARITY_CHOICES, default="neutral"),
        family=reader.string("family", default="behavior"),
        sample_size=sample_size,
        notes=reader.optional_string("notes"),
    )


def _parse_activation_findings(raw: object) -> list[ActivationFinding]:
    """Parse optional [[behavior_report.activation_findings]] entries."""
    rows = _array_of_tables(
        "[[behavior_report.activation_findings]]",
        raw,
        allow_empty=True,
    )
    return [
        _parse_activation_finding(
            row,
            f"[[behavior_report.activation_findings]][{index}]",
        )
        for index, row in enumerate(rows)
    ]


def _parse_activation_finding(
    raw: TomlDict,
    section: str,
) -> ActivationFinding:
    """Parse one activation finding table."""
    reader = SectionReader(section, raw)
    return ActivationFinding(
        name=reader.string("name"),
        summary=reader.string("summary"),
        layers=tuple(reader.int_list("layers", default=[])),
        score=reader.optional_number("score"),
        metric_name=reader.optional_string("metric_name"),
        direction_label=reader.optional_string("direction_label"),
        severity=reader.literal("severity", _SEVERITY_CHOICES, default="info"),
        evidence=tuple(reader.string_list("evidence", default=[])),
    )


def _parse_examples(raw: object) -> list[BehaviorExample]:
    """Parse optional [[behavior_report.examples]] entries."""
    rows = _array_of_tables("[[behavior_report.examples]]", raw, allow_empty=True)
    return [
        _parse_example(row, f"[[behavior_report.examples]][{index}]")
        for index, row in enumerate(rows)
    ]


def _parse_example(raw: TomlDict, section: str) -> BehaviorExample:
    """Parse one representative example table."""
    reader = SectionReader(section, raw)
    example_id = reader.optional_string("example_id")
    if example_id is None:
        example_id = reader.optional_string("id")
    if example_id is None:
        msg = f"{section}.example_id is required"
        raise ValueError(msg)
    return BehaviorExample(
        example_id=example_id,
        category=reader.string("category"),
        prompt=reader.string("prompt"),
        baseline_response=reader.optional_string("baseline_response"),
        candidate_response=reader.optional_string("candidate_response"),
        redaction=reader.literal(
            "redaction",
            _REDACTION_CHOICES,
            default="redacted",
        ),
        note=reader.optional_string("note"),
    )


def _parse_reproducibility(raw: object) -> ReproducibilityInfo | None:
    """Parse optional [behavior_report.reproducibility] table."""
    if raw is None:
        return None
    section = "[behavior_report.reproducibility]"
    reader = SectionReader(section, require_toml_table(section, raw))
    seed = reader.optional_integer("seed")
    if seed is not None and seed < 0:
        msg = f"{section}.seed must be >= 0"
        raise ValueError(msg)
    return ReproducibilityInfo(
        command=reader.string("command"),
        config_path=reader.optional_string("config_path"),
        code_revision=reader.optional_string("code_revision"),
        data_refs=tuple(reader.string_list("data_refs", default=[])),
        output_dir=reader.optional_string("output_dir"),
        seed=seed,
        notes=tuple(reader.string_list("notes", default=[])),
    )


def _array_of_tables(
    section: str,
    raw: object,
    *,
    allow_empty: bool,
) -> list[TomlDict]:
    """Read one optional or required TOML array of tables."""
    if raw is None:
        if allow_empty:
            return []
        msg = f"{section} must be a non-empty array of tables"
        raise ValueError(msg)
    if not isinstance(raw, list) or (not raw and not allow_empty):
        msg = f"{section} must be a non-empty array of tables"
        raise ValueError(msg)
    return [
        require_toml_table(f"{section}[{index}]", item)
        for index, item in enumerate(raw)
    ]
