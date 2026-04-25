# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Typed primitives for Vauban model behavior change reports.

This module intentionally contains no model runtime or TOML parsing logic. It is
the shared vocabulary used by future suite, diff, and report code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type ModelRole = Literal["baseline", "candidate", "reference", "intervention"]
type MetricPolarity = Literal["higher_is_better", "lower_is_better", "neutral"]
type MetricQuality = Literal["improved", "regressed", "unchanged", "neutral"]
type ExampleRedaction = Literal["safe", "redacted", "omitted"]
type FindingSeverity = Literal["info", "low", "medium", "high", "critical"]
type ChatRole = Literal["system", "user", "assistant"]
type ExpectedBehavior = Literal[
    "refuse",
    "comply",
    "express_uncertainty",
    "ask_clarifying_question",
    "defer",
    "unknown",
]
type MetricIdentity = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class ReportModelRef:
    """Model-side metadata for a behavior report comparison."""

    label: str
    model_path: str
    role: ModelRole = "candidate"
    checkpoint: str | None = None
    adapter_path: str | None = None
    prompt_template: str | None = None
    quantization: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate required model metadata."""
        _require_non_empty(self.label, "label")
        _require_non_empty(self.model_path, "model_path")

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return _drop_none({
            "label": self.label,
            "model_path": self.model_path,
            "role": self.role,
            "checkpoint": self.checkpoint,
            "adapter_path": self.adapter_path,
            "prompt_template": self.prompt_template,
            "quantization": self.quantization,
            "metadata": dict(self.metadata),
        })


@dataclass(frozen=True, slots=True)
class BehaviorSuiteRef:
    """Metadata describing the behavior suite used for a report."""

    name: str
    description: str
    categories: tuple[str, ...]
    metrics: tuple[str, ...]
    version: str | None = None
    source: str | None = None
    safety_policy: str = "aggregate_or_redacted_examples"

    def __post_init__(self) -> None:
        """Validate suite identity, categories, and metric names."""
        _require_non_empty(self.name, "name")
        _require_non_empty(self.description, "description")
        _require_non_empty_items(self.categories, "categories")
        _require_non_empty_items(self.metrics, "metrics")
        _require_non_empty(self.safety_policy, "safety_policy")

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return _drop_none({
            "name": self.name,
            "description": self.description,
            "categories": list(self.categories),
            "metrics": list(self.metrics),
            "version": self.version,
            "source": self.source,
            "safety_policy": self.safety_policy,
        })


@dataclass(frozen=True, slots=True)
class BehaviorMetricSpec:
    """Metric definition declared by a behavior suite."""

    name: str
    description: str
    polarity: MetricPolarity = "neutral"
    unit: str = "ratio"
    family: str = "behavior"

    def __post_init__(self) -> None:
        """Validate metric spec identity and description."""
        _require_non_empty(self.name, "name")
        _require_non_empty(self.description, "description")
        _require_non_empty(self.unit, "unit")
        _require_non_empty(self.family, "family")

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "polarity": self.polarity,
            "unit": self.unit,
            "family": self.family,
        }


@dataclass(frozen=True, slots=True)
class BehaviorChatMessage:
    """One typed chat message inside a behavior prompt."""

    role: ChatRole
    content: str

    def __post_init__(self) -> None:
        """Validate message content."""
        _require_non_empty(self.content, "content")

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "role": self.role,
            "content": self.content,
        }


@dataclass(frozen=True, slots=True)
class BehaviorPrompt:
    """One prompt record in a behavior suite."""

    prompt_id: str
    category: str
    prompt: str
    expected_behavior: ExpectedBehavior = "unknown"
    label: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    messages: tuple[BehaviorChatMessage, ...] = field(default_factory=tuple)
    source_ref: str | None = None
    redaction: ExampleRedaction = "safe"
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate prompt identity and metadata."""
        _require_non_empty(self.prompt_id, "prompt_id")
        _require_non_empty(self.category, "category")
        _require_non_empty(self.prompt, "prompt")
        _require_non_empty_items(self.tags, "tags", allow_empty=True)
        if self.label is not None:
            _require_non_empty(self.label, "label")
        if self.source_ref is not None:
            _require_non_empty(self.source_ref, "source_ref")

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return _drop_none({
            "prompt_id": self.prompt_id,
            "category": self.category,
            "prompt": self.prompt,
            "expected_behavior": self.expected_behavior,
            "label": self.label,
            "tags": list(self.tags),
            "messages": [message.to_dict() for message in self.messages],
            "source_ref": self.source_ref,
            "redaction": self.redaction,
            "metadata": dict(self.metadata),
        })


@dataclass(frozen=True, slots=True)
class BehaviorSuite:
    """A complete behavior suite definition before execution."""

    name: str
    description: str
    prompts: tuple[BehaviorPrompt, ...]
    metric_specs: tuple[BehaviorMetricSpec, ...]
    version: str | None = None
    source: str | None = None
    safety_policy: str = "aggregate_or_redacted_examples"

    def __post_init__(self) -> None:
        """Validate suite identity, prompts, and metric specs."""
        _require_non_empty(self.name, "name")
        _require_non_empty(self.description, "description")
        if not self.prompts:
            msg = "prompts must contain at least one item"
            raise ValueError(msg)
        if not self.metric_specs:
            msg = "metric_specs must contain at least one item"
            raise ValueError(msg)
        _require_non_empty(self.safety_policy, "safety_policy")
        _reject_duplicate_strings(
            tuple(prompt.prompt_id for prompt in self.prompts),
            "prompt_id",
        )
        _reject_duplicate_strings(
            tuple(metric.name for metric in self.metric_specs),
            "metric_specs.name",
        )

    @property
    def categories(self) -> tuple[str, ...]:
        """Return sorted unique prompt categories in the suite."""
        return tuple(sorted({prompt.category for prompt in self.prompts}))

    @property
    def metric_names(self) -> tuple[str, ...]:
        """Return metric names in declaration order."""
        return tuple(metric.name for metric in self.metric_specs)

    def ref(self) -> BehaviorSuiteRef:
        """Return metadata-only suite reference for report headers."""
        return BehaviorSuiteRef(
            name=self.name,
            description=self.description,
            categories=self.categories,
            metrics=self.metric_names,
            version=self.version,
            source=self.source,
            safety_policy=self.safety_policy,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return _drop_none({
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "source": self.source,
            "safety_policy": self.safety_policy,
            "prompts": [prompt.to_dict() for prompt in self.prompts],
            "metric_specs": [
                metric_spec.to_dict() for metric_spec in self.metric_specs
            ],
        })


@dataclass(frozen=True, slots=True)
class BehaviorMetric:
    """One scalar behavior or activation metric in a report."""

    name: str
    value: float
    model_label: str
    category: str | None = None
    unit: str = "ratio"
    polarity: MetricPolarity = "neutral"
    family: str = "behavior"
    sample_size: int | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        """Validate metric identity and sample-size metadata."""
        _require_non_empty(self.name, "name")
        _require_non_empty(self.model_label, "model_label")
        _require_non_empty(self.unit, "unit")
        _require_non_empty(self.family, "family")
        if self.category is not None:
            _require_non_empty(self.category, "category")
        if self.sample_size is not None and self.sample_size < 0:
            msg = "sample_size must be >= 0 when set"
            raise ValueError(msg)

    @property
    def identity(self) -> MetricIdentity:
        """Return the stable comparison key for this metric."""
        return (self.name, self.category or "", self.unit)

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return _drop_none({
            "name": self.name,
            "value": self.value,
            "model_label": self.model_label,
            "category": self.category,
            "unit": self.unit,
            "polarity": self.polarity,
            "family": self.family,
            "sample_size": self.sample_size,
            "notes": self.notes,
        })


@dataclass(frozen=True, slots=True)
class BehaviorMetricDelta:
    """A polarity-aware metric comparison between two model states."""

    name: str
    baseline_label: str
    candidate_label: str
    value_baseline: float
    value_candidate: float
    delta: float
    polarity: MetricPolarity
    category: str | None = None
    unit: str = "ratio"
    family: str = "behavior"

    @classmethod
    def from_metrics(
        cls,
        baseline: BehaviorMetric,
        candidate: BehaviorMetric,
    ) -> BehaviorMetricDelta:
        """Build a delta from two metrics with the same identity and polarity."""
        if baseline.identity != candidate.identity:
            msg = (
                "cannot compare behavior metrics with different identities:"
                f" {baseline.identity!r} != {candidate.identity!r}"
            )
            raise ValueError(msg)
        if baseline.polarity != candidate.polarity:
            msg = (
                "cannot compare behavior metrics with different polarity:"
                f" {baseline.polarity!r} != {candidate.polarity!r}"
            )
            raise ValueError(msg)
        if baseline.family != candidate.family:
            msg = (
                "cannot compare behavior metrics from different families:"
                f" {baseline.family!r} != {candidate.family!r}"
            )
            raise ValueError(msg)

        return cls(
            name=baseline.name,
            baseline_label=baseline.model_label,
            candidate_label=candidate.model_label,
            value_baseline=baseline.value,
            value_candidate=candidate.value,
            delta=candidate.value - baseline.value,
            polarity=baseline.polarity,
            category=baseline.category,
            unit=baseline.unit,
            family=baseline.family,
        )

    @property
    def percent_change(self) -> float | None:
        """Return percent change relative to baseline, if baseline is nonzero."""
        if self.value_baseline == 0.0:
            return None
        return (self.delta / abs(self.value_baseline)) * 100.0

    @property
    def quality(self) -> MetricQuality:
        """Return the polarity-aware interpretation of the metric delta."""
        if self.delta == 0.0:
            return "unchanged"
        if self.polarity == "neutral":
            return "neutral"
        if self.polarity == "higher_is_better":
            return "improved" if self.delta > 0.0 else "regressed"
        return "improved" if self.delta < 0.0 else "regressed"

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return _drop_none({
            "name": self.name,
            "baseline_label": self.baseline_label,
            "candidate_label": self.candidate_label,
            "value_baseline": self.value_baseline,
            "value_candidate": self.value_candidate,
            "delta": self.delta,
            "percent_change": self.percent_change,
            "quality": self.quality,
            "polarity": self.polarity,
            "category": self.category,
            "unit": self.unit,
            "family": self.family,
        })


@dataclass(frozen=True, slots=True)
class ActivationFinding:
    """One internal diagnostic finding connected to observed behavior."""

    name: str
    summary: str
    layers: tuple[int, ...] = field(default_factory=tuple)
    score: float | None = None
    metric_name: str | None = None
    direction_label: str | None = None
    severity: FindingSeverity = "info"
    evidence: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate finding text, layers, and evidence."""
        _require_non_empty(self.name, "name")
        _require_non_empty(self.summary, "summary")
        _require_non_negative_items(self.layers, "layers")
        _require_non_empty_items(self.evidence, "evidence", allow_empty=True)

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return _drop_none({
            "name": self.name,
            "summary": self.summary,
            "layers": list(self.layers),
            "score": self.score,
            "metric_name": self.metric_name,
            "direction_label": self.direction_label,
            "severity": self.severity,
            "evidence": list(self.evidence),
        })


@dataclass(frozen=True, slots=True)
class BehaviorExample:
    """A safe or redacted representative example for a behavior report."""

    example_id: str
    category: str
    prompt: str
    baseline_response: str | None = None
    candidate_response: str | None = None
    redaction: ExampleRedaction = "redacted"
    note: str | None = None

    def __post_init__(self) -> None:
        """Validate representative example metadata."""
        _require_non_empty(self.example_id, "example_id")
        _require_non_empty(self.category, "category")
        _require_non_empty(self.prompt, "prompt")

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return _drop_none({
            "example_id": self.example_id,
            "category": self.category,
            "prompt": self.prompt,
            "baseline_response": self.baseline_response,
            "candidate_response": self.candidate_response,
            "redaction": self.redaction,
            "note": self.note,
        })


@dataclass(frozen=True, slots=True)
class ReproducibilityInfo:
    """Provenance needed to reproduce a behavior report."""

    command: str
    config_path: str | None = None
    code_revision: str | None = None
    data_refs: tuple[str, ...] = field(default_factory=tuple)
    output_dir: str | None = None
    seed: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate reproducibility command and references."""
        _require_non_empty(self.command, "command")
        _require_non_empty_items(self.data_refs, "data_refs", allow_empty=True)
        _require_non_empty_items(self.notes, "notes", allow_empty=True)

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return _drop_none({
            "command": self.command,
            "config_path": self.config_path,
            "code_revision": self.code_revision,
            "data_refs": list(self.data_refs),
            "output_dir": self.output_dir,
            "seed": self.seed,
            "notes": list(self.notes),
        })


@dataclass(frozen=True, slots=True)
class BehaviorReport:
    """Top-level typed artifact for a model behavior change report."""

    title: str
    baseline: ReportModelRef
    candidate: ReportModelRef
    suite: BehaviorSuiteRef
    target_change: str | None = None
    metrics: tuple[BehaviorMetric, ...] = field(default_factory=tuple)
    metric_deltas: tuple[BehaviorMetricDelta, ...] = field(default_factory=tuple)
    activation_findings: tuple[ActivationFinding, ...] = field(default_factory=tuple)
    examples: tuple[BehaviorExample, ...] = field(default_factory=tuple)
    limitations: tuple[str, ...] = field(default_factory=tuple)
    recommendation: str | None = None
    reproducibility: ReproducibilityInfo | None = None
    report_version: str = "behavior_report_v1"

    def __post_init__(self) -> None:
        """Validate report title, version, and limitations."""
        _require_non_empty(self.title, "title")
        _require_non_empty(self.report_version, "report_version")
        if self.target_change is not None:
            _require_non_empty(self.target_change, "target_change")
        if self.recommendation is not None:
            _require_non_empty(self.recommendation, "recommendation")
        _require_non_empty_items(
            self.limitations,
            "limitations",
            allow_empty=True,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize the behavior report to a JSON-compatible dictionary."""
        return _drop_none({
            "report_version": self.report_version,
            "title": self.title,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "suite": self.suite.to_dict(),
            "target_change": self.target_change,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "metric_deltas": [
                delta.to_dict() for delta in self.metric_deltas
            ],
            "activation_findings": [
                finding.to_dict() for finding in self.activation_findings
            ],
            "examples": [example.to_dict() for example in self.examples],
            "limitations": list(self.limitations),
            "recommendation": self.recommendation,
            "reproducibility": (
                self.reproducibility.to_dict()
                if self.reproducibility is not None
                else None
            ),
        })

    def summary(self) -> str:
        """Return a short human-readable report summary."""
        return (
            f"BehaviorReport: {self.baseline.label} vs {self.candidate.label},"
            f" suite={self.suite.name},"
            f" metrics={len(self.metrics)},"
            f" deltas={len(self.metric_deltas)}"
        )


def compare_behavior_metrics(
    baseline_metrics: tuple[BehaviorMetric, ...],
    candidate_metrics: tuple[BehaviorMetric, ...],
) -> tuple[BehaviorMetricDelta, ...]:
    """Compare matching behavior metrics from two model states.

    Metrics are matched by ``(name, category, unit)``. Metrics present on only
    one side are ignored because absence is an evidence issue, not a numeric
    delta.
    """
    baseline_by_key = _index_metrics(baseline_metrics, "baseline_metrics")
    candidate_by_key = _index_metrics(candidate_metrics, "candidate_metrics")

    deltas: list[BehaviorMetricDelta] = []
    for key in sorted(baseline_by_key):
        candidate = candidate_by_key.get(key)
        if candidate is None:
            continue
        deltas.append(
            BehaviorMetricDelta.from_metrics(
                baseline_by_key[key],
                candidate,
            ),
        )
    return tuple(deltas)


def _index_metrics(
    metrics: tuple[BehaviorMetric, ...],
    field_name: str,
) -> dict[MetricIdentity, BehaviorMetric]:
    """Index metrics by identity and reject ambiguous duplicates."""
    indexed: dict[MetricIdentity, BehaviorMetric] = {}
    for metric in metrics:
        if metric.identity in indexed:
            msg = (
                f"{field_name} contains duplicate metric identity"
                f" {metric.identity!r}"
            )
            raise ValueError(msg)
        indexed[metric.identity] = metric
    return indexed


def _drop_none(data: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Return a copy without keys whose value is None."""
    return {key: value for key, value in data.items() if value is not None}


def _require_non_empty(value: str, field_name: str) -> None:
    """Raise ValueError when a required string is empty."""
    if not value.strip():
        msg = f"{field_name} must not be empty"
        raise ValueError(msg)


def _require_non_empty_items(
    values: tuple[str, ...],
    field_name: str,
    *,
    allow_empty: bool = False,
) -> None:
    """Validate a tuple of non-empty strings."""
    if not values and not allow_empty:
        msg = f"{field_name} must contain at least one item"
        raise ValueError(msg)
    for value in values:
        _require_non_empty(value, field_name)


def _require_non_negative_items(
    values: tuple[int, ...],
    field_name: str,
) -> None:
    """Validate a tuple of non-negative integers."""
    for value in values:
        if value < 0:
            msg = f"{field_name} must contain non-negative layer indices"
            raise ValueError(msg)


def _reject_duplicate_strings(values: tuple[str, ...], field_name: str) -> None:
    """Reject duplicate string values in one suite identity field."""
    seen: set[str] = set()
    for value in values:
        if value in seen:
            msg = f"{field_name} contains duplicate value {value!r}"
            raise ValueError(msg)
        seen.add(value)
