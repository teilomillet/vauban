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
type InterventionKind = Literal[
    "activation_steering",
    "activation_ablation",
    "activation_addition",
    "weight_projection",
    "weight_arithmetic",
    "prompt_template",
    "sampling",
    "other",
]
type InterventionEffect = Literal[
    "increased",
    "decreased",
    "mixed",
    "no_observed_change",
    "inconclusive",
]
type InterventionPolarity = Literal[
    "positive",
    "negative",
    "bidirectional",
    "control",
    "other",
]
type TransformationKind = Literal[
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
]
type AccessLevel = Literal[
    "single_snapshot",
    "paired_outputs",
    "logprobs",
    "weights",
    "activations",
    "base_and_transformed",
]
type ClaimStrength = Literal[
    "behavioral_profile",
    "black_box_behavioral_diff",
    "distributional_diff",
    "weight_diff",
    "activation_diagnostic",
    "model_change_audit",
]
type ClaimStatus = Literal[
    "planned",
    "replicated",
    "partially_replicated",
    "not_replicated",
    "inconclusive",
    "extended",
]
type EvidenceKind = Literal[
    "suite",
    "trace",
    "metric",
    "run_report",
    "logprobs",
    "activation",
    "weights",
    "paper",
    "manual_review",
    "other",
]
type ExpectedBehavior = Literal[
    "refuse",
    "comply",
    "express_uncertainty",
    "ask_clarifying_question",
    "defer",
    "unknown",
]
type MetricIdentity = tuple[str, str, str]

_ACCESS_LEVEL_RANK: dict[AccessLevel, int] = {
    "single_snapshot": 0,
    "paired_outputs": 1,
    "logprobs": 2,
    "weights": 3,
    "activations": 4,
    "base_and_transformed": 5,
}
_CLAIM_STRENGTH_RANK: dict[ClaimStrength, int] = {
    "behavioral_profile": 0,
    "black_box_behavioral_diff": 1,
    "distributional_diff": 2,
    "weight_diff": 3,
    "activation_diagnostic": 4,
    "model_change_audit": 5,
}


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
class TransformationRef:
    """The model transformation being audited by a behavior report."""

    kind: TransformationKind
    summary: str
    before: str | None = None
    after: str | None = None
    method: str | None = None
    source_ref: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate transformation metadata."""
        _require_non_empty(self.summary, "summary")
        if self.before is not None:
            _require_non_empty(self.before, "before")
        if self.after is not None:
            _require_non_empty(self.after, "after")
        if self.method is not None:
            _require_non_empty(self.method, "method")
        if self.source_ref is not None:
            _require_non_empty(self.source_ref, "source_ref")
        _require_non_empty_items(self.notes, "notes", allow_empty=True)

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return _drop_none({
            "kind": self.kind,
            "summary": self.summary,
            "before": self.before,
            "after": self.after,
            "method": self.method,
            "source_ref": self.source_ref,
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        })


@dataclass(frozen=True, slots=True)
class AccessPolicy:
    """Epistemic boundary for what a behavior report can claim."""

    level: AccessLevel
    claim_strength: ClaimStrength
    available_evidence: tuple[str, ...] = field(default_factory=tuple)
    missing_evidence: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate access-policy evidence labels."""
        _require_non_empty_items(
            self.available_evidence,
            "available_evidence",
            allow_empty=True,
        )
        _require_non_empty_items(
            self.missing_evidence,
            "missing_evidence",
            allow_empty=True,
        )
        _require_non_empty_items(self.notes, "notes", allow_empty=True)

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "level": self.level,
            "claim_strength": self.claim_strength,
            "available_evidence": list(self.available_evidence),
            "missing_evidence": list(self.missing_evidence),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """One named evidence artifact used by claims in a report."""

    evidence_id: str
    kind: EvidenceKind
    path_or_url: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        """Validate evidence identity and optional reference text."""
        _require_non_empty(self.evidence_id, "evidence_id")
        if self.path_or_url is not None:
            _require_non_empty(self.path_or_url, "path_or_url")
        if self.description is not None:
            _require_non_empty(self.description, "description")

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return _drop_none({
            "id": self.evidence_id,
            "kind": self.kind,
            "path_or_url": self.path_or_url,
            "description": self.description,
        })


@dataclass(frozen=True, slots=True)
class BehaviorClaim:
    """A report claim with explicit access level, strength, and evidence."""

    claim_id: str
    statement: str
    strength: ClaimStrength
    access_level: AccessLevel
    status: ClaimStatus = "planned"
    evidence: tuple[str, ...] = field(default_factory=tuple)
    limitations: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate claim identity, statement, and evidence references."""
        _require_non_empty(self.claim_id, "claim_id")
        _require_non_empty(self.statement, "statement")
        _require_non_empty_items(self.evidence, "evidence", allow_empty=True)
        _require_non_empty_items(
            self.limitations,
            "limitations",
            allow_empty=True,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "id": self.claim_id,
            "statement": self.statement,
            "strength": self.strength,
            "access_level": self.access_level,
            "status": self.status,
            "evidence": list(self.evidence),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class ReproductionTarget:
    """A paper or external claim that this report calibrates or extends."""

    target_id: str
    title: str
    original_claim: str
    planned_extension: str
    source_url: str | None = None
    status: ClaimStatus = "planned"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate reproduction-target metadata."""
        _require_non_empty(self.target_id, "target_id")
        _require_non_empty(self.title, "title")
        _require_non_empty(self.original_claim, "original_claim")
        _require_non_empty(self.planned_extension, "planned_extension")
        if self.source_url is not None:
            _require_non_empty(self.source_url, "source_url")
        _require_non_empty_items(self.notes, "notes", allow_empty=True)

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return _drop_none({
            "id": self.target_id,
            "title": self.title,
            "source_url": self.source_url,
            "original_claim": self.original_claim,
            "planned_extension": self.planned_extension,
            "status": self.status,
            "notes": list(self.notes),
        })


@dataclass(frozen=True, slots=True)
class ReproductionResult:
    """Observed reproduction outcome for one declared target."""

    target_id: str
    status: ClaimStatus
    summary: str
    replicated_claims: tuple[str, ...] = field(default_factory=tuple)
    failed_claims: tuple[str, ...] = field(default_factory=tuple)
    extensions: tuple[str, ...] = field(default_factory=tuple)
    evidence: tuple[str, ...] = field(default_factory=tuple)
    limitations: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate reproduction result text and evidence labels."""
        _require_non_empty(self.target_id, "target_id")
        _require_non_empty(self.summary, "summary")
        _require_non_empty_items(
            self.replicated_claims,
            "replicated_claims",
            allow_empty=True,
        )
        _require_non_empty_items(
            self.failed_claims,
            "failed_claims",
            allow_empty=True,
        )
        _require_non_empty_items(self.extensions, "extensions", allow_empty=True)
        _require_non_empty_items(self.evidence, "evidence", allow_empty=True)
        _require_non_empty_items(
            self.limitations,
            "limitations",
            allow_empty=True,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "target_id": self.target_id,
            "status": self.status,
            "summary": self.summary,
            "replicated_claims": list(self.replicated_claims),
            "failed_claims": list(self.failed_claims),
            "extensions": list(self.extensions),
            "evidence": list(self.evidence),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class InterventionResult:
    """Observed behavior or activation outcome from a controlled intervention."""

    intervention_id: str
    kind: InterventionKind
    summary: str
    target: str
    effect: InterventionEffect = "inconclusive"
    polarity: InterventionPolarity = "other"
    layers: tuple[int, ...] = field(default_factory=tuple)
    strength: float | None = None
    baseline_condition: str | None = None
    intervention_condition: str | None = None
    behavior_metric: str | None = None
    activation_metric: str | None = None
    evidence: tuple[str, ...] = field(default_factory=tuple)
    limitations: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate intervention identity, target, and evidence labels."""
        _require_non_empty(self.intervention_id, "intervention_id")
        _require_non_empty(self.summary, "summary")
        _require_non_empty(self.target, "target")
        _require_non_negative_items(self.layers, "layers")
        if self.baseline_condition is not None:
            _require_non_empty(self.baseline_condition, "baseline_condition")
        if self.intervention_condition is not None:
            _require_non_empty(
                self.intervention_condition,
                "intervention_condition",
            )
        if self.behavior_metric is not None:
            _require_non_empty(self.behavior_metric, "behavior_metric")
        if self.activation_metric is not None:
            _require_non_empty(self.activation_metric, "activation_metric")
        _require_non_empty_items(self.evidence, "evidence", allow_empty=True)
        _require_non_empty_items(
            self.limitations,
            "limitations",
            allow_empty=True,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return _drop_none({
            "id": self.intervention_id,
            "kind": self.kind,
            "summary": self.summary,
            "target": self.target,
            "effect": self.effect,
            "polarity": self.polarity,
            "layers": list(self.layers),
            "strength": self.strength,
            "baseline_condition": self.baseline_condition,
            "intervention_condition": self.intervention_condition,
            "behavior_metric": self.behavior_metric,
            "activation_metric": self.activation_metric,
            "evidence": list(self.evidence),
            "limitations": list(self.limitations),
            "metadata": dict(self.metadata),
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
    transformation: TransformationRef | None = None
    access: AccessPolicy | None = None
    claims: tuple[BehaviorClaim, ...] = field(default_factory=tuple)
    evidence: tuple[EvidenceRef, ...] = field(default_factory=tuple)
    findings: tuple[str, ...] = field(default_factory=tuple)
    metrics: tuple[BehaviorMetric, ...] = field(default_factory=tuple)
    metric_deltas: tuple[BehaviorMetricDelta, ...] = field(default_factory=tuple)
    activation_findings: tuple[ActivationFinding, ...] = field(default_factory=tuple)
    examples: tuple[BehaviorExample, ...] = field(default_factory=tuple)
    reproduction_targets: tuple[ReproductionTarget, ...] = field(
        default_factory=tuple,
    )
    reproduction_results: tuple[ReproductionResult, ...] = field(
        default_factory=tuple,
    )
    intervention_results: tuple[InterventionResult, ...] = field(
        default_factory=tuple,
    )
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
        _reject_duplicate_strings(
            tuple(claim.claim_id for claim in self.claims),
            "claims.id",
        )
        _reject_duplicate_strings(
            tuple(evidence.evidence_id for evidence in self.evidence),
            "evidence.id",
        )
        _reject_duplicate_strings(
            tuple(target.target_id for target in self.reproduction_targets),
            "reproduction_targets.id",
        )
        _reject_duplicate_strings(
            tuple(result.target_id for result in self.reproduction_results),
            "reproduction_results.target_id",
        )
        _reject_duplicate_strings(
            tuple(
                result.intervention_id for result in self.intervention_results
            ),
            "intervention_results.id",
        )
        _validate_claim_evidence_refs(self.claims, self.evidence)
        _validate_intervention_result_evidence_refs(
            self.intervention_results,
            self.evidence,
        )
        _validate_claim_bounds(self.access, self.claims)
        _validate_reproduction_results(
            self.reproduction_targets,
            self.reproduction_results,
        )
        _require_non_empty_items(
            self.findings,
            "findings",
            allow_empty=True,
        )
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
            "transformation": (
                self.transformation.to_dict()
                if self.transformation is not None
                else None
            ),
            "access": self.access.to_dict() if self.access is not None else None,
            "claims": [claim.to_dict() for claim in self.claims],
            "evidence": [evidence.to_dict() for evidence in self.evidence],
            "findings": list(self.findings),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "metric_deltas": [
                delta.to_dict() for delta in self.metric_deltas
            ],
            "activation_findings": [
                finding.to_dict() for finding in self.activation_findings
            ],
            "examples": [example.to_dict() for example in self.examples],
            "reproduction_targets": [
                target.to_dict() for target in self.reproduction_targets
            ],
            "reproduction_results": [
                result.to_dict() for result in self.reproduction_results
            ],
            "intervention_results": [
                result.to_dict() for result in self.intervention_results
            ],
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
            f" deltas={len(self.metric_deltas)},"
            f" claims={len(self.claims)},"
            f" interventions={len(self.intervention_results)}"
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


def _validate_claim_evidence_refs(
    claims: tuple[BehaviorClaim, ...],
    evidence_refs: tuple[EvidenceRef, ...],
) -> None:
    """Require claim evidence IDs to exist when a report declares evidence."""
    if not evidence_refs:
        return
    known = {evidence.evidence_id for evidence in evidence_refs}
    for claim in claims:
        missing = tuple(ref for ref in claim.evidence if ref not in known)
        if missing:
            msg = (
                f"claim {claim.claim_id!r} references undeclared evidence"
                f" ids: {missing!r}"
            )
            raise ValueError(msg)


def _validate_intervention_result_evidence_refs(
    results: tuple[InterventionResult, ...],
    evidence_refs: tuple[EvidenceRef, ...],
) -> None:
    """Require intervention evidence IDs to exist when evidence is declared."""
    if not evidence_refs:
        return
    known = {evidence.evidence_id for evidence in evidence_refs}
    for result in results:
        missing = tuple(ref for ref in result.evidence if ref not in known)
        if missing:
            msg = (
                f"intervention result {result.intervention_id!r} references"
                f" undeclared evidence ids: {missing!r}"
            )
            raise ValueError(msg)


def _validate_claim_bounds(
    access: AccessPolicy | None,
    claims: tuple[BehaviorClaim, ...],
) -> None:
    """Reject claims stronger than the report's declared access policy."""
    if access is None:
        return
    max_strength = _CLAIM_STRENGTH_RANK[access.claim_strength]
    max_access = _ACCESS_LEVEL_RANK[access.level]
    for claim in claims:
        if _CLAIM_STRENGTH_RANK[claim.strength] > max_strength:
            msg = (
                f"claim {claim.claim_id!r} strength {claim.strength!r}"
                f" exceeds report claim_strength {access.claim_strength!r}"
            )
            raise ValueError(msg)
        if _ACCESS_LEVEL_RANK[claim.access_level] > max_access:
            msg = (
                f"claim {claim.claim_id!r} access_level {claim.access_level!r}"
                f" exceeds report access level {access.level!r}"
            )
            raise ValueError(msg)


def _validate_reproduction_results(
    targets: tuple[ReproductionTarget, ...],
    results: tuple[ReproductionResult, ...],
) -> None:
    """Require result target IDs to match declared targets when present."""
    if not targets:
        return
    known = {target.target_id for target in targets}
    for result in results:
        if result.target_id not in known:
            msg = (
                f"reproduction result references undeclared target"
                f" {result.target_id!r}"
            )
            raise ValueError(msg)


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
