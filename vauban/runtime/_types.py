# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Typed runtime primitives for Vauban backend execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

type RuntimeBackendName = Literal["mlx", "torch", "max"]
type SupportLevel = Literal["unsupported", "partial", "full"]
type DeviceKind = Literal["cpu", "gpu", "cuda", "mps"]
type TraceArtifactKind = Literal[
    "text",
    "tokens",
    "logits",
    "logprobs",
    "activation",
    "intervention_result",
    "weight_snapshot",
    "metric",
    "report_evidence",
    "profile",
    "other",
]
TRACE_ARTIFACT_KINDS: tuple[TraceArtifactKind, ...] = (
    "text",
    "tokens",
    "logits",
    "logprobs",
    "activation",
    "intervention_result",
    "weight_snapshot",
    "metric",
    "report_evidence",
    "profile",
    "other",
)
type RuntimeScalar = str | int | float | bool | None
type RuntimeValue = RuntimeScalar | list[RuntimeValue] | dict[str, RuntimeValue]
type ProfileSweepAxis = Literal["token_count", "batch_size", "queue_depth"]


@runtime_checkable
class TensorLike(Protocol):
    """Minimal structural tensor surface exposed by runtime traces."""

    @property
    def shape(self) -> tuple[int, ...]: ...


@runtime_checkable
class ActivationIntervention(Protocol):
    """Reversible activation intervention applied at one runtime layer."""

    name: str
    layer_index: int

    def apply(self, activation: TensorLike) -> TensorLike:
        """Return the intervened activation."""
        ...


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Declared evidence-producing support for a Vauban runtime backend."""

    name: RuntimeBackendName
    device_kinds: tuple[DeviceKind, ...]
    logits: SupportLevel
    logprobs: SupportLevel
    activations: SupportLevel
    interventions: SupportLevel
    kv_cache: SupportLevel
    weight_access: SupportLevel
    mutable_weights: SupportLevel

    def __post_init__(self) -> None:
        """Validate capability declarations."""
        if not self.device_kinds:
            msg = "device_kinds must not be empty"
            raise ValueError(msg)

    def supports(self, capability: str) -> bool:
        """Return whether a named capability is at least partially supported."""
        level = self.support_level(capability)
        return level != "unsupported"

    def supports_artifact(self, kind: TraceArtifactKind) -> bool:
        """Return whether this backend can emit one trace artifact kind."""
        return self.support_level_for_artifact(kind) != "unsupported"

    def support_level(self, capability: str) -> SupportLevel:
        """Return a named support level."""
        if capability == "logits":
            return self.logits
        if capability == "logprobs":
            return self.logprobs
        if capability == "activations":
            return self.activations
        if capability == "interventions":
            return self.interventions
        if capability == "kv_cache":
            return self.kv_cache
        if capability == "weight_access":
            return self.weight_access
        if capability == "mutable_weights":
            return self.mutable_weights
        msg = f"Unknown runtime capability: {capability!r}"
        raise ValueError(msg)

    def support_level_for_artifact(
        self,
        kind: TraceArtifactKind,
    ) -> SupportLevel:
        """Return support level for one trace artifact kind."""
        if kind == "tokens":
            return _adapter_support(self.name)
        if kind == "logits":
            return self.logits
        if kind == "logprobs":
            return self.logprobs
        if kind == "activation":
            return self.activations
        if kind == "intervention_result":
            return self.interventions
        if kind == "weight_snapshot":
            return self.weight_access
        if kind == "profile":
            return _adapter_support(self.name)
        return "unsupported"

    def supported_artifact_kinds(self) -> tuple[TraceArtifactKind, ...]:
        """Return trace artifact kinds this backend can emit at least partially."""
        return tuple(
            kind
            for kind in TRACE_ARTIFACT_KINDS
            if self.supports_artifact(kind)
        )

    def to_dict(self) -> dict[str, RuntimeValue]:
        """Serialize capability declarations for reports and diagnostics."""
        return {
            "name": self.name,
            "device_kinds": list(self.device_kinds),
            "logits": self.logits,
            "logprobs": self.logprobs,
            "activations": self.activations,
            "interventions": self.interventions,
            "kv_cache": self.kv_cache,
            "weight_access": self.weight_access,
            "mutable_weights": self.mutable_weights,
            "artifact_support": {
                kind: self.support_level_for_artifact(kind)
                for kind in TRACE_ARTIFACT_KINDS
            },
        }


@dataclass(frozen=True, slots=True)
class DeviceRef:
    """Runtime device metadata attached to traces."""

    kind: DeviceKind
    label: str

    def __post_init__(self) -> None:
        """Validate device metadata."""
        if not self.label.strip():
            msg = "device label must not be empty"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class StageProfile:
    """Timing and optional memory metadata for one runtime stage."""

    name: str
    duration_s: float
    device: DeviceRef | None = None
    memory_bytes: int | None = None
    metadata: dict[str, RuntimeValue] = field(default_factory=dict)
    batch_size: int | None = None
    token_count: int | None = None
    input_bytes: int | None = None
    output_bytes: int | None = None
    host_device_copies: int = 0
    sync_points: int = 0
    queue_depth: int | None = None

    def __post_init__(self) -> None:
        """Validate stage profile metadata."""
        if not self.name.strip():
            msg = "stage profile name must not be empty"
            raise ValueError(msg)
        if self.duration_s < 0.0:
            msg = "stage profile duration_s must be non-negative"
            raise ValueError(msg)
        if self.memory_bytes is not None and self.memory_bytes < 0:
            msg = "stage profile memory_bytes must be non-negative"
            raise ValueError(msg)
        if self.batch_size is not None and self.batch_size < 0:
            msg = "stage profile batch_size must be non-negative"
            raise ValueError(msg)
        if self.token_count is not None and self.token_count < 0:
            msg = "stage profile token_count must be non-negative"
            raise ValueError(msg)
        if self.input_bytes is not None and self.input_bytes < 0:
            msg = "stage profile input_bytes must be non-negative"
            raise ValueError(msg)
        if self.output_bytes is not None and self.output_bytes < 0:
            msg = "stage profile output_bytes must be non-negative"
            raise ValueError(msg)
        if self.host_device_copies < 0:
            msg = "stage profile host_device_copies must be non-negative"
            raise ValueError(msg)
        if self.sync_points < 0:
            msg = "stage profile sync_points must be non-negative"
            raise ValueError(msg)
        if self.queue_depth is not None and self.queue_depth < 0:
            msg = "stage profile queue_depth must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TraceRequest:
    """Request for an evidence-bearing runtime trace."""

    trace_id: str
    input_text: str | None = None
    prompt_ids: tuple[int, ...] = ()
    requested_artifacts: tuple[TraceArtifactKind, ...] = ()
    metadata: dict[str, RuntimeValue] = field(default_factory=dict)
    collect_layers: tuple[int, ...] = ()
    interventions: tuple[ActivationIntervention, ...] = ()
    apply_chat_template: bool = False
    return_logits: bool = True
    return_logprobs: bool = False

    def __post_init__(self) -> None:
        """Validate trace request metadata."""
        _require_non_empty(self.trace_id, "trace_id")
        _reject_duplicate_strings(self.requested_artifacts, "requested_artifacts")
        if self.input_text is not None:
            _require_non_empty(self.input_text, "input_text")
        if any(token_id < 0 for token_id in self.prompt_ids):
            msg = "prompt_ids must be non-negative"
            raise ValueError(msg)
        if any(layer < 0 for layer in self.collect_layers):
            msg = "collect_layers must be non-negative"
            raise ValueError(msg)
        if any(intervention.layer_index < 0 for intervention in self.interventions):
            msg = "intervention layer indexes must be non-negative"
            raise ValueError(msg)
        if any(not intervention.name.strip() for intervention in self.interventions):
            msg = "intervention names must not be empty"
            raise ValueError(msg)
        if self.return_logprobs and not self.return_logits:
            msg = "return_logprobs requires return_logits"
            raise ValueError(msg)
        if self.input_text is None and not self.prompt_ids:
            msg = "trace request must include input_text or prompt_ids"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TraceArtifact:
    """Summary of one value produced or consumed during an inference trace."""

    artifact_id: str
    kind: TraceArtifactKind
    producer_span_id: str | None = None
    shape: tuple[int, ...] | None = None
    metadata: dict[str, RuntimeValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate artifact identity and shape metadata."""
        _require_non_empty(self.artifact_id, "artifact_id")
        _validate_artifact_kind(self.kind)
        if self.producer_span_id is not None:
            _require_non_empty(self.producer_span_id, "producer_span_id")
        if self.shape is not None and any(dim < 0 for dim in self.shape):
            msg = "artifact shape dimensions must be non-negative"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, RuntimeValue]:
        """Serialize the artifact summary."""
        data: dict[str, RuntimeValue] = {
            "id": self.artifact_id,
            "kind": self.kind,
            "metadata": dict(self.metadata),
        }
        if self.producer_span_id is not None:
            data["producer_span_id"] = self.producer_span_id
        if self.shape is not None:
            data["shape"] = [int(dim) for dim in self.shape]
        return data


@dataclass(frozen=True, slots=True)
class TraceSpan:
    """One stage inside an evidence-bearing runtime trace."""

    span_id: str
    name: str
    profile: StageProfile | None = None
    input_artifact_ids: tuple[str, ...] = ()
    output_artifact_ids: tuple[str, ...] = ()
    metadata: dict[str, RuntimeValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate span identity and artifact references."""
        _require_non_empty(self.span_id, "span_id")
        _require_non_empty(self.name, "name")
        _require_non_empty_items(
            self.input_artifact_ids,
            "input_artifact_ids",
            allow_empty=True,
        )
        _require_non_empty_items(
            self.output_artifact_ids,
            "output_artifact_ids",
            allow_empty=True,
        )
        _reject_duplicate_strings(self.input_artifact_ids, "input_artifact_ids")
        _reject_duplicate_strings(self.output_artifact_ids, "output_artifact_ids")

    def to_dict(self) -> dict[str, RuntimeValue]:
        """Serialize the trace span."""
        data: dict[str, RuntimeValue] = {
            "id": self.span_id,
            "name": self.name,
            "input_artifacts": list(self.input_artifact_ids),
            "output_artifacts": list(self.output_artifact_ids),
            "metadata": dict(self.metadata),
        }
        if self.profile is not None:
            data["profile"] = _stage_profile_to_dict(self.profile)
        return data


@dataclass(frozen=True, slots=True)
class Trace:
    """Evidence-bearing record of one runtime execution."""

    trace_id: str
    device: DeviceRef
    spans: tuple[TraceSpan, ...]
    artifacts: tuple[TraceArtifact, ...]
    metadata: dict[str, RuntimeValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate trace graph references."""
        _require_non_empty(self.trace_id, "trace_id")
        _reject_duplicate_strings(
            tuple(span.span_id for span in self.spans),
            "spans.id",
        )
        _reject_duplicate_strings(
            tuple(artifact.artifact_id for artifact in self.artifacts),
            "artifacts.id",
        )
        span_ids = {span.span_id for span in self.spans}
        artifacts_by_id = {
            artifact.artifact_id: artifact for artifact in self.artifacts
        }
        artifact_ids = set(artifacts_by_id)
        for artifact in self.artifacts:
            if (
                artifact.producer_span_id is not None
                and artifact.producer_span_id not in span_ids
            ):
                msg = (
                    f"artifact {artifact.artifact_id!r} references unknown"
                    f" producer span {artifact.producer_span_id!r}"
                )
                raise ValueError(msg)
        for span in self.spans:
            _validate_artifact_refs(
                span.input_artifact_ids,
                artifact_ids,
                span.span_id,
                "input",
            )
            _validate_artifact_refs(
                span.output_artifact_ids,
                artifact_ids,
                span.span_id,
                "output",
            )
            _validate_span_outputs_match_producers(
                span,
                artifacts_by_id,
            )

    def artifacts_by_kind(
        self,
        kind: TraceArtifactKind,
    ) -> tuple[TraceArtifact, ...]:
        """Return artifacts of one kind."""
        return tuple(artifact for artifact in self.artifacts if artifact.kind == kind)

    def to_dict(self) -> dict[str, RuntimeValue]:
        """Serialize the trace summary."""
        return {
            "id": self.trace_id,
            "device": {
                "kind": self.device.kind,
                "label": self.device.label,
            },
            "spans": [span.to_dict() for span in self.spans],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TraceProfileSummary:
    """USL-ready aggregate profile over trace spans."""

    total_duration_s: float
    profiled_spans: int
    token_count: int | None = None
    batch_size: int | None = None
    tokens_per_second: float | None = None
    peak_memory_bytes: int | None = None
    total_input_bytes: int = 0
    total_output_bytes: int = 0
    host_device_copies: int = 0
    sync_points: int = 0
    max_queue_depth: int | None = None

    def __post_init__(self) -> None:
        """Validate aggregate profile counters."""
        if self.total_duration_s < 0.0:
            msg = "total_duration_s must be non-negative"
            raise ValueError(msg)
        if self.profiled_spans < 0:
            msg = "profiled_spans must be non-negative"
            raise ValueError(msg)
        if self.token_count is not None and self.token_count < 0:
            msg = "token_count must be non-negative"
            raise ValueError(msg)
        if self.batch_size is not None and self.batch_size < 0:
            msg = "batch_size must be non-negative"
            raise ValueError(msg)
        if self.tokens_per_second is not None and self.tokens_per_second < 0.0:
            msg = "tokens_per_second must be non-negative"
            raise ValueError(msg)
        if self.peak_memory_bytes is not None and self.peak_memory_bytes < 0:
            msg = "peak_memory_bytes must be non-negative"
            raise ValueError(msg)
        if self.total_input_bytes < 0:
            msg = "total_input_bytes must be non-negative"
            raise ValueError(msg)
        if self.total_output_bytes < 0:
            msg = "total_output_bytes must be non-negative"
            raise ValueError(msg)
        if self.host_device_copies < 0:
            msg = "host_device_copies must be non-negative"
            raise ValueError(msg)
        if self.sync_points < 0:
            msg = "sync_points must be non-negative"
            raise ValueError(msg)
        if self.max_queue_depth is not None and self.max_queue_depth < 0:
            msg = "max_queue_depth must be non-negative"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, RuntimeValue]:
        """Serialize aggregate trace profile metrics."""
        return {
            "total_duration_s": self.total_duration_s,
            "profiled_spans": self.profiled_spans,
            "token_count": self.token_count,
            "batch_size": self.batch_size,
            "tokens_per_second": self.tokens_per_second,
            "peak_memory_bytes": self.peak_memory_bytes,
            "total_input_bytes": self.total_input_bytes,
            "total_output_bytes": self.total_output_bytes,
            "host_device_copies": self.host_device_copies,
            "sync_points": self.sync_points,
            "max_queue_depth": self.max_queue_depth,
        }


@dataclass(frozen=True, slots=True)
class TraceProfileSweepPoint:
    """One controlled profile sweep point over comparable traces."""

    axis_value: int
    trace_ids: tuple[str, ...]
    samples: int
    mean_total_duration_s: float
    mean_tokens_per_second: float | None = None
    peak_memory_bytes: int | None = None
    total_host_device_copies: int = 0
    total_sync_points: int = 0

    def __post_init__(self) -> None:
        """Validate sweep point counters."""
        if self.axis_value < 0:
            msg = "axis_value must be non-negative"
            raise ValueError(msg)
        if self.samples < 1:
            msg = "samples must be at least one"
            raise ValueError(msg)
        _require_non_empty_items(self.trace_ids, "trace_ids")
        _reject_duplicate_strings(self.trace_ids, "trace_ids")
        if self.mean_total_duration_s < 0.0:
            msg = "mean_total_duration_s must be non-negative"
            raise ValueError(msg)
        if (
            self.mean_tokens_per_second is not None
            and self.mean_tokens_per_second < 0.0
        ):
            msg = "mean_tokens_per_second must be non-negative"
            raise ValueError(msg)
        if self.peak_memory_bytes is not None and self.peak_memory_bytes < 0:
            msg = "peak_memory_bytes must be non-negative"
            raise ValueError(msg)
        if self.total_host_device_copies < 0:
            msg = "total_host_device_copies must be non-negative"
            raise ValueError(msg)
        if self.total_sync_points < 0:
            msg = "total_sync_points must be non-negative"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, RuntimeValue]:
        """Serialize the sweep point."""
        return {
            "axis_value": self.axis_value,
            "trace_ids": list(self.trace_ids),
            "samples": self.samples,
            "mean_total_duration_s": self.mean_total_duration_s,
            "mean_tokens_per_second": self.mean_tokens_per_second,
            "peak_memory_bytes": self.peak_memory_bytes,
            "total_host_device_copies": self.total_host_device_copies,
            "total_sync_points": self.total_sync_points,
        }


@dataclass(frozen=True, slots=True)
class TraceProfileSweep:
    """Controlled USL-ready profile sweep over stable trace coverage."""

    sweep_id: str
    axis: ProfileSweepAxis
    artifact_kinds: tuple[TraceArtifactKind, ...]
    points: tuple[TraceProfileSweepPoint, ...]
    metadata: dict[str, RuntimeValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate sweep identity and coverage metadata."""
        _require_non_empty(self.sweep_id, "sweep_id")
        if not self.artifact_kinds:
            msg = "artifact_kinds must not be empty"
            raise ValueError(msg)
        for kind in self.artifact_kinds:
            _validate_artifact_kind(kind)
        if not self.points:
            msg = "points must not be empty"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, RuntimeValue]:
        """Serialize the profile sweep without claiming a USL fit."""
        return {
            "id": self.sweep_id,
            "axis": self.axis,
            "artifact_kinds": list(self.artifact_kinds),
            "points": [point.to_dict() for point in self.points],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ModelRef:
    """Reference to a model runtime should load."""

    model_path: str
    revision: str | None = None
    metadata: dict[str, RuntimeValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate model reference fields."""
        if not self.model_path.strip():
            msg = "model_path must not be empty"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """Loaded backend-specific model handle with honest capabilities."""

    ref: ModelRef
    backend: RuntimeBackendName
    capabilities: BackendCapabilities
    model: object
    tokenizer: object | None = None
    metadata: dict[str, RuntimeValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate loaded model metadata."""
        if self.backend != self.capabilities.name:
            msg = (
                "loaded model backend must match capability backend: "
                f"{self.backend!r} != {self.capabilities.name!r}"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TokenizeRequest:
    """Request to tokenize text through a loaded model tokenizer."""

    text: str
    apply_chat_template: bool = False

    def __post_init__(self) -> None:
        """Validate tokenize request fields."""
        if not self.text:
            msg = "text must not be empty"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TokenizedPrompt:
    """Tokenized prompt emitted by a runtime tokenizer stage."""

    token_ids: tuple[int, ...]
    text: str
    profile: tuple[StageProfile, ...] = ()

    def __post_init__(self) -> None:
        """Validate tokenized prompt fields."""
        if not self.token_ids:
            msg = "token_ids must not be empty"
            raise ValueError(msg)
        if any(token_id < 0 for token_id in self.token_ids):
            msg = "token_ids must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ForwardRequest:
    """Runtime forward-pass request."""

    prompt_ids: tuple[int, ...]
    collect_layers: tuple[int, ...] = ()
    interventions: tuple[ActivationIntervention, ...] = ()
    return_logits: bool = True
    return_logprobs: bool = False

    def __post_init__(self) -> None:
        """Validate forward request fields."""
        if not self.prompt_ids:
            msg = "prompt_ids must not be empty"
            raise ValueError(msg)
        if any(token_id < 0 for token_id in self.prompt_ids):
            msg = "prompt_ids must be non-negative"
            raise ValueError(msg)
        if any(layer < 0 for layer in self.collect_layers):
            msg = "collect_layers must be non-negative"
            raise ValueError(msg)
        if any(intervention.layer_index < 0 for intervention in self.interventions):
            msg = "intervention layer indexes must be non-negative"
            raise ValueError(msg)
        if any(not intervention.name.strip() for intervention in self.interventions):
            msg = "intervention names must not be empty"
            raise ValueError(msg)
        if self.return_logprobs and not self.return_logits:
            msg = "return_logprobs requires return_logits"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class InterventionRecord:
    """Metadata for one activation intervention applied during forward."""

    name: str
    layer_index: int
    metadata: dict[str, RuntimeValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate intervention metadata."""
        if not self.name.strip():
            msg = "intervention record name must not be empty"
            raise ValueError(msg)
        if self.layer_index < 0:
            msg = "intervention record layer_index must be non-negative"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, RuntimeValue]:
        """Serialize intervention metadata."""
        data: dict[str, RuntimeValue] = {
            "name": self.name,
            "layer_index": self.layer_index,
        }
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


@dataclass(frozen=True, slots=True)
class ForwardTrace:
    """Observed tensors and metadata from one forward pass."""

    logits: TensorLike | None
    logprobs: TensorLike | None
    activations: dict[int, TensorLike]
    device: DeviceRef
    interventions: tuple[InterventionRecord, ...] = ()
    profile: tuple[StageProfile, ...] = ()

    def __post_init__(self) -> None:
        """Validate forward trace consistency."""
        if any(layer < 0 for layer in self.activations):
            msg = "activation layer indexes must be non-negative"
            raise ValueError(msg)
        if any(record.layer_index < 0 for record in self.interventions):
            msg = "intervention layer indexes must be non-negative"
            raise ValueError(msg)
        if self.logprobs is not None and self.logits is None:
            msg = "logprobs require logits in the same trace"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class RuntimeTraceResult:
    """Runtime trace plus the concrete forward evidence that produced it."""

    request: TraceRequest
    trace: Trace
    forward: ForwardTrace
    tokenized: TokenizedPrompt | None = None

    def __post_init__(self) -> None:
        """Validate trace result consistency."""
        if self.trace.trace_id != self.request.trace_id:
            msg = "runtime trace id must match request trace_id"
            raise ValueError(msg)


def _stage_profile_to_dict(profile: StageProfile) -> dict[str, RuntimeValue]:
    """Serialize a stage profile for trace spans."""
    device: dict[str, RuntimeValue] | None = None
    if profile.device is not None:
        device = {
            "kind": profile.device.kind,
            "label": profile.device.label,
        }
    return {
        "name": profile.name,
        "duration_s": profile.duration_s,
        "device": device,
        "memory_bytes": profile.memory_bytes,
        "batch_size": profile.batch_size,
        "token_count": profile.token_count,
        "input_bytes": profile.input_bytes,
        "output_bytes": profile.output_bytes,
        "host_device_copies": profile.host_device_copies,
        "sync_points": profile.sync_points,
        "queue_depth": profile.queue_depth,
        "metadata": dict(profile.metadata),
    }


def _adapter_support(name: RuntimeBackendName) -> SupportLevel:
    """Return support for artifacts requiring an implemented adapter."""
    if name == "max":
        return "unsupported"
    return "full"


def _validate_artifact_refs(
    refs: tuple[str, ...],
    artifact_ids: set[str],
    span_id: str,
    direction: str,
) -> None:
    """Reject span artifact references that do not exist in the trace."""
    for ref in refs:
        if ref not in artifact_ids:
            msg = f"span {span_id!r} references unknown {direction} artifact {ref!r}"
            raise ValueError(msg)


def _validate_span_outputs_match_producers(
    span: TraceSpan,
    artifacts: dict[str, TraceArtifact],
) -> None:
    """Reject inconsistent span-output and artifact-producer links."""
    for artifact_id in span.output_artifact_ids:
        artifact = artifacts[artifact_id]
        if artifact.producer_span_id is None:
            continue
        if artifact.producer_span_id != span.span_id:
            msg = (
                f"span {span.span_id!r} outputs artifact {artifact_id!r}, but"
                f" artifact producer is {artifact.producer_span_id!r}"
            )
            raise ValueError(msg)


def _validate_artifact_kind(kind: str) -> None:
    """Reject artifact kinds outside the stable trace vocabulary."""
    if kind not in TRACE_ARTIFACT_KINDS:
        msg = f"unknown trace artifact kind: {kind!r}"
        raise ValueError(msg)


def _require_non_empty(value: str, field_name: str) -> None:
    """Reject empty strings in trace metadata."""
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


def _reject_duplicate_strings(values: tuple[str, ...], field_name: str) -> None:
    """Reject duplicate string values."""
    seen: set[str] = set()
    for value in values:
        if value in seen:
            msg = f"{field_name} contains duplicate value {value!r}"
            raise ValueError(msg)
        seen.add(value)
