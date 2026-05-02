# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Runtime evidence serialization for regression-safe report plumbing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vauban.behavior import AccessPolicy, EvidenceRef, JsonValue
from vauban.runtime._capabilities import access_policy_for_trace
from vauban.runtime._types import (
    TRACE_ARTIFACT_KINDS,
    ProfileSweepAxis,
    Trace,
    TraceArtifact,
    TraceArtifactKind,
    TraceProfileSummary,
    TraceProfileSweep,
    TraceProfileSweepPoint,
    TraceSpan,
)

if TYPE_CHECKING:
    from vauban.runtime._types import (
        BackendCapabilities,
        ForwardTrace,
        RuntimeValue,
        StageProfile,
        TensorLike,
        TokenizedPrompt,
    )


@dataclass(frozen=True, slots=True)
class RuntimeReportEvidence:
    """Report-ready runtime evidence with an explicit access policy."""

    access: AccessPolicy
    evidence: tuple[EvidenceRef, ...]
    capabilities: dict[str, RuntimeValue]
    trace: dict[str, RuntimeValue]

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize runtime evidence for JSON report payloads."""
        return {
            "access": self.access.to_dict(),
            "evidence": [ref.to_dict() for ref in self.evidence],
            "capabilities": _runtime_dict_to_json(self.capabilities),
            "trace": _runtime_dict_to_json(self.trace),
        }


def runtime_report_evidence(
    capabilities: BackendCapabilities,
    trace: ForwardTrace,
    *,
    prefix: str = "runtime",
    runtime_trace: Trace | None = None,
) -> RuntimeReportEvidence:
    """Build a report-ready evidence package from one runtime trace."""
    capability_ref = EvidenceRef(
        evidence_id=f"{prefix}.capabilities",
        kind="run_report",
        description="Runtime capability declaration used for access boundaries.",
    )
    return RuntimeReportEvidence(
        access=access_policy_for_trace(capabilities, trace),
        evidence=(
            capability_ref,
            *runtime_evidence_refs(
                trace,
                prefix=prefix,
                runtime_trace=runtime_trace,
            ),
        ),
        capabilities=runtime_capability_snapshot(capabilities),
        trace=forward_trace_summary(trace, runtime_trace=runtime_trace),
    )


def runtime_capability_snapshot(
    capabilities: BackendCapabilities,
) -> dict[str, RuntimeValue]:
    """Return a stable serialized capability snapshot."""
    return capabilities.to_dict()


def forward_trace_summary(
    trace: ForwardTrace,
    *,
    runtime_trace: Trace | None = None,
) -> dict[str, RuntimeValue]:
    """Return a stable, JSON-compatible summary of runtime evidence."""
    resolved_trace = (
        trace_from_forward_trace(trace)
        if runtime_trace is None
        else runtime_trace
    )
    device: dict[str, RuntimeValue] = {
        "kind": trace.device.kind,
        "label": trace.device.label,
    }
    activation_shapes: dict[str, RuntimeValue] = {
        str(layer): _shape(tensor)
        for layer, tensor in sorted(trace.activations.items())
    }
    interventions: list[RuntimeValue] = [
        record.to_dict() for record in trace.interventions
    ]
    profile: list[RuntimeValue] = [
        _stable_stage_profile(stage) for stage in trace.profile
    ]
    logits_shape: RuntimeValue = _shape_or_none(trace.logits)
    logprobs_shape: RuntimeValue = _shape_or_none(trace.logprobs)
    summary: dict[str, RuntimeValue] = {}
    summary["device"] = device
    summary["logits_shape"] = logits_shape
    summary["logprobs_shape"] = logprobs_shape
    summary["activation_shapes"] = activation_shapes
    summary["interventions"] = interventions
    summary["profile"] = profile
    summary["spans"] = _trace_span_summaries(resolved_trace)
    summary["artifacts"] = [
        artifact.to_dict() for artifact in resolved_trace.artifacts
    ]
    summary["profile_summary"] = summarize_trace_profile(resolved_trace).to_dict()
    return summary


def trace_from_forward_trace(
    trace: ForwardTrace,
    *,
    trace_id: str = "forward",
) -> Trace:
    """Build the trace-first primitive from an existing forward trace."""
    return trace_from_runtime_execution(None, trace, trace_id=trace_id)


def trace_from_runtime_execution(
    tokenized: TokenizedPrompt | None,
    trace: ForwardTrace,
    *,
    trace_id: str = "runtime",
    metadata: dict[str, RuntimeValue] | None = None,
) -> Trace:
    """Build a trace from optional tokenization plus forward evidence."""
    token_profiles: tuple[StageProfile, ...] = (
        () if tokenized is None else tokenized.profile
    )
    profiles = token_profiles + trace.profile
    raw_spans = tuple(
        TraceSpan(
            span_id=_span_id(index, stage.name),
            name=stage.name,
            profile=stage,
            metadata=dict(stage.metadata),
        )
        for index, stage in enumerate(profiles)
    )
    producer_by_name = {span.name: span.span_id for span in raw_spans}
    artifacts = (
        _token_artifacts(tokenized, producer_by_name)
        + _forward_trace_artifacts(trace, producer_by_name)
    )
    outputs_by_span = _artifact_outputs_by_span(artifacts)
    spans = tuple(
        TraceSpan(
            span_id=span.span_id,
            name=span.name,
            profile=span.profile,
            input_artifact_ids=_span_inputs(span.name, tokenized),
            output_artifact_ids=outputs_by_span.get(span.span_id, ()),
            metadata=dict(span.metadata),
        )
        for span in raw_spans
    )
    return Trace(
        trace_id=trace_id,
        device=trace.device,
        spans=spans,
        artifacts=artifacts,
        metadata={} if metadata is None else dict(metadata),
    )


def summarize_trace_profile(trace: Trace) -> TraceProfileSummary:
    """Summarize USL-ready profile counters across trace spans."""
    profiles = tuple(
        span.profile for span in trace.spans if span.profile is not None
    )
    total_duration_s = sum(profile.duration_s for profile in profiles)
    token_counts = tuple(
        profile.token_count
        for profile in profiles
        if profile.token_count is not None
    )
    batch_sizes = tuple(
        profile.batch_size
        for profile in profiles
        if profile.batch_size is not None
    )
    memory_values = tuple(
        profile.memory_bytes
        for profile in profiles
        if profile.memory_bytes is not None
    )
    queue_depths = tuple(
        profile.queue_depth
        for profile in profiles
        if profile.queue_depth is not None
    )
    token_count = max(token_counts) if token_counts else None
    tokens_per_second: float | None = None
    if token_count is not None and total_duration_s > 0.0:
        tokens_per_second = token_count / total_duration_s
    return TraceProfileSummary(
        total_duration_s=total_duration_s,
        profiled_spans=len(profiles),
        token_count=token_count,
        batch_size=max(batch_sizes) if batch_sizes else None,
        tokens_per_second=tokens_per_second,
        peak_memory_bytes=max(memory_values) if memory_values else None,
        total_input_bytes=sum(
            profile.input_bytes or 0 for profile in profiles
        ),
        total_output_bytes=sum(
            profile.output_bytes or 0 for profile in profiles
        ),
        host_device_copies=sum(
            profile.host_device_copies for profile in profiles
        ),
        sync_points=sum(profile.sync_points for profile in profiles),
        max_queue_depth=max(queue_depths) if queue_depths else None,
    )


def summarize_trace_profile_sweep(
    traces: tuple[Trace, ...],
    *,
    sweep_id: str = "profile_sweep",
    axis: ProfileSweepAxis = "token_count",
    require_stable_artifacts: bool = True,
) -> TraceProfileSweep:
    """Summarize a controlled profile sweep over comparable traces."""
    if not traces:
        msg = "profile sweep requires at least one trace"
        raise ValueError(msg)
    if axis not in ("token_count", "batch_size", "queue_depth"):
        msg = f"unsupported profile sweep axis: {axis!r}"
        raise ValueError(msg)
    coverage = _artifact_kind_coverage(traces[0])
    if require_stable_artifacts:
        _validate_stable_artifact_coverage(traces, coverage)
    grouped: dict[int, list[TraceProfileSummary]] = {}
    grouped_ids: dict[int, list[str]] = {}
    for trace in traces:
        summary = summarize_trace_profile(trace)
        axis_value = _profile_axis_value(summary, axis)
        grouped.setdefault(axis_value, []).append(summary)
        grouped_ids.setdefault(axis_value, []).append(trace.trace_id)
    return TraceProfileSweep(
        sweep_id=sweep_id,
        axis=axis,
        artifact_kinds=coverage,
        points=tuple(
            _sweep_point(axis_value, grouped[axis_value], grouped_ids[axis_value])
            for axis_value in sorted(grouped)
        ),
        metadata={
            "requires_stable_artifacts": require_stable_artifacts,
            "fit": "not_computed",
            "reason": "profile sweep records scaling counters without fitting USL",
        },
    )


def runtime_evidence_refs(
    trace: ForwardTrace,
    *,
    prefix: str = "runtime",
    runtime_trace: Trace | None = None,
) -> tuple[EvidenceRef, ...]:
    """Return report evidence references justified by a forward trace."""
    refs = [
        EvidenceRef(
            evidence_id=f"{prefix}.trace",
            kind="trace",
            description="Runtime forward trace summary.",
        ),
    ]
    if runtime_trace is not None and runtime_trace.artifacts_by_kind("tokens"):
        refs.append(
            EvidenceRef(
                evidence_id=f"{prefix}.tokens",
                kind="trace",
                description="Runtime tokenization evidence.",
            ),
        )
    if trace.logprobs is not None:
        refs.append(
            EvidenceRef(
                evidence_id=f"{prefix}.logprobs",
                kind="logprobs",
                description="Runtime token-probability evidence.",
            ),
        )
    if trace.activations:
        refs.append(
            EvidenceRef(
                evidence_id=f"{prefix}.activations",
                kind="activation",
                description="Runtime activation evidence.",
            ),
        )
    return tuple(refs)


def _forward_trace_artifacts(
    trace: ForwardTrace,
    producer_by_name: dict[str, str],
) -> tuple[TraceArtifact, ...]:
    """Return artifact summaries for an existing forward trace."""
    artifacts: list[TraceArtifact] = []
    head_span = producer_by_name.get("lm_head")
    forward_span = producer_by_name.get("forward")
    if trace.logits is not None:
        artifacts.append(
            TraceArtifact(
                artifact_id="logits",
                kind="logits",
                producer_span_id=head_span,
                shape=_shape_tuple(trace.logits),
            ),
        )
    if trace.logprobs is not None:
        artifacts.append(
            TraceArtifact(
                artifact_id="logprobs",
                kind="logprobs",
                producer_span_id=head_span,
                shape=_shape_tuple(trace.logprobs),
            ),
        )
    for layer, tensor in sorted(trace.activations.items()):
        artifacts.append(
            TraceArtifact(
                artifact_id=f"activation.layer_{layer}",
                kind="activation",
                producer_span_id=forward_span,
                shape=_shape_tuple(tensor),
                metadata={"layer_index": layer},
            ),
        )
    for index, record in enumerate(trace.interventions):
        artifacts.append(
            TraceArtifact(
                artifact_id=f"intervention.{index}",
                kind="intervention_result",
                producer_span_id=forward_span,
                metadata=record.to_dict(),
            ),
        )
    return tuple(artifacts)


def _token_artifacts(
    tokenized: TokenizedPrompt | None,
    producer_by_name: dict[str, str],
) -> tuple[TraceArtifact, ...]:
    """Return token artifact summaries when tokenization was observed."""
    if tokenized is None:
        return ()
    return (
        TraceArtifact(
            artifact_id="tokens",
            kind="tokens",
            producer_span_id=producer_by_name.get("tokenize"),
            shape=(len(tokenized.token_ids),),
            metadata={
                "token_count": len(tokenized.token_ids),
                "text_length_chars": len(tokenized.text),
            },
        ),
    )


def _span_inputs(
    span_name: str,
    tokenized: TokenizedPrompt | None,
) -> tuple[str, ...]:
    """Return stable input artifact references for known trace spans."""
    if tokenized is not None and span_name == "prepare_batch":
        return ("tokens",)
    return ()


def _artifact_outputs_by_span(
    artifacts: tuple[TraceArtifact, ...],
) -> dict[str, tuple[str, ...]]:
    """Index artifact IDs by producer span."""
    outputs: dict[str, list[str]] = {}
    for artifact in artifacts:
        if artifact.producer_span_id is None:
            continue
        outputs.setdefault(artifact.producer_span_id, []).append(
            artifact.artifact_id,
        )
    return {
        span_id: tuple(artifact_ids)
        for span_id, artifact_ids in outputs.items()
    }


def _artifact_kind_coverage(trace: Trace) -> tuple[TraceArtifactKind, ...]:
    """Return stable unique artifact kind coverage for one trace."""
    return tuple(
        kind for kind in TRACE_ARTIFACT_KINDS if trace.artifacts_by_kind(kind)
    )


def _validate_stable_artifact_coverage(
    traces: tuple[Trace, ...],
    expected: tuple[TraceArtifactKind, ...],
) -> None:
    """Reject profile sweeps whose traces do not expose comparable evidence."""
    for trace in traces[1:]:
        actual = _artifact_kind_coverage(trace)
        if actual != expected:
            msg = (
                "profile sweep requires stable artifact coverage: "
                f"{trace.trace_id!r} has {actual!r}, expected {expected!r}"
            )
            raise ValueError(msg)


def _profile_axis_value(
    summary: TraceProfileSummary,
    axis: ProfileSweepAxis,
) -> int:
    """Return the concrete axis value for one profile summary."""
    value: int | None
    if axis == "token_count":
        value = summary.token_count
    elif axis == "batch_size":
        value = summary.batch_size
    else:
        value = summary.max_queue_depth
    if value is None:
        msg = f"profile sweep axis {axis!r} is missing from one trace"
        raise ValueError(msg)
    return value


def _sweep_point(
    axis_value: int,
    summaries: list[TraceProfileSummary],
    trace_ids: list[str],
) -> TraceProfileSweepPoint:
    """Aggregate trace profile summaries into one sweep point."""
    token_rates = [
        summary.tokens_per_second
        for summary in summaries
        if summary.tokens_per_second is not None
    ]
    memory_values = [
        summary.peak_memory_bytes
        for summary in summaries
        if summary.peak_memory_bytes is not None
    ]
    return TraceProfileSweepPoint(
        axis_value=axis_value,
        trace_ids=tuple(trace_ids),
        samples=len(summaries),
        mean_total_duration_s=(
            sum(summary.total_duration_s for summary in summaries)
            / len(summaries)
        ),
        mean_tokens_per_second=(
            None if not token_rates else sum(token_rates) / len(token_rates)
        ),
        peak_memory_bytes=None if not memory_values else max(memory_values),
        total_host_device_copies=sum(
            summary.host_device_copies for summary in summaries
        ),
        total_sync_points=sum(summary.sync_points for summary in summaries),
    )


def _trace_span_summaries(trace: Trace) -> list[RuntimeValue]:
    """Return stable span summaries without volatile durations."""
    spans: list[RuntimeValue] = []
    for span in trace.spans:
        profile: dict[str, RuntimeValue] | None = None
        if span.profile is not None:
            profile = _stable_stage_profile(span.profile)
        spans.append({
            "id": span.span_id,
            "name": span.name,
            "input_artifacts": list(span.input_artifact_ids),
            "output_artifacts": list(span.output_artifact_ids),
            "profile": profile,
            "metadata": dict(span.metadata),
        })
    return spans


def _stable_stage_profile(profile: StageProfile) -> dict[str, RuntimeValue]:
    """Return stable profile counters without volatile wall-clock duration."""
    return {
        "name": profile.name,
        "device": _device_summary(profile.device),
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


def _device_summary(device: object) -> dict[str, RuntimeValue] | None:
    """Serialize optional device-like metadata for stable summaries."""
    if device is None:
        return None
    typed = device
    kind = getattr(typed, "kind", None)
    label = getattr(typed, "label", None)
    if not isinstance(kind, str) or not isinstance(label, str):
        return None
    return {
        "kind": kind,
        "label": label,
    }


def _span_id(index: int, name: str) -> str:
    """Return a stable span ID for one profile entry."""
    normalized = name.strip().replace(" ", "_")
    return f"span.{index}.{normalized}"


def _shape_or_none(tensor: TensorLike | None) -> list[RuntimeValue] | None:
    """Serialize tensor shape if present."""
    if tensor is None:
        return None
    return _shape(tensor)


def _shape(tensor: TensorLike) -> list[RuntimeValue]:
    """Serialize tensor shape as JSON-compatible runtime values."""
    return [int(dim) for dim in tensor.shape]


def _shape_tuple(tensor: TensorLike) -> tuple[int, ...]:
    """Serialize tensor shape as a tuple for trace artifacts."""
    return tuple(int(dim) for dim in tensor.shape)


def _runtime_dict_to_json(data: dict[str, RuntimeValue]) -> dict[str, JsonValue]:
    """Convert runtime JSON-compatible values to behavior JSON-compatible values."""
    return {key: _runtime_value_to_json(value) for key, value in data.items()}


def _runtime_value_to_json(value: RuntimeValue) -> JsonValue:
    """Convert a runtime value into the behavior-report JSON alias."""
    if isinstance(value, list):
        return [_runtime_value_to_json(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _runtime_value_to_json(item)
            for key, item in value.items()
        }
    return value
