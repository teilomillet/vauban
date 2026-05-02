# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Backend-independent tests for trace-first runtime primitives."""

from __future__ import annotations

from typing import cast

import pytest

from vauban.runtime import (
    TRACE_ARTIFACT_KINDS,
    DeviceRef,
    ForwardTrace,
    InterventionRecord,
    StageProfile,
    TokenizedPrompt,
    Trace,
    TraceArtifact,
    TraceArtifactKind,
    TraceRequest,
    TraceSpan,
    forward_trace_summary,
    max_capabilities,
    mlx_capabilities,
    summarize_trace_profile,
    summarize_trace_profile_sweep,
    trace_from_forward_trace,
    trace_from_runtime_execution,
)


class FakeTensor:
    """Small tensor-like object with only shape metadata."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize fake tensor shape."""
        self._shape = shape

    @property
    def shape(self) -> tuple[int, ...]:
        """Return fake tensor shape."""
        return self._shape


def _device() -> DeviceRef:
    """Return stable fake device metadata."""
    return DeviceRef(kind="cpu", label="unit-cpu")


def _forward_trace() -> ForwardTrace:
    """Build a forward trace with all first-slice artifact kinds."""
    device = _device()
    return ForwardTrace(
        logits=FakeTensor((1, 2, 4)),
        logprobs=FakeTensor((1, 2, 4)),
        activations={0: FakeTensor((1, 2, 4))},
        device=device,
        interventions=(InterventionRecord(name="shift", layer_index=0),),
        profile=(
            StageProfile(
                name="prepare_batch",
                duration_s=0.1,
                device=device,
                batch_size=1,
                token_count=2,
                input_bytes=16,
                host_device_copies=1,
                sync_points=1,
                metadata={"tokens": 2},
            ),
            StageProfile(
                name="forward",
                duration_s=0.2,
                device=device,
                batch_size=1,
                token_count=2,
                memory_bytes=128,
                sync_points=1,
                metadata={"collect_layers": [0]},
            ),
            StageProfile(
                name="lm_head",
                duration_s=0.3,
                device=device,
                batch_size=1,
                token_count=2,
                output_bytes=32,
            ),
        ),
    )


def _tokenized_prompt(token_count: int = 2) -> TokenizedPrompt:
    """Build a tokenized prompt with one tokenization span."""
    return TokenizedPrompt(
        token_ids=tuple(range(1, token_count + 1)),
        text="x" * token_count,
        profile=(
            StageProfile(
                name="tokenize",
                duration_s=0.05,
                device=_device(),
                token_count=token_count,
            ),
        ),
    )


def test_trace_from_forward_trace_promotes_artifacts_and_spans() -> None:
    """ForwardTrace should lift into trace/span/artifact primitives."""
    trace = trace_from_forward_trace(_forward_trace(), trace_id="unit")

    assert trace.trace_id == "unit"
    assert tuple(span.name for span in trace.spans) == (
        "prepare_batch",
        "forward",
        "lm_head",
    )
    assert tuple(artifact.kind for artifact in trace.artifacts) == (
        "logits",
        "logprobs",
        "activation",
        "intervention_result",
    )
    assert trace.artifacts_by_kind("activation")[0].metadata == {
        "layer_index": 0,
    }
    assert trace.spans[1].output_artifact_ids == (
        "activation.layer_0",
        "intervention.0",
    )
    assert trace.spans[2].output_artifact_ids == ("logits", "logprobs")


def test_trace_from_runtime_execution_adds_token_stage_and_artifact() -> None:
    """Runtime execution traces should include observed tokenization evidence."""
    trace = trace_from_runtime_execution(
        _tokenized_prompt(),
        _forward_trace(),
        trace_id="unit.runtime",
        metadata={"suite": "unit"},
    )

    assert trace.trace_id == "unit.runtime"
    assert trace.metadata == {"suite": "unit"}
    assert tuple(span.name for span in trace.spans) == (
        "tokenize",
        "prepare_batch",
        "forward",
        "lm_head",
    )
    assert tuple(artifact.kind for artifact in trace.artifacts) == (
        "tokens",
        "logits",
        "logprobs",
        "activation",
        "intervention_result",
    )
    assert trace.spans[0].output_artifact_ids == ("tokens",)
    assert trace.spans[1].input_artifact_ids == ("tokens",)
    token_artifact = trace.artifacts_by_kind("tokens")[0]
    assert token_artifact.shape == (2,)
    assert token_artifact.metadata["token_count"] == 2


def test_forward_trace_summary_keeps_legacy_keys_and_adds_trace_view() -> None:
    """Runtime report payloads keep old shape keys while exposing trace view."""
    summary = forward_trace_summary(_forward_trace())

    assert summary["logits_shape"] == [1, 2, 4]
    assert summary["logprobs_shape"] == [1, 2, 4]
    assert summary["activation_shapes"] == {"0": [1, 2, 4]}
    assert "profile" in summary
    assert "profile_summary" in summary
    assert "spans" in summary
    assert "artifacts" in summary

    spans = summary["spans"]
    assert isinstance(spans, list)
    first_span = spans[0]
    assert isinstance(first_span, dict)
    first_profile = first_span["profile"]
    assert isinstance(first_profile, dict)
    assert "duration_s" not in first_profile
    assert first_profile["token_count"] == 2
    assert first_profile["host_device_copies"] == 1

    artifacts = summary["artifacts"]
    assert isinstance(artifacts, list)
    first_artifact = artifacts[0]
    assert isinstance(first_artifact, dict)
    assert first_artifact["id"] == "logits"
    assert first_artifact["shape"] == [1, 2, 4]

    profile_summary = summary["profile_summary"]
    assert isinstance(profile_summary, dict)
    assert profile_summary["profiled_spans"] == 3
    assert profile_summary["token_count"] == 2
    assert profile_summary["batch_size"] == 1
    assert profile_summary["peak_memory_bytes"] == 128
    assert profile_summary["total_input_bytes"] == 16
    assert profile_summary["total_output_bytes"] == 32
    assert profile_summary["host_device_copies"] == 1
    assert profile_summary["sync_points"] == 2


def test_summarize_trace_profile_exposes_usl_ready_counters() -> None:
    """Trace profile summaries expose scaling counters without fitting USL yet."""
    trace = trace_from_forward_trace(_forward_trace(), trace_id="unit")
    summary = summarize_trace_profile(trace)

    assert summary.profiled_spans == 3
    assert summary.token_count == 2
    assert summary.batch_size == 1
    assert summary.tokens_per_second == pytest.approx(2 / 0.6)
    assert summary.peak_memory_bytes == 128
    assert summary.host_device_copies == 1
    assert summary.sync_points == 2
    assert summary.to_dict()["total_duration_s"] == pytest.approx(0.6)


def test_summarize_trace_profile_sweep_requires_stable_artifact_coverage() -> None:
    """Profile sweeps should only compare traces with the same evidence kinds."""
    comparable = (
        trace_from_runtime_execution(
            _tokenized_prompt(2),
            _forward_trace(),
            trace_id="tokens-2",
        ),
        trace_from_runtime_execution(
            _tokenized_prompt(3),
            ForwardTrace(
                logits=FakeTensor((1, 3, 4)),
                logprobs=FakeTensor((1, 3, 4)),
                activations={0: FakeTensor((1, 3, 4))},
                device=_device(),
                interventions=(InterventionRecord(name="shift", layer_index=0),),
                profile=(
                    StageProfile(
                        name="prepare_batch",
                        duration_s=0.1,
                        device=_device(),
                        batch_size=1,
                        token_count=3,
                        host_device_copies=1,
                    ),
                    StageProfile(
                        name="forward",
                        duration_s=0.2,
                        device=_device(),
                        batch_size=1,
                        token_count=3,
                        memory_bytes=256,
                        sync_points=1,
                    ),
                    StageProfile(
                        name="lm_head",
                        duration_s=0.3,
                        device=_device(),
                        batch_size=1,
                        token_count=3,
                    ),
                ),
            ),
            trace_id="tokens-3",
        ),
    )
    sweep = summarize_trace_profile_sweep(comparable, sweep_id="unit")

    assert sweep.sweep_id == "unit"
    assert sweep.axis == "token_count"
    assert sweep.artifact_kinds == (
        "tokens",
        "logits",
        "logprobs",
        "activation",
        "intervention_result",
    )
    assert tuple(point.axis_value for point in sweep.points) == (2, 3)
    assert sweep.points[1].peak_memory_bytes == 256
    assert sweep.metadata["fit"] == "not_computed"

    mismatched = trace_from_runtime_execution(
        _tokenized_prompt(4),
        ForwardTrace(
            logits=FakeTensor((1, 4, 4)),
            logprobs=None,
            activations={},
            device=_device(),
            profile=(
                StageProfile(
                    name="prepare_batch",
                    duration_s=0.1,
                    device=_device(),
                    batch_size=1,
                    token_count=4,
                ),
            ),
        ),
        trace_id="tokens-4",
    )
    with pytest.raises(ValueError, match="stable artifact coverage"):
        summarize_trace_profile_sweep((*comparable, mismatched), sweep_id="bad")


def test_trace_rejects_duplicate_artifact_ids() -> None:
    """Trace artifact identity must be unambiguous."""
    artifact = TraceArtifact(artifact_id="logits", kind="logits")

    with pytest.raises(ValueError, match=r"artifacts\.id contains duplicate"):
        Trace(
            trace_id="bad",
            device=_device(),
            spans=(),
            artifacts=(artifact, artifact),
        )


def test_trace_rejects_unknown_span_artifact_refs() -> None:
    """Spans must only reference artifacts declared by the trace."""
    with pytest.raises(ValueError, match="unknown output artifact"):
        Trace(
            trace_id="bad",
            device=_device(),
            spans=(
                TraceSpan(
                    span_id="span.forward",
                    name="forward",
                    output_artifact_ids=("missing",),
                ),
            ),
            artifacts=(),
        )


def test_trace_rejects_output_producer_mismatch() -> None:
    """Span outputs must agree with artifact producer metadata."""
    artifact = TraceArtifact(
        artifact_id="logits",
        kind="logits",
        producer_span_id="span.lm_head",
    )

    with pytest.raises(ValueError, match="artifact producer"):
        Trace(
            trace_id="bad",
            device=_device(),
            spans=(
                TraceSpan(
                    span_id="span.forward",
                    name="forward",
                    output_artifact_ids=("logits",),
                ),
                TraceSpan(span_id="span.lm_head", name="lm_head"),
            ),
            artifacts=(artifact,),
        )


def test_trace_rejects_unknown_artifact_kind() -> None:
    """Trace artifacts should stay inside the declared artifact vocabulary."""
    with pytest.raises(ValueError, match="unknown trace artifact kind"):
        TraceArtifact(
            artifact_id="bad",
            kind=cast("TraceArtifactKind", "backend_blob"),
        )


def test_trace_request_rejects_duplicate_requested_artifacts() -> None:
    """Requested artifact kinds should be explicit and non-duplicated."""
    with pytest.raises(ValueError, match="requested_artifacts contains duplicate"):
        TraceRequest(
            trace_id="bad",
            prompt_ids=(1,),
            requested_artifacts=("logits", "logits"),
        )


def test_trace_request_requires_model_input() -> None:
    """A trace request should identify the runtime input being traced."""
    with pytest.raises(ValueError, match="input_text or prompt_ids"):
        TraceRequest(trace_id="bad")


def test_trace_request_accepts_tokenized_input() -> None:
    """Token IDs are enough input for a backend forward trace."""
    request = TraceRequest(
        trace_id="unit",
        prompt_ids=(1, 2),
        requested_artifacts=("logits", "activation"),
    )

    assert request.prompt_ids == (1, 2)
    assert request.requested_artifacts == ("logits", "activation")


def test_backend_capabilities_are_artifact_oriented() -> None:
    """Backends should declare support in the same artifact vocabulary as traces."""
    caps = mlx_capabilities()

    assert "logits" in TRACE_ARTIFACT_KINDS
    assert caps.support_level_for_artifact("logits") == "full"
    assert caps.support_level_for_artifact("activation") == "full"
    assert caps.support_level_for_artifact("intervention_result") == "full"
    assert caps.support_level_for_artifact("profile") == "full"
    assert caps.support_level_for_artifact("text") == "unsupported"
    assert "activation" in caps.supported_artifact_kinds()


def test_placeholder_max_capabilities_do_not_claim_runtime_artifacts() -> None:
    """Declared MAX support should stay honest until an adapter exists."""
    caps = max_capabilities()

    assert caps.support_level_for_artifact("tokens") == "unsupported"
    assert caps.support_level_for_artifact("profile") == "unsupported"
    assert caps.supported_artifact_kinds() == ()
