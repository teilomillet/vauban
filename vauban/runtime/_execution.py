# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Trace-first runtime execution helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vauban.runtime._evidence import trace_from_runtime_execution
from vauban.runtime._types import (
    ForwardRequest,
    RuntimeTraceResult,
    RuntimeValue,
    TokenizedPrompt,
    TokenizeRequest,
    Trace,
    TraceArtifactKind,
)

if TYPE_CHECKING:
    from vauban.runtime._protocols import ModelRuntime
    from vauban.runtime._types import LoadedModel, TraceRequest


def run_runtime_trace(
    runtime: ModelRuntime,
    loaded: LoadedModel,
    request: TraceRequest,
) -> RuntimeTraceResult:
    """Execute tokenization and forward stages as one trace-first run."""
    tokenized: TokenizedPrompt | None = None
    prompt_ids: tuple[int, ...] = request.prompt_ids
    if not prompt_ids:
        if request.input_text is None:
            msg = "trace request requires input_text when prompt_ids are absent"
            raise ValueError(msg)
        tokenized = runtime.tokenize(
            loaded,
            TokenizeRequest(
                request.input_text,
                apply_chat_template=request.apply_chat_template,
            ),
        )
        prompt_ids = tokenized.token_ids

    return_logprobs = (
        request.return_logprobs or "logprobs" in request.requested_artifacts
    )
    return_logits = (
        request.return_logits
        or return_logprobs
        or "logits" in request.requested_artifacts
    )
    forward = runtime.forward(
        loaded,
        ForwardRequest(
            prompt_ids=prompt_ids,
            collect_layers=request.collect_layers,
            interventions=request.interventions,
            return_logits=return_logits,
            return_logprobs=return_logprobs,
        ),
    )
    trace = trace_from_runtime_execution(
        tokenized,
        forward,
        trace_id=request.trace_id,
        metadata=_request_metadata(request),
    )
    trace = _record_requested_artifact_gaps(loaded, request, trace)
    return RuntimeTraceResult(
        request=request,
        trace=trace,
        forward=forward,
        tokenized=tokenized,
    )


def _request_metadata(request: TraceRequest) -> dict[str, RuntimeValue]:
    """Return trace metadata derived from the request."""
    metadata = dict(request.metadata)
    if request.requested_artifacts:
        metadata["requested_artifacts"] = list(request.requested_artifacts)
    return metadata


def _record_requested_artifact_gaps(
    loaded: LoadedModel,
    request: TraceRequest,
    trace: Trace,
) -> Trace:
    """Record missing or unsupported requested artifacts on the trace."""
    if not request.requested_artifacts:
        return trace
    missing = _missing_requested_artifacts(request.requested_artifacts, trace)
    unsupported = _unsupported_requested_artifacts(
        request.requested_artifacts,
        loaded,
    )
    if not missing and not unsupported:
        return trace
    metadata = dict(trace.metadata)
    if missing:
        metadata["missing_requested_artifacts"] = list(missing)
    if unsupported:
        metadata["unsupported_requested_artifacts"] = list(unsupported)
    return Trace(
        trace_id=trace.trace_id,
        device=trace.device,
        spans=trace.spans,
        artifacts=trace.artifacts,
        metadata=metadata,
    )


def _missing_requested_artifacts(
    requested: tuple[TraceArtifactKind, ...],
    trace: Trace,
) -> tuple[TraceArtifactKind, ...]:
    """Return requested artifact kinds absent from the emitted trace."""
    return tuple(
        kind for kind in requested if not trace.artifacts_by_kind(kind)
    )


def _unsupported_requested_artifacts(
    requested: tuple[TraceArtifactKind, ...],
    loaded: LoadedModel,
) -> tuple[TraceArtifactKind, ...]:
    """Return requested artifact kinds unsupported by the loaded backend."""
    return tuple(
        kind
        for kind in requested
        if loaded.capabilities.support_level_for_artifact(kind) == "unsupported"
    )
