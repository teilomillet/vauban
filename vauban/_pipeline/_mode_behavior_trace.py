# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Behavior trace collection early-mode runner."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import (
    ModeReport,
    finish_mode_run,
    write_mode_report,
)
from vauban.api_trace import ApiTraceResponse, call_api_behavior_trace
from vauban.behavior import (
    BehaviorObservation,
    BehaviorTrace,
    ExampleRedaction,
    ExpectedBehavior,
    JsonValue,
    artifact_hashes,
    is_refusal_text,
    reproducibility_payload,
    score_behavior_output,
    write_behavior_trace,
)
from vauban.evaluate import _generate
from vauban.runtime import (
    DirectionInterventionMode,
    LoadedModel,
    ModelRef,
    TorchActivationPrimitiveRequest,
    TorchActivationTensor,
    TorchDirectionIntervention,
    Trace,
    TraceArtifact,
    TraceArtifactKind,
    TraceRequest,
    TraceSpan,
    create_runtime,
    run_torch_activation_primitive,
    runtime_report_evidence,
    summarize_trace_profile_sweep,
)

if TYPE_CHECKING:
    from pathlib import Path

    from vauban.runtime import (
        RuntimeBackendName,
        RuntimeReportEvidence,
        RuntimeTraceResult,
        TensorLike,
    )
    from vauban.types import (
        BehaviorTraceActivationPrimitiveConfig,
        BehaviorTraceConfig,
        BehaviorTracePromptConfig,
        CausalLM,
        Tokenizer,
    )


def _run_behavior_trace_mode(context: EarlyModeContext) -> None:
    """Run [behavior_trace] and write reusable JSONL trace artifacts."""
    config = context.config
    trace_config = config.behavior_trace
    if trace_config is None:
        msg = "[behavior_trace] section is required for behavior_trace mode"
        raise ValueError(msg)

    log(
        (
            "Collecting behavior trace"
            f" — suite={trace_config.suite_name!r},"
            f" prompts={len(trace_config.prompts)}"
        ),
        verbose=config.verbose,
        elapsed=time.monotonic() - context.t0,
    )

    runtime_evidence: list[JsonValue] | None = None
    runtime_profile_sweep: JsonValue | None = None
    if trace_config.runtime_backend == "api":
        observations, runtime_evidence = _collect_api_observations(trace_config)
        runtime_profile_sweep = {
            "status": "not_applicable",
            "reason": "API behavior traces do not expose local runtime spans.",
        }
        model_path = trace_config.api.model if trace_config.api is not None else ""
    else:
        if context.model is None or context.tokenizer is None:
            msg = "local behavior traces require a loaded model and tokenizer"
            raise ValueError(msg)
        model = cast("CausalLM", context.model)
        tokenizer = cast("Tokenizer", context.tokenizer)
        observations = _collect_observations(model, tokenizer, trace_config)
        model_path = context.config.model_path
    if (
        trace_config.collect_runtime_evidence
        and trace_config.runtime_backend != "api"
    ):
        runtime_evidence, runtime_profile_sweep = _collect_runtime_evidence(
            model,
            tokenizer,
            trace_config,
            model_path=model_path,
        )
    trace_path = _trace_path(context.config.output_dir, trace_config)
    trace = BehaviorTrace(
        trace_id=trace_path.stem,
        model_label=trace_config.model_label,
        model_path=model_path,
        suite_name=trace_config.suite_name,
        source_path=str(trace_path),
        observations=tuple(observations),
        metadata={
            "suite_description": trace_config.suite_description,
            "suite_version": trace_config.suite_version,
            "suite_source": trace_config.suite_source,
            "safety_policy": trace_config.safety_policy,
            "scorers": list(trace_config.scorers),
            "record_outputs": trace_config.record_outputs,
            "max_tokens": trace_config.max_tokens,
            "runtime_backend": trace_config.runtime_backend,
        },
    )
    written_trace_path = write_behavior_trace(trace_path, trace)
    json_path = write_mode_report(
        config.output_dir,
        ModeReport(
            trace_config.json_filename,
            _report_payload(
                trace,
                trace_config,
                written_trace_path,
                config_path=context.config_path,
                output_dir=config.output_dir,
                runtime_evidence=runtime_evidence,
                runtime_profile_sweep=runtime_profile_sweep,
            ),
        ),
    )

    refused_count = sum(1 for observation in observations if observation.refused)
    finish_mode_run(
        context,
        "behavior_trace",
        [str(written_trace_path), str(json_path)],
        {
            "n_observations": len(observations),
            "n_refusals": refused_count,
            "refusal_rate": (
                refused_count / len(observations)
                if observations
                else 0.0
            ),
        },
    )


def _collect_observations(
    model: CausalLM,
    tokenizer: Tokenizer,
    config: BehaviorTraceConfig,
) -> list[BehaviorObservation]:
    """Generate outputs for configured prompts and convert them to observations."""
    observations: list[BehaviorObservation] = []
    for prompt in config.prompts:
        output = _generate(model, tokenizer, prompt.text, config.max_tokens)
        refused = is_refusal_text(output, config.refusal_phrases)
        expected_behavior = cast("ExpectedBehavior", prompt.expected_behavior)
        observations.append(
            BehaviorObservation(
                observation_id=f"{config.model_label}:{prompt.prompt_id}",
                model_label=config.model_label,
                prompt_id=prompt.prompt_id,
                category=prompt.category,
                prompt=_prompt_for_trace(prompt),
                output_text=_output_for_trace(
                    output,
                    prompt.redaction,
                    record_outputs=config.record_outputs,
                ),
                expected_behavior=expected_behavior,
                refused=refused,
                metrics=score_behavior_output(
                    output,
                    refused=refused,
                    expected_behavior=expected_behavior,
                    scorer_names=tuple(config.scorers),
                ),
                redaction=cast("ExampleRedaction", prompt.redaction),
                metadata={
                    "tags": list(prompt.tags),
                    "scorers": list(config.scorers),
                },
            ),
        )
    return observations


def _collect_api_observations(
    config: BehaviorTraceConfig,
) -> tuple[list[BehaviorObservation], list[JsonValue]]:
    """Collect behavior observations from an OpenAI-compatible API endpoint."""
    endpoint = config.api
    if endpoint is None:
        msg = '[behavior_trace] runtime_backend = "api" requires api settings'
        raise ValueError(msg)
    api_key = os.environ.get(endpoint.api_key_env)
    if not api_key:
        msg = (
            f"Environment variable {endpoint.api_key_env!r} is not set"
            " or empty — required for API behavior traces"
        )
        raise ValueError(msg)

    observations: list[BehaviorObservation] = []
    runtime_rows: list[JsonValue] = []
    for prompt in config.prompts:
        response = call_api_behavior_trace(
            endpoint=endpoint,
            api_key=api_key,
            prompt=prompt.text,
            max_tokens=config.max_tokens,
            refusal_phrases=config.refusal_phrases,
            return_logprobs=config.return_logprobs,
        )
        expected_behavior = cast("ExpectedBehavior", prompt.expected_behavior)
        observations.append(
            BehaviorObservation(
                observation_id=f"{config.model_label}:{prompt.prompt_id}",
                model_label=config.model_label,
                prompt_id=prompt.prompt_id,
                category=prompt.category,
                prompt=_prompt_for_trace(prompt),
                output_text=_output_for_trace(
                    response.text,
                    prompt.redaction,
                    record_outputs=config.record_outputs,
                ),
                expected_behavior=expected_behavior,
                refused=response.refused,
                metrics=score_behavior_output(
                    response.text,
                    refused=response.refused,
                    expected_behavior=expected_behavior,
                    scorer_names=tuple(config.scorers),
                ),
                redaction=cast("ExampleRedaction", prompt.redaction),
                metadata=_api_observation_metadata(config, prompt, response),
            ),
        )
        runtime_rows.append(_api_runtime_evidence_row(config, prompt, response))
    return observations, runtime_rows


def _collect_runtime_evidence(
    model: CausalLM,
    tokenizer: Tokenizer,
    config: BehaviorTraceConfig,
    *,
    model_path: str,
) -> tuple[list[JsonValue], JsonValue]:
    """Collect opt-in runtime evidence for behavior trace report payloads."""
    runtime = create_runtime(config.runtime_backend)
    backend = cast("RuntimeBackendName", config.runtime_backend)
    loaded = LoadedModel(
        ref=ModelRef(model_path),
        backend=backend,
        capabilities=runtime.capabilities,
        model=model,
        tokenizer=tokenizer,
    )
    rows: list[JsonValue] = []
    traces: list[Trace] = []
    collect_layers = _runtime_collect_layers(config)
    interventions = _runtime_interventions(config, collect_layers)
    for prompt in config.prompts:
        for warmup_index in range(config.runtime_profile_sweep.warmup):
            runtime.trace(
                loaded,
                _runtime_trace_request(
                    prompt,
                    config,
                    collect_layers=collect_layers,
                    interventions=interventions,
                    trace_id=(
                        f"behavior_trace.{prompt.prompt_id}"
                        f".warmup_{warmup_index}"
                    ),
                    sample_index=None,
                ),
            )
        samples: list[RuntimeTraceResult] = []
        for sample_index in range(config.runtime_profile_sweep.samples):
            result = runtime.trace(
                loaded,
                _runtime_trace_request(
                    prompt,
                    config,
                    collect_layers=collect_layers,
                    interventions=interventions,
                    trace_id=_sample_trace_id(
                        prompt.prompt_id,
                        sample_index,
                        config.runtime_profile_sweep.samples,
                    ),
                    sample_index=sample_index,
                ),
            )
            samples.append(result)
            traces.append(
                _with_activation_primitive_artifacts(
                    result.trace,
                    result.forward.activations,
                    config.activation_primitive,
                ),
            )
        result = samples[0]
        runtime_trace = _with_activation_primitive_artifacts(
            result.trace,
            result.forward.activations,
            config.activation_primitive,
        )
        package = runtime_report_evidence(
            runtime.capabilities,
            result.forward,
            prefix=f"behavior_trace.{prompt.prompt_id}",
            runtime_trace=runtime_trace,
        )
        rows.append(_runtime_evidence_row(prompt.prompt_id, package))
    return rows, _runtime_profile_sweep(config, traces)


def _api_observation_metadata(
    config: BehaviorTraceConfig,
    prompt: BehaviorTracePromptConfig,
    response: ApiTraceResponse,
) -> dict[str, JsonValue]:
    """Return endpoint metadata for one API behavior observation."""
    endpoint = config.api
    if endpoint is None:
        return {}
    metadata: dict[str, JsonValue] = {
        "tags": list(prompt.tags),
        "scorers": list(config.scorers),
        "access_level": "endpoint",
        "runtime_backend": "api",
        "endpoint": endpoint.name,
        "api_model": endpoint.model,
        "return_logprobs": config.return_logprobs,
    }
    metadata.update(response.metadata())
    return metadata


def _api_runtime_evidence_row(
    config: BehaviorTraceConfig,
    prompt: BehaviorTracePromptConfig,
    response: ApiTraceResponse,
) -> dict[str, JsonValue]:
    """Serialize endpoint evidence for the behavior trace JSON sidecar."""
    endpoint = config.api
    if endpoint is None:
        msg = "API runtime evidence requires endpoint settings"
        raise ValueError(msg)
    artifacts: list[JsonValue] = [
        {
            "id": "output_text",
            "kind": "text",
            "metadata": {
                "redaction": prompt.redaction,
                "recorded": (
                    config.record_outputs and prompt.redaction == "safe"
                ),
            },
        },
    ]
    if response.logprobs:
        artifacts.append({
            "id": "logprobs",
            "kind": "logprobs",
            "metadata": {
                "token_count": len(response.logprobs),
            },
        })
    return {
        "prompt_id": prompt.prompt_id,
        "runtime": {
            "backend": "api",
            "access_level": "endpoint",
            "endpoint": endpoint.name,
            "model": endpoint.model,
            "return_logprobs": config.return_logprobs,
            "trace": {
                "artifacts": artifacts,
                "metadata": response.metadata(),
            },
        },
    }


def _runtime_trace_request(
    prompt: BehaviorTracePromptConfig,
    config: BehaviorTraceConfig,
    *,
    collect_layers: tuple[int, ...],
    interventions: tuple[TorchDirectionIntervention, ...],
    trace_id: str,
    sample_index: int | None,
) -> TraceRequest:
    """Build one runtime trace request for a behavior prompt."""
    metadata: dict[str, JsonValue] = {
        "prompt_id": prompt.prompt_id,
    }
    if sample_index is not None:
        metadata["sample_index"] = sample_index
    return TraceRequest(
        trace_id=trace_id,
        input_text=prompt.text,
        requested_artifacts=_runtime_requested_artifacts(config),
        metadata=metadata,
        collect_layers=collect_layers,
        interventions=interventions,
        return_logits=True,
        return_logprobs=config.return_logprobs,
    )


def _sample_trace_id(prompt_id: str, sample_index: int, samples: int) -> str:
    """Return a stable trace ID for one measured runtime sample."""
    if samples == 1:
        return f"behavior_trace.{prompt_id}"
    return f"behavior_trace.{prompt_id}.sample_{sample_index}"


def _runtime_profile_sweep(
    config: BehaviorTraceConfig,
    traces: list[Trace],
) -> JsonValue:
    """Return a controlled profile sweep for stable runtime trace coverage."""
    sweep_config = config.runtime_profile_sweep
    if not sweep_config.enabled:
        return {
            "status": "disabled",
            "reason": "runtime profile sweep disabled by config",
        }
    if not traces:
        return {
            "status": "not_computed",
            "reason": "no runtime traces were collected",
        }
    try:
        return cast(
            "JsonValue",
            summarize_trace_profile_sweep(
                tuple(traces),
                sweep_id=f"behavior_trace.{config.model_label}.profile_sweep",
                axis=sweep_config.axis,
                require_stable_artifacts=sweep_config.require_stable_artifacts,
            ).to_dict(),
        )
    except ValueError as exc:
        return {
            "status": "not_computed",
            "reason": str(exc),
        }


def _runtime_requested_artifacts(
    config: BehaviorTraceConfig,
) -> tuple[TraceArtifactKind, ...]:
    """Return trace artifacts requested by the behavior trace config."""
    artifacts: list[TraceArtifactKind] = ["tokens", "logits"]
    if config.return_logprobs:
        artifacts.append("logprobs")
    if config.collect_layers or config.activation_primitive.enabled:
        artifacts.append("activation")
    return tuple(artifacts)


def _runtime_collect_layers(config: BehaviorTraceConfig) -> tuple[int, ...]:
    """Return activation layers needed by runtime evidence and primitives."""
    layers = list(config.collect_layers)
    primitive = config.activation_primitive
    if primitive.enabled:
        if config.runtime_backend != "torch":
            msg = (
                "[behavior_trace.activation_primitive] currently requires"
                ' runtime_backend = "torch"'
            )
            raise ValueError(msg)
        primitive_layers = primitive.layers or config.collect_layers
        if not primitive_layers:
            msg = (
                "[behavior_trace.activation_primitive] requires layers or"
                " collect_layers"
            )
            raise ValueError(msg)
        layers.extend(primitive_layers)
    deduped: list[int] = []
    seen: set[int] = set()
    for layer in layers:
        if layer not in seen:
            deduped.append(layer)
            seen.add(layer)
    return tuple(deduped)


def _runtime_interventions(
    config: BehaviorTraceConfig,
    collect_layers: tuple[int, ...],
) -> tuple[TorchDirectionIntervention, ...]:
    """Return primitive-backed interventions requested by behavior trace config."""
    primitive = config.activation_primitive
    if not primitive.enabled or primitive.mode in ("project", "subspace_project"):
        return ()
    import torch

    tensor = cast("TorchActivationTensor", torch.tensor(_primitive_values(primitive)))
    mode = _intervention_mode(primitive.mode)
    layers = tuple(primitive.layers) if primitive.layers else collect_layers
    return tuple(
        TorchDirectionIntervention(
            name=f"{primitive.name}.layer_{layer}",
            layer_index=layer,
            direction=tensor,
            alpha=primitive.alpha,
            mode=mode,
        )
        for layer in layers
    )


def _with_activation_primitive_artifacts(
    trace: Trace,
    activations: dict[int, TensorLike],
    primitive: BehaviorTraceActivationPrimitiveConfig,
) -> Trace:
    """Attach activation projection artifacts to a runtime trace when requested."""
    if not primitive.enabled or primitive.mode not in ("project", "subspace_project"):
        return trace
    import torch

    tensor = cast("TorchActivationTensor", torch.tensor(_primitive_values(primitive)))
    layers = tuple(primitive.layers) if primitive.layers else tuple(activations)
    forward_span_id = _forward_span_id(trace)
    artifacts = list(trace.artifacts)
    new_artifact_ids: list[str] = []
    for layer in layers:
        activation = activations.get(layer)
        if activation is None:
            continue
        result = run_torch_activation_primitive(
            TorchActivationPrimitiveRequest(
                activation=cast("TorchActivationTensor", activation),
                direction=tensor,
                layer_index=layer,
                mode=primitive.mode,
                alpha=primitive.alpha,
                name=primitive.name,
            ),
        )
        artifact = TraceArtifact(
            artifact_id=f"activation_projection.layer_{layer}",
            kind="metric",
            producer_span_id=forward_span_id,
            shape=tuple(int(dim) for dim in result.projection.shape),
            metadata=result.artifact_metadata(),
        )
        artifacts.append(artifact)
        new_artifact_ids.append(artifact.artifact_id)
    if not new_artifact_ids:
        return trace
    return Trace(
        trace_id=trace.trace_id,
        device=trace.device,
        spans=_append_span_outputs(trace, forward_span_id, tuple(new_artifact_ids)),
        artifacts=tuple(artifacts),
        metadata=dict(trace.metadata),
    )


def _primitive_values(
    primitive: BehaviorTraceActivationPrimitiveConfig,
) -> list[float] | list[list[float]]:
    """Return direction or basis values for a Torch primitive tensor."""
    if primitive.mode in ("project", "subtract", "add"):
        return primitive.direction
    return primitive.basis


def _intervention_mode(
    mode: str,
) -> DirectionInterventionMode:
    """Map behavior trace primitive modes to intervention modes."""
    if mode == "subtract":
        return "subtract"
    if mode == "add":
        return "add"
    if mode == "subspace_remove":
        return "subspace_remove"
    if mode == "subspace_add":
        return "subspace_add"
    msg = f"mode {mode!r} is not an intervention mode"
    raise ValueError(msg)


def _forward_span_id(trace: Trace) -> str | None:
    """Return the forward span id for primitive artifacts."""
    for span in trace.spans:
        if span.name == "forward":
            return span.span_id
    return None


def _append_span_outputs(
    trace: Trace,
    span_id: str | None,
    artifact_ids: tuple[str, ...],
) -> tuple[TraceSpan, ...]:
    """Return spans with primitive artifacts added to the forward output list."""
    if span_id is None:
        return trace.spans
    spans: list[TraceSpan] = []
    for span in trace.spans:
        if span.span_id != span_id:
            spans.append(span)
            continue
        spans.append(
            TraceSpan(
                span_id=span.span_id,
                name=span.name,
                profile=span.profile,
                input_artifact_ids=span.input_artifact_ids,
                output_artifact_ids=span.output_artifact_ids + artifact_ids,
                metadata=dict(span.metadata),
            ),
        )
    return tuple(spans)


def _runtime_evidence_row(
    prompt_id: str,
    package: RuntimeReportEvidence,
) -> dict[str, JsonValue]:
    """Serialize one prompt runtime evidence package."""
    return {
        "prompt_id": prompt_id,
        "runtime": package.to_dict(),
    }


def _prompt_for_trace(prompt: BehaviorTracePromptConfig) -> str | None:
    """Return prompt text according to its redaction policy."""
    if prompt.redaction == "safe":
        return prompt.text
    if prompt.redaction == "redacted":
        return "[redacted prompt]"
    return None


def _output_for_trace(
    output: str,
    redaction: str,
    *,
    record_outputs: bool,
) -> str | None:
    """Return output text only when both config and prompt redaction allow it."""
    if record_outputs and redaction == "safe":
        return output
    return None


def _trace_path(
    output_dir: Path,
    config: BehaviorTraceConfig,
) -> Path:
    """Resolve the trace output path."""
    if config.output_trace is not None:
        return config.output_trace
    return output_dir / config.trace_filename


def _report_payload(
    trace: BehaviorTrace,
    config: BehaviorTraceConfig,
    trace_path: Path,
    *,
    config_path: str | Path,
    output_dir: Path,
    runtime_evidence: list[JsonValue] | None,
    runtime_profile_sweep: JsonValue | None,
) -> dict[str, JsonValue]:
    """Build a compact JSON report for the trace collection run."""
    reproducibility = reproducibility_payload(
        command=f"vauban {config_path}",
        config_path=config_path,
        output_dir=output_dir,
        data_refs=(str(trace_path),),
        artifact_hashes_value=artifact_hashes({
            "config": config_path,
            "trace": trace_path,
        }),
        scorers=tuple(config.scorers),
        generation={
            "max_tokens": config.max_tokens,
            "record_outputs": config.record_outputs,
            "n_prompts": len(config.prompts),
        },
    )
    payload: dict[str, JsonValue] = {
        "report_version": "behavior_trace_v1",
        "trace": trace.summary_dict(),
        "trace_path": str(trace_path),
        "reproducibility": reproducibility,
        "suite": {
            "name": config.suite_name,
            "description": config.suite_description,
            "version": config.suite_version,
            "source": config.suite_source,
            "safety_policy": config.safety_policy,
            "categories": list(trace.categories),
            "metric_names": list(trace.metric_names),
            "scorers": list(config.scorers),
            "metric_specs": [
                {
                    "name": metric.name,
                    "description": metric.description,
                    "polarity": metric.polarity,
                    "unit": metric.unit,
                    "family": metric.family,
                }
                for metric in config.metrics
            ],
        },
        "config": {
            "model_label": config.model_label,
            "max_tokens": config.max_tokens,
            "record_outputs": config.record_outputs,
            "n_prompts": len(config.prompts),
            "scorers": list(config.scorers),
        },
    }
    if runtime_evidence is not None:
        payload["runtime_evidence"] = {
            "enabled": True,
            "backend": config.runtime_backend,
            "collect_layers": list(config.collect_layers),
            "return_logprobs": config.return_logprobs,
            "profile_sweep_config": {
                "enabled": config.runtime_profile_sweep.enabled,
                "axis": config.runtime_profile_sweep.axis,
                "samples": config.runtime_profile_sweep.samples,
                "warmup": config.runtime_profile_sweep.warmup,
                "require_stable_artifacts": (
                    config.runtime_profile_sweep.require_stable_artifacts
                ),
            },
            "prompts": runtime_evidence,
            "profile_sweep": runtime_profile_sweep,
        }
    return payload
