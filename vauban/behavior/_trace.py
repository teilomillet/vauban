# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Behavior trace JSONL I/O and trace-diff report construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from vauban.behavior._primitives import (
    AccessLevel,
    BehaviorDiffResult,
    BehaviorExample,
    BehaviorMetric,
    BehaviorMetricDelta,
    BehaviorMetricSpec,
    BehaviorObservation,
    BehaviorReport,
    BehaviorSuiteRef,
    BehaviorTrace,
    ClaimStrength,
    EvidenceRef,
    ExampleRedaction,
    ExpectedBehavior,
    JsonValue,
    ReportModelRef,
    ReproducibilityInfo,
    TransformationKind,
    TransformationRef,
    access_policy_for_level,
    compare_behavior_metrics,
)

_EXPECTED_BEHAVIOR_CHOICES: tuple[ExpectedBehavior, ...] = (
    "refuse",
    "comply",
    "express_uncertainty",
    "ask_clarifying_question",
    "defer",
    "unknown",
)
_REDACTION_CHOICES: tuple[ExampleRedaction, ...] = (
    "safe",
    "redacted",
    "omitted",
)


def load_behavior_trace(
    path: str | Path,
    *,
    model_label: str,
    trace_id: str | None = None,
    model_path: str | None = None,
    suite_name: str | None = None,
) -> BehaviorTrace:
    """Load a behavior trace from JSONL observations."""
    path_obj = Path(path)
    observations: list[BehaviorObservation] = []
    with path_obj.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            raw_obj: object = json.loads(stripped)
            raw = _require_mapping(path_obj, line_number, raw_obj)
            observations.append(
                _parse_observation(
                    raw,
                    default_model_label=model_label,
                    line_number=line_number,
                ),
            )

    inferred_trace_id = trace_id or path_obj.stem
    return BehaviorTrace(
        trace_id=inferred_trace_id,
        model_label=model_label,
        model_path=model_path,
        suite_name=suite_name,
        source_path=str(path_obj),
        observations=tuple(observations),
    )


def write_behavior_trace(path: str | Path, trace: BehaviorTrace) -> Path:
    """Write a behavior trace as one JSON object per line."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(observation.to_dict(), sort_keys=True)
        for observation in trace.observations
    ]
    path_obj.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path_obj


def infer_behavior_metric_specs(
    baseline_trace: BehaviorTrace,
    candidate_trace: BehaviorTrace,
) -> tuple[BehaviorMetricSpec, ...]:
    """Infer metric specs from observed trace metric names."""
    names = tuple(
        sorted(set(baseline_trace.metric_names) | set(candidate_trace.metric_names)),
    )
    return tuple(
        BehaviorMetricSpec(
            name=name,
            description=f"Observed metric inferred from behavior traces: {name}.",
            polarity="neutral",
            unit=_default_unit_for_metric(name),
            family="behavior",
        )
        for name in names
    )


def build_behavior_diff_result(
    baseline_trace: BehaviorTrace,
    candidate_trace: BehaviorTrace,
    *,
    title: str,
    suite_name: str,
    suite_description: str,
    metric_specs: tuple[BehaviorMetricSpec, ...],
    target_change: str | None = None,
    suite_version: str | None = None,
    suite_source: str | None = None,
    safety_policy: str = "aggregate_or_redacted_examples",
    transformation_kind: TransformationKind = "evaluation_only",
    transformation_summary: str | None = None,
    access_level: AccessLevel = "black_box",
    claim_strength: ClaimStrength | None = None,
    limitations: tuple[str, ...] = (),
    recommendation: str | None = None,
    include_examples: bool = True,
    max_examples: int = 3,
    record_outputs: bool = False,
    command: str | None = None,
    config_path: str | None = None,
    output_dir: str | None = None,
    tool_version: str | None = None,
    artifact_hashes_value: dict[str, str] | None = None,
    scorer_refs: tuple[str, ...] = (),
    generation: dict[str, JsonValue] | None = None,
    extra_evidence: tuple[EvidenceRef, ...] = (),
) -> BehaviorDiffResult:
    """Build a behavior diff result and embedded behavior report."""
    effective_metric_specs = metric_specs or infer_behavior_metric_specs(
        baseline_trace,
        candidate_trace,
    )
    metrics = (
        aggregate_trace_metrics(baseline_trace, effective_metric_specs)
        + aggregate_trace_metrics(candidate_trace, effective_metric_specs)
    )
    metric_deltas = compare_behavior_metrics(
        tuple(
            metric
            for metric in metrics
            if metric.model_label == baseline_trace.model_label
        ),
        tuple(
            metric
            for metric in metrics
            if metric.model_label == candidate_trace.model_label
        ),
    )
    findings = _findings_from_deltas(metric_deltas)
    examples = _examples_from_traces(
        baseline_trace,
        candidate_trace,
        include_examples=include_examples,
        max_examples=max_examples,
        record_outputs=record_outputs,
    )
    suite_ref = BehaviorSuiteRef(
        name=suite_name,
        description=suite_description,
        categories=_suite_categories(baseline_trace, candidate_trace),
        metrics=tuple(spec.name for spec in effective_metric_specs),
        version=suite_version,
        source=suite_source,
        safety_policy=safety_policy,
    )
    report = _build_report(
        baseline_trace,
        candidate_trace,
        title=title,
        suite=suite_ref,
        target_change=target_change,
        transformation_kind=transformation_kind,
        transformation_summary=transformation_summary,
        access_level=access_level,
        claim_strength=claim_strength,
        metrics=metrics,
        metric_deltas=metric_deltas,
        findings=findings,
        examples=examples,
        limitations=limitations,
        recommendation=recommendation,
        command=command,
        config_path=config_path,
        output_dir=output_dir,
        tool_version=tool_version,
        artifact_hashes_value=artifact_hashes_value or {},
        scorer_refs=scorer_refs,
        generation=generation or {},
        extra_evidence=extra_evidence,
    )
    return BehaviorDiffResult(
        title=title,
        baseline_trace=baseline_trace,
        candidate_trace=candidate_trace,
        suite=suite_ref,
        metrics=metrics,
        metric_deltas=metric_deltas,
        findings=findings,
        examples=examples,
        target_change=target_change,
        limitations=limitations,
        recommendation=recommendation,
        report=report,
    )


def aggregate_trace_metrics(
    trace: BehaviorTrace,
    metric_specs: tuple[BehaviorMetricSpec, ...],
) -> tuple[BehaviorMetric, ...]:
    """Aggregate per-observation metrics into per-category and overall means."""
    metrics: list[BehaviorMetric] = []
    for spec in metric_specs:
        for category in (*trace.categories, "overall"):
            values = _metric_values_for_category(trace, spec.name, category)
            if not values:
                continue
            metrics.append(
                BehaviorMetric(
                    name=spec.name,
                    value=sum(values) / len(values),
                    model_label=trace.model_label,
                    category=category,
                    unit=spec.unit,
                    polarity=spec.polarity,
                    family=spec.family,
                    sample_size=len(values),
                    notes=(
                        "Mean over behavior-trace observations"
                        f" for category {category!r}."
                    ),
                ),
            )
    return tuple(metrics)


def _build_report(
    baseline_trace: BehaviorTrace,
    candidate_trace: BehaviorTrace,
    *,
    title: str,
    suite: BehaviorSuiteRef,
    target_change: str | None,
    transformation_kind: TransformationKind,
    transformation_summary: str | None,
    access_level: AccessLevel,
    claim_strength: ClaimStrength | None,
    metrics: tuple[BehaviorMetric, ...],
    metric_deltas: tuple[BehaviorMetricDelta, ...],
    findings: tuple[str, ...],
    examples: tuple[BehaviorExample, ...],
    limitations: tuple[str, ...],
    recommendation: str | None,
    command: str | None,
    config_path: str | None,
    output_dir: str | None,
    tool_version: str | None,
    artifact_hashes_value: dict[str, str],
    scorer_refs: tuple[str, ...],
    generation: dict[str, JsonValue],
    extra_evidence: tuple[EvidenceRef, ...],
) -> BehaviorReport:
    """Build the standard Model Behavior Change Report for a trace diff."""
    summary = transformation_summary or (
        f"Compared behavior traces for {baseline_trace.model_label}"
        f" and {candidate_trace.model_label}."
    )
    evidence = (
        EvidenceRef(
            evidence_id="baseline_trace",
            kind="trace",
            path_or_url=baseline_trace.source_path,
            description="Baseline behavior trace JSONL.",
        ),
        EvidenceRef(
            evidence_id="candidate_trace",
            kind="trace",
            path_or_url=candidate_trace.source_path,
            description="Candidate behavior trace JSONL.",
        ),
        *extra_evidence,
    )
    access = access_policy_for_level(
        access_level,
        claim_strength=claim_strength,
        available_evidence=("paired_outputs", "behavior_traces"),
        missing_evidence=("weights", "activations", "training_data"),
        notes=(
            "Trace diffs support behavioral claims, not internal causal claims.",
        ),
    )
    return BehaviorReport(
        title=title,
        baseline=ReportModelRef(
            label=baseline_trace.model_label,
            model_path=baseline_trace.model_path or baseline_trace.model_label,
            role="baseline",
        ),
        candidate=ReportModelRef(
            label=candidate_trace.model_label,
            model_path=candidate_trace.model_path or candidate_trace.model_label,
            role="candidate",
        ),
        suite=suite,
        target_change=target_change,
        transformation=TransformationRef(
            kind=transformation_kind,
            summary=summary,
            before=baseline_trace.model_label,
            after=candidate_trace.model_label,
            method="behavior_trace_diff",
        ),
        access=access,
        evidence=evidence,
        findings=findings,
        metrics=metrics,
        metric_deltas=metric_deltas,
        examples=examples,
        claims=(),
        limitations=limitations,
        recommendation=recommendation,
        reproducibility=(
            ReproducibilityInfo(
                command=command,
                config_path=config_path,
                tool_version=tool_version,
                data_refs=(
                    baseline_trace.source_path or baseline_trace.trace_id,
                    candidate_trace.source_path or candidate_trace.trace_id,
                ),
                output_dir=output_dir,
                artifact_hashes=artifact_hashes_value,
                scorers=scorer_refs,
                generation=generation,
            )
            if command is not None
            else None
        ),
    )


def _parse_observation(
    raw: dict[str, object],
    *,
    default_model_label: str,
    line_number: int,
) -> BehaviorObservation:
    """Parse one JSONL observation object."""
    prompt_id = _string_from_keys(
        raw,
        ("prompt_id", "id"),
        default=f"prompt-{line_number:04d}",
    )
    observation_id = _string_from_keys(
        raw,
        ("observation_id", "id"),
        default=f"{default_model_label}:{prompt_id}",
    )
    model_label = _string_from_keys(
        raw,
        ("model_label",),
        default=default_model_label,
    )
    category = _string_from_keys(raw, ("category",), default="default")
    expected_behavior = _literal_from_keys(
        raw,
        ("expected_behavior",),
        _EXPECTED_BEHAVIOR_CHOICES,
        default="unknown",
    )
    redaction = _literal_from_keys(
        raw,
        ("redaction",),
        _REDACTION_CHOICES,
        default="redacted",
    )
    return BehaviorObservation(
        observation_id=observation_id,
        model_label=model_label,
        prompt_id=prompt_id,
        category=category,
        prompt=_optional_string_from_keys(raw, ("prompt", "text")),
        output_text=_optional_string_from_keys(raw, ("output_text", "response")),
        expected_behavior=expected_behavior,
        refused=_optional_bool_from_keys(raw, ("refused",)),
        metrics=_metrics_from_raw(raw.get("metrics")),
        redaction=redaction,
        metadata=_metadata_from_raw(raw.get("metadata")),
    )


def _metric_values_for_category(
    trace: BehaviorTrace,
    metric_name: str,
    category: str,
) -> list[float]:
    """Return metric values for a category or the synthetic overall category."""
    values: list[float] = []
    for observation in trace.observations:
        if category != "overall" and observation.category != category:
            continue
        value = observation.metric_values().get(metric_name)
        if value is not None:
            values.append(value)
    return values


def _findings_from_deltas(
    deltas: tuple[BehaviorMetricDelta, ...],
) -> tuple[str, ...]:
    """Render concise findings from the largest behavior deltas."""
    if not deltas:
        return ("No matched behavior metrics were available for comparison.",)

    sorted_deltas = tuple(
        sorted(deltas, key=lambda delta: abs(delta.delta), reverse=True),
    )
    findings: list[str] = []
    for delta in sorted_deltas[:5]:
        sign = "+" if delta.delta >= 0.0 else ""
        category = delta.category or "overall"
        findings.append(
            (
                f"{delta.name} changed by {sign}{delta.delta:.3f}"
                f" in {category}"
                f" ({delta.baseline_label}={delta.value_baseline:.3f},"
                f" {delta.candidate_label}={delta.value_candidate:.3f},"
                f" quality={delta.quality})."
            ),
        )
    return tuple(findings)


def _examples_from_traces(
    baseline_trace: BehaviorTrace,
    candidate_trace: BehaviorTrace,
    *,
    include_examples: bool,
    max_examples: int,
    record_outputs: bool,
) -> tuple[BehaviorExample, ...]:
    """Build safe/redacted representative examples from paired prompt IDs."""
    if not include_examples or max_examples <= 0:
        return ()

    candidate_by_prompt = {
        observation.prompt_id: observation
        for observation in candidate_trace.observations
    }
    examples: list[BehaviorExample] = []
    for baseline in baseline_trace.observations:
        candidate = candidate_by_prompt.get(baseline.prompt_id)
        if candidate is None:
            continue
        redaction = _combined_redaction(baseline.redaction, candidate.redaction)
        examples.append(
            BehaviorExample(
                example_id=baseline.prompt_id,
                category=baseline.category,
                prompt=_render_prompt(baseline.prompt, redaction),
                baseline_response=_render_response(
                    baseline.output_text,
                    redaction,
                    record_outputs=record_outputs,
                ),
                candidate_response=_render_response(
                    candidate.output_text,
                    redaction,
                    record_outputs=record_outputs,
                ),
                redaction=redaction,
                note="Trace-derived representative example.",
            ),
        )
        if len(examples) >= max_examples:
            break
    return tuple(examples)


def _suite_categories(
    baseline_trace: BehaviorTrace,
    candidate_trace: BehaviorTrace,
) -> tuple[str, ...]:
    """Return sorted unique categories across both traces."""
    return tuple(
        sorted(set(baseline_trace.categories) | set(candidate_trace.categories)),
    )


def _require_mapping(
    path: Path,
    line_number: int,
    raw: object,
) -> dict[str, object]:
    """Require one JSONL row to decode to an object with string keys."""
    if not isinstance(raw, dict):
        msg = f"{path}:{line_number}: JSONL row must be an object"
        raise TypeError(msg)
    parsed: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            msg = f"{path}:{line_number}: JSON object keys must be strings"
            raise TypeError(msg)
        parsed[key] = value
    return parsed


def _string_from_keys(
    raw: dict[str, object],
    keys: tuple[str, ...],
    *,
    default: str,
) -> str:
    """Read the first present string key from a raw observation."""
    value = _optional_string_from_keys(raw, keys)
    return default if value is None else value


def _optional_string_from_keys(
    raw: dict[str, object],
    keys: tuple[str, ...],
) -> str | None:
    """Read the first present string key from a raw observation."""
    for key in keys:
        value = raw.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            msg = f"observation field {key!r} must be a string"
            raise TypeError(msg)
        return value
    return None


def _optional_bool_from_keys(
    raw: dict[str, object],
    keys: tuple[str, ...],
) -> bool | None:
    """Read the first present bool key from a raw observation."""
    for key in keys:
        value = raw.get(key)
        if value is None:
            continue
        if not isinstance(value, bool):
            msg = f"observation field {key!r} must be a boolean"
            raise TypeError(msg)
        return value
    return None


def _literal_from_keys[LiteralT: str](
    raw: dict[str, object],
    keys: tuple[str, ...],
    choices: tuple[LiteralT, ...],
    *,
    default: LiteralT,
) -> LiteralT:
    """Read and validate a literal string from raw observation data."""
    value = _optional_string_from_keys(raw, keys)
    if value is None:
        return default
    if value not in choices:
        msg = f"observation literal must be one of {choices!r}, got {value!r}"
        raise ValueError(msg)
    return cast("LiteralT", value)


def _metrics_from_raw(raw: object) -> dict[str, float]:
    """Parse an optional metrics object from a JSONL observation."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        msg = "observation metrics must be an object"
        raise TypeError(msg)
    metrics: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            msg = "observation metric keys must be strings"
            raise TypeError(msg)
        if isinstance(value, bool) or not isinstance(value, int | float):
            msg = f"observation metric {key!r} must be numeric"
            raise TypeError(msg)
        metrics[key] = float(value)
    return metrics


def _metadata_from_raw(raw: object) -> dict[str, JsonValue]:
    """Parse optional JSON-compatible metadata."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        msg = "observation metadata must be an object"
        raise TypeError(msg)
    metadata: dict[str, JsonValue] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            msg = "observation metadata keys must be strings"
            raise TypeError(msg)
        metadata[key] = _json_value(value)
    return metadata


def _json_value(raw: object) -> JsonValue:
    """Validate and return a JSON-compatible value."""
    if raw is None or isinstance(raw, str | int | float | bool):
        return raw
    if isinstance(raw, list):
        return [_json_value(item) for item in raw]
    if isinstance(raw, dict):
        value: dict[str, JsonValue] = {}
        for key, item in raw.items():
            if not isinstance(key, str):
                msg = "JSON-compatible object keys must be strings"
                raise TypeError(msg)
            value[key] = _json_value(item)
        return value
    msg = f"metadata value is not JSON-compatible: {type(raw).__name__}"
    raise TypeError(msg)


def _default_unit_for_metric(name: str) -> str:
    """Infer a useful unit for a metric name."""
    if name.endswith("_rate") or name in {"refusal_rate"}:
        return "ratio"
    if name.endswith("_chars") or name.endswith("_count"):
        return "count"
    return "score"


def _combined_redaction(
    baseline: ExampleRedaction,
    candidate: ExampleRedaction,
) -> ExampleRedaction:
    """Return the strictest redaction level across a paired example."""
    if "omitted" in {baseline, candidate}:
        return "omitted"
    if "redacted" in {baseline, candidate}:
        return "redacted"
    return "safe"


def _render_prompt(prompt: str | None, redaction: ExampleRedaction) -> str:
    """Render a prompt without violating the observation redaction policy."""
    if redaction == "omitted":
        return "[prompt omitted]"
    if redaction == "redacted":
        return "[redacted prompt]"
    return prompt or "[prompt unavailable]"


def _render_response(
    response: str | None,
    redaction: ExampleRedaction,
    *,
    record_outputs: bool,
) -> str | None:
    """Render a response only when output recording and redaction allow it."""
    if not record_outputs:
        return None
    if redaction != "safe":
        return None
    return response
