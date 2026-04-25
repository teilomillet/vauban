# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Standalone behavior trace diff runner."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import (
    ModeReport,
    finish_mode_run,
    write_mode_report,
)
from vauban.behavior import (
    BehaviorMetricSpec,
    BehaviorThresholdResult,
    BehaviorThresholdSpec,
    BehaviorTrace,
    EvidenceRef,
    JsonValue,
    MetricPolarity,
    ThresholdSeverity,
    TransformationKind,
    artifact_hashes,
    behavior_threshold_summary,
    build_behavior_diff_result,
    evaluate_behavior_thresholds,
    load_behavior_trace,
    render_behavior_report_markdown,
    vauban_version,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from vauban.behavior import AccessLevel, ClaimStrength


@dataclass(frozen=True, slots=True)
class _RuntimeEvidenceSidecar:
    """Compact runtime evidence summary loaded from a behavior-trace report."""

    path: Path
    backend: str
    prompt_ids: tuple[str, ...]
    access_levels: tuple[str, ...]
    activation_layers: tuple[str, ...]
    logprobs_prompt_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize sidecar coverage for behavior-diff reports."""
        return {
            "path": str(self.path),
            "backend": self.backend,
            "n_prompts": len(self.prompt_ids),
            "prompt_ids": _json_string_list(self.prompt_ids),
            "access_levels": _json_string_list(self.access_levels),
            "activation_layers": _json_string_list(self.activation_layers),
            "n_logprobs_prompts": len(self.logprobs_prompt_ids),
            "logprobs_prompt_ids": _json_string_list(self.logprobs_prompt_ids),
        }


def _run_behavior_diff_mode(context: EarlyModeContext) -> None:
    """Run standalone [behavior_diff] mode and write report artifacts."""
    config = context.config
    diff_cfg = config.behavior_diff
    if diff_cfg is None:
        msg = "[behavior_diff] section is required for behavior diff mode"
        raise ValueError(msg)

    baseline_trace = load_behavior_trace(
        diff_cfg.baseline_trace,
        model_label=diff_cfg.baseline_label,
        model_path=diff_cfg.baseline_model_path,
        suite_name=diff_cfg.suite_name,
    )
    candidate_trace = load_behavior_trace(
        diff_cfg.candidate_trace,
        model_label=diff_cfg.candidate_label,
        model_path=diff_cfg.candidate_model_path,
        suite_name=diff_cfg.suite_name,
    )
    metric_specs = tuple(
        BehaviorMetricSpec(
            name=metric.name,
            description=metric.description
            or f"Behavior metric from trace field {metric.name!r}.",
            polarity=cast("MetricPolarity", metric.polarity),
            unit=metric.unit,
            family=metric.family,
        )
        for metric in diff_cfg.metrics
    )
    scorer_refs = _trace_scorers(baseline_trace, candidate_trace)
    runtime_diff = _load_runtime_evidence_diff(
        diff_cfg.baseline_report,
        diff_cfg.candidate_report,
    )
    artifact_refs = {
        "config": context.config_path,
        "baseline_trace": diff_cfg.baseline_trace,
        "candidate_trace": diff_cfg.candidate_trace,
    }
    if diff_cfg.baseline_report is not None:
        artifact_refs["baseline_report"] = diff_cfg.baseline_report
    if diff_cfg.candidate_report is not None:
        artifact_refs["candidate_report"] = diff_cfg.candidate_report
    artifact_hashes_value = artifact_hashes(artifact_refs)
    result = build_behavior_diff_result(
        baseline_trace,
        candidate_trace,
        title=diff_cfg.title,
        suite_name=diff_cfg.suite_name,
        suite_description=diff_cfg.suite_description,
        metric_specs=metric_specs,
        target_change=diff_cfg.target_change,
        suite_version=diff_cfg.suite_version,
        suite_source=diff_cfg.suite_source,
        safety_policy=diff_cfg.safety_policy,
        transformation_kind=cast(
            "TransformationKind",
            diff_cfg.transformation_kind,
        ),
        transformation_summary=diff_cfg.transformation_summary,
        access_level=cast("AccessLevel", diff_cfg.access_level),
        claim_strength=(
            cast("ClaimStrength", diff_cfg.claim_strength)
            if diff_cfg.claim_strength is not None
            else None
        ),
        limitations=tuple(diff_cfg.limitations),
        recommendation=diff_cfg.recommendation,
        include_examples=diff_cfg.include_examples,
        max_examples=diff_cfg.max_examples,
        record_outputs=diff_cfg.record_outputs,
        command=f"vauban {context.config_path}",
        config_path=str(context.config_path),
        output_dir=str(config.output_dir),
        tool_version=vauban_version(),
        artifact_hashes_value=artifact_hashes_value,
        scorer_refs=scorer_refs,
        generation={
            "include_examples": diff_cfg.include_examples,
            "max_examples": diff_cfg.max_examples,
            "record_outputs": diff_cfg.record_outputs,
            "markdown_report": diff_cfg.markdown_report,
        },
        extra_evidence=_runtime_sidecar_evidence_refs(
            diff_cfg.baseline_report,
            diff_cfg.candidate_report,
        ),
    )
    threshold_results = evaluate_behavior_thresholds(
        result.metric_deltas,
        tuple(
            BehaviorThresholdSpec(
                metric=threshold.metric,
                category=threshold.category,
                max_delta=threshold.max_delta,
                min_delta=threshold.min_delta,
                max_absolute_delta=threshold.max_absolute_delta,
                severity=cast("ThresholdSeverity", threshold.severity),
                description=threshold.description or None,
            )
            for threshold in diff_cfg.thresholds
        ),
    )
    threshold_summary = behavior_threshold_summary(threshold_results)
    threshold_failures = sum(
        1 for threshold in threshold_results
        if not threshold.passed and threshold.severity == "fail"
    )

    log(
        (
            "Behavior diff"
            f" — suite={result.suite.name!r},"
            f" baseline={baseline_trace.model_label!r},"
            f" candidate={candidate_trace.model_label!r},"
            f" deltas={len(result.metric_deltas)}"
        ),
        verbose=config.verbose,
        elapsed=time.monotonic() - context.t0,
    )

    payload = result.to_dict()
    payload.update({
        "thresholds": [
            threshold.to_dict() for threshold in threshold_results
        ],
        "threshold_summary": threshold_summary,
    })
    if result.report is not None and result.report.reproducibility is not None:
        payload["reproducibility"] = result.report.reproducibility.to_dict()
    if runtime_diff is not None:
        payload["runtime_evidence_diff"] = runtime_diff

    json_path = write_mode_report(
        config.output_dir,
        ModeReport(
            filename=diff_cfg.json_filename,
            payload=payload,
        ),
    )
    report_files = [str(json_path)]
    if diff_cfg.markdown_report and result.report is not None:
        markdown_path = config.output_dir / diff_cfg.markdown_filename
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown = render_behavior_report_markdown(result.report)
        markdown += _render_runtime_evidence_markdown(runtime_diff)
        markdown += _render_threshold_markdown(threshold_results)
        markdown_path.write_text(markdown, encoding="utf-8")
        report_files.append(str(markdown_path))

    finish_mode_run(
        context,
        "behavior_diff",
        report_files,
        {
            "n_baseline_observations": len(baseline_trace.observations),
            "n_candidate_observations": len(candidate_trace.observations),
            "n_metrics": len(result.metrics),
            "n_metric_deltas": len(result.metric_deltas),
            "n_examples": len(result.examples),
            "n_thresholds": len(threshold_results),
            "n_threshold_failures": threshold_failures,
        },
    )

    if threshold_failures > 0:
        msg = (
            "behavior_diff thresholds failed:"
            f" {threshold_failures} failing gate(s)"
        )
        raise ValueError(msg)


def _render_threshold_markdown(
    threshold_results: tuple[BehaviorThresholdResult, ...],
) -> str:
    """Render a Markdown section for behavior regression gates."""
    if not threshold_results:
        return ""
    lines = ["", "## Regression Gates", ""]
    for result in threshold_results:
        status = "PASS" if result.passed else result.severity.upper()
        category = result.category or "overall"
        delta = "n/a" if result.delta is None else f"{result.delta:.3f}"
        reason = f" - {result.reason}" if result.reason else ""
        lines.append(
            f"- {status}: {result.metric} / {category} delta={delta}{reason}",
        )
    lines.append("")
    return "\n".join(lines)


def _load_runtime_evidence_diff(
    baseline_report: Path | None,
    candidate_report: Path | None,
) -> dict[str, JsonValue] | None:
    """Load and compare optional behavior-trace runtime evidence sidecars."""
    if baseline_report is None or candidate_report is None:
        return None
    baseline = _load_runtime_evidence_sidecar(baseline_report)
    candidate = _load_runtime_evidence_sidecar(candidate_report)
    baseline_prompts = set(baseline.prompt_ids)
    candidate_prompts = set(candidate.prompt_ids)
    baseline_layers = set(baseline.activation_layers)
    candidate_layers = set(candidate.activation_layers)
    result: dict[str, JsonValue] = {}
    result["baseline"] = baseline.to_dict()
    result["candidate"] = candidate.to_dict()
    result["shared_prompt_ids"] = _json_string_list(
        sorted(baseline_prompts & candidate_prompts),
    )
    result["missing_in_baseline"] = _json_string_list(
        sorted(candidate_prompts - baseline_prompts),
    )
    result["missing_in_candidate"] = _json_string_list(
        sorted(baseline_prompts - candidate_prompts),
    )
    result["shared_activation_layers"] = _json_string_list(
        sorted(baseline_layers & candidate_layers),
    )
    result["baseline_only_activation_layers"] = _json_string_list(
        sorted(baseline_layers - candidate_layers),
    )
    result["candidate_only_activation_layers"] = _json_string_list(
        sorted(candidate_layers - baseline_layers),
    )
    result["limitations"] = _json_string_list([
        "Behavior diff consumed runtime sidecar summaries, not raw tensors.",
        (
            "This supports evidence-coverage checks, not activation-value"
            " causal claims."
        ),
    ])
    return result


def _load_runtime_evidence_sidecar(path: Path) -> _RuntimeEvidenceSidecar:
    """Load compact runtime evidence coverage from one trace report sidecar."""
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    report = _object_dict(raw, str(path))
    runtime_raw = report.get("runtime_evidence")
    runtime = _object_dict(runtime_raw, f"{path}:runtime_evidence")
    prompts_raw = runtime.get("prompts")
    if not isinstance(prompts_raw, list):
        msg = f"{path}:runtime_evidence.prompts must be a list"
        raise TypeError(msg)

    prompt_ids: list[str] = []
    access_levels: dict[str, None] = {}
    activation_layers: dict[str, None] = {}
    logprobs_prompt_ids: list[str] = []
    for index, prompt_raw in enumerate(prompts_raw):
        prompt = _object_dict(prompt_raw, f"{path}:runtime_evidence.prompts[{index}]")
        prompt_id = _optional_string(prompt.get("prompt_id"))
        if prompt_id is None:
            msg = f"{path}:runtime_evidence.prompts[{index}].prompt_id is required"
            raise ValueError(msg)
        prompt_ids.append(prompt_id)

        runtime_payload = _object_dict(
            prompt.get("runtime"),
            f"{path}:runtime_evidence.prompts[{index}].runtime",
        )
        access = _object_dict(
            runtime_payload.get("access"),
            f"{path}:runtime_evidence.prompts[{index}].runtime.access",
        )
        level = _optional_string(access.get("level"))
        if level is not None:
            access_levels[level] = None

        trace = _object_dict(
            runtime_payload.get("trace"),
            f"{path}:runtime_evidence.prompts[{index}].runtime.trace",
        )
        activation_shapes = trace.get("activation_shapes")
        if isinstance(activation_shapes, dict):
            for layer in activation_shapes:
                activation_layers[str(layer)] = None
        if trace.get("logprobs_shape") is not None:
            logprobs_prompt_ids.append(prompt_id)

    return _RuntimeEvidenceSidecar(
        path=path,
        backend=_optional_string(runtime.get("backend")) or "unknown",
        prompt_ids=tuple(prompt_ids),
        access_levels=tuple(access_levels),
        activation_layers=tuple(sorted(activation_layers)),
        logprobs_prompt_ids=tuple(logprobs_prompt_ids),
    )


def _runtime_sidecar_evidence_refs(
    baseline_report: Path | None,
    candidate_report: Path | None,
) -> tuple[EvidenceRef, ...]:
    """Return behavior-report evidence refs for optional runtime sidecars."""
    if baseline_report is None or candidate_report is None:
        return ()
    return (
        EvidenceRef(
            evidence_id="baseline_runtime_report",
            kind="run_report",
            path_or_url=str(baseline_report),
            description="Baseline behavior trace runtime-evidence sidecar.",
        ),
        EvidenceRef(
            evidence_id="candidate_runtime_report",
            kind="run_report",
            path_or_url=str(candidate_report),
            description="Candidate behavior trace runtime-evidence sidecar.",
        ),
    )


def _render_runtime_evidence_markdown(
    runtime_diff: dict[str, JsonValue] | None,
) -> str:
    """Render optional runtime-evidence sidecar coverage."""
    if runtime_diff is None:
        return ""
    baseline = _json_object_from_value(runtime_diff["baseline"], "baseline")
    candidate = _json_object_from_value(runtime_diff["candidate"], "candidate")
    shared_layers = _string_list_from_value(
        runtime_diff["shared_activation_layers"],
        "shared_activation_layers",
    )
    shared_prompts = _string_list_from_value(
        runtime_diff["shared_prompt_ids"],
        "shared_prompt_ids",
    )
    lines = ["", "## Runtime Evidence Sidecars", ""]
    lines.append(f"- Baseline backend: {baseline.get('backend', 'unknown')}")
    lines.append(f"- Candidate backend: {candidate.get('backend', 'unknown')}")
    lines.append(f"- Shared prompts with runtime evidence: {len(shared_prompts)}")
    lines.append(
        "- Shared activation layers: "
        f"{', '.join(shared_layers) if shared_layers else 'none'}",
    )
    lines.append(
        "- Limitation: sidecars contain runtime summaries, not raw activation tensors.",
    )
    lines.append("")
    return "\n".join(lines)


def _object_dict(raw: object, section: str) -> dict[str, object]:
    """Validate a JSON object and return string-keyed items."""
    if not isinstance(raw, dict):
        msg = f"{section} must be an object"
        raise TypeError(msg)
    result: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            msg = f"{section} keys must be strings"
            raise TypeError(msg)
        result[key] = value
    return result


def _json_string_list(values: Iterable[str]) -> list[JsonValue]:
    """Return string values as an explicitly typed JSON list."""
    result: list[JsonValue] = []
    for value in values:
        result.append(value)
    return result


def _optional_string(raw: object) -> str | None:
    """Return a string JSON field when present."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    msg = f"expected string or null, got {type(raw).__name__}"
    raise TypeError(msg)


def _json_object_from_value(
    value: JsonValue,
    field: str,
) -> dict[str, JsonValue]:
    """Return a nested JSON object from an already typed payload."""
    if not isinstance(value, dict):
        msg = f"runtime_evidence_diff.{field} must be an object"
        raise TypeError(msg)
    return value


def _string_list_from_value(
    value: JsonValue,
    field: str,
) -> list[str]:
    """Return a string list from an already typed payload."""
    if not isinstance(value, list):
        msg = f"runtime_evidence_diff.{field} must be a list"
        raise TypeError(msg)
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            msg = f"runtime_evidence_diff.{field} elements must be strings"
            raise TypeError(msg)
        result.append(item)
    return result


def _trace_scorers(*traces: BehaviorTrace) -> tuple[str, ...]:
    """Return scorer names recorded in trace observation metadata."""
    scorers: dict[str, None] = {}
    for trace in traces:
        for observation in trace.observations:
            raw = observation.metadata.get("scorers")
            for scorer in _scorer_names_from_metadata(raw):
                scorers[scorer] = None
    return tuple(scorers)


def _scorer_names_from_metadata(raw: JsonValue | None) -> tuple[str, ...]:
    """Parse scorer metadata from one observation."""
    if isinstance(raw, str):
        return (raw,)
    if not isinstance(raw, list):
        return ()
    names: list[str] = []
    for item in raw:
        if isinstance(item, str):
            names.append(item)
    return tuple(names)
