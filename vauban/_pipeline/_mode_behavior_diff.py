# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Standalone behavior trace diff runner."""

from __future__ import annotations

import time
from typing import cast

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
    MetricPolarity,
    ThresholdSeverity,
    TransformationKind,
    behavior_threshold_summary,
    build_behavior_diff_result,
    evaluate_behavior_thresholds,
    load_behavior_trace,
    render_behavior_report_markdown,
)


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
        limitations=tuple(diff_cfg.limitations),
        recommendation=diff_cfg.recommendation,
        include_examples=diff_cfg.include_examples,
        max_examples=diff_cfg.max_examples,
        record_outputs=diff_cfg.record_outputs,
        command=f"vauban {context.config_path}",
        config_path=str(context.config_path),
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

    json_path = write_mode_report(
        config.output_dir,
        ModeReport(
            filename=diff_cfg.json_filename,
            payload={
                **result.to_dict(),
                "thresholds": [
                    threshold.to_dict() for threshold in threshold_results
                ],
                "threshold_summary": threshold_summary,
            },
        ),
    )
    report_files = [str(json_path)]
    if diff_cfg.markdown_report and result.report is not None:
        markdown_path = config.output_dir / diff_cfg.markdown_filename
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown = render_behavior_report_markdown(result.report)
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
