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
    MetricPolarity,
    TransformationKind,
    build_behavior_diff_result,
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
            payload=result.to_dict(),
        ),
    )
    report_files = [str(json_path)]
    if diff_cfg.markdown_report and result.report is not None:
        markdown_path = config.output_dir / diff_cfg.markdown_filename
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            render_behavior_report_markdown(result.report),
            encoding="utf-8",
        )
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
        },
    )
