# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Standalone behavior-report runner."""

from __future__ import annotations

import time

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import (
    ModeReport,
    finish_mode_run,
    write_mode_report,
)
from vauban.behavior import render_behavior_report_markdown


def _run_behavior_report_mode(context: EarlyModeContext) -> None:
    """Run standalone [behavior_report] mode and write report artifacts."""
    config = context.config
    report_cfg = config.behavior_report
    if report_cfg is None:
        msg = "[behavior_report] section is required for behavior report mode"
        raise ValueError(msg)

    report = report_cfg.report
    log(
        (
            "Behavior report"
            f" — title={report.title!r},"
            f" suite={report.suite.name!r},"
            f" metrics={len(report.metrics)}"
        ),
        verbose=config.verbose,
        elapsed=time.monotonic() - context.t0,
    )

    report_path = write_mode_report(
        config.output_dir,
        ModeReport(
            filename=report_cfg.json_filename,
            payload=report.to_dict(),
        ),
    )
    report_files = [str(report_path)]
    if report_cfg.markdown_report:
        markdown_path = config.output_dir / report_cfg.markdown_filename
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            render_behavior_report_markdown(report),
            encoding="utf-8",
        )
        report_files.append(str(markdown_path))

    finish_mode_run(
        context,
        "behavior_report",
        report_files,
        {
            "n_metrics": len(report.metrics),
            "n_metric_deltas": len(report.metric_deltas),
            "n_activation_findings": len(report.activation_findings),
            "n_claims": len(report.claims),
            "n_evidence_refs": len(report.evidence),
            "n_examples": len(report.examples),
            "n_reproduction_targets": len(report.reproduction_targets),
        },
    )
